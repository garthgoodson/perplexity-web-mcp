"""ReAct-style tool calling support for Perplexity models.

Since Perplexity's web interface doesn't support native tool calling,
we implement it via ReAct (Reasoning + Acting) prompt engineering:

1. Inject tool definitions with ReAct format instructions
2. Model outputs Thought/Action/Observation sequences
3. Parse Action lines to extract tool calls
4. Convert to Anthropic tool_use format

ReAct is a well-established prompting pattern that works across many LLMs
because it's present in training data (LangChain, agent frameworks, etc.)

Reference: https://react-lm.github.io/
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any


# =============================================================================
# ReAct Format Constants
# =============================================================================

# ReAct uses natural language patterns instead of XML
ACTION_PREFIX = "Action:"
ACTION_INPUT_PREFIX = "Action Input:"
THOUGHT_PREFIX = "Thought:"
FINAL_ANSWER_PREFIX = "Final Answer:"

# Minimal instruction - just list tools, let model respond naturally
REACT_INSTRUCTIONS = """Available tools you can use:
{tool_definitions}

To use a tool, say which one and what parameters. Otherwise just respond normally."""

# Condensed behavioral instructions  
CLAUDE_CODE_BEHAVIOR = ""


# =============================================================================
# Tool Definition Formatting
# =============================================================================

def format_tool_schema(tool: dict[str, Any]) -> str:
    """Format a single tool definition for the prompt.
    
    Args:
        tool: Anthropic-format tool definition with name, description, input_schema
        
    Returns:
        Human-readable tool description
    """
    name = tool.get("name", "unknown")
    description = tool.get("description", "No description")
    input_schema = tool.get("input_schema", {})
    
    # Extract parameters
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    
    params = []
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "any")
        param_desc = param_info.get("description", "")
        req_marker = " (required)" if param_name in required else ""
        params.append(f"    {param_name}: {param_type}{req_marker} - {param_desc}")
    
    params_str = "\n".join(params) if params else "    (no parameters)"
    
    return f"""- {name}: {description}
  Parameters:
{params_str}"""


def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Format all tool definitions for injection into the prompt.
    
    Args:
        tools: List of Anthropic-format tool definitions
        
    Returns:
        Complete tool instructions to inject into prompt
    """
    if not tools:
        return ""
    
    tool_definitions = "\n\n".join(format_tool_schema(t) for t in tools)
    return REACT_INSTRUCTIONS.format(tool_definitions=tool_definitions)


# =============================================================================
# Tool Call Parsing (ReAct Format)
# =============================================================================

def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse tool calls from ReAct-format model response.
    
    Looks for patterns like:
        Thought: ...
        Action: tool_name
        Action Input: {"param": "value"}
    
    Args:
        text: Model response potentially containing ReAct tool calls
        
    Returns:
        Tuple of (cleaned_text, list_of_tool_calls)
        where cleaned_text has ReAct sequences removed/cleaned and
        tool_calls are dicts with 'name' and 'input' keys
    """
    tool_calls = []
    
    # Pattern to match Action + Action Input (multiline)
    # This handles various formatting styles the model might use
    action_pattern = re.compile(
        r'(?:^|\n)\s*Action:\s*([^\n]+)\s*\n\s*Action Input:\s*(.+?)(?=\n\s*(?:Thought:|Action:|Final Answer:|Observation:)|$)',
        re.DOTALL | re.MULTILINE
    )
    
    for match in action_pattern.finditer(text):
        tool_name = match.group(1).strip()
        input_str = match.group(2).strip()
        
        # Try to parse the input as JSON
        try:
            # Handle case where input might be on multiple lines or have extra whitespace
            input_str = input_str.split('\n')[0].strip() if '\n' in input_str else input_str
            
            # Try parsing as JSON
            tool_input = json.loads(input_str)
        except json.JSONDecodeError:
            # Try to extract JSON from the string
            json_match = re.search(r'\{[^}]*\}', input_str)
            if json_match:
                try:
                    tool_input = json.loads(json_match.group())
                except json.JSONDecodeError:
                    tool_input = {}
            else:
                # If input looks like a simple string, treat it as a query param
                tool_input = {"query": input_str} if input_str else {}
        
        tool_calls.append({
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": tool_name,
            "input": tool_input,
        })
    
    # Clean text: extract Final Answer if present, otherwise return everything
    # before the first Action
    final_answer_match = re.search(r'Final Answer:\s*(.+?)(?=\n\s*(?:Thought:|Action:)|$)', text, re.DOTALL)
    if final_answer_match:
        cleaned_text = final_answer_match.group(1).strip()
    elif tool_calls:
        # If there are tool calls but no final answer, extract any text before the ReAct sequence
        first_thought = re.search(r'^(.*?)(?:Thought:|Action:)', text, re.DOTALL)
        cleaned_text = first_thought.group(1).strip() if first_thought else ""
    else:
        # No tool calls and no Final Answer - return original text
        cleaned_text = text.strip()
    
    return cleaned_text, tool_calls


# =============================================================================
# Anthropic Format Conversion
# =============================================================================

def create_tool_use_block(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Create an Anthropic-format tool_use content block.
    
    Args:
        tool_call: Dict with id, name, and input
        
    Returns:
        Anthropic tool_use content block
    """
    return {
        "type": "tool_use",
        "id": tool_call["id"],
        "name": tool_call["name"],
        "input": tool_call["input"],
    }


def convert_response_with_tools(
    text: str,
    tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert parsed response to Anthropic content blocks.
    
    Args:
        text: Cleaned text (Final Answer or pre-Action text)
        tool_calls: List of parsed tool calls
        
    Returns:
        List of Anthropic content blocks (text and tool_use)
    """
    content = []
    
    # Add text block if there's any text
    if text:
        content.append({"type": "text", "text": text})
    
    # Add tool_use blocks
    for tool_call in tool_calls:
        content.append(create_tool_use_block(tool_call))
    
    return content


# =============================================================================
# Query Building
# =============================================================================

def build_query_with_tools(
    user_message: str,
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    include_behavior: bool = True,
) -> str:
    """Build a complete query with ReAct tool instructions.
    
    Args:
        user_message: The user's actual message/query
        tools: Optional list of tool definitions
        system_prompt: Optional system prompt (will be distilled)
        include_behavior: Whether to include behavioral instructions
        
    Returns:
        Complete query string to send to Perplexity
    """
    parts = []
    
    # Add condensed behavioral instructions
    if include_behavior:
        parts.append(CLAUDE_CODE_BEHAVIOR)
    
    # Add tool instructions if tools are provided
    if tools:
        parts.append(format_tools_for_prompt(tools))
    
    # Add the actual user message
    parts.append(f"User: {user_message}")
    
    return "\n\n".join(parts)


# =============================================================================
# Streaming Support
# =============================================================================

class ToolCallStreamParser:
    """Stateful parser for detecting ReAct tool calls in streaming responses."""
    
    def __init__(self):
        self.buffer = ""
        self.in_action = False
        self.action_name = ""
        self.pending_tool_calls: list[dict[str, Any]] = []
        self.emitted_text = ""
        self.found_final_answer = False
    
    def feed(self, chunk: str) -> tuple[str, list[dict[str, Any]]]:
        """Feed a chunk and return any complete text/tool calls.
        
        Args:
            chunk: New text chunk from stream
            
        Returns:
            Tuple of (text_to_emit, completed_tool_calls)
        """
        self.buffer += chunk
        text_to_emit = ""
        completed_tools = []
        
        # Check for Final Answer
        if not self.found_final_answer:
            final_match = re.search(r'Final Answer:\s*', self.buffer)
            if final_match:
                self.found_final_answer = True
                # Emit text after Final Answer:
                after_final = self.buffer[final_match.end():]
                # Don't emit anything before Final Answer that we haven't emitted
                text_to_emit = after_final
                self.buffer = after_final
                return text_to_emit, completed_tools
        
        if self.found_final_answer:
            # After Final Answer, emit everything
            text_to_emit = self.buffer
            self.buffer = ""
            return text_to_emit, completed_tools
        
        # Look for Action patterns
        action_match = re.search(r'Action:\s*([^\n]+)\s*\nAction Input:\s*(\{[^}]*\})', self.buffer)
        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                tool_input = json.loads(action_match.group(2))
            except json.JSONDecodeError:
                tool_input = {}
            
            completed_tools.append({
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": tool_name,
                "input": tool_input,
            })
            
            # Clear buffer up to end of action
            self.buffer = self.buffer[action_match.end():]
        
        # Check if we're in a potential Action/Thought sequence
        if re.search(r'(?:^|\n)\s*(?:Thought:|Action:)', self.buffer):
            # Don't emit - we're in a ReAct sequence
            pass
        else:
            # Emit buffered text if it's safe (not starting a ReAct sequence)
            safe_len = max(0, len(self.buffer) - 20)  # Keep last 20 chars in case of partial pattern
            if safe_len > 0:
                text_to_emit = self.buffer[:safe_len]
                self.buffer = self.buffer[safe_len:]
        
        return text_to_emit, completed_tools
    
    def finish(self) -> tuple[str, list[dict[str, Any]]]:
        """Flush any remaining buffer at end of stream.
        
        Returns:
            Tuple of (remaining_text, any_remaining_tool_calls)
        """
        remaining_text = ""
        remaining_tools = []
        
        # Try to parse any remaining complete actions
        action_match = re.search(r'Action:\s*([^\n]+)\s*\nAction Input:\s*(\{[^}]*\})', self.buffer)
        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                tool_input = json.loads(action_match.group(2))
            except json.JSONDecodeError:
                tool_input = {}
            
            remaining_tools.append({
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": tool_name,
                "input": tool_input,
            })
            self.buffer = self.buffer[action_match.end():]
        
        # Check for Final Answer in remaining buffer
        final_match = re.search(r'Final Answer:\s*(.+?)$', self.buffer, re.DOTALL)
        if final_match:
            remaining_text = final_match.group(1).strip()
        elif not remaining_tools:
            # No tools and no final answer - emit whatever's left
            # But filter out ReAct artifacts
            remaining_text = re.sub(r'(?:^|\n)\s*Thought:.*?(?=\n|$)', '', self.buffer)
            remaining_text = remaining_text.strip()
        
        self.buffer = ""
        return remaining_text, remaining_tools
