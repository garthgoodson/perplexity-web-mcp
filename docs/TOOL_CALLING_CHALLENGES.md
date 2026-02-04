# Tool Calling Challenges

This document captures our attempts to enable tool calling through Perplexity's web interface, what we learned, and potential future directions.

## The Problem

Claude Code (and similar AI coding assistants) rely on **native tool calling** to interact with the local environment - reading files, running commands, editing code, etc. When using Perplexity models through the web interface, these tools don't work because:

1. Perplexity's web UI doesn't expose native function/tool calling APIs
2. We only have access to a chat interface that returns text responses
3. The models need to output structured tool calls that can be parsed and executed

## Approaches Tried

### 1. XML Tag Format

**Approach:** Instruct models to output tool calls in XML format:
```xml
<tool_call>
{"name": "tool_name", "input": {"param": "value"}}
</tool_call>
```

**Result:** Models ignored the format entirely. They would acknowledge they had tools but respond conversationally instead of outputting the XML structure.

**Example prompt:**
```
IMPORTANT: You have access to the following tools. When a user asks you to do something that requires one of these tools, you MUST use the tool by outputting EXACTLY this format:

<tool_call>
{"name": "TOOL_NAME_HERE", "input": {"parameter": "value"}}
</tool_call>
```

### 2. ReAct Format (Detailed)

**Approach:** Use the well-known ReAct (Reasoning + Acting) pattern that's common in agent frameworks:
```
Thought: I need to list the user's notebooks
Action: notebook_list
Action Input: {}
```

**Result:** Models explained what ReAct is instead of using it. One response said "Perplexity's tools are currently disabled for this turn, so I'll respond using general knowledge of AI agent workflows like ReAct."

The model interpreted our instructions as content to discuss rather than rules to follow.

### 3. ReAct Format (Minimal with Examples)

**Approach:** Stripped down instructions with few-shot examples:
```
FORMAT FOR TOOL CALLS:
Action: tool_name
Action Input: {"param": "value"}

EXAMPLES:
User: List my notebooks
Action: notebook_list
Action Input: {}

User: Hello!
Final Answer: Hello! How can I help you today?
```

**Result:** Models still didn't follow the format. They would explain the Task tool or other concepts instead of using the Action/Final Answer format.

### 4. Minimal Instructions

**Approach:** Bare minimum - just list tools and ask model to use them:
```
Available tools you can use:
[tool list]

To use a tool, say which one and what parameters. Otherwise just respond normally.
```

**Result:** Same behavior - conversational responses, no structured output.

## Root Cause Analysis

The fundamental issue is that **Perplexity's web UI models are optimized for conversational search, not structured output generation.**

1. **Training objective mismatch:** These models are trained to be helpful assistants that explain things, not to follow strict output formats
2. **No system prompt distinction:** Everything we inject appears as user content, so models treat it as "information to discuss" rather than "instructions to execute"
3. **Search-first behavior:** Models default to searching and explaining rather than executing actions

## What Does Work

- **Basic chat:** Models respond well to questions and conversations
- **Web search:** Models use Perplexity's built-in search to find information
- **Thinking modes:** Extended thinking/reasoning works for complex questions
- **Model selection:** Can switch between GPT-5.2, Claude 4.5, Gemini 3, etc.

## Potential Future Solutions

### 1. Ollama as Orchestrator

Use a local model (via Ollama) that supports native tool calling as the orchestrator:

```
User → Claude Code → Ollama (tool calling) → Perplexity MCP (web search)
                          ↓
                    Local tools (Read, Write, Bash)
```

- Ollama models like Llama 3.1 70B support native function calling
- Perplexity becomes just a "web search" tool, not the main model
- Local tools work because Ollama handles them natively

### 2. Official Perplexity API

Perplexity's official API supports native function calling:
- Endpoint: `https://api.perplexity.ai/chat/completions`
- Supports OpenAI-compatible tool/function definitions
- Requires separate API key and billing

**Note:** An official Perplexity MCP already exists using this API, so this project's value is leveraging the web UI (included with Pro subscription) without additional API costs.

### 3. Fine-tuned Parsing

More aggressive parsing of natural language responses:
- Extract intent from conversational responses
- Pattern match for tool-like requests
- Use an intermediate LLM to translate responses to tool calls

**Downsides:** Unreliable, adds latency, may misinterpret intent.

### 4. Browser Automation

Instead of API calls, automate the actual browser:
- Use Playwright/Puppeteer to interact with perplexity.ai
- Leverage whatever tool calling the web UI might support in future
- More fragile but potentially more capable

### 5. Wait for Perplexity Updates

Perplexity may add tool calling support to the web interface in the future. Their "Spaces" feature and other updates suggest they're building more agentic capabilities.

## Current State (Feb 2026)

Tool calling is **disabled** in the API compatibility layer. The project provides:

- Chat with any Perplexity model through Claude Code
- Model selection (GPT-5.2, Claude 4.5 Sonnet, Gemini 3, Grok 4.1, Kimi K2.5)
- Thinking mode toggle
- Streaming responses
- Anthropic and OpenAI API compatibility

No local tool access (Read, Write, Bash, MCP tools) when using Perplexity models.

## Files Reference

- `src/perplexity_web_mcp/api/tool_calling.py` - Tool calling implementation (currently unused)
- `src/perplexity_web_mcp/api/server.py` - API server with tool calling disabled
- `src/perplexity_web_mcp/api/session_manager.py` - Session/conversation management

## Lessons Learned

1. **Prompt engineering has limits:** No amount of instruction can force a model to output structured formats if it's not trained for that
2. **Few-shot examples help but aren't magic:** Even with clear examples, models may explain them instead of following them
3. **Architecture matters:** The right solution may be changing the architecture (Ollama orchestrator) rather than fighting the model
4. **Web UI ≠ API:** Web interfaces are designed for humans, not programmatic access - limitations are expected
