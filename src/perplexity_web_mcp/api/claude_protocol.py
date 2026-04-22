"""Claude protocol helpers for Anthropic-style tool use and MCP approval flows.

This module centralizes protocol-specific parsing and response shaping so that
server.py can call into it without embedding all Claude/Anthropic tool logic
inline.

Current scope:
- Extract tool definitions from incoming requests
- Detect Claude-style tool_result blocks coming back from the client
- Build Anthropic-compatible tool_use content blocks
- Build tool_result user content blocks
- Provide a minimal approval request/result abstraction for write tools

This does not execute tools itself. It only normalizes protocol payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass(slots=True)
class ToolDefinition:
    """Normalized tool definition from Claude/Anthropic requests."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolUse:
    """Normalized tool call emitted by the assistant."""

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """Normalized tool result sent back by the client."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class ApprovalRequest:
    """Approval request for sensitive tools such as file writes."""

    id: str
    tool_name: str
    input: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(slots=True)
class ApprovalResponse:
    """Approval decision corresponding to an ApprovalRequest."""

    approval_request_id: str
    approve: bool
    reason: str | None = None


def extract_tool_definitions(tools: list[dict[str, Any]] | None) -> list[ToolDefinition]:
    """Normalize incoming Anthropic/OpenAI tool definitions."""
    if not tools:
        return []

    normalized: list[ToolDefinition] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", "")).strip()
        if not name:
            continue
        normalized.append(
            ToolDefinition(
                name=name,
                description=str(tool.get("description", "")),
                input_schema=tool.get("input_schema", {}) or {},
            )
        )
    return normalized


def extract_tool_results_from_message_content(content: Any) -> list[ToolResult]:
    """Extract Claude-style tool_result blocks from message content."""
    if not isinstance(content, list):
        return []

    results: list[ToolResult] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue

        tool_use_id = str(block.get("tool_use_id", "")).strip()
        if not tool_use_id:
            continue

        raw_content = block.get("content", "")
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            text = "\n".join(part for part in parts if part)
        else:
            text = str(raw_content)

        results.append(
            ToolResult(
                tool_use_id=tool_use_id,
                content=text,
                is_error=bool(block.get("is_error", False)),
            )
        )
    return results


def build_tool_use_block(name: str, tool_input: dict[str, Any], tool_use_id: str | None = None) -> dict[str, Any]:
    """Build an Anthropic-compatible tool_use content block."""
    return {
        "type": "tool_use",
        "id": tool_use_id or f"toolu_{uuid.uuid4().hex[:24]}",
        "name": name,
        "input": tool_input or {},
    }


def build_tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    """Build an Anthropic-compatible tool_result block."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }


def build_tool_result_user_message(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    """Build a user message wrapping a tool_result block."""
    return {
        "role": "user",
        "content": [build_tool_result_block(tool_use_id, content, is_error=is_error)],
    }


def requires_approval(tool_name: str) -> bool:
    """Return True for tools that should require explicit approval.

    This is intentionally conservative for file-system write style operations.
    """
    normalized = tool_name.lower().strip()
    approval_markers = (
        "write",
        "edit",
        "create",
        "delete",
        "rename",
        "move",
        "replace",
        "patch",
    )
    return any(marker in normalized for marker in approval_markers)


def build_approval_request(tool_name: str, tool_input: dict[str, Any], reason: str = "") -> ApprovalRequest:
    """Create a normalized approval request."""
    return ApprovalRequest(
        id=f"apr_{uuid.uuid4().hex[:24]}",
        tool_name=tool_name,
        input=tool_input or {},
        reason=reason,
    )


def build_openai_mcp_approval_request(tool_name: str, tool_input: dict[str, Any], reason: str = "") -> dict[str, Any]:
    """Build a minimal OpenAI Responses/MCP-style approval request payload."""
    approval = build_approval_request(tool_name, tool_input, reason=reason)
    return {
        "type": "mcp_approval_request",
        "approval_request_id": approval.id,
        "tool_name": approval.tool_name,
        "input": approval.input,
        "reason": approval.reason,
    }


def build_openai_mcp_approval_response(approval_request_id: str, approve: bool, reason: str | None = None) -> dict[str, Any]:
    """Build a minimal OpenAI Responses/MCP-style approval response payload."""
    response = ApprovalResponse(
        approval_request_id=approval_request_id,
        approve=approve,
        reason=reason,
    )
    payload = {
        "type": "mcp_approval_response",
        "approval_request_id": response.approval_request_id,
        "approve": response.approve,
    }
    if response.reason is not None:
        payload["reason"] = response.reason
    return payload


def tool_use_stop_response(message_id: str, model: str, content: list[dict[str, Any]], input_tokens: int = 0, output_tokens: int = 0) -> dict[str, Any]:
    """Build a Claude-compatible response that stops for tool use."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
