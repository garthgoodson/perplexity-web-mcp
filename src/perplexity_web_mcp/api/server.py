"""Dual API-compatible server for Perplexity Web MCP.

This server provides both Anthropic Messages API and OpenAI Chat Completions API
compatible interfaces, allowing Claude Code, OpenAI SDK clients, and other tools
to use Perplexity models as a backend.

Supported APIs:
- Anthropic Messages API: POST /v1/messages
  Reference: https://docs.anthropic.com/en/api/messages

- OpenAI Chat Completions API: POST /v1/chat/completions
  Reference: https://platform.openai.com/docs/api-reference/chat

- OpenAI Responses API: POST /v1/responses
  Minimal compatibility layer for Codex/OpenAI clients that prefer Responses API.

Both APIs support streaming (SSE) and non-streaming responses.

Claude Code Integration:
  Set these environment variables:
    export ANTHROPIC_AUTH_TOKEN=perplexity
    export ANTHROPIC_BASE_URL=http://localhost:8080
    export ANTHROPIC_API_KEY=""

  Then run Claude Code with any supported model:
    claude --model claude-sonnet-4-6      # Use Claude 4.6 Sonnet via Perplexity
    claude --model gpt-5.4                # Use GPT-5.4 via Perplexity
    claude --model perplexity-auto        # Use Perplexity's auto model selection
    claude --model claude-3-5-sonnet      # Legacy name, maps to Claude 4.6 Sonnet
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from perplexity_web_mcp import Perplexity, ConversationConfig, Models
from perplexity_web_mcp.api.session_manager import (
    ConversationManager,
    distill_system_prompt,
)
from perplexity_web_mcp.api import claude_protocol
from perplexity_web_mcp.enums import CitationMode
from perplexity_web_mcp.models import Model
from perplexity_web_mcp.token_store import load_token

# Tool calling disabled for now - models don't reliably follow format instructions
# from perplexity_web_mcp.api.tool_calling import (...)

# Supported Anthropic API version
ANTHROPIC_API_VERSION = "2023-06-01"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ServerConfig:
    """Server configuration from environment variables."""

    session_token: str
    api_key: str | None = None  # Optional auth
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    default_model: str = "auto"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load from environment."""
        # Try to load from token store (env var or ~/.config/perplexity-web-mcp/token)
        session_token = load_token()
        if not session_token:
            raise ValueError(
                "No Perplexity session token found. "
                "Run 'pwm-auth' to authenticate."
            )

        return cls(
            session_token=session_token,
            api_key=os.getenv("ANTHROPIC_API_KEY"),  # For auth validation
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            default_model=os.getenv("DEFAULT_MODEL", "auto"),
        )


# =============================================================================
# Model Mapping
# =============================================================================

# Map model names to Perplexity models
# Supports Anthropic, OpenAI, and standard Claude Code model naming conventions
# Updated Mar 2026 to match Perplexity UI offerings
MODEL_MAP: dict[str, tuple[Model, Model | None]] = {
    # ==========================================================================
    # Perplexity Native Models
    # ==========================================================================
    "perplexity-auto": (Models.BEST, None),
    "auto": (Models.BEST, None),
    "best": (Models.BEST, None),
    "perplexity-sonar": (Models.SONAR, None),
    "sonar": (Models.SONAR, None),
    "perplexity-research": (Models.DEEP_RESEARCH, None),
    "deep-research": (Models.DEEP_RESEARCH, None),
    # ==========================================================================
    # Anthropic Claude Models (via Perplexity)
    # Claude Sonnet 4.6 - supports thinking toggle
    # Claude Opus 4.6 - supports thinking (requires Max subscription)
    # ==========================================================================
    # Current model names
    "claude-sonnet-4-6": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "claude-4-6-sonnet": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "claude-sonnet-4-6-20260217": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    # Legacy Sonnet 4.5 aliases (map to 4.6)
    "claude-sonnet-4-5": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "claude-4-5-sonnet": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "claude-opus-4-6": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    "claude-4-6-opus": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    "claude-opus-4-6-20260203": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    # Legacy Opus 4.5 aliases (map to 4.6)
    "claude-opus-4-5": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    "claude-4-5-opus": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    "claude-opus-4-5-20251101": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    # Claude Code default model aliases (for compatibility)
    "claude-3-5-sonnet": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "claude-3-5-sonnet-20241022": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "claude-3-opus": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    "claude-3-opus-20240229": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    "claude-3-5-haiku": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "claude-haiku-4-5-20251001": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "claude-haiku": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    # Generic aliases
    "claude": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "sonnet": (Models.CLAUDE_46_SONNET, Models.CLAUDE_46_SONNET_THINKING),
    "opus": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    # ==========================================================================
    # OpenAI GPT Models (via Perplexity) - support thinking toggle
    # ==========================================================================
    "gpt-5.4": (Models.GPT_54, Models.GPT_54_THINKING),
    "gpt-5-4": (Models.GPT_54, Models.GPT_54_THINKING),
    "gpt-54": (Models.GPT_54, Models.GPT_54_THINKING),
    "gpt54": (Models.GPT_54, Models.GPT_54_THINKING),
    # ==========================================================================
    # Google Gemini Models (via Perplexity)
    # Gemini 3.1 Pro: thinking ALWAYS enabled (no toggle in UI)
    # ==========================================================================
    "gemini-3.1-pro": (Models.GEMINI_31_PRO_THINKING, Models.GEMINI_31_PRO_THINKING),
    "gemini-3-pro": (Models.GEMINI_31_PRO_THINKING, Models.GEMINI_31_PRO_THINKING),
    "gemini-pro": (Models.GEMINI_31_PRO_THINKING, Models.GEMINI_31_PRO_THINKING),
    # ==========================================================================
    # NVIDIA Nemotron 3 Super (via Perplexity)
    # Thinking ALWAYS enabled (no toggle in UI) - reasoning only
    # ==========================================================================
    "nemotron-3-super": (Models.NEMOTRON_3_SUPER, Models.NEMOTRON_3_SUPER),
    "nemotron-3": (Models.NEMOTRON_3_SUPER, Models.NEMOTRON_3_SUPER),
    "nemotron": (Models.NEMOTRON_3_SUPER, Models.NEMOTRON_3_SUPER),
}

# Models we expose via /v1/models
# Ordered to match Perplexity UI (Mar 2026)
AVAILABLE_MODELS = [
    {"id": "perplexity-auto", "description": "Best - Automatically selects optimal model"},
    {"id": "perplexity-sonar", "description": "Sonar - Perplexity's latest model"},
    {"id": "perplexity-research", "description": "Deep Research - In-depth reports with sources"},
    {"id": "gemini-3.1-pro", "description": "Gemini 3.1 Pro - Advanced, thinking always on"},
    {"id": "gpt-5.4", "description": "GPT-5.4 - OpenAI's latest, thinking toggle available"},
    {"id": "claude-sonnet-4-6", "description": "Claude Sonnet 4.6 - Fast, thinking toggle available"},
    {"id": "claude-opus-4-6", "description": "Claude Opus 4.6 - Advanced reasoning, Max tier required"},
    {"id": "nemotron-3-super", "description": "Nemotron 3 Super - NVIDIA 120B, thinking always on"},
]


def get_model(name: str, thinking: bool = False) -> Model:
    """Get Perplexity model from name.

    Supports both Anthropic and OpenAI model naming conventions.
    Falls back to perplexity-auto for unknown models.
    """
    key = name.lower().strip()
    if key in MODEL_MAP:
        base, thinking_model = MODEL_MAP[key]
        if thinking and thinking_model:
            return thinking_model
        return base
    logging.warning(f"Unknown model '{name}', using perplexity-auto")
    return Models.BEST


# =============================================================================
# Pydantic Models (Anthropic API format)
# =============================================================================


class TextContent(BaseModel):
    """Text content block in a message."""
    type: str = "text"
    text: str


class ImageSource(BaseModel):
    """Image source for multimodal content."""
    type: str
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class ImageContent(BaseModel):
    """Image content block."""
    type: str = "image"
    source: ImageSource


class MessageParam(BaseModel):
    """Input message parameter (Anthropic format)."""

    role: str
    content: str | list[dict[str, Any]]

    model_config = ConfigDict(extra="allow")

    def get_text(self) -> str:
        """Extract text content from message."""
        if isinstance(self.content, str):
            return self.content
        texts = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)


class MessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    model: str = Field(..., description="Model to use (e.g., 'claude-sonnet-4-6')")
    max_tokens: int = Field(..., description="Maximum tokens to generate")
    messages: list[MessageParam] = Field(..., description="Conversation messages")

    system: str | list[dict[str, Any]] | None = Field(None, description="System prompt")
    stream: bool = Field(False, description="Enable streaming response")
    temperature: float | None = Field(None, ge=0, le=1, description="Sampling temperature")
    top_p: float | None = Field(None, ge=0, le=1, description="Nucleus sampling")
    top_k: int | None = Field(None, ge=0, description="Top-k sampling")
    stop_sequences: list[str] | None = Field(None, description="Stop sequences")
    metadata: dict[str, Any] | None = Field(None, description="Request metadata")

    thinking: dict[str, Any] | None = Field(
        None,
        description="Extended thinking config. Set type='enabled' to use thinking models.",
    )

    tools: list[dict[str, Any]] | None = Field(None, description="Available tools")
    tool_choice: dict[str, Any] | str | None = Field(None, description="Tool choice config")

    model_config = ConfigDict(extra="allow")

    def get_system_text(self) -> str | None:
        """Extract system prompt text, handling both string and array formats."""
        if self.system is None:
            return None
        if isinstance(self.system, str):
            return self.system
        texts = []
        for block in self.system:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts) if texts else None


class Usage(BaseModel):
    """Token usage statistics."""
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")


class TextBlock(BaseModel):
    """Response text content block."""
    type: str = "text"
    text: str


class MessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str = Field(..., description="Unique message identifier")
    type: str = Field("message", description="Object type")
    role: str = Field("assistant", description="Message role")
    content: list[TextBlock] = Field(..., description="Response content blocks")
    model: str = Field(..., description="Model used")
    stop_reason: str | None = Field("end_turn", description="Reason generation stopped")
    stop_sequence: str | None = Field(None, description="Stop sequence if triggered")
    usage: Usage = Field(..., description="Token usage")


class ModelObject(BaseModel):
    """Model object (OpenAI-compatible format for /v1/models)."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "perplexity-web-mcp"


class ModelsListResponse(BaseModel):
    """Models list response (OpenAI-compatible format)."""
    object: str = "list"
    data: list[ModelObject]


class ErrorDetail(BaseModel):
    """Error detail."""
    type: str
    message: str


class ErrorResponse(BaseModel):
    """Anthropic API error response."""
    type: str = "error"
    error: ErrorDetail


# =============================================================================
# OpenAI API Models (Chat Completions + Responses format)
# =============================================================================


class OpenAIChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str | list[dict[str, Any]] | None = Field(None, description="Message content")
    name: str | None = Field(None, description="Optional name for the participant")

    model_config = ConfigDict(extra="allow")

    def get_text(self) -> str:
        """Extract text content from message."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        texts = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)


class OpenAIChatRequest(BaseModel):
    """OpenAI Chat Completions API request."""

    model: str = Field(..., description="Model ID (e.g., 'gpt-4o')")
    messages: list[OpenAIChatMessage] = Field(..., description="Conversation messages")

    max_tokens: int | None = Field(None, description="Max tokens (deprecated, use max_completion_tokens)")
    max_completion_tokens: int | None = Field(None, description="Max completion tokens")
    temperature: float | None = Field(None, ge=0, le=2, description="Sampling temperature")
    top_p: float | None = Field(None, ge=0, le=1, description="Nucleus sampling")
    n: int | None = Field(1, description="Number of completions")
    stream: bool = Field(False, description="Enable streaming")
    stream_options: dict[str, Any] | None = Field(None, description="Streaming options")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    frequency_penalty: float | None = Field(None, ge=-2, le=2)
    user: str | None = Field(None, description="User identifier")

    reasoning_effort: str | None = Field(None, description="Reasoning effort level")

    model_config = ConfigDict(extra="allow")

class OpenAIResponsesRequest(BaseModel):
    """Minimal OpenAI Responses API request.

    Supports the common subset used by Codex/OpenAI-compatible clients.
    """

    model: str = Field(..., description="Model ID")
    input: str | list[dict[str, Any]] = Field(..., description="Input text or structured input items")
    stream: bool = Field(False, description="Enable streaming response")
    instructions: str | None = Field(None, description="Optional system/instructions text")
    reasoning: dict[str, Any] | None = Field(None, description="Optional reasoning config")
    tools: list[dict[str, Any]] | None = Field(None, description="Optional MCP/tool definitions")

    model_config = ConfigDict(extra="allow")


class OpenAIChoiceMessage(BaseModel):
    """Message in a chat completion choice."""
    role: str = "assistant"
    content: str | None = None


class OpenAIChoice(BaseModel):
    """A chat completion choice."""
    index: int = 0
    message: OpenAIChoiceMessage
    finish_reason: str | None = "stop"
    logprobs: Any | None = None


class OpenAIUsage(BaseModel):
    """Token usage for OpenAI format."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(BaseModel):
    """OpenAI Chat Completions API response."""

    id: str = Field(..., description="Unique completion identifier")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: list[OpenAIChoice] = Field(..., description="Completion choices")
    usage: OpenAIUsage = Field(..., description="Token usage")
    system_fingerprint: str | None = Field(None, description="System fingerprint")


class OpenAIResponseOutputText(BaseModel):
    """Responses API output text content block."""
    type: str = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class OpenAIResponseOutputMessage(BaseModel):
    """Responses API output message item."""
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[OpenAIResponseOutputText]


class OpenAIResponsesResponse(BaseModel):
    """Minimal OpenAI Responses API response."""
    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    model: str
    output: list[OpenAIResponseOutputMessage]
    output_text: str
    usage: OpenAIUsage

class OpenAIStreamChoice(BaseModel):
    """A streaming chat completion choice."""
    index: int = 0
    delta: dict[str, Any]
    finish_reason: str | None = None
    logprobs: Any | None = None


class OpenAIStreamResponse(BaseModel):
    """OpenAI streaming chunk response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIStreamChoice]
    system_fingerprint: str | None = None


# =============================================================================
# Global State
# =============================================================================

config: ServerConfig
client: Perplexity
conversation_manager: ConversationManager
start_time: datetime
perplexity_semaphore: asyncio.Semaphore
last_request_time: float = 0.0
MIN_REQUEST_INTERVAL: float = 5.0
pending_tool_calls: dict[str, "PendingToolCall"] = {}
pending_approvals: dict[str, "PendingApproval"] = {}
pending_state_lock = asyncio.Lock()
PENDING_TOOL_TTL_SECONDS: float = 1800.0


@dataclass
class PendingToolCall:
    """Pending Claude/OpenAI-compatible tool call awaiting result."""
    tool_use_id: str
    model_name: str
    tool_name: str
    tool_input: dict[str, Any]
    original_user_text: str
    created_at: float = field(default_factory=time.time)
    approved: bool = False
    approval_request_id: str | None = None


@dataclass
class PendingApproval:
    """Approval request tied to a pending tool call."""
    approval_request_id: str
    tool_use_id: str
    tool_name: str
    reason: str
    created_at: float = field(default_factory=time.time)


# =============================================================================
# Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    global config, client, conversation_manager, start_time, perplexity_semaphore

    start_time = datetime.now()
    config = ServerConfig.from_env()
    client = Perplexity(session_token=config.session_token)
    conversation_manager = ConversationManager(
        client=client,
        max_sessions=50,
        session_timeout_seconds=1800,
    )
    perplexity_semaphore = asyncio.Semaphore(1)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Starting Anthropic API server on http://{config.host}:{config.port}")
    logging.info(f"Auth required: {'Yes' if config.api_key else 'No'}")
    logging.info("Fresh client per request, single ask (system prepended), serialized")

    yield

    conversation_manager.clear_all()
    client.close()


app = FastAPI(
    title="Perplexity Web MCP - Anthropic API",
    description="Anthropic & OpenAI API compatible server powered by Perplexity AI",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helpers
# =============================================================================


def verify_auth(request: Request) -> None:
    """Verify API key if configured."""
    if not config.api_key:
        return

    auth = request.headers.get("x-api-key", "")
    if not auth:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            auth = auth[7:]

    if auth != config.api_key:
        raise HTTPException(
            status_code=401,
            detail={"type": "authentication_error", "message": "Invalid API key"},
        )


def check_anthropic_version(request: Request) -> None:
    """Log anthropic-version header if present (informational only)."""
    version = request.headers.get("anthropic-version")
    if version and version != ANTHROPIC_API_VERSION:
        logging.debug(
            f"Client requested anthropic-version {version}, "
            f"server implements {ANTHROPIC_API_VERSION}"
        )


def messages_to_query(messages: list[MessageParam]) -> str:
    """Convert Anthropic messages to Perplexity query."""
    parts = []
    for msg in messages:
        text = msg.get_text()
        if msg.role == "user":
            parts.append(text if len(messages) == 1 else f"User: {text}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {text}")
    return "\n\n".join(parts)


def openai_messages_to_query(messages: list[OpenAIChatMessage]) -> str:
    """Convert OpenAI chat messages to Perplexity query."""
    conversation_msgs = [m for m in messages if m.role in ("user", "assistant")]

    user_msgs = [m for m in conversation_msgs if m.role == "user"]
    if len(user_msgs) == 1 and len(conversation_msgs) == 1:
        return user_msgs[0].get_text()

    parts = []
    for msg in conversation_msgs:
        text = msg.get_text()
        if msg.role == "user":
            parts.append(f"User: {text}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {text}")

    return "\n\n".join(parts)


def responses_input_to_query(value: str | list[dict[str, Any]]) -> str:
    """Convert OpenAI Responses API input to a Perplexity query."""
    if isinstance(value, str):
        return value

    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "user"))
        content = item.get("content", "")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in ("input_text", "text", "output_text"):
                    text_parts.append(str(block.get("text", "")))
            text = "\n".join(part for part in text_parts if part)
        else:
            text = ""

        if not text:
            continue

        if role == "user":
            parts.append(f"User: {text}")
        elif role == "assistant":
            parts.append(f"Assistant: {text}")
        elif role == "system":
            continue
        else:
            parts.append(text)

    return "\n\n".join(parts).strip()


def responses_input_to_instructions(value: str | list[dict[str, Any]]) -> str | None:
    """Extract system/instructions text from structured Responses input if present."""
    if isinstance(value, str):
        return None

    instructions: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if str(item.get("role")) != "system":
            continue

        content = item.get("content", "")
        if isinstance(content, str):
            if content.strip():
                instructions.append(content.strip())
            continue

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("input_text", "text"):
                    text = str(block.get("text", "")).strip()
                    if text:
                        instructions.append(text)

    return "\n".join(instructions).strip() or None


def extract_latest_user_text_from_responses_input(value: str | list[dict[str, Any]]) -> str:
    """Extract the latest meaningful user text from Responses API structured input."""
    if isinstance(value, str):
        return value.strip()

    latest_user_text = ""

    for item in value:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        if role != "user":
            continue

        content = item.get("content", "")
        if isinstance(content, str):
            candidate = content.strip()
            if candidate:
                latest_user_text = candidate
            continue

        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("input_text", "text", "output_text"):
                    part = str(block.get("text", "")).strip()
                    if part:
                        text_parts.append(part)
            candidate = "\n".join(text_parts).strip()
            if candidate:
                latest_user_text = candidate

    return latest_user_text


def is_thinking_enabled(
    reasoning_effort: str | None = None,
    reasoning: dict[str, Any] | None = None,
) -> bool:
    """Map OpenAI-style reasoning controls to Perplexity thinking models."""
    if reasoning_effort in ("medium", "high", "xhigh"):
        return True
    if reasoning:
        effort = str(reasoning.get("effort", "")).lower()
        if effort in ("medium", "high", "xhigh"):
            return True
    return False


def estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    return len(text) // 4


def format_citations(search_results: list) -> str:
    """Format search results as citations to append to response."""
    if not search_results:
        return ""

    citations = ["\n\nCitations:"]
    for i, result in enumerate(search_results, 1):
        url = getattr(result, "url", "") or ""
        if url:
            citations.append(f"\n[{i}]: {url}")

    return "".join(citations) if len(citations) > 1 else ""

def extract_last_user_text(messages: list[MessageParam]) -> str:
    """Extract the most recent user text message."""
    for msg in reversed(messages):
        if msg.role == "user":
            text = msg.get_text().strip()
            if text:
                return text
    return ""


def extract_tool_results(messages: list[MessageParam]) -> list[claude_protocol.ToolResult]:
    """Extract Claude-style tool_result blocks from incoming messages."""
    results: list[claude_protocol.ToolResult] = []
    for msg in messages:
        if msg.role != "user":
            continue
        if isinstance(msg.content, list):
            results.extend(claude_protocol.extract_tool_results_from_message_content(msg.content))
    return results


async def cleanup_expired_pending_state() -> None:
    """Remove expired pending tool calls and approvals."""
    now = time.time()
    expired_tool_ids: list[str] = []
    expired_approval_ids: list[str] = []

    async with pending_state_lock:
        for tool_use_id, pending in pending_tool_calls.items():
            if now - pending.created_at > PENDING_TOOL_TTL_SECONDS:
                expired_tool_ids.append(tool_use_id)

        for approval_request_id, approval in pending_approvals.items():
            if now - approval.created_at > PENDING_TOOL_TTL_SECONDS:
                expired_approval_ids.append(approval_request_id)

        for tool_use_id in expired_tool_ids:
            pending = pending_tool_calls.pop(tool_use_id, None)
            if pending and pending.approval_request_id:
                pending_approvals.pop(pending.approval_request_id, None)

        for approval_request_id in expired_approval_ids:
            approval = pending_approvals.pop(approval_request_id, None)
            if approval:
                pending_tool_calls.pop(approval.tool_use_id, None)

    if expired_tool_ids or expired_approval_ids:
        logging.debug(
            "Expired pending state cleanup: expired_tools=%s expired_approvals=%s",
            expired_tool_ids,
            expired_approval_ids,
        )


async def register_pending_tool_call(
    model_name: str,
    tool_name: str,
    tool_input: dict[str, Any],
    original_user_text: str,
    requires_approval: bool,
    approval_reason: str = "",
) -> tuple[str, str | None]:
    """Register a pending tool call and optional approval request."""
    await cleanup_expired_pending_state()

    tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
    approval_request_id = f"apr_{uuid.uuid4().hex[:24]}" if requires_approval else None

    async with pending_state_lock:
        pending_tool_calls[tool_use_id] = PendingToolCall(
            tool_use_id=tool_use_id,
            model_name=model_name,
            tool_name=tool_name,
            tool_input=tool_input,
            original_user_text=original_user_text,
            approved=not requires_approval,
            approval_request_id=approval_request_id,
        )
        if approval_request_id:
            pending_approvals[approval_request_id] = PendingApproval(
                approval_request_id=approval_request_id,
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                reason=approval_reason,
            )

    logging.debug(
        "Registered pending tool call: tool_use_id=%s tool_name=%s approval_required=%s approval_request_id=%s",
        tool_use_id,
        tool_name,
        requires_approval,
        approval_request_id,
    )

    return tool_use_id, approval_request_id


async def get_pending_tool_call(tool_use_id: str) -> PendingToolCall | None:
    """Look up a pending tool call."""
    await cleanup_expired_pending_state()
    async with pending_state_lock:
        return pending_tool_calls.get(tool_use_id)


async def approve_pending_tool_call(approval_request_id: str, approve: bool) -> PendingToolCall | None:
    """Approve or reject a pending tool call."""
    await cleanup_expired_pending_state()
    async with pending_state_lock:
        approval = pending_approvals.get(approval_request_id)
        if not approval:
            logging.debug(
                "Approval request not found: approval_request_id=%s approve=%s",
                approval_request_id,
                approve,
            )
            return None

        pending = pending_tool_calls.get(approval.tool_use_id)
        if not pending:
            pending_approvals.pop(approval_request_id, None)
            logging.debug(
                "Approval request had no pending tool call: approval_request_id=%s tool_use_id=%s",
                approval_request_id,
                approval.tool_use_id,
            )
            return None

        if approve:
            pending.approved = True
        else:
            pending_tool_calls.pop(pending.tool_use_id, None)

        pending_approvals.pop(approval_request_id, None)
        logging.debug(
            "Approval processed: approval_request_id=%s approve=%s tool_use_id=%s tool_name=%s",
            approval_request_id,
            approve,
            pending.tool_use_id,
            pending.tool_name,
        )
        return pending


async def pop_pending_tool_call(tool_use_id: str) -> PendingToolCall | None:
    """Remove and return a pending tool call."""
    await cleanup_expired_pending_state()
    async with pending_state_lock:
        pending = pending_tool_calls.pop(tool_use_id, None)
        if pending and pending.approval_request_id:
            pending_approvals.pop(pending.approval_request_id, None)

    logging.debug(
        "Popped pending tool call: tool_use_id=%s found=%s",
        tool_use_id,
        pending is not None,
    )
    return pending


async def maybe_build_tool_protocol_response(
    request_model: str,
    messages: list[MessageParam],
    tools: list[dict[str, Any]] | None,
    input_tokens: int,
) -> dict[str, Any] | None:
    """Return a Claude tool_use response when protocol handling should take over."""
    normalized_tools = claude_protocol.extract_tool_definitions(tools)
    if not normalized_tools:
        return None

    response_id = f"msg_{uuid.uuid4().hex[:24]}"
    tool_results = extract_tool_results(messages)
    if tool_results:
        rendered_responses: list[str] = []

        for result in tool_results:
            pending = await get_pending_tool_call(result.tool_use_id)
            if pending is None:
                rendered_responses.append(
                    f"Received tool result for unknown or expired tool call {result.tool_use_id}."
                )
                continue

            if not pending.approved:
                rendered_responses.append(
                    f"Received tool result for unapproved tool call {result.tool_use_id}."
                )
                continue

            popped = await pop_pending_tool_call(result.tool_use_id)
            if popped is None:
                rendered_responses.append(
                    f"Tool call {result.tool_use_id} was no longer available."
                )
                continue

            continuation_prompt = (
                f"Original user request:\n{popped.original_user_text}\n\n"
                f"Tool executed: {popped.tool_name}\n"
                f"Tool status: {'error' if result.is_error else 'success'}\n\n"
                f"Tool output:\n{result.content}\n\n"
                "Please continue by responding to the user with the next helpful assistant message. "
                "If the tool output indicates success, summarize what was done and any relevant next steps. "
                "If the tool output indicates an error, explain the failure and suggest how to fix it."
            )

            try:
                continued_response = await run_perplexity_query(
                    model=get_model(request_model),
                    query=continuation_prompt,
                    system_text=None,
                )
            except Exception as e:
                continued_response = (
                    f"Tool {popped.tool_name} completed, but continuation generation failed: {e}"
                )

            rendered_responses.append(continued_response)

        summary_text = "\n\n".join(rendered_responses)
        return {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": summary_text}],
            "model": request_model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": estimate_tokens(summary_text),
            },
        }

    user_text = extract_last_user_text(messages).lower()
    write_markers = (
        "write",
        "edit",
        "modify",
        "update",
        "create file",
        "save",
        "delete",
        "rename",
        "move file",
        "replace",
        "patch",
    )
    if not any(marker in user_text for marker in write_markers):
        return None

    selected_tool = normalized_tools[0]
    original_user_text = extract_last_user_text(messages)
    tool_input = {"query": original_user_text}

    approval_required = claude_protocol.requires_approval(selected_tool.name)
    tool_use_id, approval_request_id = await register_pending_tool_call(
        model_name=request_model,
        tool_name=selected_tool.name,
        tool_input=tool_input,
        original_user_text=original_user_text,
        requires_approval=approval_required,
        approval_reason="This tool appears to modify files or workspace state.",
    )

    if approval_required and approval_request_id is not None:
        approval_payload = claude_protocol.build_openai_mcp_approval_request(
            selected_tool.name,
            tool_input,
            reason="This tool appears to modify files or workspace state.",
        )
        approval_payload["approval_request_id"] = approval_request_id
        tool_input = {
            **tool_input,
            "_approval": approval_payload,
        }

    tool_block = claude_protocol.build_tool_use_block(
        name=selected_tool.name,
        tool_input=tool_input,
        tool_use_id=tool_use_id,
    )
    return claude_protocol.tool_use_stop_response(
        message_id=response_id,
        model=request_model,
        content=[tool_block],
        input_tokens=input_tokens,
        output_tokens=estimate_tokens(selected_tool.name),
    )


def extract_openai_approval_response(input_value: str | list[dict[str, Any]]) -> tuple[str, bool] | None:
    """Extract an OpenAI Responses approval reply if present."""
    if isinstance(input_value, str):
        return None

    for item in input_value:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "mcp_approval_response":
            continue
        approval_request_id = str(item.get("approval_request_id", "")).strip()
        approve = bool(item.get("approve", False))
        if approval_request_id:
            return approval_request_id, approve

    return None


def extract_openai_tool_result(input_value: str | list[dict[str, Any]]) -> tuple[str, str, bool] | None:
    """Extract an OpenAI Responses tool result if present."""
    if isinstance(input_value, str):
        return None

    for item in input_value:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in ("mcp_tool_result", "tool_result"):
            continue

        tool_use_id = str(item.get("tool_use_id", item.get("call_id", ""))).strip()
        content = str(item.get("content", ""))
        is_error = bool(item.get("is_error", False))
        if tool_use_id:
            return tool_use_id, content, is_error

    return None


def responses_tools_to_claude_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize OpenAI-style tool definitions into Claude helper input."""
    if not tools:
        return []
    return [tool for tool in tools if isinstance(tool, dict)]


async def maybe_build_responses_protocol_output(
    model_name: str,
    user_input: str,
    tools: list[dict[str, Any]] | None,
    raw_input: str | list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a minimal MCP/tool-call style protocol response for /v1/responses."""
    approval_response = extract_openai_approval_response(raw_input)
    if approval_response is not None:
        approval_request_id, approve = approval_response
        pending = await approve_pending_tool_call(approval_request_id, approve)
        if pending is None:
            return {
                "id": f"resp_{uuid.uuid4().hex[:24]}",
                "object": "response",
                "created_at": int(time.time()),
                "status": "failed",
                "model": model_name,
                "output": [],
                "output_text": "Approval request not found.",
                "usage": {
                    "prompt_tokens": estimate_tokens(user_input),
                    "completion_tokens": 0,
                    "total_tokens": estimate_tokens(user_input),
                },
            }

        if not approve:
            return {
                "id": f"resp_{uuid.uuid4().hex[:24]}",
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": model_name,
                "output": [],
                "output_text": f"Tool call {pending.tool_name} was rejected by the user.",
                "usage": {
                    "prompt_tokens": estimate_tokens(user_input),
                    "completion_tokens": 0,
                    "total_tokens": estimate_tokens(user_input),
                },
            }

        return {
            "id": f"resp_{uuid.uuid4().hex[:24]}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "requires_action",
            "model": model_name,
            "output": [
                {
                    "type": "mcp_call",
                    "id": pending.tool_use_id,
                    "tool_name": pending.tool_name,
                    "arguments": pending.tool_input,
                    "status": "approved",
                }
            ],
            "output_text": "",
            "usage": {
                "prompt_tokens": estimate_tokens(user_input),
                "completion_tokens": 0,
                "total_tokens": estimate_tokens(user_input),
            },
        }

    tool_result = extract_openai_tool_result(raw_input)
    if tool_result is not None:
        tool_use_id, content, is_error = tool_result
        pending = await get_pending_tool_call(tool_use_id)

        if pending is None:
            message = f"Tool result received for unknown or expired tool call {tool_use_id}."
        elif not pending.approved:
            message = f"Tool result received for unapproved tool call {tool_use_id}."
        else:
            popped = await pop_pending_tool_call(tool_use_id)
            if popped is None:
                message = f"Tool call {tool_use_id} was no longer available."
            else:
                continuation_prompt = (
                    f"Original user request:\n{popped.original_user_text}\n\n"
                    f"Tool executed: {popped.tool_name}\n"
                    f"Tool status: {'error' if is_error else 'success'}\n\n"
                    f"Tool output:\n{content}\n\n"
                    "Please continue by responding to the user with the next helpful assistant message. "
                    "If the tool output indicates success, summarize what was done and any relevant next steps. "
                    "If the tool output indicates an error, explain the failure and suggest how to fix it."
                )

                try:
                    message = await run_perplexity_query(
                        model=get_model(model_name),
                        query=continuation_prompt,
                        system_text=None,
                    )
                except Exception as e:
                    message = f"Tool {popped.tool_name} completed, but continuation generation failed: {e}"

        return {
            "id": f"resp_{uuid.uuid4().hex[:24]}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model_name,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": message}],
                }
            ],
            "output_text": message,
            "usage": {
                "prompt_tokens": estimate_tokens(user_input),
                "completion_tokens": estimate_tokens(message),
                "total_tokens": estimate_tokens(user_input) + estimate_tokens(message),
            },
        }

    normalized_tools = claude_protocol.extract_tool_definitions(
        responses_tools_to_claude_tools(tools)
    )
    if not normalized_tools:
        return None

    lowered = user_input.lower()
    write_markers = (
        "write",
        "edit",
        "modify",
        "update",
        "create file",
        "save",
        "delete",
        "rename",
        "move file",
        "replace",
        "patch",
    )
    if not any(marker in lowered for marker in write_markers):
        return None

    tool = normalized_tools[0]
    tool_input = {"query": user_input}
    approval_required = claude_protocol.requires_approval(tool.name)

    tool_use_id, approval_request_id = await register_pending_tool_call(
        model_name=model_name,
        tool_name=tool.name,
        tool_input=tool_input,
        original_user_text=user_input,
        requires_approval=approval_required,
        approval_reason="This tool appears to modify files or workspace state.",
    )

    if approval_required and approval_request_id is not None:
        approval = claude_protocol.build_openai_mcp_approval_request(
            tool.name,
            tool_input,
            reason="This tool appears to modify files or workspace state.",
        )
        approval["approval_request_id"] = approval_request_id
        return {
            "id": f"resp_{uuid.uuid4().hex[:24]}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "requires_action",
            "model": model_name,
            "output": [
                approval,
                {
                    "type": "mcp_call",
                    "id": tool_use_id,
                    "tool_name": tool.name,
                    "arguments": tool_input,
                    "status": "pending_approval",
                },
            ],
            "output_text": "",
            "usage": {
                "prompt_tokens": estimate_tokens(user_input),
                "completion_tokens": 0,
                "total_tokens": estimate_tokens(user_input),
            },
        }

    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "requires_action",
        "model": model_name,
        "output": [
            {
                "type": "mcp_call",
                "id": tool_use_id,
                "tool_name": tool.name,
                "arguments": tool_input,
                "status": "approved",
            }
        ],
        "output_text": "",
        "usage": {
            "prompt_tokens": estimate_tokens(user_input),
            "completion_tokens": 0,
            "total_tokens": estimate_tokens(user_input),
        },
    }

def responses_tools_to_claude_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize OpenAI-style tool definitions into Claude helper input."""
    if not tools:
        return []
    return [tool for tool in tools if isinstance(tool, dict)]


async def run_perplexity_query(
    model: Model,
    query: str,
    system_text: str | None = None,
) -> str:
    """Run a single non-streaming Perplexity query with shared serialization logic."""
    global last_request_time
    import time as time_module

    async with perplexity_semaphore:
        now = time_module.time()
        wait_needed = MIN_REQUEST_INTERVAL - (now - last_request_time)
        if wait_needed > 0:
            logging.debug(f"Rate limiting: waiting {wait_needed:.1f}s")
            await asyncio.sleep(wait_needed)
        last_request_time = time_module.time()

        full_query = query
        if system_text:
            distilled = distill_system_prompt(system_text)
            full_query = f"[Instructions: {distilled}]\n\n{query}"

        fresh_client = Perplexity(session_token=config.session_token)
        try:
            conversation = fresh_client.create_conversation(
                ConversationConfig(model=model, citation_mode=CitationMode.DEFAULT)
            )
            await asyncio.to_thread(conversation.ask, full_query)
            answer = conversation.answer or ""
            citations = format_citations(conversation.search_results)
            return answer + citations
        finally:
            fresh_client.close()


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/")
async def root():
    """Server info (for discovery)."""
    return {
        "name": "perplexity-web-mcp",
        "version": "0.1.0",
        "description": "Anthropic & OpenAI API compatible server powered by Perplexity",
        "endpoints": {
            "anthropic": "/v1/messages",
            "openai": "/v1/chat/completions",
            "responses": "/v1/responses",
            "models": "/v1/models",
            "health": "/health",
        },
    }


class CountTokensRequest(BaseModel):
    """Request model for count_tokens endpoint."""
    model: str
    messages: list[MessageParam]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request, body: CountTokensRequest):
    """Count tokens in a messages request (Anthropic beta endpoint)."""
    verify_auth(request)
    query = messages_to_query(body.messages)
    input_tokens = estimate_tokens(query)
    return {
        "input_tokens": input_tokens,
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
        "backend": "perplexity",
        "sessions": conversation_manager.get_stats(),
    }


@app.get("/v1/models")
async def list_models(request: Request):
    """List available models (OpenAI-compatible format)."""
    verify_auth(request)

    now = int(time.time())
    return ModelsListResponse(
        data=[ModelObject(id=m["id"], created=now) for m in AVAILABLE_MODELS]
    )


@app.post("/v1/messages")
async def create_message(body: MessagesRequest, request: Request):
    anthropic_version = request.headers.get("anthropic-version", "")
    query = claude_input_to_query(body.messages)
    input_value = getattr(body, "input", None)
    system_text = (
        getattr(body, "system", None)
        or getattr(body, "instructions", None)
        or (responses_input_to_instructions(input_value) if input_value is not None else None)
    )
    thinking_enabled = is_thinking_enabled(reasoning=getattr(body, "reasoning", None))
    model = get_model(body.model, thinking=thinking_enabled)

    # Claude-native tools should exist on MessagesRequest, but guard anyway so this
    # path can tolerate mixed payloads during compatibility work.
    body_tools = getattr(body, "tools", None)
    protocol_output = await maybe_build_claude_protocol_response(
        model_name=body.model,
        messages=body.messages,
        system_text=system_text,
        tools=body_tools,
    )
    if protocol_output is not None:
        if body.stream:
            return StreamingResponse(
                claude_stream_protocol_response(protocol_output),
                media_type="text/event-stream",
            )
        return JSONResponse(content=protocol_output)

    logger.info(
        "Anthropic Request: model=%s, thinking=%s, stream=%s",
        body.model,
        thinking_enabled,
        body.stream,
    )

    response_text = await run_perplexity_query(
        model=model,
        query=query,
        system_text=system_text,
    )

    response = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": body.model,
        "content": [{"type": "text", "text": response_text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": estimate_tokens(query if isinstance(query, str) else str(query)),
            "output_tokens": estimate_tokens(response_text),
        },
    }

    if body.stream:
        return StreamingResponse(
            claude_stream_text_response(response, response_text),
            media_type="text/event-stream",
        )
    return JSONResponse(content=response)

async def stream_response(
    response_id: str,
    model_name: str,
    model: Model,
    query: str,
    input_tokens: int,
    system_text: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream Anthropic-format SSE response."""
    import threading
    import time as time_module

    await perplexity_semaphore.acquire()
    semaphore_released = False

    message_start = {
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

    queue: asyncio.Queue[tuple[str, str | tuple[str, str]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def release_semaphore():
        nonlocal semaphore_released
        if not semaphore_released:
            semaphore_released = True
            loop.call_soon_threadsafe(perplexity_semaphore.release)

    def producer():
        """Background thread to stream from Perplexity with retry."""
        global last_request_time

        now = time_module.time()
        wait_needed = MIN_REQUEST_INTERVAL - (now - last_request_time)
        if wait_needed > 0:
            logging.info(f"Rate limiting: waiting {wait_needed:.1f}s before next Perplexity request")
            time_module.sleep(wait_needed)

        last = ""
        max_retries = 3

        full_query = query
        if system_text:
            distilled = distill_system_prompt(system_text)
            full_query = f"[Instructions: {distilled}]\n\n{query}"
            logging.debug(f"Query with system context ({len(distilled)} chars)")

        for attempt in range(max_retries):
            try:
                fresh_client = Perplexity(session_token=config.session_token)
                conversation = fresh_client.create_conversation(
                    ConversationConfig(model=model, citation_mode=CitationMode.DEFAULT)
                )

                for resp in conversation.ask(full_query, stream=True):
                    current = resp.answer or ""
                    if len(current) > len(last):
                        delta = current[len(last):]
                        last = current
                        loop.call_soon_threadsafe(queue.put_nowait, ("delta", delta))

                citations = format_citations(conversation.search_results)
                loop.call_soon_threadsafe(queue.put_nowait, ("done", (last, citations)))
                fresh_client.close()
                last_request_time = time_module.time()
                break
            except Exception as e:
                error_str = str(e)
                last_request_time = time_module.time()
                if "curl" in error_str.lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logging.warning(f"Curl error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s")
                    time_module.sleep(wait_time)
                    last = ""
                    continue
                loop.call_soon_threadsafe(queue.put_nowait, ("error", error_str))
                break

        release_semaphore()

    threading.Thread(target=producer, daemon=True).start()

    total_output = ""
    citations_text = ""
    delta_count = 0

    while True:
        kind, payload = await queue.get()
        if kind == "delta":
            assert isinstance(payload, str)
            total_output += payload
            delta_count += 1

            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": payload},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

            if delta_count % 10 == 0:
                yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        elif kind == "error":
            assert isinstance(payload, str)
            logging.error(f"Stream error: {payload}")
            error_msg = payload
            if "403" in payload or "forbidden" in payload.lower():
                error_msg = (
                    "Session token expired (403). "
                    "Re-authenticate: pwm-auth --email EMAIL, then pwm-auth --email EMAIL --code CODE"
                )
            error_delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": f"\n\n[Error: {error_msg}]"},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(error_delta)}\n\n"
            break
        else:
            assert isinstance(payload, tuple)
            total_output, citations_text = payload
            break

    if citations_text:
        citation_delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": citations_text},
        }
        yield f"event: content_block_delta\ndata: {json.dumps(citation_delta)}\n\n"
        total_output += citations_text

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    output_tokens = estimate_tokens(total_output)
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    if not semaphore_released:
        perplexity_semaphore.release()

    logging.info(f"Stream complete: {output_tokens} output tokens")


# =============================================================================
# OpenAI Chat Completions Endpoint
# =============================================================================


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request, body: OpenAIChatRequest):
    """Create a chat completion (OpenAI Chat Completions API)."""
    verify_auth(request)

    if not body.messages:
        raise HTTPException(
            status_code=400,
            detail={"error": {"type": "invalid_request_error", "message": "messages is required"}},
        )

    thinking_enabled = is_thinking_enabled(reasoning_effort=body.reasoning_effort)
    model = get_model(body.model, thinking=thinking_enabled)
    query = openai_messages_to_query(body.messages)
    input_tokens = estimate_tokens(query)
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    logging.info(f"OpenAI Request: model={body.model}, thinking={thinking_enabled}, stream={body.stream}")

    if body.stream:
        return StreamingResponse(
            stream_openai_response(response_id, body.model, model, query, created),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        full_response = await run_perplexity_query(model=model, query=query, system_text=None)
        output_tokens = estimate_tokens(full_response)

        return OpenAIChatResponse(
            id=response_id,
            created=created,
            model=body.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChoiceMessage(role="assistant", content=full_response),
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

    except Exception as e:
        logging.error(f"Error creating chat completion: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": {"type": "api_error", "message": str(e)}},
        )


async def stream_openai_response(
    response_id: str,
    model_name: str,
    model: Model,
    query: str,
    created: int,
) -> AsyncGenerator[str, None]:
    """Stream OpenAI-format SSE response."""
    import threading

    initial_chunk = OpenAIStreamResponse(
        id=response_id,
        created=created,
        model=model_name,
        choices=[
            OpenAIStreamChoice(
                index=0,
                delta={"role": "assistant", "content": ""},
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    queue: asyncio.Queue[tuple[str, str | tuple[str, str]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def producer():
        """Background thread to stream from Perplexity."""
        last = ""
        try:
            conversation = client.create_conversation(
                ConversationConfig(model=model, citation_mode=CitationMode.DEFAULT)
            )
            for resp in conversation.ask(query, stream=True):
                current = resp.answer or ""
                if len(current) > len(last):
                    delta = current[len(last):]
                    last = current
                    loop.call_soon_threadsafe(queue.put_nowait, ("delta", delta))
            citations = format_citations(conversation.search_results)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", (last, citations)))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

    threading.Thread(target=producer, daemon=True).start()

    citations_text = ""
    while True:
        kind, payload = await queue.get()
        if kind == "delta":
            assert isinstance(payload, str)
            delta_chunk = OpenAIStreamResponse(
                id=response_id,
                created=created,
                model=model_name,
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta={"content": payload},
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {delta_chunk.model_dump_json()}\n\n"

        elif kind == "error":
            assert isinstance(payload, str)
            logging.error(f"Stream error: {payload}")
            error_chunk = OpenAIStreamResponse(
                id=response_id,
                created=created,
                model=model_name,
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta={"content": f"\n\n[Error: {payload}]"},
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            break
        else:
            assert isinstance(payload, tuple)
            _, citations_text = payload
            break

    if citations_text:
        citation_chunk = OpenAIStreamResponse(
            id=response_id,
            created=created,
            model=model_name,
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta={"content": citations_text},
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {citation_chunk.model_dump_json()}\n\n"

    final_chunk = OpenAIStreamResponse(
        id=response_id,
        created=created,
        model=model_name,
        choices=[
            OpenAIStreamChoice(
                index=0,
                delta={},
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

    logging.info("OpenAI stream complete")

@app.post("/v1/responses")
async def create_response(request: Request, body: OpenAIResponsesRequest):
    """Create a response (minimal OpenAI Responses API compatibility layer)."""
    verify_auth(request)

    logging.debug(
        "Responses POST received: model=%s stream=%s tools=%s input_type=%s",
        body.model,
        body.stream,
        len(body.tools or []),
        type(body.input).__name__,
    )

    query = responses_input_to_query(body.input)
    logging.debug("Responses POST parsed query: %r", query[:500])

    if not query:
        raise HTTPException(
            status_code=400,
            detail={"error": {"type": "invalid_request_error", "message": "input is required"}},
        )

    system_text = body.instructions or responses_input_to_instructions(body.input)
    thinking_enabled = is_thinking_enabled(reasoning=body.reasoning)
    model = get_model(body.model, thinking=thinking_enabled)

    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    input_tokens = estimate_tokens(query)

    logging.info(
        f"OpenAI Responses Request: model={body.model}, thinking={thinking_enabled}, stream={body.stream}"
    )

    if body.stream:
        return StreamingResponse(
            stream_openai_responses_response(
                response_id=response_id,
                model_name=body.model,
                model=model,
                query=query,
                created=created,
                system_text=system_text,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        full_response = await run_perplexity_query(model=model, query=query, system_text=system_text)
        output_tokens = estimate_tokens(full_response)

        return OpenAIResponsesResponse(
            id=response_id,
            created_at=created,
            model=body.model,
            output=[
                OpenAIResponseOutputMessage(
                    id=f"msg_{uuid.uuid4().hex[:24]}",
                    content=[OpenAIResponseOutputText(text=full_response)],
                )
            ],
            output_text=full_response,
            usage=OpenAIUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
    except Exception as e:
        logging.error(f"Error creating response: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": {"type": "api_error", "message": str(e)}},
        )


async def stream_openai_responses_response(
    response_id: str,
    model_name: str,
    model: Model,
    query: str,
    created: int,
    system_text: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a minimal Responses-API-compatible SSE response."""
    import threading
    import time as time_module

    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def producer():
        global last_request_time
        now = time_module.time()
        wait_needed = MIN_REQUEST_INTERVAL - (now - last_request_time)
        if wait_needed > 0:
            time_module.sleep(wait_needed)

        full_query = query
        if system_text:
            distilled = distill_system_prompt(system_text)
            full_query = f"[Instructions: {distilled}]\n\n{query}"

        last = ""
        try:
            fresh_client = Perplexity(session_token=config.session_token)
            try:
                conversation = fresh_client.create_conversation(
                    ConversationConfig(model=model, citation_mode=CitationMode.DEFAULT)
                )
                for resp in conversation.ask(full_query, stream=True):
                    current = resp.answer or ""
                    if len(current) > len(last):
                        delta = current[len(last):]
                        last = current
                        loop.call_soon_threadsafe(queue.put_nowait, ("delta", delta))
                citations = format_citations(conversation.search_results)
                loop.call_soon_threadsafe(queue.put_nowait, ("done", last + citations))
                last_request_time = time_module.time()
            finally:
                fresh_client.close()
        except Exception as e:
            last_request_time = time_module.time()
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

    threading.Thread(target=producer, daemon=True).start()

    yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': {'id': response_id, 'object': 'response', 'created_at': created, 'model': model_name, 'status': 'in_progress'}})}\n\n"

    accumulated = ""
    while True:
        kind, payload = await queue.get()
        if kind == "delta":
            accumulated += payload
            yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'delta': payload})}\n\n"
        elif kind == "error":
            yield f"event: response.error\ndata: {json.dumps({'type': 'response.error', 'error': {'message': payload}})}\n\n"
            break
        else:
            if payload.startswith(accumulated):
                remainder = payload[len(accumulated):]
                if remainder:
                    yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'delta': remainder})}\n\n"
                accumulated = payload
            else:
                accumulated = payload
            break

    yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': {'id': response_id, 'object': 'response', 'created_at': created, 'model': model_name, 'status': 'completed', 'output_text': accumulated}})}\n\n"


@app.websocket("/v1/responses")
async def responses_websocket(websocket: WebSocket):
    """Minimal websocket compatibility endpoint for clients probing /v1/responses."""
    await websocket.accept()
    logging.debug("Responses websocket accepted")

    try:
        logging.debug("Responses websocket waiting for payload")

        try:
            payload = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            logging.debug("Responses websocket payload: %s", payload)
        except asyncio.TimeoutError:
            logging.debug("Responses websocket timeout waiting for payload")
            payload = None
        except Exception as e:
            logging.debug("Responses websocket receive error: %s", e)
            payload = None

        if not payload:
            logging.debug("Responses websocket no payload; closing")
            await websocket.close(code=1000)
            return

        payload_type = str(payload.get("type", "")).strip()
        model_name = str(payload.get("model", config.default_model))
        raw_input = payload.get("input", "")
        top_level_instructions = payload.get("instructions")
        structured_instructions = responses_input_to_instructions(raw_input)
        system_text = top_level_instructions or structured_instructions
        query = extract_latest_user_text_from_responses_input(raw_input)
        thinking_enabled = is_thinking_enabled(reasoning=payload.get("reasoning"))
        model = get_model(model_name, thinking=thinking_enabled)
        tools = payload.get("tools")

        logging.debug(
            "Responses websocket parsed: type=%s model=%s thinking=%s query=%r tools_present=%s",
            payload_type,
            model_name,
            thinking_enabled,
            query[:500],
            bool(tools),
        )

        if payload_type and payload_type != "response.create":
            error_payload = {
                "type": "error",
                "error": {"message": f"Unsupported websocket payload type: {payload_type}"},
            }
            logging.debug("Responses websocket unsupported payload type: %s", error_payload)
            await websocket.send_json(error_payload)
            await websocket.close(code=1003)
            return

        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        event_created_id = f"event_{uuid.uuid4().hex[:24]}"
        input_token_count = estimate_tokens(query)

        created_payload = {
            "event_id": event_created_id,
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "status_details": None,
                "model": model_name,
                "output": [],
                "usage": {
                    "input_tokens": input_token_count,
                    "output_tokens": 0,
                    "total_tokens": input_token_count,
                },
                "metadata": None,
            },
        }

        if not query:
            logging.debug("Responses websocket no actionable user input; sending created event only")
            await websocket.send_json(created_payload)
            logging.debug("Responses websocket closing cleanly")
            await websocket.close(code=1000)
            return

        logging.debug("Responses websocket sending created event")
        await websocket.send_json(created_payload)

        protocol_output = await maybe_build_responses_protocol_output(
            model_name=model_name,
            user_input=query,
            tools=tools if isinstance(tools, list) else None,
            raw_input=raw_input,
        )

        if protocol_output is not None:
            logging.debug(
                "Responses websocket using protocol output: status=%s output_types=%s",
                protocol_output.get("status"),
                [item.get("type") for item in protocol_output.get("output", [])],
            )

            output_items = protocol_output.get("output", [])
            for idx, item in enumerate(output_items):
                item_id = str(item.get("id", f"item_{uuid.uuid4().hex[:24]}"))
                item_type = str(item.get("type", ""))

                if item_type == "mcp_approval_request":
                    approval_event = {
                        "event_id": f"event_{uuid.uuid4().hex[:24]}",
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": idx,
                        "item": {
                            "id": item_id,
                            "object": "realtime.item",
                            "type": "mcp_approval_request",
                            "status": "completed",
                            "approval_request_id": item.get("approval_request_id"),
                            "tool_name": item.get("tool_name"),
                            "input": item.get("input"),
                            "reason": item.get("reason"),
                        },
                    }
                    logging.debug("Responses websocket sending approval request item")
                    await websocket.send_json(approval_event)

                elif item_type == "mcp_call":
                    tool_call_event = {
                        "event_id": f"event_{uuid.uuid4().hex[:24]}",
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": idx,
                        "item": {
                            "id": item_id,
                            "object": "realtime.item",
                            "type": "function_call",
                            "status": item.get("status", "completed"),
                            "name": item.get("tool_name"),
                            "call_id": item.get("id"),
                            "arguments": json.dumps(item.get("arguments", {})),
                        },
                    }
                    logging.debug("Responses websocket sending tool call item")
                    await websocket.send_json(tool_call_event)

                elif item_type == "message":
                    content = item.get("content", [])
                    text_value = ""
                    if isinstance(content, list) and content:
                        first = content[0]
                        if isinstance(first, dict):
                            text_value = str(first.get("text", ""))

                    part_added_event = {
                        "event_id": f"event_{uuid.uuid4().hex[:24]}",
                        "type": "response.content_part.added",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": idx,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    }
                    await websocket.send_json(part_added_event)

                    delta_event = {
                        "event_id": f"event_{uuid.uuid4().hex[:24]}",
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": idx,
                        "content_index": 0,
                        "delta": text_value,
                    }
                    await websocket.send_json(delta_event)

                    done_event = {
                        "event_id": f"event_{uuid.uuid4().hex[:24]}",
                        "type": "response.output_text.done",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": idx,
                        "content_index": 0,
                        "text": text_value,
                    }
                    await websocket.send_json(done_event)

                    item_done_event = {
                        "event_id": f"event_{uuid.uuid4().hex[:24]}",
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": idx,
                        "item": {
                            "id": item_id,
                            "object": "realtime.item",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": text_value,
                                    "annotations": [],
                                }
                            ],
                        },
                    }
                    await websocket.send_json(item_done_event)

            protocol_output_text = str(protocol_output.get("output_text", ""))
            protocol_usage = protocol_output.get("usage") or {
                "input_tokens": input_token_count,
                "output_tokens": estimate_tokens(protocol_output_text),
                "total_tokens": input_token_count + estimate_tokens(protocol_output_text),
            }

            completed_payload = {
                "event_id": f"event_{uuid.uuid4().hex[:24]}",
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": protocol_output.get("status", "completed"),
                    "status_details": None,
                    "model": model_name,
                    "output": output_items,
                    "usage": protocol_usage,
                    "metadata": None,
                    "output_text": protocol_output_text,
                },
            }
            logging.debug("Responses websocket sending protocol completed event")
            await websocket.send_json(completed_payload)
            logging.debug("Responses websocket closing cleanly")
            await websocket.close(code=1000)
            return

        event_part_added_id = f"event_{uuid.uuid4().hex[:24]}"
        event_text_delta_id = f"event_{uuid.uuid4().hex[:24]}"
        event_text_done_id = f"event_{uuid.uuid4().hex[:24]}"
        event_part_done_id = f"event_{uuid.uuid4().hex[:24]}"
        event_item_done_id = f"event_{uuid.uuid4().hex[:24]}"
        event_completed_id = f"event_{uuid.uuid4().hex[:24]}"

        response_text = await run_perplexity_query(model=model, query=query, system_text=system_text)
        output_token_count = estimate_tokens(response_text)

        content_part_added_payload = {
            "event_id": event_part_added_id,
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
            },
        }
        logging.debug("Responses websocket sending content_part.added")
        await websocket.send_json(content_part_added_payload)

        delta_payload = {
            "event_id": event_text_delta_id,
            "type": "response.output_text.delta",
            "response_id": response_id,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "delta": response_text,
        }
        logging.debug("Responses websocket sending output_text.delta")
        await websocket.send_json(delta_payload)

        done_payload = {
            "event_id": event_text_done_id,
            "type": "response.output_text.done",
            "response_id": response_id,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "text": response_text,
        }
        logging.debug("Responses websocket sending output_text.done")
        await websocket.send_json(done_payload)

        content_part_done_payload = {
            "event_id": event_part_done_id,
            "type": "response.content_part.done",
            "response_id": response_id,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": response_text,
                "annotations": [],
            },
        }
        logging.debug("Responses websocket sending content_part.done")
        await websocket.send_json(content_part_done_payload)

        output_item = {
            "id": message_id,
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": response_text,
                    "annotations": [],
                }
            ],
        }

        output_item_done_payload = {
            "event_id": event_item_done_id,
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": 0,
            "item": output_item,
        }
        logging.debug("Responses websocket sending output_item.done")
        await websocket.send_json(output_item_done_payload)

        completed_payload = {
            "event_id": event_completed_id,
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "completed",
                "status_details": None,
                "model": model_name,
                "output": [output_item],
                "usage": {
                    "input_tokens": input_token_count,
                    "output_tokens": output_token_count,
                    "total_tokens": input_token_count + output_token_count,
                },
                "metadata": None,
                "output_text": response_text,
            },
        }
        logging.debug("Responses websocket sending completed event")
        await websocket.send_json(completed_payload)
        logging.debug("Responses websocket closing cleanly")
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        logging.debug("Responses websocket disconnected by client")
        logging.info("Responses websocket disconnected")
    except Exception:
        logging.exception("Responses websocket error")
        raise


# =============================================================================
# Main
# =============================================================================


def run_server():
    """Run the API server."""
    cfg = ServerConfig.from_env()
    uvicorn.run(
        "perplexity_web_mcp.api.server:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()

