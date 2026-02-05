# Perplexity Web MCP

## What This Is

An MCP server and Anthropic API-compatible interface that provides access to Perplexity AI models (GPT-5.2, Claude 4.5, Gemini 3, Grok 4.1, Kimi K2.5) through the web interface. Currently supports chat with streaming, model selection, and thinking modes — but not tool calling.

## Core Value

Enable Perplexity web models to execute local tools (Read, Write, Bash) through Claude Code, unlocking agentic workflows without requiring the official Perplexity API.

## Requirements

### Validated

- ✓ MCP server for Claude Code/Cursor integration — existing
- ✓ Anthropic API-compatible server (POST /v1/messages) — existing
- ✓ OpenAI API-compatible endpoint (POST /v1/chat/completions) — existing
- ✓ Email-based authentication with session token storage — existing
- ✓ Multiple model support (GPT-5.2, Claude 4.5 Sonnet/Opus, Gemini 3, Grok 4.1, Kimi K2.5) — existing
- ✓ Thinking mode toggle for all compatible models — existing
- ✓ Streaming responses with SSE — existing
- ✓ Citation extraction and formatting — existing
- ✓ Deep Research mode support — existing
- ✓ File upload support (PDF, text, audio) — existing
- ✓ Search focus options (Web, Academic, Social, Finance) — existing

### Active

- [ ] Perplexity models output parseable tool calls when given tool definitions
- [ ] Tool calls execute locally (Read, Write, Bash) through Claude Code
- [ ] Tool results can be fed back to continue the conversation

### Out of Scope

- Intermediate parsing model — user prefers pure solution without second model
- Official Perplexity API — defeats the purpose of leveraging web interface access
- Browser automation — too fragile for production use

## Context

**Previous attempts that failed:**
1. XML tag format (`<tool_call>...</tool_call>`) — models ignored format entirely
2. ReAct format (detailed) — models explained ReAct instead of using it
3. ReAct format (minimal with examples) — models still conversed instead of acting
4. Bare minimum instructions — same conversational behavior

**Root cause identified:**
- Perplexity web UI models are optimized for conversational search, not structured output
- No system prompt distinction — everything appears as user content
- Models default to explaining rather than executing

**Potential research directions:**
- How Ollama enables tool calling (grammars? constrained decoding?)
- Hermes function calling format (trained into many open models)
- Other approaches that have made non-tool-trained models produce structured output
- Whether Perplexity's internal tool support could be leveraged

## Constraints

- **No second model**: Solution must not require running another LLM to parse output
- **Web interface only**: Must work through Perplexity's web UI, not official API
- **Maintain existing functionality**: Cannot break working chat/search features

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on pure solution | User preference — avoid latency/complexity of intermediate model | — Pending |
| Research-first approach | Previous attempts failed — need new angles before more implementation | — Pending |

---
*Last updated: 2026-02-04 after initialization*
