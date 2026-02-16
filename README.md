<p align="center">
  <img src="assets/logo.png" alt="Perplexity Web MCP" width="700">
</p>

# Perplexity Web MCP

MCP server, CLI, and API-compatible interface for Perplexity AI's web interface.

Use your Perplexity Pro/Max subscription to access premium models (GPT-5.2, Claude 4.6 Opus, Claude 4.5 Sonnet, Gemini 3, Grok 4.1, Kimi K2.5) from the terminal, through MCP tools, or as an API endpoint.

## Features

- **CLI**: Query Perplexity models directly from the terminal (`pwm ask`, `pwm research`)
- **MCP Server**: 17 MCP tools for AI agents with citations and rate limit checking
- **API Server**: Drop-in Anthropic Messages API and OpenAI Chat Completions API
- **10 Models**: GPT-5.2, Claude 4.6 Opus, Claude 4.5 Sonnet, Gemini 3 Flash/Pro, Grok 4.1, Kimi K2.5, Sonar
- **Thinking Mode**: Extended thinking support for all compatible models
- **Deep Research**: Full support for Perplexity's Deep Research mode
- **Setup & Skill Management**: Auto-configure MCP for Claude, Cursor, Windsurf, Gemini CLI; install Agent Skills across platforms
- **Doctor**: Diagnose installation, auth, config, rate limits, and skill status

---

## Installation

### From PyPI (recommended)

**Using uv:**

```bash
uv tool install "perplexity-web-mcp-cli[all]"
```

**Using pipx:**

```bash
pipx install "perplexity-web-mcp-cli[all]"
```

**Using pip:**

```bash
pip install "perplexity-web-mcp-cli[all]"
```

> **Note:** Requires Python 3.10-3.13.

### From source (for development)

```bash
git clone https://github.com/jacob-bd/perplexity-web-mcp.git
cd perplexity-web-mcp
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"
```

### Install variants

- `[all]` -- MCP server + API server (recommended)
- `[mcp]` -- MCP server only
- `[api]` -- API server only
- No extras -- CLI + Python library only

### Upgrading

```bash
pip install --upgrade perplexity-web-mcp-cli[all]
```

After upgrading, restart your MCP client (Claude Code, Cursor, etc.) to reload the server.

---

## Quick Start

```bash
# 1. Authenticate
pwm login

# 2. Ask a question
pwm ask "What is quantum computing?"

# 3. Deep research
pwm research "agentic AI trends 2026"

# 4. Check your remaining quotas
pwm usage

# 5. Set up MCP for your AI tools
pwm setup add claude-code
pwm setup add cursor

# 6. Install the Agent Skill
pwm skill install claude-code

# 7. Diagnose any issues
pwm doctor
```

---

## CLI Reference

### Querying

```bash
pwm ask "What is quantum computing?"                      # Auto-select best model
pwm ask "latest AI news" -m gpt52 -s academic             # GPT-5.2 + academic sources
pwm ask "explain transformers" -m claude_sonnet --thinking # Claude 4.5 + thinking
pwm ask "query" --json                                    # JSON output
pwm ask "query" --no-citations                            # No citation URLs
```

### Deep Research

```bash
pwm research "agentic AI trends 2026"                     # Full research report
pwm research "climate policy" -s academic --json           # Academic + JSON output
```

### Authentication

```bash
pwm login                                    # Interactive login (email + OTP)
pwm login --check                            # Check if authenticated
pwm login --email user@example.com           # Send verification code (non-interactive)
pwm login --email user@example.com --code 123456  # Complete auth with code
```

### Usage & Limits

```bash
pwm usage                  # Check remaining rate limits
pwm usage --refresh        # Force-refresh from Perplexity servers
```

### MCP Setup

```bash
pwm setup list             # Show supported tools and MCP configuration status
pwm setup add claude-code  # Add MCP server to Claude Code
pwm setup add cursor       # Add MCP server to Cursor
pwm setup add windsurf     # Add MCP server to Windsurf
pwm setup add gemini       # Add MCP server to Gemini CLI
pwm setup remove cursor    # Remove MCP server from a tool
```

### Skill Management

```bash
pwm skill list                            # Show installation status per platform
pwm skill install claude-code             # Install skill for Claude Code
pwm skill install cursor --level project  # Install at project level
pwm skill uninstall gemini-cli            # Remove skill
pwm skill update                          # Update all outdated skills
pwm skill show                            # Display skill content
```

### Doctor

```bash
pwm doctor                 # Diagnose installation, auth, config, limits
pwm doctor -v              # Verbose (includes security + per-platform skill status)
```

### AI Documentation

```bash
pwm --ai                   # Print comprehensive AI-optimized reference
```

---

## Models

| CLI Name | Provider | Thinking | Notes |
|----------|----------|----------|-------|
| `auto` | Perplexity | No | Auto-selects best model |
| `sonar` | Perplexity | No | Perplexity's latest model |
| `deep_research` | Perplexity | No | Monthly quota, in-depth reports |
| `gpt52` | OpenAI | Toggle | GPT-5.2 |
| `claude_sonnet` | Anthropic | Toggle | Claude 4.5 Sonnet |
| `claude_opus` | Anthropic | Toggle | Claude 4.6 Opus (Max tier required) |
| `gemini_flash` | Google | Toggle | Gemini 3 Flash |
| `gemini_pro` | Google | Always | Gemini 3 Pro |
| `grok` | xAI | Toggle | Grok 4.1 |
| `kimi` | Moonshot | Always | Kimi K2.5 |

**Source focus options:** `web` (default), `academic`, `social`, `finance`, `all`

---

## MCP Server

### Setup

The easiest way to configure MCP:

```bash
pwm setup add claude-code
```

Or configure manually for any MCP client:

**Claude Code CLI:**
```bash
claude mcp add perplexity pwm-mcp
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "perplexity": {
      "command": "pwm-mcp"
    }
  }
}
```

**Cursor** (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "perplexity": {
      "command": "pwm-mcp"
    }
  }
}
```

### Available MCP Tools

**Query tools (14):**

| Tool | Description |
|------|-------------|
| `pplx_query` | Flexible: model selection + thinking toggle |
| `pplx_ask` | Quick Q&A (auto-selects best model) |
| `pplx_deep_research` | In-depth reports with sources |
| `pplx_sonar` | Perplexity Sonar |
| `pplx_gpt52` / `pplx_gpt52_thinking` | GPT-5.2 |
| `pplx_claude_sonnet` / `pplx_claude_sonnet_think` | Claude 4.5 Sonnet |
| `pplx_gemini_flash` / `pplx_gemini_flash_think` | Gemini 3 Flash |
| `pplx_gemini_pro_think` | Gemini 3 Pro (thinking always on) |
| `pplx_grok` / `pplx_grok_thinking` | Grok 4.1 |
| `pplx_kimi_thinking` | Kimi K2.5 (thinking always on) |

**Usage & auth tools (4):**

| Tool | Description |
|------|-------------|
| `pplx_usage` | Check remaining quotas |
| `pplx_auth_status` | Check authentication status |
| `pplx_auth_request_code` | Send verification code to email |
| `pplx_auth_complete` | Complete auth with 6-digit code |

All query tools support `source_focus`: `web`, `academic`, `social`, `finance`, `all`.

---

## API Server

Use Perplexity models through Anthropic or OpenAI compatible API endpoints.

### Start the server

```bash
pwm-api
```

### Anthropic API (Claude Code)

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=perplexity
claude --model gpt-5.2
```

### OpenAI API

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=anything
```

### API Model Names

| API Name | Perplexity Model | Thinking |
|----------|------------------|----------|
| `perplexity-auto` | Best (auto-select) | No |
| `gpt-5.2` | GPT-5.2 | Toggle |
| `claude-sonnet-4-5` | Claude 4.5 Sonnet | Toggle |
| `claude-opus-4-6` | Claude 4.6 Opus | Toggle |
| `gemini-3-flash` | Gemini 3 Flash | Toggle |
| `gemini-3-pro` | Gemini 3 Pro | Always |
| `grok-4.1` | Grok 4.1 | Toggle |
| `kimi-k2.5` | Kimi K2.5 | Always |

Legacy aliases (`claude-3-5-sonnet`, `claude-3-opus`) are supported for compatibility.

---

## Python API

```python
from perplexity_web_mcp import Perplexity, ConversationConfig, Models

client = Perplexity(session_token="your_token")
conversation = client.create_conversation(
    ConversationConfig(model=Models.CLAUDE_45_SONNET)
)

conversation.ask("What is quantum computing?")
print(conversation.answer)

for result in conversation.search_results:
    print(f"Source: {result.url}")

# Follow-up (context preserved)
conversation.ask("Explain it simpler")
print(conversation.answer)
```

---

## Subscription Tiers & Rate Limits

| Tier | Cost | Pro Search | Deep Research | Labs |
|------|------|------------|---------------|------|
| Free | $0 | 3/day | 1/month | No |
| Pro | $20/mo | Weekly pool | Monthly pool | Monthly pool |
| Max | $200/mo | Weekly pool | Monthly pool | Monthly pool |

The MCP server checks quotas before each query. Use `pwm usage` or `pplx_usage` to check your limits.

---

## Troubleshooting

### Authentication Errors (403)

Session tokens last ~30 days. Re-authenticate when expired:

```bash
pwm login
```

**Non-interactive (for AI agents):**

```bash
pwm login --email your@email.com
```
```bash
pwm login --email your@email.com --code 123456
```

**Via MCP tools (for AI agents without shell):**

1. Call `pplx_auth_request_code(email="your@email.com")`
2. Check email for 6-digit code
3. Call `pplx_auth_complete(email="your@email.com", code="123456")`

### Diagnose Issues

```bash
pwm doctor
```

This checks installation, authentication, rate limits, MCP configuration, and skill installation -- with fix suggestions for every issue found.

### Rate Limiting

- **CLI/MCP**: Auto-checks quotas before each query, blocks if exhausted
- **API server**: Enforces 5-second minimum between requests

---

## Agent Skill

This project includes a portable [Agent Skill](https://agentskills.io/) (SKILL.md) that teaches AI agents how to use the CLI and MCP tools. Install it for your platform:

```bash
pwm skill install claude-code
pwm skill install cursor
pwm skill install gemini-cli
```

The skill follows Anthropic's Agent Skills open standard and works across any compliant AI platform.

---

## Credits

Originally forked from [perplexity-webui-scraper](https://github.com/henrique-coder/perplexity-webui-scraper) by [henrique-coder](https://github.com/henrique-coder).

## License

MIT
