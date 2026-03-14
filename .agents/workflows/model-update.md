---
description: How to detect and apply Perplexity model changes (add/remove models)
---

# Model Update Workflow

## Step 1: Detect Model Changes (Automated)

Run the model detection script that fetches the live config from Perplexity's API and diffs it against our stored reference:

```bash
// turbo
python scripts/detect_model_changes.py
```

This will:
- Fetch `https://www.perplexity.ai/rest/models/config?config_schema=v1`
- Compare against `scripts/reference_model_config.json`
- Print a diff showing added/removed/changed models
- Show exactly which files need updating

> **Note:** This API is public and does NOT require authentication. It returns the full model catalog including identifiers, labels, providers, and the active UI config.

## Step 2: Review Changes

The script output will show:
- **New models** — need to be added to the codebase
- **Removed models** — need to be removed from the codebase
- **Changed models** — labels/descriptions updated
- **Config changes** — models added/removed from the active model selector

Pay attention to the `config` array — this defines what actually appears in the model selector dropdown. A model can exist in the `models` dict but not be in the active UI selector.

## Step 3: Update the Codebase

For each model change, update these files in order:

### Core (always required)
1. `src/perplexity_web_mcp/models.py` — Add/remove `Model` entries in `Models` class
2. `src/perplexity_web_mcp/shared.py` — Update `MODEL_MAP` and `ALL_SHORTCUTS`

### MCP Server
3. `src/perplexity_web_mcp/mcp/server.py` — Add/remove `pplx_<model>` tool functions, update `pplx_query` docstring

### API Server
4. `src/perplexity_web_mcp/api/server.py` — Update `MODEL_ALIASES`, `AVAILABLE_MODELS` list

### CLI
5. `src/perplexity_web_mcp/cli/ai_doc.py` — Update the model table in help text

### Documentation
6. `README.md` — Update model counts, model tables, and tool tables
7. `CLAUDE.md` — Update model list
8. `src/perplexity_web_mcp/data/SKILL.md` — Update skill docs
9. `src/perplexity_web_mcp/data/references/models.md` — Update model reference
10. `src/perplexity_web_mcp/data/references/mcp-tools.md` — Update MCP tools reference
11. `src/perplexity_web_mcp/data/references/api-endpoints.md` — Update API docs
12. `skills/perplexity-web-mcp/` — Mirror changes from `src/.../data/`

### Tests
13. `tests/test_shared.py` — Update model shortcut tests
14. `tests/test_rate_limits.py` — Update rate limit tests

## Step 4: Update Reference Snapshot

After applying changes, update the reference config:

```bash
// turbo
python scripts/detect_model_changes.py --save
```

## Step 5: Verify

```bash
// turbo
python -m pytest tests/ -x -q
```

## Key Technical Details

### API Endpoint
- **URL:** `https://www.perplexity.ai/rest/models/config?config_schema=v1`
- **Method:** GET (no auth required)
- **Returns:** JSON with `models` (full catalog), `config` (active selector items), `default_models`

### Model Config Structure
Each entry in `config` array:
```json
{
  "label": "Display Name",
  "description": "Model description",
  "has_new_tag": false,
  "subscription_tier": "pro" | "max",
  "non_reasoning_model": "identifier" | null,
  "reasoning_model": "identifier" | null,
  "text_only_model": false
}
```

- If `non_reasoning_model` is null → model is **always thinking** (reasoning only)
- If `reasoning_model` is null → model has **no thinking mode**
- If both are set → model has **toggle thinking**

### Provider Enum
`PERPLEXITY`, `ANTHROPIC`, `OPENAI`, `GOOGLE`, `XAI`, `MOONSHOT_AI`, `NVIDIA`
