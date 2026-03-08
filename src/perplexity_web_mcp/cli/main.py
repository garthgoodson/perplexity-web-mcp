"""Unified CLI for Perplexity Web MCP.

Entry point: pwm

Subcommands:
    pwm login           Authenticate with Perplexity (interactive or non-interactive)
    pwm ask "query"     Ask a question (web search + AI model)
    pwm research "q"    Deep research on a topic
    pwm api             Start the Anthropic/OpenAI API-compatible server
    pwm usage           Check remaining rate limits and quotas
    pwm hack claude     Launch Claude Code connected to Perplexity models
    pwm skill           Manage skill installation across AI platforms
    pwm doctor          Diagnose installation, auth, config, and limits
    pwm --ai            Print AI-optimized documentation (for LLM agents)
    pwm --help          Show help
    pwm --version       Show version
"""

from __future__ import annotations

import sys
from importlib import metadata
from typing import NoReturn

import rich_click as click

from perplexity_web_mcp.exceptions import AuthenticationError, RateLimitError
from perplexity_web_mcp.shared import (
    MODEL_MAP,
    MODEL_NAMES,
    SOURCE_FOCUS_NAMES,
    Models,
    SourceFocusName,
    ask,
    get_limit_cache,
    resolve_model,
)
from perplexity_web_mcp.token_store import load_token


# ── Click configuration ────────────────────────────────────────────────────


def _print_ai_docs(ctx, param, value):
    """Callback for --ai flag. Outputs AI-friendly documentation."""
    if not value or ctx.resilient_parsing:
        return
    from perplexity_web_mcp.cli.ai_doc import print_ai_doc
    print_ai_doc()
    ctx.exit(0)


def _print_version(ctx, param, value):
    """Callback for --version flag."""
    if not value or ctx.resilient_parsing:
        return
    version = metadata.version("perplexity-web-mcp-cli")
    click.echo(f"perplexity-web-mcp-cli {version}")
    ctx.exit(0)


@click.group(invoke_without_command=True)
@click.option("--version", "-v", is_flag=True, callback=_print_version,
              expose_value=False, is_eager=True, help="Show version.")
@click.option("--ai", is_flag=True, callback=_print_ai_docs,
              expose_value=False, is_eager=True,
              help="Print AI-optimized documentation (for LLM agents).")
@click.pass_context
def cli(ctx):
    """pwm — Perplexity Web MCP CLI.

    Ask questions, run deep research, manage MCP server setup,
    and more — all powered by Perplexity AI.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ── Ask ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("-m", "--model", "model_name", default="auto",
              help=f"Model to use ({', '.join(MODEL_NAMES)}).")
@click.option("-t", "--thinking", is_flag=True, help="Enable extended thinking mode.")
@click.option("-s", "--source", "source", default="web",
              help=f"Source focus ({', '.join(SOURCE_FOCUS_NAMES)}).")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@click.option("--no-citations", is_flag=True, help="Suppress citation URLs.")
@click.option("--intent", default="standard",
              help="Routing intent: quick, standard, detailed, research.")
def ask_cmd(query, model_name, thinking, source, json_output, no_citations, intent):
    """Ask a question using Perplexity AI.

    \b
    Examples:
      pwm ask "What is quantum computing?"
      pwm ask "latest AI news" -m gpt52 -s academic
      pwm ask "explain transformers" -m claude_sonnet --thinking
    """
    code = _cmd_ask_impl(query, model_name, thinking, source, json_output, no_citations, intent)
    raise SystemExit(code)


def _cmd_ask_impl(query, model_name, thinking, source, json_output, no_citations, intent):
    """Implementation for ask command (kept separate for testability)."""
    if source not in SOURCE_FOCUS_NAMES:
        print(f"Error: Unknown source '{source}'. Available: {', '.join(SOURCE_FOCUS_NAMES)}", file=sys.stderr)
        return 1

    try:
        explicit_model = model_name != "auto"
        if explicit_model:
            if model_name not in MODEL_MAP:
                print(f"Error: Unknown model '{model_name}'. Available: {', '.join(MODEL_NAMES)}", file=sys.stderr)
                return 1

            model = resolve_model(model_name, thinking=thinking)
            result = ask(query, model, source)

            if json_output:
                import orjson

                parts = result.split("\n\nCitations:")
                answer_text = parts[0]
                citations = []
                if len(parts) > 1:
                    for line in parts[1].strip().split("\n"):
                        line = line.strip()
                        if line.startswith("[") and "]: " in line:
                            url = line.split("]: ", 1)[1]
                            citations.append(url)
                data = {"answer": answer_text, "citations": citations, "model": model_name, "source": source}
                sys.stdout.buffer.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
                sys.stdout.buffer.write(b"\n")
            elif no_citations:
                parts = result.split("\n\nCitations:")
                print(parts[0])
            else:
                print(result)
        else:
            from perplexity_web_mcp.shared import smart_ask

            response = smart_ask(query, intent=intent, source_focus=source)
            if json_output:
                import orjson

                sys.stdout.buffer.write(orjson.dumps(response.to_dict(), option=orjson.OPT_INDENT_2))
                sys.stdout.buffer.write(b"\n")
            elif no_citations:
                print(response.answer)
            else:
                print(response.format_response())
    except (AuthenticationError, RateLimitError) as e:
        print(str(e), file=sys.stderr)
        return 1

    return 0


# ── Research ───────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("-s", "--source", "source", default="web",
              help=f"Source focus ({', '.join(SOURCE_FOCUS_NAMES)}).")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
def research(query, source, json_output):
    """Deep research on a topic.

    Uses the in-depth research model for comprehensive reports (monthly quota).

    \b
    Examples:
      pwm research "agentic AI trends 2026"
      pwm research "quantum computing advances" -s academic
    """
    code = _cmd_research_impl(query, source, json_output)
    raise SystemExit(code)


def _cmd_research_impl(query, source, json_output):
    """Implementation for research command."""
    model = Models.DEEP_RESEARCH

    try:
        result = ask(query, model, source)
    except (AuthenticationError, RateLimitError) as e:
        print(str(e), file=sys.stderr)
        return 1

    if json_output:
        import orjson

        parts = result.split("\n\nCitations:")
        answer_text = parts[0]
        citations = []
        if len(parts) > 1:
            for line in parts[1].strip().split("\n"):
                line = line.strip()
                if line.startswith("[") and "]: " in line:
                    url = line.split("]: ", 1)[1]
                    citations.append(url)
        data = {"answer": answer_text, "citations": citations, "model": "deep_research", "source": source}
        sys.stdout.buffer.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        sys.stdout.buffer.write(b"\n")
    else:
        print(result)

    return 0


# ── Login ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--check", is_flag=True, help="Check current auth status (no login prompt).")
@click.option("--email", default=None, help="Send verification code to email (non-interactive).")
@click.option("--code", default=None, help="Complete auth with 6-digit code from email.")
@click.option("--no-save", is_flag=True, help="Don't save token to config.")
@click.pass_context
def login(ctx, check, email, code, no_save):
    """Authenticate with Perplexity.

    \b
    Examples:
      pwm login                                    # Interactive login
      pwm login --check                            # Check current auth status
      pwm login --email user@example.com           # Send verification code
      pwm login --email user@example.com --code 123456  # Complete auth
    """
    from perplexity_web_mcp.cli.auth import main as auth_main

    # Build args for the auth module
    auth_args = []
    if check:
        auth_args.append("--check")
    if email:
        auth_args.extend(["--email", email])
    if code:
        auth_args.extend(["--code", code])
    if no_save:
        auth_args.append("--no-save")

    sys.argv = ["pwm-auth", *auth_args]
    try:
        auth_main()
    except SystemExit as e:
        raise SystemExit(e.code if isinstance(e.code, int) else 0)


# ── Usage ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--refresh", is_flag=True, help="Force refresh rate limit data.")
def usage(refresh):
    """Check remaining rate limits and quotas.

    \b
    Examples:
      pwm usage
      pwm usage --refresh
    """
    code = _cmd_usage_impl(refresh)
    raise SystemExit(code)


def _cmd_usage_impl(refresh):
    """Implementation for usage command."""
    token = load_token()
    if not token:
        print(
            "NOT AUTHENTICATED\n\n"
            "No session token found. Authenticate first with: pwm login"
        )
        return 1

    cache = get_limit_cache()
    if cache is None:
        print("ERROR: Could not initialize limit cache.")
        return 1

    parts: list[str] = []

    limits = cache.get_rate_limits(force_refresh=refresh)
    if limits:
        parts.append("RATE LIMITS (remaining queries)")
        parts.append("=" * 40)
        parts.append(limits.format_summary())
    else:
        parts.append("WARNING: Could not fetch rate limits (network error or token issue).")

    settings = cache.get_user_settings(force_refresh=refresh)
    if settings:
        parts.append("")
        parts.append("ACCOUNT INFO")
        parts.append("=" * 40)
        parts.append(settings.format_summary())

    print("\n".join(parts))
    return 0


# ── API ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option("-p", "--port", default=8080, type=int, help="Port number.")
@click.option("--model", "default_model", default="auto", help="Default model.")
@click.option("--log-level", default="info", help="Log level: debug, info, warning, error.")
def api(host, port, default_model, log_level):
    """Start the Anthropic/OpenAI API-compatible server.

    \b
    Examples:
      pwm api
      pwm api --port 9090
      pwm api --model gpt52 --log-level debug
    """
    import os

    os.environ.setdefault("HOST", host)
    os.environ.setdefault("PORT", str(port))
    os.environ.setdefault("LOG_LEVEL", log_level)
    os.environ.setdefault("DEFAULT_MODEL", default_model)

    from perplexity_web_mcp.api import run_server

    run_server()


# ── Hack ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("tool")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def hack(ctx, tool, extra_args):
    """Launch AI tools connected to Perplexity models.

    Currently supports 'claude' — launches Claude Code using
    the local Perplexity API server as the backend.

    \b
    Examples:
      pwm hack claude
      pwm hack claude -m gpt52
    """
    from perplexity_web_mcp.cli.hack import cmd_hack

    code = cmd_hack([tool, *extra_args])
    raise SystemExit(code)


# ── Skill ──────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("args", nargs=-1)
def skill(args):
    """Manage Perplexity Web MCP skill across AI platforms.

    \b
    Actions:
      pwm skill list                          Show tools and installation status
      pwm skill install <tool>                Install skill for a tool
      pwm skill install all                   Install for all detected tools
      pwm skill install <tool> --level project  Install at project level
      pwm skill uninstall <tool>              Remove installed skill
      pwm skill show                          Display the skill content
      pwm skill update                        Update all outdated skills

    \b
    Tools: claude-code, cursor, codex, opencode, gemini-cli,
           antigravity, cline, openclaw, all
    """
    from perplexity_web_mcp.cli.skill import cmd_skill

    code = cmd_skill(list(args))
    raise SystemExit(code)


# ── Doctor ─────────────────────────────────────────────────────────────────


@cli.command()
@click.option("-v", "--verbose", is_flag=True, help="Show additional diagnostic details.")
def doctor(verbose):
    """Diagnose installation, auth, config, and limits.

    Runs a comprehensive set of checks and reports the status
    of your Perplexity Web MCP installation.

    \b
    Examples:
      pwm doctor
      pwm doctor -v
    """
    from perplexity_web_mcp.cli.doctor import cmd_doctor

    args = []
    if verbose:
        args.append("-v")
    code = cmd_doctor(args)
    raise SystemExit(code)


# ── Setup (Click subgroup) ─────────────────────────────────────────────────


def _register_setup():
    """Register the setup subgroup from the setup module."""
    from perplexity_web_mcp.cli.setup import setup
    cli.add_command(setup)


_register_setup()


# ── Legacy functions for backward compatibility (tests) ────────────────────


def _cmd_ask(args: list[str]) -> int:
    """Handle: pwm ask <query> [options] — legacy interface for tests."""
    if not args or args[0].startswith("-"):
        print("Error: pwm ask requires a query string.\n", file=sys.stderr)
        print('Usage: pwm ask "your question" [--model MODEL] [--thinking] [--source SOURCE]', file=sys.stderr)
        return 1

    query = args[0]
    model_name = "auto"
    thinking = False
    source: SourceFocusName = "web"
    json_output = False
    no_citations = False
    intent = "standard"

    i = 1
    while i < len(args):
        arg = args[i]
        if arg in ("-m", "--model") and i + 1 < len(args):
            model_name = args[i + 1]
            i += 2
        elif arg in ("-t", "--thinking"):
            thinking = True
            i += 1
        elif arg in ("-s", "--source") and i + 1 < len(args):
            source = args[i + 1]  # type: ignore[assignment]
            i += 2
        elif arg == "--json":
            json_output = True
            i += 1
        elif arg == "--no-citations":
            no_citations = True
            i += 1
        elif arg == "--intent" and i + 1 < len(args):
            intent = args[i + 1]
            i += 2
        else:
            print(f"Unknown option: {arg}", file=sys.stderr)
            return 1

    return _cmd_ask_impl(query, model_name, thinking, source, json_output, no_citations, intent)


def _cmd_research(args: list[str]) -> int:
    """Handle: pwm research <query> [options] — legacy interface for tests."""
    if not args or args[0].startswith("-"):
        print("Error: pwm research requires a query string.\n", file=sys.stderr)
        print('Usage: pwm research "your topic" [--source SOURCE]', file=sys.stderr)
        return 1

    query = args[0]
    source: SourceFocusName = "web"
    json_output = False

    i = 1
    while i < len(args):
        arg = args[i]
        if arg in ("-s", "--source") and i + 1 < len(args):
            source = args[i + 1]  # type: ignore[assignment]
            i += 2
        elif arg == "--json":
            json_output = True
            i += 1
        else:
            print(f"Unknown option: {arg}", file=sys.stderr)
            return 1

    return _cmd_research_impl(query, source, json_output)


def _cmd_usage(args: list[str]) -> int:
    """Handle: pwm usage [--refresh] — legacy interface for tests."""
    refresh = "--refresh" in args
    return _cmd_usage_impl(refresh)


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> NoReturn:
    """Main entry point for the unified pwm CLI."""
    try:
        cli(standalone_mode=True)
    except SystemExit as e:
        sys.exit(e.code if isinstance(e.code, int) else 0)
    sys.exit(0)


if __name__ == "__main__":
    main()
