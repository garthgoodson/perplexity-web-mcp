"""Tests for the skill command (cli/skill.py).

Uses tmp_path for all filesystem operations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from perplexity_web_mcp.cli.skill import (
    SKILL_DIR_NAME,
    SkillTarget,
    _find_skill_source,
    _get_current_version,
    _get_installed_version,
    _install_skill,
    _uninstall_skill,
    cmd_skill,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def skill_source(tmp_path: Path) -> Path:
    """Create a minimal skill source directory."""
    source = tmp_path / "source" / SKILL_DIR_NAME
    source.mkdir(parents=True)
    skill_md = source / "SKILL.md"
    skill_md.write_text(
        '---\nname: querying-perplexity\ndescription: "test"\n'
        'metadata:\n  version: "0.3.0"\n---\n# Test Skill\n'
    )
    refs = source / "references"
    refs.mkdir()
    (refs / "models.md").write_text("# Models\n")
    return source


@pytest.fixture
def dest_dir(tmp_path: Path) -> Path:
    """Create a destination directory."""
    d = tmp_path / "dest"
    d.mkdir()
    return d


# ============================================================================
# 1. _get_installed_version
# ============================================================================


class TestGetInstalledVersion:
    """Test version extraction from installed SKILL.md."""

    def test_returns_version_from_frontmatter(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / SKILL_DIR_NAME
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text('---\nname: test\nmetadata:\n  version: "1.2.3"\n---\n')
        assert _get_installed_version(skill_dir) == "1.2.3"

    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        assert _get_installed_version(tmp_path / "nope") is None

    def test_returns_none_when_no_version(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / SKILL_DIR_NAME
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---\n")
        assert _get_installed_version(skill_dir) is None


# ============================================================================
# 2. _install_skill / _uninstall_skill
# ============================================================================


class TestInstallUninstall:
    """Test skill file copy and removal."""

    def test_install_copies_files(self, skill_source: Path, dest_dir: Path) -> None:
        result = _install_skill(skill_source, dest_dir)
        assert result is True

        installed = dest_dir / SKILL_DIR_NAME
        assert installed.exists()
        assert (installed / "SKILL.md").exists()
        assert (installed / "references" / "models.md").exists()

    def test_install_overwrites_existing(self, skill_source: Path, dest_dir: Path) -> None:
        # Install once
        _install_skill(skill_source, dest_dir)

        # Modify installed file
        installed_skill = dest_dir / SKILL_DIR_NAME / "SKILL.md"
        installed_skill.write_text("modified")

        # Install again (should overwrite)
        _install_skill(skill_source, dest_dir)
        assert "modified" not in installed_skill.read_text()

    def test_uninstall_removes_directory(self, skill_source: Path, dest_dir: Path) -> None:
        _install_skill(skill_source, dest_dir)
        installed = dest_dir / SKILL_DIR_NAME
        assert installed.exists()

        result = _uninstall_skill(dest_dir)
        assert result is True
        assert not installed.exists()

    def test_uninstall_returns_false_when_not_installed(self, dest_dir: Path) -> None:
        assert _uninstall_skill(dest_dir) is False


# ============================================================================
# 3. cmd_skill routing
# ============================================================================


class TestCmdSkill:
    """Test the cmd_skill CLI handler."""

    def test_help_returns_0(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_skill(["--help"]) == 0
        assert "Manage Perplexity Web MCP skill" in capsys.readouterr().out

    def test_no_args_returns_0(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_skill([]) == 0

    def test_list_returns_0(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_skill(["list"]) == 0
        out = capsys.readouterr().out
        assert "claude-code" in out
        assert "cursor" in out

    @patch("perplexity_web_mcp.cli.skill._find_skill_source")
    def test_show_displays_skill(
        self, mock_source: pytest.CaptureFixture, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        skill_dir = tmp_path / SKILL_DIR_NAME
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill Content")
        mock_source.return_value = skill_dir

        assert cmd_skill(["show"]) == 0
        assert "My Skill Content" in capsys.readouterr().out

    def test_install_missing_tool_shows_usage(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_skill(["install"]) == 0
        assert "Usage:" in capsys.readouterr().out

    def test_install_unknown_tool_returns_1(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_skill(["install", "nonexistent"]) == 1
        assert "Unknown tool" in capsys.readouterr().err

    @patch("perplexity_web_mcp.cli.skill._find_skill_source", return_value=None)
    def test_install_no_source_returns_1(self, mock_src, capsys: pytest.CaptureFixture) -> None:
        assert cmd_skill(["install", "claude-code"]) == 1
        assert "Could not find" in capsys.readouterr().err

    def test_unknown_action_returns_1(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_skill(["bogus"]) == 1

    @patch("perplexity_web_mcp.cli.skill._find_skill_source", return_value=None)
    def test_update_no_source_returns_1(self, mock_src, capsys: pytest.CaptureFixture) -> None:
        assert cmd_skill(["update"]) == 1

    @patch("perplexity_web_mcp.cli.skill._find_skill_source")
    @patch("perplexity_web_mcp.cli.skill._get_targets")
    @patch("perplexity_web_mcp.cli.skill._get_current_version", return_value="0.4.0")
    def test_update_finds_outdated(
        self, mock_ver, mock_targets, mock_source, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # Set up a fake installed skill with old version
        user_dir = tmp_path / "skills"
        skill_dir = user_dir / SKILL_DIR_NAME
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text('---\nname: test\nmetadata:\n  version: "0.3.0"\n---\n')

        # Set up source
        source_dir = tmp_path / "source" / SKILL_DIR_NAME
        source_dir.mkdir(parents=True)
        (source_dir / "SKILL.md").write_text('---\nname: test\nmetadata:\n  version: "0.4.0"\n---\n')
        mock_source.return_value = source_dir

        mock_targets.return_value = [
            SkillTarget(name="test-tool", description="T", user_dir=user_dir, project_dir=".test/skills"),
        ]

        assert cmd_skill(["update"]) == 0
        out = capsys.readouterr().out
        assert "Updated" in out
        assert "0.3.0" in out
        assert "0.4.0" in out
