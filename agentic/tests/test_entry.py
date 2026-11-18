"""The entry points: the shared flags, the runner, and the exit statuses.

Each script runs in-process against a throwaway repository with the
credential-free ``test`` provider, so this covers argument parsing, the one
``asyncio.run``, printing and exit codes without a model.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG = """\
default_role = "high"
[providers.t]
kind = "test"
[roles]
high = "t/m"
[workflows.review]
agents = ["high"]
"""

#: The same config, with the level every role inherits named explicitly.
_CONFIG_THINKING = _CONFIG.replace(
    'default_role = "high"', 'default_role = "high"\nthinking = "high"', 1
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "first")
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-qam", "[agents] Second\n\nBody.")
    (repo / "config.toml").write_text(_CONFIG)
    (repo / ".gitignore").write_text("config.toml\n")
    monkeypatch.chdir(repo)
    return repo


class TestAutoreview:
    def test_a_clean_review_exits_0_and_prints_its_record(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agentic.scripts import autoreview

        assert autoreview.main(["--config", "config.toml", "--head", "--json"]) == 0
        record = json.loads(capsys.readouterr().out)
        assert record["passed"] is True
        assert record["exit_code"] == 0
        assert record["models"]["high"] == "t/m"

    def test_a_second_run_reviews_again(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Nothing is recorded between runs: the second run pays for its own turn.
        from agentic.scripts import autoreview

        argv = ["--config", "config.toml", "--head"]
        assert autoreview.main(argv) == 0
        capsys.readouterr()
        assert autoreview.main(argv) == 0
        out = capsys.readouterr().out
        assert "cached" not in out
        assert "reviewed" in out

    def test_a_missing_config_cannot_start(self, repo: Path) -> None:
        from agentic.scripts import autoreview

        assert autoreview.main(["--config", "missing.toml", "--head"]) == 4

    def test_dry_run_needs_no_turn(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agentic.scripts import autoreview

        assert autoreview.main(["--config", "config.toml", "--head", "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "reviewer:    high -> t/m" in out
        assert "thinking: (the provider's own default)" in out
        assert not (repo / "agentic" / "logs").exists()

    def test_the_thinking_level_reaches_the_dry_run_and_the_record(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The level is stated where a reader looks, and keys the record."""
        from agentic.scripts import autoreview

        (repo / "level.toml").write_text(_CONFIG_THINKING)
        assert autoreview.main(["--config", "level.toml", "--head", "--dry-run"]) == 0
        assert "thinking: high" in capsys.readouterr().out

        assert autoreview.main(["--config", "level.toml", "--head", "--json"]) == 0
        record = json.loads(capsys.readouterr().out)
        assert record["models"]["high"] == "t/m (thinking=high)"


class TestSharedFlags:
    def test_every_entry_point_takes_the_run_flags(self) -> None:
        from agentic.scripts import autoreview

        flags = ["--trace", "--timeout", "7", "--json"]
        for parse in (autoreview.parse_args,):
            args = parse(flags)
            assert (args.trace, args.json) == (True, True)
            assert args.timeout == 7.0

    def test_a_typo_exits_4_not_argparse_s_2(self) -> None:
        # 2 means "a blocker was found" to this tool's callers; a usage error
        # must not look like one.
        from agentic.scripts import autoreview

        for parse in (autoreview.parse_args,):
            with pytest.raises(SystemExit) as exited:
                parse(["--bogus"])
            assert exited.value.code == 4


def test_the_run_log_names_what_was_reviewed(repo: Path) -> None:
    from agentic.scripts import autoreview

    assert autoreview.main(["--config", "config.toml", "--head"]) == 0
    [log] = (repo / "agentic" / "logs").iterdir()
    assert (log / "index.md").read_text().startswith("# agentic autoreview --head ")
