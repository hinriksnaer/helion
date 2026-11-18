"""The agentic config: roles, providers, thinking levels, the TOML file."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError
import pytest

from agentic.core.config import Config
from agentic.core.config import ProviderConfig
from agentic.core.config import load_config

if TYPE_CHECKING:
    from pathlib import Path


class TestConfig:
    def test_role_resolves_to_provider_and_model(self) -> None:
        config = Config(
            providers={"gw": ProviderConfig(kind="test")},
            roles={"high": "gw/big-model"},
        )
        ref = config.resolve(None)
        assert (ref.provider, ref.model) == ("gw", "big-model")
        assert ref.label == "gw/big-model"

    def test_explicit_reference_overrides_the_role(self) -> None:
        config = Config(
            providers={"gw": ProviderConfig(kind="test")},
            roles={"high": "gw/big-model"},
        )
        assert config.resolve("gw/small-model").model == "small-model"

    def test_the_thinking_level_comes_from_the_top_level(self) -> None:
        config = Config(
            providers={"gw": ProviderConfig(kind="test")},
            roles={"high": "gw/big-model"},
            thinking="high",
        )
        ref = config.resolve(None)
        assert ref.thinking == "high"
        assert ref.settings_label == "thinking=high"

    def test_a_role_table_sets_its_own_thinking(self) -> None:
        config = Config(
            providers={"gw": ProviderConfig(kind="test")},
            roles={
                "high": "gw/big-model",
                "cheap": {"model": "gw/small-model", "thinking": "low"},
            },
            thinking="high",
        )
        assert config.resolve("high").thinking == "high"
        assert config.resolve("cheap").thinking == "low"
        assert config.resolve("cheap").label == "gw/small-model"

    def test_no_level_stays_unset(self) -> None:
        """Unset is not a level: it keys a record exactly as it always did."""
        config = Config(
            providers={"gw": ProviderConfig(kind="test")}, roles={"high": "gw/m"}
        )
        ref = config.resolve(None)
        assert ref.thinking is None
        assert ref.settings_label == ""

    def test_a_typo_in_a_role_table_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Config(
                providers={"gw": ProviderConfig(kind="test")},
                roles={"high": {"model": "gw/m", "effort": "high"}},
            )

    def test_an_empty_table_falls_back_to_the_default_role(self) -> None:
        from agentic.core.config import WorkflowSettings

        config = Config(
            providers={"gw": ProviderConfig(kind="test")}, roles={"high": "gw/m"}
        )
        assert config.agent_roles(config.workflow("any", WorkflowSettings)) == ["high"]

    def test_unknown_role_is_an_error(self) -> None:
        config = Config(
            providers={"gw": ProviderConfig(kind="test")}, roles={"high": "gw/m"}
        )
        with pytest.raises(ValueError, match="unknown capability role"):
            config.resolve("nonexistent")

    def test_unknown_provider_is_an_error(self) -> None:
        config = Config(providers={"gw": ProviderConfig(kind="test")}, roles={})
        with pytest.raises(ValueError, match="not configured"):
            config.resolve("other/model")

    def test_an_openai_compatible_provider_needs_a_base_url(self) -> None:
        with pytest.raises(ValueError, match="needs base_url"):
            ProviderConfig(kind="openai_compatible")

    def test_a_key_is_read_from_the_named_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENTIC_TEST_KEY", "not-a-secret")
        provider = ProviderConfig(kind="test", key_env="AGENTIC_TEST_KEY")
        assert provider.api_key() == "not-a-secret"

    def test_a_missing_key_names_the_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AGENTIC_TEST_KEY", raising=False)
        provider = ProviderConfig(kind="test", key_env="AGENTIC_TEST_KEY")
        with pytest.raises(ValueError, match="AGENTIC_TEST_KEY"):
            provider.api_key()

    def test_a_key_can_come_from_a_command(self) -> None:
        # The same convention pi's provider config uses: the secret stays in the
        # secret store and is resolved when needed.
        provider = ProviderConfig(kind="test", key_cmd="printf from-a-command")
        assert provider.api_key() == "from-a-command"

    def test_a_failing_key_command_withholds_its_output(self) -> None:
        provider = ProviderConfig(kind="test", key_cmd="false")
        with pytest.raises(ValueError, match="produced no key"):
            provider.api_key()

    def test_only_one_key_source_is_allowed(self) -> None:
        with pytest.raises(ValueError, match="only one of key_env and key_cmd"):
            ProviderConfig(kind="test", key_env="A", key_cmd="printf b")

    def test_the_configured_key_is_consulted_for_every_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `infer_model(f"{kind}:{model}")` dropped the resolved key on the floor,
        # so a key_env or key_cmd for openai/anthropic/google configured an
        # endpoint the run never authenticated to. Asking for the key is the
        # observable part of honouring it.
        from agentic.core.config import build_model

        monkeypatch.delenv("AGENTIC_ABSENT_KEY", raising=False)
        for kind in ("openai", "anthropic", "google"):
            config = Config(
                providers={
                    "p": ProviderConfig(kind=kind, key_env="AGENTIC_ABSENT_KEY")
                },
                roles={"high": "p/m"},
            )
            with pytest.raises(ValueError, match="AGENTIC_ABSENT_KEY"):
                build_model(config, config.resolve("high"))

    def test_the_status_line_names_the_variable_and_its_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A tip on stderr must not print the secret, and must not read it either:
        # the state of the variable is all `--dry-run` needs.
        monkeypatch.setenv("AGENTIC_TEST_KEY", "not-a-secret")
        provider = ProviderConfig(kind="test", key_env="AGENTIC_TEST_KEY")
        assert "AGENTIC_TEST_KEY" in provider.key_status()
        assert "set" in provider.key_status()
        assert "not-a-secret" not in provider.key_status()

        monkeypatch.delenv("AGENTIC_TEST_KEY")
        assert "UNSET" in provider.key_status()

    def test_a_key_command_is_named_but_not_run_in_a_status_line(self) -> None:
        # `false` would raise ValueError if the status line executed it, so this
        # asserts non-execution rather than the absence of a word.
        provider = ProviderConfig(kind="test", key_cmd="false")
        status = provider.key_status()
        assert "`false`" in status
        assert "not run" in status

    def test_the_endpoint_is_reported(self) -> None:
        assert ProviderConfig(kind="test").endpoint() == "the test default endpoint"
        with_url = ProviderConfig(
            kind="openai_compatible", base_url="https://gw.invalid/v1"
        )
        assert with_url.endpoint() == "https://gw.invalid/v1"

    def test_toml_keeps_settings_out_of_the_role_table(self, tmp_path: Path) -> None:
        # A plain TOML trap: every key after `[roles]` belongs to roles. Settings
        # must therefore come first, and this pins that they do.
        path = tmp_path / "c.toml"
        path.write_text(
            'skills_dirs = [".agents/skills"]\n'
            "timeout_seconds = 12.0\n"
            '\n[providers.fake]\nkind = "test"\n'
            '\n[roles]\nhigh = "fake/m"\n'
            '\n[workflows.review]\nagents = ["high"]\nmax_diff_chars = 1234\n'
        )
        config = load_config(path)
        assert config.timeout_seconds == 12.0
        assert config.workflows["review"]["max_diff_chars"] == 1234
        assert config.skills_dirs == [".agents/skills"]
        assert config.roles == {"high": "fake/m"}

    def test_a_misplaced_setting_is_rejected_rather_than_ignored(
        self, tmp_path: Path
    ) -> None:
        # The other half of the trap: written after `[roles]`, a setting becomes
        # a role. Unknown keys are forbidden, so this fails loudly instead of
        # silently reviewing with a provider nobody configured.
        path = tmp_path / "c.toml"
        path.write_text(
            '[providers.fake]\nkind = "test"\n'
            '\n[roles]\nhigh = "fake/m"\ntimeout_seconds = 12.0\n'
        )
        with pytest.raises(ValueError, match="timeout_seconds"):
            load_config(path)
