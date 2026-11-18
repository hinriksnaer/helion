"""Configuration: *how* to run a pipeline on this machine, not *what* it does.

The split is the core design rule of ``agentic/``:

- a *definition* (an agent, a workflow) names a **capability role** such as
  ``high`` and never a provider or a model;
- this module maps a role to a provider and model, from a file the contributor
  owns and never commits.

That is what lets one shared definition run for someone with an internal
OpenAI-compatible gateway, someone with an Anthropic key, and someone with a
local model, without any of them being a special case in the code.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import model_validator

if TYPE_CHECKING:
    from pydantic_ai.models import Model

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - pydantic-ai itself needs 3.11
    raise ImportError(
        "agentic needs Python 3.11 or newer (for tomllib, and because pydantic-ai "
        "requires it). The shipped package still supports 3.10; this is a dev tool."
    ) from None


#: How hard a model thinks, in the unified spelling pydantic-ai's
#: ``ModelSettings.thinking`` takes. Mirrored rather than imported from
#: pydantic-ai: this module must import in a checkout without the optional extra,
#: so that it can say what is missing.
ThinkingLevel = Literal[True, False, "minimal", "low", "medium", "high", "xhigh", "max"]


class ProviderConfig(BaseModel):
    """Where a model lives and how to authenticate to it."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["openai", "anthropic", "google", "openai_compatible", "test"] = Field(
        ...,
        description=(
            "openai_compatible covers any /v1/chat/completions endpoint: an "
            "internal gateway, OpenRouter, vLLM, or ollama. 'test' builds "
            "pydantic-ai's TestModel, so a pipeline can be exercised end to end "
            "with no credentials and no network."
        ),
    )
    base_url: str | None = Field(
        None,
        description=(
            "Required for openai_compatible; optional for anthropic and google, "
            "where both SDKs take one (a proxy in front of the vendor API)."
        ),
    )
    key_env: str | None = Field(
        None,
        description=(
            "Environment variable holding the key. Never put a key in this file. "
            "Unset means the provider SDK's own variable is used."
        ),
    )
    key_cmd: str | None = Field(
        None,
        description=(
            "A command that prints the key on stdout, e.g. 'rbw get my-key' or "
            "'pass show gateway'. Resolved when the key is needed, so the secret "
            "never lands in a file or a shell history. The same convention pi's "
            "provider config uses ('!rbw get ...')."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> ProviderConfig:
        if self.kind == "openai_compatible" and not self.base_url:
            raise ValueError("an openai_compatible provider needs base_url")
        # `build_model` passes base_url to the Anthropic and Google clients too,
        # both of which take one (an internal proxy in front of a vendor API). It
        # used to be refused here, which turned a working endpoint into a config
        # error. `test` has no endpoint to point anywhere.
        if self.base_url and self.kind == "test":
            raise ValueError("a test provider has no endpoint to override")
        if self.key_env is not None and self.key_cmd is not None:
            raise ValueError("set only one of key_env and key_cmd")
        return self

    def api_key(self) -> str | None:
        """The key, from a command or an environment variable.

        Raising ``ValueError`` rather than ``KeyError`` is deliberate: this is a
        configuration problem, and the message is what the operator reads. A key
        is never read from the config file itself, so a local config can be shared
        without carrying a secret.
        """
        if self.key_cmd is not None:
            return _run_key_command(self.key_cmd)
        if self.key_env is None:
            return None
        value = os.environ.get(self.key_env)
        if not value:
            raise ValueError(
                f"provider {self.kind!r} needs {self.key_env}, which is unset or "
                f"empty. Export it, or point the local config's key_env at the "
                f"variable you do have."
            )
        return value

    def fingerprint(self) -> str:
        """What this provider *is*, for a cache key: kind, endpoint, credential source.

        Never the secret's value. A role repointed from a `test` provider at a real
        one kept serving the earlier empty answers, because the record was keyed by
        the model label alone: the same question, a different answer, served as if
        nothing had changed.
        """
        source = self.key_env or ("<command>" if self.key_cmd else "")
        return " ".join(
            part for part in (self.kind, self.base_url or "", source) if part
        )

    def endpoint(self) -> str:
        """Where requests go, for ``--dry-run``. Never a secret."""
        return self.base_url or f"the {self.kind} default endpoint"

    def key_status(self) -> str:
        """One line for ``--dry-run``: which credential this provider expects.

        A command is named but not run: ``--dry-run`` stays free of side effects
        and of secrets. Its output is what a real run would use.
        """
        if self.key_cmd is not None:
            return f"`{self.key_cmd}` (not run)"
        if self.key_env is None:
            return "the provider SDK's own environment variable"
        state = "set" if os.environ.get(self.key_env) else "UNSET"
        return f"{self.key_env} ({state})"


def _run_key_command(command: str) -> str:
    """Run a key command and return its first line of stdout.

    Nothing from the command's own output is included in an error: it is a
    credential by definition, and a failed secret lookup is not worth leaking one
    for. The exit status is enough to act on.
    """
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"key_cmd {command!r} is not a valid command: {exc}") from None
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ValueError(f"key_cmd {command!r} could not start: {exc}") from None
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise ValueError(
            f"key_cmd {command!r} produced no key (exit {completed.returncode}); "
            "its output is withheld because it may be a secret"
        )
    return value


class RoleConfig(BaseModel):
    """A role written as a table, ``[roles.<name>]``.

    The string form, ``name = "provider/model"``, still works and means "the
    config's own thinking level". A table is for the role that wants its own: a
    second reviewer on a cheaper model does not need the first one's effort.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        ..., description="'provider/model', exactly as the string form writes it."
    )
    thinking: ThinkingLevel | None = Field(
        None,
        description=(
            "How hard this role's model thinks. Unset means the top-level "
            "`thinking`, and with that unset too, the provider's own default."
        ),
    )


class ModelRef(BaseModel):
    """A resolved role: which provider, which model, and how hard it thinks."""

    provider: str
    model: str
    thinking: ThinkingLevel | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def settings_label(self) -> str:
        """The model settings, as a record's key and the run record write them.

        Empty when unset, so a role that names no level keys its records exactly
        as it did before a level could be configured -- the level is not a field
        that was silently added to every existing record's identity. It *does*
        take part in the key when set: the same question asked at ``low`` and at
        ``high`` is two answers, and serving one for the other is the mistake
        this key exists to prevent.
        """
        if self.thinking is None:
            return ""
        # `True` is a level too ("the provider's default effort"); spelled lower
        # so a level and a boolean read alike in a record.
        return f"thinking={str(self.thinking).lower()}"


class WorkflowSettings(BaseModel):
    """One workflow's table, ``[workflows.<name>]``. Workflows subclass it.

    Validated by the workflow that reads it (:meth:`Config.workflow`), with
    unknown keys rejected, so a typo in a workflow's table is as loud as one at
    the top level -- without the runtime having to know every workflow's knobs.
    """

    model_config = ConfigDict(extra="forbid")

    agents: list[str] = Field(
        default_factory=list,
        description=(
            "The roles (or 'provider/model' strings) this workflow's agents run "
            "on, in order. What one entry means is the workflow's: a reviewer. "
            "Empty means the config's default_role."
        ),
    )
    timeout_seconds: float | None = Field(
        None, description="Wall-clock budget per turn; unset means the top level."
    )
    max_total_tokens: int | None = Field(
        None, description="Token ceiling per turn; unset means the top level."
    )


S = TypeVar("S", bound=WorkflowSettings)


class Config(BaseModel):
    """The contributor's local configuration.

    Unknown keys are rejected: under TOML scoping a stray key is nearly always
    a setting written after a `[table]` header, where it silently belongs to
    that table instead of the document.
    """

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig]
    roles: dict[str, str | RoleConfig] = Field(
        ...,
        description=(
            "Capability role -> 'provider/model', or a table with that model and "
            "its own `thinking`. A definition asks for a role (high, mid, cheap); "
            "this is the only place a model name appears."
        ),
    )
    thinking: ThinkingLevel | None = Field(
        None,
        description=(
            "How hard every role's model thinks. Unset leaves each provider's own "
            "default, which is what a review inherited before this existed: the "
            "level is also the one model setting a run record cannot infer, so a "
            "workflow that cares should name it."
        ),
    )
    default_role: str = Field(
        "high", description="The role a workflow uses when its table names none."
    )
    skills_dirs: list[str] = Field(
        default_factory=lambda: [".agents/skills"],
        description="Skill libraries offered to an agent; paths are repo-relative.",
    )
    timeout_seconds: float = Field(1800.0, description="Wall-clock budget per turn.")
    max_total_tokens: int | None = Field(
        None,
        description=(
            "Ceiling on tokens for one turn. Unset means the provider's own "
            "limit; set it to keep a runaway agent from spending without bound."
        ),
    )
    workflows: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-workflow tables, validated by each workflow on use.",
    )

    def workflow(self, name: str, settings: type[S]) -> S:
        """The ``[workflows.<name>]`` table, validated by the workflow's own model."""
        try:
            return settings.model_validate(self.workflows.get(name, {}))
        except ValidationError as exc:
            raise ValueError(f"invalid [workflows.{name}] table:\n{exc}") from None

    def agent_roles(
        self, settings: WorkflowSettings, requested: list[str] | None = None
    ) -> list[str]:
        """The roles a run uses: the command line's, the table's, or the default."""
        return list(requested or settings.agents or [self.default_role])

    def resolve(self, role: str | None) -> ModelRef:
        """Map a capability role to a provider and model.

        ``role`` may also be a literal ``provider/model`` string, which is what
        ``--reviewer`` passes: an explicit override for one run.
        """
        if role is None:
            role = self.default_role
        if "/" in role:
            provider, _, model = role.partition("/")
            if provider not in self.providers:
                raise ValueError(
                    f"model reference {role!r} names provider {provider!r}, which is "
                    f"not configured (have: {', '.join(sorted(self.providers))})"
                )
            return ModelRef(provider=provider, model=model, thinking=self.thinking)
        try:
            entry = self.roles[role]
        except KeyError:
            raise ValueError(
                f"unknown capability role {role!r} (configured roles: "
                f"{', '.join(sorted(self.roles))})"
            ) from None
        if isinstance(entry, str):
            return self.resolve(entry)
        ref = self.resolve(entry.model)
        if entry.thinking is None:
            return ref
        return ref.model_copy(update={"thinking": entry.thinking})

    def provider_of(self, ref: ModelRef) -> ProviderConfig:
        try:
            return self.providers[ref.provider]
        except KeyError:
            raise ValueError(f"provider {ref.provider!r} is not configured") from None


#: Where ``load_config`` looks, in order. The first two are gitignored.
_SEARCH = (
    Path("agentic/config.local.toml"),
    Path("config.local.toml"),
    Path("agentic/config.toml"),
)


def load_config(path: str | Path | None = None, *, root: Path | None = None) -> Config:
    """Load and validate the local configuration.

    ``AGENTIC_CONFIG`` overrides the search. The search is relative to ``root``
    (the repository, when the caller has one): a run's other paths are
    repo-rooted, so starting the tool from a subdirectory must not change which
    config it finds. Errors name the file, so a malformed config fails here
    rather than halfway through a run.
    """
    if path is None:
        path = os.environ.get("AGENTIC_CONFIG")
    candidate: Path | None
    if path is not None:
        candidate = Path(path)
    else:
        base = Path(".") if root is None else root
        candidate = next((base / p for p in _SEARCH if (base / p).is_file()), None)

    if candidate is None:
        searched = ", ".join(str(p) for p in _SEARCH)
        raise FileNotFoundError(
            f"no agentic config found (searched: {searched}). Copy "
            f"agentic/config.example.toml to agentic/config.local.toml and fill in "
            f"one provider and role."
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"agentic config {candidate} does not exist")

    with candidate.open("rb") as handle:
        data = tomllib.load(handle)
    try:
        return Config.model_validate(data)
    except Exception as exc:
        raise ValueError(f"invalid agentic config {candidate}:\n{exc}") from None


def build_model(config: Config, ref: ModelRef) -> Model:
    """Build the pydantic-ai model a run will use.

    Imported lazily so a checkout without the optional ``agentic`` extra can
    still import this module and explain what is missing.
    """
    provider = config.provider_of(ref)
    base_url = provider.base_url

    if provider.kind == "test":
        # A credential-free path through the whole pipeline: same composition,
        # same contract, same artifacts, synthetic answers. This is what lets
        # the tool be reviewed -- and smoke-tested -- without a key.
        from pydantic_ai.models.test import TestModel

        # No tool calls: this path exists to exercise the contract, the
        # artifact and the gate, not the tools, and a TestModel left to its
        # own devices calls `load_capability` with invented arguments forever.
        return TestModel(call_tools=[])

    api_key = provider.api_key()

    # The key is passed to every provider, not just the OpenAI-compatible one.
    # `infer_model(f"{kind}:{model}")` looks like the general case but discards
    # the key entirely -- it builds a provider from the SDK's own environment --
    # so a `key_env` for openai/anthropic/google configured an endpoint the run
    # never used.
    if provider.kind == "openai_compatible" or provider.kind == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            ref.model,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )

    if provider.kind == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            ref.model,
            provider=AnthropicProvider(api_key=api_key, base_url=base_url),
        )

    if provider.kind == "google":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            ref.model,
            provider=GoogleProvider(api_key=api_key, base_url=base_url),
        )

    raise ValueError(f"unsupported provider kind {provider.kind!r}")
