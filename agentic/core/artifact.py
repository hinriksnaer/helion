"""Content-addressed records of what an agent produced, and from what.

A record is stored under a key derived from everything that determines it: the
workflow's contract version, the agent's name, the model that answered, the
prompt, the named digests of the task's inputs, and the digests of the
environment and the agent definition. Recording it is what turns a run from a
terminal transcript into an artifact with three properties the project needs:

- **reuse** -- an unchanged task costs nothing to re-run;
- **auditability** -- a record names the content its outcome is about, so a stale
  ``autoreview passed:`` line becomes detectable rather than trusted;
- **comparability** -- two agents answering the same inputs land under the same
  input digests and can be compared as data.

Nothing here knows about models, providers, capabilities, or the shape of a
workflow's answer. The outcome is stored as plain JSON so that the *workflow*
owns its schema; this module stores the identifiers around it.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

#: Where records live, relative to the repository root. Gitignored: a run of a
#: local tree is machine-local state, not a repository artifact.
STORE_DIR = Path("agentic/.cache/artifacts")


def diff_digest(text: str) -> str:
    """A stable digest of some content a task is about.

    Named for its first caller (a diff). Used for anything whose text determines
    an outcome: a diff, a file list, a plan, an architecture sketch.
    """
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def environment_digest(
    root: Path, skills_dirs: list[str], versions: dict[str, str]
) -> str:
    """A digest of what a run reads that the task's inputs do not show.

    An agent is given AGENTS.md, the skill library and a harness, and all three
    change what it finds. None of them is in the diff, so without this a record
    made under one policy would be reused under another. It is recorded on every
    artifact and compared before reuse (see
    :meth:`Artifact.provenance_stale`).
    """
    entries: list[str] = []
    for name in ("AGENTS.md", "CLAUDE.md"):
        path = root / name
        if path.is_file():
            entries.append(f"{name}:{diff_digest(path.read_text(errors='replace'))}")
    for directory in skills_dirs:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                entries.append(f"{rel}:{diff_digest(path.read_text(errors='replace'))}")
    entries.extend(f"{name}=={found}" for name, found in sorted(versions.items()))
    return diff_digest("\n".join(entries))


def key_for(
    *,
    agent: str,
    model: str,
    prompt: str,
    inputs: dict[str, str],
    schema_version: str,
    provider: str = "",
    settings: str = "",
) -> str:
    """The artifact key: the *question* a record answers, and nothing else.

    The prompt, so a prompt edit asks a different question instead of silently
    serving an answer to the old one; the model, because two models answering the
    same question are two records; and ``inputs``, as named digests rather than
    one diff hash, so a workflow that is about a tree, a plan or a set of reviews
    names its own inputs and they all take part.

    The *provenance* of a record -- the environment digest (AGENTS.md, the skill
    library, the harness version) and the definition digest (instructions,
    composition, tool set) -- is deliberately **not** in the key. Those describe
    the answer, not the question, and putting them in the key made the cache
    unusable: an edit to a tool docstring re-keyed every record and re-ran a
    multi-million-token review, and two answers to the same diff landed under
    different keys and could not be compared. They are stored on the record
    instead, and :meth:`Artifact.provenance_stale` decides what to do about a
    record from an older policy.
    """
    material = json.dumps(
        {
            "schema_version": schema_version,
            "agent": agent,
            "model": model,
            # Where the model is served from, without the secret: a role
            # repointed at another provider (a `test` one included) is a
            # different question, and its earlier answers must not be served.
            "provider": provider,
            # How the model was asked to think, when a level was configured:
            # the same question at `low` and at `high` are two answers. Absent
            # when unset, so records made before a level existed keep their key.
            **({"settings": settings} if settings else {}),
            "prompt": hashlib.sha256(prompt.encode("utf-8", "replace")).hexdigest(),
            "inputs": dict(sorted(inputs.items())),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Artifact(BaseModel):
    """One recorded outcome of one agent on one task.

    pydantic for the same reason the answers are: the record is written and read
    by a *later* version of this code, and a field that changed shape has to fail
    as a cache miss rather than as an attribute error.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    #: When the answer was recorded. A default, so a caller never writes it and
    #: a record read back carries the timestamp it was written with.
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    agent: str
    model: str
    schema_version: str
    #: The workflow's name for this unit of work.
    task: str
    #: Named digests of what the task was about (a diff, a tree, a plan, ...),
    #: plus whatever names it (revisions, paths) for a reader.
    inputs: dict[str, str] = Field(default_factory=dict)
    environment: str = ""
    definition: str = ""
    elapsed_seconds: float = 0.0
    #: The workflow's answer, as JSON. The workflow owns the schema and validates
    #: it on load; this module never imports it.
    outcome: dict[str, Any] = Field(default_factory=dict)

    def provenance_stale(self, *, environment: str, definition: str) -> bool:
        """Whether this record came from a policy other than the current one.

        A record is reused by *question*, so a reviewer that answers the same diff
        after its instructions, composition or environment changed would otherwise
        serve an answer produced by a different reviewer. Comparing the digests
        the record carries is what keeps that from happening, without making a
        policy edit re-key -- and therefore lose -- every record.

        An empty digest on either side means "unknown" and is not treated as a
        change: a record written before these digests existed is still the best
        evidence available, and refusing it would be a cache miss dressed up as
        safety.
        """
        return bool(
            (environment and self.environment and environment != self.environment)
            or (definition and self.definition and definition != self.definition)
        )


def store_dir(root: Path) -> Path:
    return root / STORE_DIR


def path_for(root: Path, key: str) -> Path:
    return store_dir(root) / f"{key}.json"


def save(root: Path, artifact: Artifact) -> Path:
    """Write the artifact, replacing any record of an identical input."""
    destination = path_for(root, artifact.key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".json.tmp")
    tmp.write_text(artifact.model_dump_json(indent=2) + "\n")
    tmp.replace(destination)
    return destination


def load(root: Path, key: str, *, schema_version: str) -> Artifact | None:
    """Read a record by key, or ``None``.

    A record written under a different contract version is treated as absent: an
    outcome about a different schema is not an outcome about this one, and the
    safe direction is to run again.
    """
    path = path_for(root, key)
    if not path.is_file():
        return None
    try:
        artifact = Artifact.model_validate_json(path.read_text())
    except ValueError:  # unreadable, or a shape from an older version
        return None
    if artifact.schema_version != schema_version:
        return None
    return artifact
