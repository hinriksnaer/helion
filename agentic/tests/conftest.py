"""Make `agentic` importable for its own tests without installing it.

Also decides whether this directory is collected at all. `agentic` is an
optional extra that needs Python 3.11, pydantic, and pydantic-ai with the
harness: a bare `pytest` from a checkout that did not install `.[agentic]` would
otherwise fail at collection on an import instead of skipping a directory of
tools it never asked for.

Every module of the extra is checked, not just the two the test module happens
to import at its top: `agentic.workflows.review` imports its agent (and therefore
pydantic-ai and the harness) eagerly, so a predicate that stopped at `pydantic`
collected the suite and then died on the first test file's import.
"""

from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: What importing the package transitively needs. `pydantic-ai-slim` installs as
#: `pydantic_ai`; the harness is a separate distribution.
_MISSING = [
    name
    for name in ("pydantic", "tomllib", "pydantic_ai", "pydantic_ai_harness")
    if sys.version_info < (3, 11) or find_spec(name) is None
]

if _MISSING:
    collect_ignore_glob = ["*.py"]


@pytest.fixture(autouse=True)
def _no_git_signing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixtures must not depend on the developer's global git config.

    With `commit.gpgsign = true` and no forwarded agent, every `git commit` in a
    test repository fails ("Couldn't get agent socket?"), which says nothing
    about the code under test.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
