"""The improve workflow: a candidate in, one reviewed branch out.

Four prose steps and one typed one, in ordinary `async` code:

    scan -> choose -> plan -> write -> review <-> refine

The review's answer is the only contract, because it is the only answer the run
acts on. `pipeline.py` is the run (read that first); `state.py` is the record and
the gate; `instructions/` is what each step is told.
"""

from __future__ import annotations

from .pipeline import REPORT
from .pipeline import ImproveOptions
from .pipeline import ImproveSettings
from .pipeline import Summary
from .pipeline import dry_run
from .pipeline import run_improve
from .state import Delivered
from .state import Nothing
from .state import Rejected
from .state import Round
from .state import independent
from .state import roles

__all__ = [
    "REPORT",
    "Delivered",
    "ImproveOptions",
    "ImproveSettings",
    "Nothing",
    "Rejected",
    "Round",
    "Summary",
    "dry_run",
    "independent",
    "roles",
    "run_improve",
]
