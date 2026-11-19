"""Metal ``sign`` and trunc-div MSL forms, rendered on CPU without a Metal device."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

CONFIG = helion.Config(block_sizes=[16], num_threads=[16])


@helion.kernel(backend="metal", static_shapes=True)
def _sign(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.sign(x[tile])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def _trunc_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.div(x[tile], y[tile], rounding_mode="trunc")
    return out


# The three float widths share metal::sign; the two integer widths are what
# Metal loads as int and long, and take the comparison form instead.
@pytest.mark.parametrize(
    ("dtype", "expected", "absent"),
    [
        (torch.float16, "metal.sign(load)", "decltype("),
        (torch.bfloat16, "metal.sign(load)", "decltype("),
        (torch.float32, "metal.sign(load)", "decltype("),
        (torch.int32, "(load > 0) - (load < 0)", "metal.sign("),
        (torch.int64, "(load > 0) - (load < 0)", "metal.sign("),
    ],
)
@forbid_cuda_work()
def test_sign_renders_by_dtype(dtype, expected, absent):
    x = torch.arange(16, dtype=dtype)
    code = _sign.bind((x,)).to_code(CONFIG)
    assert expected in code
    assert absent not in code


# An integer quotient is the truncation itself; only a float one is wrapped.
@pytest.mark.parametrize(
    ("dtype", "truncated"),
    [(torch.int32, False), (torch.int64, False), (torch.float32, True)],
)
@forbid_cuda_work()
def test_trunc_div_renders_by_dtype(dtype, truncated):
    x = torch.arange(1, 17, dtype=dtype)
    y = torch.arange(1, 17, dtype=dtype)
    code = _trunc_div.bind((x, y)).to_code(CONFIG)
    assert "load / load_1" in code
    assert ("metal.trunc(" in code) is truncated
