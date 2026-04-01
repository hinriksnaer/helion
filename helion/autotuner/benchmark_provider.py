from __future__ import annotations

import abc
import contextlib
import datetime
import functools
from itertools import count
from itertools import starmap
import math
from math import inf
import os
import tempfile
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import NamedTuple
from typing import NoReturn
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map_only
from torch.utils._pytree import tree_unflatten

from .. import exc
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import already_compiled_fail
from ..runtime.precompile_shim import make_precompiler
from .benchmarking import do_bench
from .benchmarking import interleaved_bench
from .benchmarking import sync_object
from .logger import SUPPRESSED_TRITON_CODE_MSG
from .logger import AutotuneLogEntry
from .logger import _get_failure_dump_dir
from .logger import capture_output
from .logger import classify_triton_exception
from .logger import format_triton_compile_failure
from .logger import log_generated_triton_code_debug
from .logger import match_unrecoverable_runtime_error
from .logger import maybe_dump_triton_failure
from .precompile_future import PrecompileContext
from .precompile_future import PrecompileFuture
from .precompile_future import _ExtractedLaunchArgs
from .progress_bar import iter_with_progress
from helion._dist_utils import all_gather_object
from helion._dist_utils import get_signal_pad_ptrs_dev
from helion._dist_utils import is_symm_mem_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import CompiledConfig
    from ..runtime.settings import Settings
    from . import ConfigSpec
    from .base_search import _AutotunableKernel
    from .logger import AutotuningLogger
    from .metrics import AutotuneMetrics


# ---------------------------------------------------------------------------
# Helper functions (moved from base_search.py)
# ---------------------------------------------------------------------------

_FP8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
}


def _assert_close(actual: object, expected: object, atol: float, rtol: float) -> None:
    """Like torch.testing.assert_close but handles fp8 and uses chunked comparison for large tensors."""

    def convert(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.uint8) if t.dtype in _FP8_DTYPES else t

    actual_flat, actual_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, actual)
    )
    expected_flat, expected_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, expected)
    )

    if actual_spec != expected_spec:
        raise AssertionError(
            f"Output tree structure mismatch during autotuner accuracy check:\n"
            f"  actual:   {actual_spec} ({len(actual_flat)} leaves)\n"
            f"  expected: {expected_spec} ({len(expected_flat)} leaves)"
        )

    for a, e in zip(actual_flat, expected_flat, strict=True):
        if isinstance(a, torch.Tensor):
            _chunked_assert_close(a, e, atol=atol, rtol=rtol)
        elif isinstance(a, str):
            if not isinstance(e, str):
                raise AssertionError(f"Type mismatch {a} vs {e}")
            if a != e:
                raise AssertionError(f"string mismatch {a} vs {e}")
        else:
            torch.testing.assert_close(a, e, atol=atol, rtol=rtol)


def _chunked_assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    chunk_size: int = 2**22,  # ~4M elements per chunk
) -> None:
    """Memory-efficient assert_close for large tensors.

    Processes the comparison in chunks to avoid allocating multiple
    full-size temporary tensors.  Uses torch.testing.assert_close on
    each chunk so error messages retain full detail.
    """
    if actual.numel() <= chunk_size:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    a_flat = actual.reshape(-1)
    e_flat = expected.reshape(-1)
    for i in range(0, a_flat.numel(), chunk_size):
        a_chunk = a_flat[i : i + chunk_size]
        e_chunk = e_flat[i : i + chunk_size]
        torch.testing.assert_close(a_chunk, e_chunk, atol=atol, rtol=rtol)


def _clone_symm_mem_tensor(t: torch.Tensor) -> torch.Tensor:
    assert t.is_contiguous(), "Only support cloning contiguous symm mem tensor for now"
    new_tensor = symm_mem.empty(
        *t.shape,
        dtype=t.dtype,
        device=t.device,
    )
    new_tensor.copy_(t)
    # rendezvous so we don't count the time in benchmarking
    assert dist.group.WORLD is not None
    symm_mem.rendezvous(new_tensor, dist.group.WORLD.group_name)
    return new_tensor


def _clone_args(
    args: Sequence[object],
    idx_to_clone: Sequence[int] | None = None,
) -> Sequence[object]:
    """
    Clone the given arguments, but cloning only the tensors specified by
      idx_to_clone. If idx_to_clone is None, clone all tensors.
    """

    def _should_clone(idx: int) -> bool:
        return idx_to_clone is None or idx in idx_to_clone

    args_flat, tree_spec = tree_flatten(args)
    old_arg_to_new_arg = {}

    for i, arg in enumerate(args_flat):
        if _should_clone(i) and is_symm_mem_tensor(arg):
            new_arg = _clone_symm_mem_tensor(arg)
            old_arg_to_new_arg[get_signal_pad_ptrs_dev(arg)] = get_signal_pad_ptrs_dev(
                new_arg
            )
            old_arg_to_new_arg[arg] = new_arg  # pyrefly: ignore[unsupported-operation]

    for i, arg in enumerate(args_flat):
        if arg in old_arg_to_new_arg:
            args_flat[i] = old_arg_to_new_arg[arg]
            continue
        if not isinstance(arg, torch.Tensor):
            continue
        if _should_clone(i):
            clone = arg.detach().clone()
            clone.requires_grad_(arg.requires_grad)
            args_flat[i] = clone

    return tree_unflatten(args_flat, tree_spec)


def _estimate_tree_bytes(obj: object) -> int:
    """Estimate the memory usage of a pytree of objects, counting shared storage only once."""
    total = 0
    seen_ptrs: set[int] = set()

    def _accumulate(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        size = tensor.element_size() * tensor.numel()
        try:
            storage = tensor.untyped_storage()
        except RuntimeError:
            pass
        else:
            ptr = storage.data_ptr()
            if ptr in seen_ptrs:
                return tensor
            seen_ptrs.add(ptr)
            size = storage.nbytes()
        total += size
        return tensor

    tree_map_only(torch.Tensor, _accumulate, obj)
    return total


def _triton_compile(
    fn: CompiledConfig,
    args: Sequence[object],
    config: Config,
    kernel: _AutotunableKernel,
) -> bool:
    """Trigger Triton JIT compilation without running the kernel.

    Extracts the Triton kernel and its launch arguments from fn, then
    invokes the precompiler so the compiled binary is cached before the
    actual benchmark run.

    The function requires the availability of CUDA.
    """

    def extract_launcher(
        triton_kernel: object,
        grid: tuple[int, ...],
        *launch_args: object,
        **launch_kwargs: object,
    ) -> NoReturn:
        raise _ExtractedLaunchArgs(triton_kernel, grid, launch_args, launch_kwargs)

    try:
        fn(*args, _launcher=extract_launcher)
        raise RuntimeError("Expected _ExtractedLaunchArgs to be raised")
    except _ExtractedLaunchArgs as extracted:
        precompiler = make_precompiler(
            cast("Any", extracted.kernel),
            config,
            cast("BoundKernel", kernel),
        )(*extracted.args, **extracted.kwargs)
        if precompiler is already_compiled:
            return True
        if precompiler is already_compiled_fail:
            return False
        return precompiler(False)  # pyrefly: ignore[bad-argument-count]
    except Exception:
        return False


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class BenchmarkResult(NamedTuple):
    """Result tuple returned by benchmark_batch."""

    config: Config
    fn: Callable[..., object]
    perf: float
    status: Literal["ok", "error", "timeout", "peer_compilation_fail"]
    compile_time: float | None


# ---------------------------------------------------------------------------
# BenchmarkProvider (abstract base)
# ---------------------------------------------------------------------------


class BenchmarkProvider(abc.ABC):
    """Abstract interface for the compile -> benchmark pipeline.

    Search algorithms access this via ``self.benchmark_provider``.
    Subclass this to provide alternative benchmarking strategies
    (e.g. cross-node precompilation, overlapped precompile+benchmark).

    Lifecycle::

        provider = LocalBenchmarkProvider(...)  # __init__ computes baseline
        provider.setup()  # prepare resources (tmpdir, etc.)
        try:
            provider.benchmark_batch(configs)  # compile and benchmark
            provider.verify_benchmark(results)  # re-measure with higher accuracy
        finally:
            provider.cleanup()  # release resources

    ``BaseSearch.autotune()`` manages this lifecycle automatically.
    """

    best_perf_so_far: float
    mutated_arg_indices: Sequence[int]

    @abc.abstractmethod
    def __init__(
        self,
        kernel: _AutotunableKernel,
        settings: Settings,
        config_spec: ConfigSpec,
        args: Sequence[object],
        log: AutotuningLogger,
        autotune_metrics: AutotuneMetrics,
    ) -> None:
        """Initialize the provider with kernel context and benchmarking state."""
        ...

    @abc.abstractmethod
    def benchmark_batch(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        """Compile and benchmark a batch of configurations."""
        ...

    @abc.abstractmethod
    def benchmark(self, config: Config) -> BenchmarkResult:
        """Compile and benchmark a single configuration."""
        ...

    @abc.abstractmethod
    def benchmark_function(self, config: Config, fn: CompiledConfig) -> float:
        """Benchmark a single compiled function.  Returns time in ms or inf."""
        ...

    @abc.abstractmethod
    def post_initial_benchmark(
        self,
        results: list[BenchmarkResult],
    ) -> None:
        """Hook called after the initial population has been benchmarked.

        Allows the provider to optimize its pipeline based on observed
        data (e.g. adjusting compile timeouts).
        """
        ...

    @abc.abstractmethod
    def verify_benchmark(
        self,
        results: list[BenchmarkResult],
        *,
        desc: str = "Verifying",
    ) -> list[float]:
        """Re-measure previously benchmarked configs with higher accuracy.

        Used when comparing configs with small performance differences.
        The provider decides the measurement strategy (e.g. interleaved
        execution, repeated trials).  Returns a list of timings (one
        per result).
        """
        ...

    @abc.abstractmethod
    def setup(self) -> None:
        """Prepare resources needed before benchmarking begins (e.g. tmpdir)."""
        ...

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Release resources (tmpdir, subprocesses, etc.)."""
        ...


# ---------------------------------------------------------------------------
# LocalBenchmarkProvider
# ---------------------------------------------------------------------------


class LocalBenchmarkProvider(BenchmarkProvider):
    """Local single-machine benchmark provider.

    Compiles kernels locally, optionally precompiles in subprocesses
    (fork/spawn), and benchmarks on the local GPU.  This is the default
    provider created by ``BaseSearch._prepare()``.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        settings: Settings,
        config_spec: ConfigSpec,
        args: Sequence[object],
        log: AutotuningLogger,
        autotune_metrics: AutotuneMetrics,
    ) -> None:
        self.kernel = kernel
        self.settings = settings
        self.config_spec = config_spec
        self.args = args
        self.log = log
        self._autotune_metrics = autotune_metrics
        self.best_perf_so_far: float = inf
        self._precompile_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._precompile_args_path: str | None = None
        self._precompile_result_counter: count[int] = count()

        # Compute baseline and derived state
        (
            self._baseline_output,
            self.mutated_arg_indices,
            self._baseline_post_args,
        ) = self._compute_baseline()
        self._effective_atol, self._effective_rtol = (
            self._compute_effective_tolerances()
        )
        self._jobs = self._decide_num_jobs()

    # ------------------------------------------------------------------
    # Internal helpers (baseline, tolerances, jobs, precompile context)
    # ------------------------------------------------------------------

    def _compute_baseline(
        self,
    ) -> tuple[object, Sequence[int], Sequence[object] | None]:
        """Compute baseline output and detect mutated arguments."""
        new_args = _clone_args(self.args)

        # Use custom baseline function if provided
        if self.settings.autotune_baseline_fn is not None:
            try:
                baseline_output = self.settings.autotune_baseline_fn(*new_args)
                torch.accelerator.synchronize()
            except Exception as e:
                raise exc.AutotuneError(
                    "Custom baseline function failed while computing baseline.\n"
                    f"Baseline function: {self.settings.autotune_baseline_fn}\n"
                ) from e
        else:
            # Use default config
            baseline_config = self.config_spec.default_config()
            try:
                baseline_output = self.kernel.compile_config(
                    baseline_config, allow_print=False
                )(*new_args)
                torch.accelerator.synchronize()
            except Exception as e:
                decorator = self.kernel.format_kernel_decorator(
                    baseline_config, self.settings
                )
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    baseline_config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.kernel.maybe_log_repro(self.log.error, new_args, baseline_config)
                raise exc.InvalidConfig(
                    "Default config failed while computing baseline.\n"
                    f"Default config: {decorator}\n"
                    f"{SUPPRESSED_TRITON_CODE_MSG}\n"
                    "To work around this error, you could set `@helion.kernel(autotune_baseline_fn=...)` "
                    "to provide a custom baseline function (e.g. PyTorch eager implementation of your kernel)."
                ) from e

        original_args_flat, _ = tree_flatten(self.args)
        new_args_flat, _ = tree_flatten(new_args)
        mutated_tensor_idxs = []
        # we should only count tensors, since they won't be bound or removed
        tensor_idx = 0
        for old, new in zip(original_args_flat, new_args_flat, strict=False):
            if not (isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor)):
                continue
            try:
                equal = torch.equal(new, old)
            except RuntimeError:
                # torch.equal and device-to-host copies can fail on some
                # devices (e.g., TPU for large tensors).  Conservatively
                # assume the argument was not mutated.
                equal = True
            if not equal:
                mutated_tensor_idxs.append(tensor_idx)
            tensor_idx += 1
        baseline_post_args = _clone_args(new_args, idx_to_clone=mutated_tensor_idxs)
        return baseline_output, mutated_tensor_idxs, baseline_post_args

    def _compute_effective_tolerances(self) -> tuple[float, float]:
        """Compute atol/rtol for accuracy validation based on output dtypes."""
        # Default tolerance when not user-specified
        DEFAULT_TOL = 1e-2

        # Get user-specified or default tolerances
        atol = self.settings.autotune_baseline_atol
        rtol = self.settings.autotune_baseline_rtol

        # Collect all dtypes from baseline output and mutated args
        dtypes = set()

        def collect_dtypes(obj: object) -> object:
            if isinstance(obj, torch.Tensor):
                dtypes.add(obj.dtype)
            return obj

        tree_map_only(torch.Tensor, collect_dtypes, self._baseline_output)
        if len(self.mutated_arg_indices) > 0 and self._baseline_post_args is not None:
            tree_map_only(torch.Tensor, collect_dtypes, self._baseline_post_args)

        # Only apply strict tolerances if ALL dtypes are fp8
        # Mixed dtypes (fp8 + fp32) would be too strict with atol=0.0, rtol=0.0
        all_dtypes_are_fp8 = dtypes and all(dtype in _FP8_DTYPES for dtype in dtypes)

        if all_dtypes_are_fp8:
            # All dtypes are fp8 - use bitwise comparison
            # unless the user explicitly set either tolerance value (i.e., not None)
            if atol is None and rtol is None:
                self.log(
                    f"Detected fp8 dtype(s) in output: {dtypes}. "
                    "Using bitwise comparison (atol=0.0, rtol=0.0) for autotuning accuracy check."
                )
                return 0.0, 0.0

        # Use user-specified values or defaults
        return (
            atol if atol is not None else DEFAULT_TOL,
            rtol if rtol is not None else DEFAULT_TOL,
        )

    def _decide_num_jobs(self) -> int:
        """Determine the number of concurrent precompile jobs."""
        if not self.settings.autotune_precompile:
            return 1

        jobs = self.settings.autotune_precompile_jobs
        if not jobs:
            jobs = os.cpu_count() or 1

        if self.settings.autotune_precompile != "spawn":
            return jobs

        memory_per_job = _estimate_tree_bytes(self.args) + _estimate_tree_bytes(
            self._baseline_output
        )
        memory_per_job *= 2  # safety factor
        if memory_per_job <= 0:
            return jobs

        device = self.kernel.env.device
        if device.type != "cuda":
            # TODO(jansel): support non-cuda devices
            return jobs

        available_memory, _ = torch.cuda.mem_get_info(device)
        jobs_by_memory = available_memory // memory_per_job
        if jobs_by_memory < jobs:
            gib_per_job = memory_per_job / (1024**3)
            available_gib = available_memory / (1024**3)
            if jobs_by_memory > 0:
                self.log.warning(
                    f"Reducing autotune precompile spawn jobs from {jobs} to {jobs_by_memory} "
                    f"due to limited GPU memory (estimated {gib_per_job:.2f} GiB per job, "
                    f"{available_gib:.2f} GiB free). "
                    f"Set HELION_AUTOTUNE_PRECOMPILE_JOBS={jobs_by_memory} "
                    "to make this lower cap persistent, "
                    'set HELION_AUTOTUNE_PRECOMPILE="fork" to disable spawning, or reduce GPU memory usage.'
                )
            else:
                raise exc.AutotuneError(
                    "Autotune precompile spawn mode requires at least one job, but estimated "
                    "memory usage exceeds available GPU memory."
                    f"Estimated {gib_per_job:.2f} GiB per job, but only "
                    f"{available_gib:.2f} GiB free. "
                    'Set HELION_AUTOTUNE_PRECOMPILE="fork" to disable spawning, or reduce GPU memory usage.'
                )
            jobs = jobs_by_memory

        return jobs

    def _precompile_context(self) -> PrecompileContext:
        """Build the narrow context that PrecompileFuture needs."""
        return PrecompileContext(
            settings=self.settings,
            log=self.log,
            kernel=self.kernel,
            args=self.args,
            jobs=self._jobs,
        )

    def setup(self) -> None:
        """Prepare precompile tmpdir and args for spawn mode."""
        if self._precompile_tmpdir is None:
            self._precompile_tmpdir = tempfile.TemporaryDirectory()
        if self.settings.autotune_precompile == "spawn":
            args_path = os.path.join(self._precompile_tmpdir.name, "args.pt")
            torch.save(self.args, args_path)
            self._precompile_args_path = args_path

    def _next_precompile_result_path(self) -> str:
        """Return a fresh path for a precompile result file."""
        if self._precompile_tmpdir is None:
            self._precompile_tmpdir = tempfile.TemporaryDirectory()
        return os.path.join(
            self._precompile_tmpdir.name,
            f"result_{next(self._precompile_result_counter)}.pkl",
        )

    def cleanup(self) -> None:
        """Release precompile tmpdir and related resources."""
        if self._precompile_tmpdir is not None:
            self._precompile_tmpdir.cleanup()
            self._precompile_tmpdir = None
        self._precompile_args_path = None
        self._precompile_result_counter = count()

    # ------------------------------------------------------------------
    # Accuracy validation
    # ------------------------------------------------------------------

    def _validate_against_baseline(
        self, config: Config, output: object, args: Sequence[object]
    ) -> bool:
        """Return True if ``output`` matches the baseline within tolerances."""
        try:
            custom_check = self.settings.autotune_baseline_accuracy_check_fn
            if custom_check is not None:
                custom_check(output, self._baseline_output)
                if len(self.mutated_arg_indices) > 0:
                    custom_check(args, self._baseline_post_args)
            else:
                _assert_close(
                    output,
                    self._baseline_output,
                    atol=self._effective_atol,
                    rtol=self._effective_rtol,
                )
                if os.getenv("CHECK_INPUT_ACCURACY", "1") == "1":
                    if len(self.mutated_arg_indices) > 0:
                        # For distributed kernel, group_name may also be a argument.
                        # torch.testing.assert_close does not handle str argument.
                        # Filter needed.
                        assert self._baseline_post_args is not None
                        _assert_close(
                            args,
                            self._baseline_post_args,
                            atol=self._effective_atol,
                            rtol=self._effective_rtol,
                        )
        except AssertionError as e:
            if not self.settings.autotune_ignore_errors:
                self.log.warning(
                    f"Skipping config with accuracy mismatch: {config!r}\n{e!s}\nUse HELION_AUTOTUNE_ACCURACY_CHECK=0 to disable this check.\n"
                )
            return False
        return True

    # ------------------------------------------------------------------
    # Single-config benchmarking
    # ------------------------------------------------------------------

    def benchmark_function(self, config: Config, fn: CompiledConfig) -> float:
        """Benchmark a single compiled function.  Returns time in ms or inf."""
        self._autotune_metrics.num_configs_tested += 1
        self.log.debug(lambda: f"Running benchmark for {config!r}")
        _captured_output: list[str] = [""]
        _capture_ctx = (
            capture_output()
            if _get_failure_dump_dir()
            else contextlib.nullcontext(_captured_output)
        )

        if len(self.mutated_arg_indices) > 0:
            working_args = _clone_args(self.args, idx_to_clone=self.mutated_arg_indices)
        else:
            working_args = self.args

        # precompile in the current process for distributed kernels.
        # The reason we need this is due to some tricky distributed kernels
        # like https://gist.github.com/shunting314/81f13ce00f835b21ab6466e21454b7c5 . We specialize the RANK argument for each GPU,
        # some rank may get out of resource errors while others don't
        # due to the specialization.
        #
        # Without precompilation here, some rank may fail and skip running
        # the kernel while outer ranks waiting for its peers. It
        # results in a stuck job.
        #
        # Precompiilation happening in child process is not enough because
        # CUDA is not available there. We can not check resource usage
        # like shared-memory, tmem, max-threads etc.
        #
        # This precompilation has overhead. Only do it if distributed is
        # initialized.

        if dist.is_initialized():
            # Trigger Triton JIT compilation before running the kernel
            compile_success = _triton_compile(fn, working_args, config, self.kernel)
            compile_success_all = all(all_gather_object(compile_success))

            if not compile_success_all:
                return inf

        try:
            # TODO(jansel): early exit with fewer trials if early runs are slow
            self.log.debug(lambda: f"Running {config} at {datetime.datetime.now()}")
            t0 = time.perf_counter()
            torch.accelerator.synchronize()

            with _capture_ctx as _captured_output:
                output = fn(*working_args)  # make sure the kernel is compiled

            torch.accelerator.synchronize()

            pass_accuracy_check = (
                not self.settings.autotune_accuracy_check
                or self._validate_against_baseline(config, output, working_args)
            )
            if not pass_accuracy_check:
                self._autotune_metrics.num_accuracy_failures += 1
            if not all(all_gather_object(pass_accuracy_check)):
                # for distributed kernels like matmul-reduce-scatter, different ranks compute
                # a different chunk. It's possible that some ranks pass the accuracy check while
                # others don't. Skip the config if any rank fails the accuracy check.
                # Without this synchronization, some ranks go on to call the benchmark function
                # while other ranks return immediately, this will cause stuck jobs!
                return inf

            bench_fn = self.kernel.bench_compile_config(config, allow_print=False)
            bench_fn(*working_args)  # warmup benchmark kernel

            t1 = time.perf_counter()
            _backend = getattr(self.config_spec, "backend", None)
            _bench_fn = (
                _backend.get_do_bench() if _backend is not None else None
            ) or do_bench
            res = _bench_fn(
                functools.partial(bench_fn, *working_args),
                return_mode="median",
                warmup=1,  # we are already warmed up above
                rep=50,
            )
            res = sync_object(res)
            t2 = time.perf_counter()
            assert isinstance(res, float)

            self.log.debug(
                lambda: f"result: {res:.4f}ms (took {t1 - t0:.1f}s + {t2 - t1:.1f}s)",
            )
            if res < self.best_perf_so_far:
                self.best_perf_so_far = res
            return res
        except Exception as e:
            # e.__traceback__ holds references to all local variables in the call stack frames.
            # When a Triton kernel fails, the output tensors allocated by the Helion kernel function
            # were being held by the traceback, preventing them from being freed.
            e.__traceback__ = None
            maybe_dump_triton_failure(
                self.kernel,
                config,
                e,
                captured_output=_captured_output[0] or None,
            )
            if match_unrecoverable_runtime_error(e):
                self.kernel.maybe_log_repro(self.log.error, self.args, config)
                raise exc.TritonUnrecoverableRuntimeError(
                    reason=str(e),
                    decorator=self.kernel.format_kernel_decorator(
                        config, self.settings
                    ),
                    error=f"{type(e).__qualname__}: {e}",
                ) from e
            _backend = getattr(self.config_spec, "backend", None)
            action = (
                _backend.classify_autotune_exception(e)
                if _backend is not None
                else None
            ) or classify_triton_exception(e)
            if self.settings.autotune_ignore_errors:
                pass
            elif action == "raise":
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.kernel.maybe_log_repro(self.log.error, self.args, config)
                raise exc.TritonError(
                    error=f"{type(e).__qualname__}: {e}",
                    decorator=decorator,
                    code=SUPPRESSED_TRITON_CODE_MSG,
                ) from e
            elif action == "warn":
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.log.warning(format_triton_compile_failure(config, e, self.kernel))
                self.kernel.maybe_log_repro(self.log.warning, self.args, config)
            else:
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.log.debug(f"Benchmarking failed: {type(e).__name__}: {e}")
                self.kernel.maybe_log_repro(self.log.debug, self.args, config)

            self._autotune_metrics.num_compile_failures += 1
            return inf

    # ------------------------------------------------------------------
    # Precompilation
    # ------------------------------------------------------------------

    def create_precompile_future(
        self, config: Config, fn: CompiledConfig
    ) -> PrecompileFuture:
        """Create a subprocess-based precompile future for ``fn``."""
        ctx = self._precompile_context()
        if not self.settings.autotune_precompile:
            return PrecompileFuture.skip(ctx, config, True)
        mode = self.settings.autotune_precompile
        if mode not in {"fork", "spawn"}:
            raise exc.InvalidAPIUsage("autotune_precompile must be 'fork' or 'spawn'")
        if len(self.mutated_arg_indices) > 0:
            args = _clone_args(self.args, idx_to_clone=self.mutated_arg_indices)
        else:
            args = self.args

        return PrecompileFuture.create(
            ctx=ctx,
            config=config,
            fn=fn,
            args=args,
            result_path=self._next_precompile_result_path(),
            args_path=self._precompile_args_path,
        )

    # Defaults for adaptive compile timeout adjustment.
    _ADAPTIVE_TIMEOUT_LOWER_BOUND: float = 30.0
    _ADAPTIVE_TIMEOUT_QUANTILE: float = 0.9

    def post_initial_benchmark(
        self,
        results: list[BenchmarkResult],
    ) -> None:
        """Hook called after the initial population has been benchmarked.

        Allows the provider to optimize its pipeline based on observed
        data.  The default implementation adjusts the compile timeout
        using the observed compile times from *results*.
        """
        if not self.settings.autotune_adaptive_timeout:
            return

        min_seconds = self._ADAPTIVE_TIMEOUT_LOWER_BOUND
        quantile = self._ADAPTIVE_TIMEOUT_QUANTILE

        # Collect valid compile times (non-None and positive)
        compile_times = [
            r.compile_time
            for r in results
            if r.compile_time is not None and r.compile_time > 0
        ]

        if not compile_times:
            self.log("No valid compile times found, keeping default timeout")
            return

        original_timeout = self.settings.autotune_compile_timeout

        # Compute the quantile
        compile_times_sorted = sorted(compile_times)
        quantile_index = min(
            int(len(compile_times_sorted) * quantile),
            len(compile_times_sorted) - 1,
        )
        quantile_value = compile_times_sorted[quantile_index]

        adaptive_timeout = int(min(max(quantile_value, min_seconds), original_timeout))

        self.settings.autotune_compile_timeout = adaptive_timeout

        self.log(
            f"Adaptive compile timeout: {adaptive_timeout}s "
            f"({quantile:.0%} percentile={quantile_value:.1f}s, "
            f"bounds=[{min_seconds}s, {original_timeout}s])"
        )

    # ------------------------------------------------------------------
    # Batch benchmarking
    # ------------------------------------------------------------------

    def _benchmark(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        """Internal: compile, precompile, and benchmark a list of configs."""
        fns: list[Callable[..., object]] = []
        valid_configs: list[Config] = []
        futures: list[PrecompileFuture] | None = None
        for i, config in enumerate(configs):
            try:
                fn = self.kernel.compile_config(config, allow_print=False)
            except Exception:
                # If all configs failed, raise error
                if not valid_configs and i == len(configs) - 1:
                    raise
                self.log.warning(
                    "Skipping config that failed to compile: %s",
                    self.kernel.format_kernel_decorator(config, self.settings),
                    exc_info=True,
                )
                continue
            fns.append(fn)
            valid_configs.append(config)
        configs = valid_configs
        if self.settings.autotune_precompile:
            futures = list(
                starmap(
                    self.create_precompile_future,
                    zip(configs, fns, strict=True),
                )
            )
            precompile_desc = (
                f"{desc} precompiling" if self.settings.autotune_progress_bar else None
            )
            is_workings = PrecompileFuture.wait_for_all(futures, desc=precompile_desc)
            precompile_status: list[Literal["ok", "error", "timeout"]] = []
            for future, ok in zip(futures, is_workings, strict=True):
                reason = future.failure_reason
                if ok:
                    precompile_status.append("ok")
                elif reason == "timeout":
                    precompile_status.append("timeout")
                else:
                    precompile_status.append("error")
        else:
            is_workings = [True] * len(configs)
            precompile_status = ["ok"] * len(configs)

        results: list[BenchmarkResult] = []

        # Render a progress bar only when the user requested it.
        iterator = iter_with_progress(
            enumerate(zip(fns, is_workings, precompile_status, strict=True)),
            total=len(configs),
            description=f"{desc} exploring neighbors",
            enabled=self.settings.autotune_progress_bar,
        )
        for index, (fn, is_working, reason) in iterator:
            config = configs[index]
            if futures is not None:
                future = futures[index]
                compile_time = (
                    future.elapsed
                    if future.process is not None and future.started
                    else None
                )
            else:
                compile_time = None
            status: Literal["ok", "error", "timeout", "peer_compilation_fail"]
            if all(all_gather_object(is_working)):
                # Log started before benchmarking to help identify hangs
                self.log.record_autotune_entry(
                    AutotuneLogEntry(
                        generation=self._autotune_metrics.num_generations,
                        status="started",
                        perf_ms=None,
                        compile_time=compile_time,
                        config=config,
                    )
                )
                # benchmark one-by-one to avoid noisy results
                perf = self.benchmark_function(config, fn)
                status = "ok" if math.isfinite(perf) else "error"
                # Log completion after benchmarking
                self.log.record_autotune_entry(
                    AutotuneLogEntry(
                        generation=self._autotune_metrics.num_generations,
                        status=status,
                        perf_ms=perf if math.isfinite(perf) else None,
                        compile_time=compile_time,
                        config=config,
                    )
                )
                results.append(
                    BenchmarkResult(
                        config=config,
                        fn=fn,
                        perf=perf,
                        status=status,
                        compile_time=compile_time,
                    )
                )
            else:
                status = "timeout" if reason == "timeout" else "error"
                if is_working:
                    status = "peer_compilation_fail"
                results.append(
                    BenchmarkResult(
                        config=config,
                        fn=fn,
                        perf=inf,
                        status=status,
                        compile_time=compile_time,
                    )
                )
        return results

    def benchmark_batch(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        """Compile and benchmark a batch of configurations."""
        return self._benchmark(configs, desc=desc)

    def benchmark(self, config: Config) -> BenchmarkResult:
        """Compile and benchmark a single configuration."""
        return self.benchmark_batch([config])[0]

    def verify_benchmark(
        self,
        results: list[BenchmarkResult],
        *,
        desc: str = "Verifying",
    ) -> list[float]:
        """Re-measure previously benchmarked configs with higher accuracy."""
        if len(self.mutated_arg_indices) > 0:
            bench_args = _clone_args(self.args, idx_to_clone=self.mutated_arg_indices)
        else:
            bench_args = self.args
        iterator = [functools.partial(r.fn, *bench_args) for r in results]

        # Calculate repeat count based on best performance
        base_repeat = (
            int(200 / self.best_perf_so_far)
            if math.isfinite(self.best_perf_so_far) and self.best_perf_so_far > 0
            else 1000
        )
        repeat = min(1000, max(3, base_repeat))
        if (capstr := os.getenv("HELION_CAP_REBENCHMARK_REPEAT")) is not None:
            repeat = min(repeat, int(capstr))

        _backend = getattr(self.config_spec, "backend", None)
        _ib = (
            _backend.get_interleaved_bench() if _backend is not None else None
        ) or interleaved_bench
        bench_fn: Callable[..., list[float]] = (
            self.settings.autotune_benchmark_fn or _ib
        )
        if self.settings.autotune_progress_bar:
            new_timings = bench_fn(iterator, repeat=repeat, desc=desc)
        else:
            new_timings = bench_fn(iterator, repeat=repeat)
        new_timings = sync_object(new_timings)
        for t in new_timings:
            if t < self.best_perf_so_far:
                self.best_perf_so_far = t
        return new_timings
