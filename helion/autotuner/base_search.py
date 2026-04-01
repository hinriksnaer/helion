from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import logging
import math
import os
import pprint
import random
import re
import sys
import time
import types
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal
from typing import Protocol
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map_only

from .. import exc
from .._compat import extract_device
from .._compat import get_device_name
from .benchmark_provider import BenchmarkProvider
from .benchmark_provider import BenchmarkResult as BenchmarkResult
from .benchmark_provider import LocalBenchmarkProvider
from .logger import AutotuningLogger
from .metrics import AutotuneMetrics
from .metrics import _run_post_autotune_hooks
from .precompile_future import PrecompileFuture as PrecompileFuture
from helion._dist_utils import all_gather_object
from helion._dist_utils import is_master_rank

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from . import ConfigSpec
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig
    from .local_cache import SavedBestConfig
    from .precompile_future import PrecompileFuture as PrecompileFuture
    from helion.autotuner.effort_profile import AutotuneEffortProfile


class _HasDevice(Protocol):
    device: torch.device


class _AutotunableKernel(Protocol):
    @property
    def config_spec(self) -> ConfigSpec: ...

    @property
    def settings(self) -> Settings: ...

    @property  # pyrefly: ignore[bad-return]
    def env(self) -> _HasDevice: ...

    @property
    def configs(self) -> Sequence[Config]: ...

    def compile_config(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        allow_print: bool = True,
    ) -> Callable[..., object]:
        """Compile a kernel for the given config, used for accuracy checking."""
        ...

    def bench_compile_config(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        allow_print: bool = True,
    ) -> Callable[..., object]:
        """Compile a kernel for the given config, used for benchmarking.

        By default this is the same as compile_config. Override to return
        a different callable for benchmarking, e.g. a fused kernel that
        includes prologue/epilogue code from Inductor.
        """
        ...

    def format_kernel_decorator(self, config: Config, settings: Settings) -> str: ...

    def get_cached_path(self, config: Config | None = None) -> str | None: ...

    def to_triton_code(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str | None: ...

    def maybe_log_repro(
        self,
        log_func: Callable[[str], None],
        args: Sequence[object],
        config: Config | None = None,
    ) -> None: ...


_CODE_OBJECT_RE = re.compile(r"<code object .+?, line \d+>")


class _CodeSentinel:
    """Stable stand-in for types.CodeType so spec key comparison is repr-independent."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<code>"


_CODE_SENTINEL = _CodeSentinel()


def _normalize_spec_key(key: object) -> object:
    """Replace types.CodeType with a stable sentinel in a spec key tree."""
    return tree_map_only(types.CodeType, lambda _: _CODE_SENTINEL, key)


def _normalize_spec_key_str(s: str) -> str:
    """Normalize a specialization_key string for cache comparison.

    Replaces code object repr strings with a stable '<code>' sentinel,
    allowing FROM_BEST_AVAILABLE to match function arguments based
    on their closure values only, ignoring code object identity.
    """
    return _CODE_OBJECT_RE.sub("<code>", s)


class BaseAutotuner(abc.ABC):
    """
    Abstract base class for all autotuners and classes that wrap autotuners, like caching.
    """

    @abc.abstractmethod
    def autotune(self, *, skip_cache: bool = False) -> Config:
        raise NotImplementedError


class BaseSearch(BaseAutotuner):
    """Base class for all search algorithms.

    Holds kernel context, settings, benchmark provider, and the
    autotune orchestration loop.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        benchmark_provider_cls: type[BenchmarkProvider] = LocalBenchmarkProvider,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.settings: Settings = kernel.settings
        self.config_spec: ConfigSpec = kernel.config_spec
        self.args: Sequence[object] = args
        self.log = AutotuningLogger(self.settings)
        self._benchmark_provider_cls = benchmark_provider_cls
        self._prepared = False

    def _prepare(self) -> None:
        """Some initialization deferred until autotuning actually runs.

        This is called at the start of autotune() so that cache hits skip it.
        """
        if self._prepared:
            return
        self._prepared = True
        seed = self.settings.autotune_random_seed
        random.seed(seed)
        self.log(f"Autotune random seed: {seed}")
        self._autotune_metrics: AutotuneMetrics = AutotuneMetrics(
            kernel_name=getattr(getattr(self.kernel, "kernel", None), "name", ""),
            input_shapes=str(
                [tuple(arg.shape) for arg in self.args if isinstance(arg, torch.Tensor)]
            ),
            hardware=get_device_name(extract_device(self.args)) or "",
            random_seed=self.settings.autotune_random_seed,
            search_algorithm=type(self).__name__,
        )
        self.benchmark_provider = self._benchmark_provider_cls(
            kernel=self.kernel,
            settings=self.settings,
            config_spec=self.config_spec,
            args=self.args,
            log=self.log,
            autotune_metrics=self._autotune_metrics,
        )

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        """Retrieve extra kwargs from the effort profile for the autotuner."""
        kwargs: dict[str, object] = {}
        if settings.autotune_max_generations is not None:
            kwargs.setdefault("max_generations", settings.autotune_max_generations)
        return kwargs

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """Perform autotuning to find the best configuration."""
        self._skip_cache = skip_cache
        self._prepare()
        start = time.perf_counter()
        exit_stack = contextlib.ExitStack()
        with exit_stack:
            if self.settings.autotune_log:
                exit_stack.enter_context(self.log.autotune_logging())
            self.log.reset()
            env_overrides = {"TRITON_LOCAL_BUILD": "1"}
            if "TRITON_STORE_BINARY_ONLY" not in os.environ:
                env_overrides["TRITON_STORE_BINARY_ONLY"] = "1"
            exit_stack.enter_context(patch.dict(os.environ, env_overrides, clear=False))
            self.benchmark_provider.setup()
            exit_stack.callback(self.benchmark_provider.cleanup)
            try:
                best = self._autotune()
            finally:
                self._finalize_autotune_metrics()
        end = time.perf_counter()
        kernel_decorator = self.kernel.format_kernel_decorator(best, self.settings)

        self.log(
            f"Autotuning complete in {end - start:.1f}s after searching {self._autotune_metrics.num_configs_tested} configs.\n"
            "One can hardcode the best config and skip autotuning with:\n"
            f"    {kernel_decorator}\n",
            level=logging.INFO + 5,
        )
        cached_path = self.kernel.get_cached_path(best)
        if cached_path is not None and is_master_rank():
            self.log(f"Code of selected kernel: {cached_path}")
        self.kernel.maybe_log_repro(self.log.warning, self.args, best)
        if self.settings.print_output_code:
            triton_code = self.kernel.to_triton_code(best)
            if triton_code is not None:
                print(triton_code, file=sys.stderr)
        return best

    def _autotune(self) -> Config:
        """Subclasses implement this to perform the actual search."""
        raise NotImplementedError

    def set_generation(self, generation: int) -> None:
        self._autotune_metrics.num_generations = generation

    def _finalize_autotune_metrics(self) -> None:
        self._autotune_metrics.best_perf_ms = (
            self.benchmark_provider.best_perf_so_far
            if math.isfinite(self.benchmark_provider.best_perf_so_far)
            else 0.0
        )
        self._autotune_metrics.finalize()
        _run_post_autotune_hooks(self._autotune_metrics)


def check_population_consistency(population: Sequence[PopulationMember]) -> None:
    if os.getenv("HELION_DEBUG_DISTRIBUTED") != "1" or not dist.is_initialized():
        return

    # remove unpickled fields
    sanitized_population = tuple((p.config, p.perfs) for p in population)
    all_sanitized_population = all_gather_object(sanitized_population)
    if all_sanitized_population != all_sanitized_population[:1] * len(
        all_sanitized_population
    ):
        raise exc.InconsistantConfigsAcrossRanks


@dataclasses.dataclass
class PopulationMember:
    flat_values: FlatConfig
    config: Config
    perfs: list[float] = dataclasses.field(default_factory=list)
    _result: BenchmarkResult | None = None

    def update(self, result: BenchmarkResult) -> None:
        """Record a benchmark result."""
        assert result.config is self.config
        self._result = result
        self.perfs.append(result.perf)

    @property
    def result(self) -> BenchmarkResult:
        """The latest benchmark result."""
        assert self._result is not None
        return self._result

    @property
    def perf(self) -> float:
        return self.perfs[-1]

    @property
    def fn(self) -> Callable[..., object]:
        return self.result.fn

    @property
    def status(
        self,
    ) -> Literal["ok", "error", "timeout", "peer_compilation_fail", "unknown"]:
        if self._result is None:
            return "unknown"
        return self._result.status

    @property
    def compile_time(self) -> float | None:
        if self._result is None:
            return None
        return self._result.compile_time


def performance(member: PopulationMember) -> float:
    """
    Retrieve the performance of a population member.  Used as a sort key.

    Args:
        member: The population member.

    Returns:
        The performance of the member.
    """
    return member.perf


class PopulationBasedSearch(BaseSearch):
    """Base class for population-based search algorithms (e.g. PatternSearch, DifferentialEvolution).

    Adds population management, rebenchmarking, and finishing phases
    on top of the shared ``BaseSearch`` infrastructure.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        finishing_rounds: int = 0,
    ) -> None:
        """
        Initialize the PopulationBasedSearch object.

        Args:
            kernel: The kernel to be tuned.
            args: The arguments to be passed to the kernel.
            finishing_rounds: Number of finishing rounds to run after the main search.
        """
        super().__init__(kernel, args)
        self.finishing_rounds = finishing_rounds
        self.population: list[PopulationMember] = []
        self.config_gen: ConfigGeneration = self.config_spec.create_config_generation(
            overrides=self.settings.autotune_config_overrides or None,
            advanced_controls_files=self.settings.autotune_search_acf or None,
        )

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        """
        Retrieve extra kwargs from the effort profile for the autotuner.
        """
        from ..runtime.settings import _env_get_optional_int

        finishing_rounds = _env_get_optional_int("HELION_AUTOTUNE_FINISHING_ROUNDS")
        if finishing_rounds is None:
            finishing_rounds = profile.finishing_rounds

        return {
            "finishing_rounds": finishing_rounds,
            **super().get_kwargs_from_profile(profile, settings),
        }

    @property
    def best(self) -> PopulationMember:
        """
        Retrieve the best configuration in the population.

        Returns:
            The best population member.
        """
        return min(self.population, key=performance)

    @best.setter
    def best(self, value: PopulationMember) -> None:
        """Replace the current best member in the population."""
        idx = min(range(len(self.population)), key=lambda i: self.population[i].perf)
        self.population[idx] = value

    def benchmark_flat(self, to_check: list[FlatConfig]) -> list[PopulationMember]:
        """
        Benchmark multiple flat configurations.

        Args:
            to_check: A list of flat configurations to benchmark.

        Returns:
            A list of population members with the benchmark results.
        """
        result = [*map(self.make_unbenchmarked, to_check)]
        return self._benchmark_members(result)

    def make_unbenchmarked(self, flat_values: FlatConfig) -> PopulationMember:
        """
        Create a population member with unbenchmarked configuration.  You
        should pass the result of this to _benchmark_members.

        Args:
            flat_values: The flat configuration values.

        Returns:
            A population member with undefined performance.
        """
        config = self.config_gen.unflatten(flat_values)
        return PopulationMember(flat_values, config)

    def _get_current_hardware_and_specialization(
        self,
    ) -> tuple[str | None, str | None]:
        """
        Get the current hardware and specialization_key for matching cached configs.

        Returns:
            A tuple of (hardware, specialization_key) strings.
        """
        hardware = get_device_name(extract_device(self.args))

        inner_kernel = getattr(self.kernel, "kernel", None)
        if inner_kernel is None or not hasattr(inner_kernel, "specialization_key"):
            return hardware, None
        spec_key = inner_kernel.specialization_key(self.args)
        specialization_key = str(_normalize_spec_key(spec_key))

        return hardware, specialization_key

    def _find_similar_cached_configs(self, max_configs: int) -> list[SavedBestConfig]:
        """
        Find cached configs that match hardware, specialization_key, and
        structural fingerprint (config_spec_hash).

        Returns an empty list when cache is skipped (via HELION_SKIP_CACHE
        or the skip_cache parameter), so that "skip cache" consistently
        means no cache reads of any kind.

        Args:
            max_configs: Maximum number of configs to return.

        Returns:
            List of matching SavedBestConfig objects, sorted by file modification time (most recent first).
        """
        from .base_cache import should_skip_cache

        if self._skip_cache or should_skip_cache():
            return []

        from .local_cache import get_helion_cache_dir
        from .local_cache import iter_cache_entries

        current_hardware, current_spec_key = (
            self._get_current_hardware_and_specialization()
        )
        if current_hardware is None or current_spec_key is None:
            return []

        current_fingerprint_hash = self.config_spec.structural_fingerprint_hash()

        matching: list[SavedBestConfig] = []
        for entry in iter_cache_entries(
            get_helion_cache_dir(),
            max_scan=self.settings.autotune_best_available_max_cache_scan,
        ):
            if entry.hardware != current_hardware:
                continue
            if _normalize_spec_key_str(entry.specialization_key) != current_spec_key:
                continue
            # Skip entries without a matching structural fingerprint or flat_config.
            if entry.config_spec_hash != current_fingerprint_hash:
                continue
            if entry.flat_config is None:
                continue
            matching.append(entry)
            if len(matching) >= max_configs:
                break

        return matching

    def _generate_best_available_population_flat(self) -> list[FlatConfig]:
        """
        Generate initial population using default config plus cached configs.

        Always starts with the default configuration, then adds up to
        MAX_BEST_AVAILABLE_CONFIGS matching cached configs from previous runs.
        No random configs are added.  Duplicate configs are discarded.

        Returns:
            A list of unique FlatConfig values for the initial population.
            Minimum size is 1 (just default), maximum is 1 + autotune_best_available_max_configs setting.
        """
        # Always start with the default config
        default_flat = self.config_gen.default_flat()
        default_config = self.config_gen.unflatten(default_flat)
        seen: set[Config] = {default_config}
        result: list[FlatConfig] = [default_flat]
        self.log("Starting with default config")

        max_configs = self.settings.autotune_best_available_max_configs
        cached_entries = self._find_similar_cached_configs(max_configs)

        if cached_entries:
            self.log.debug(
                f"Found {len(cached_entries)} cached config(s) from previous runs"
            )

        duplicates = 0
        for i, entry in enumerate(cached_entries):
            try:
                self.log.debug(f"Cached config {i + 1}: {entry.config}")
                flat = entry.to_mutable_flat_config()
                transferred_config = self.config_gen.unflatten(flat)
                if transferred_config in seen:
                    duplicates += 1
                    self.log.debug(
                        f"Cached config {i + 1} is a duplicate, skipping: {transferred_config}"
                    )
                    continue
                seen.add(transferred_config)
                result.append(flat)
                self.log.debug(
                    f"Cached config {i + 1} (transferred): {transferred_config}"
                )
            except (ValueError, TypeError, KeyError, AssertionError) as e:
                self.log(f"Failed to transfer cached config {i + 1}: {e}")
                continue

        if duplicates > 0:
            self.log.debug(f"Discarded {duplicates} duplicate config(s)")

        self.log(
            f"Initial population: 1 default + {len(result) - 1} unique cached = {len(result)} total"
        )

        return result

    def _benchmark_members(
        self, members: list[PopulationMember], *, desc: str = "Benchmarking"
    ) -> list[PopulationMember]:
        """Benchmark population members via benchmark_batch and map results back."""
        results = self.benchmark_provider.benchmark_batch(
            [m.config for m in members], desc=desc
        )
        for member, result in zip(members, results, strict=True):
            member.update(result)
        return members

    def compare(self, a: PopulationMember, b: PopulationMember) -> int:
        """Compare two population members based on their performance."""
        if self.should_rebenchmark(a) and self.should_rebenchmark(b):
            self.rebenchmark([a, b])
        return (a.perf > b.perf) - (a.perf < b.perf)

    def should_rebenchmark(self, member: PopulationMember) -> bool:
        """Determine if a population member should be re-benchmarked."""
        threshold = self.settings.get_rebenchmark_threshold()
        return (
            member.perf < threshold * self.benchmark_provider.best_perf_so_far
            and math.isfinite(member.perf)
        )

    def rebenchmark(
        self, members: list[PopulationMember], *, desc: str = "Rebenchmarking"
    ) -> None:
        """Re-benchmark a list of population members to avoid outliers."""
        if len(members) < 2:
            return
        results = [m.result for m in members]
        new_timings = self.benchmark_provider.verify_benchmark(results, desc=desc)
        for m, t in zip(members, new_timings, strict=True):
            m.perfs.append(t)

    def verify_members(
        self,
        members: list[PopulationMember] | None = None,
        *,
        desc: str = "Rebenchmarking",
    ) -> None:
        """Re-benchmark the entire population to avoid outliers."""
        if members is None:
            members = self.population
        self.rebenchmark([p for p in members if self.should_rebenchmark(p)], desc=desc)

    def statistics(self) -> str:
        """Generate statistics for the current population."""
        return population_statistics(self.population)

    def run_finishing_phase(
        self, best: PopulationMember, rounds: int
    ) -> PopulationMember:
        """Simplify the best config by resetting parameters to defaults.

        Attempts to reset each parameter to its default value while
        maintaining performance, producing a minimal configuration.
        """
        if rounds <= 0:
            return best

        self.log(f"Starting finishing phase with {rounds} rounds")
        default_flat = self.config_gen.default_flat()
        current = best

        for round_num in range(1, rounds + 1):
            simplified = False
            candidates: list[PopulationMember] = [current]

            # Generate candidates by resetting each parameter to its default
            for i in range(len(current.flat_values)):
                if current.flat_values[i] != default_flat[i]:
                    new_flat = [*current.flat_values]
                    new_flat[i] = default_flat[i]
                    candidate = self.make_unbenchmarked(new_flat)
                    if candidate.config != current.config:
                        candidates.append(candidate)

            if len(candidates) <= 1:
                self.log(f"Finishing round {round_num}: no more parameters to simplify")
                break

            # Benchmark the candidates
            unbenchmarked = [m for m in candidates if len(m.perfs) == 0]
            if unbenchmarked:
                self.set_generation(self._autotune_metrics.num_generations + 1)
                self._benchmark_members(unbenchmarked)

            # Rebenchmark all candidates (including current) for fair comparison
            self.rebenchmark(candidates)

            # Log performance of each candidate at debug level
            current_perf = current.perf
            for candidate in candidates[1:]:
                delta = candidate.perf - current_perf
                delta_pct = (delta / current_perf * 100) if current_perf != 0 else 0
                status = "ok" if candidate.perf <= current_perf else "worse"
                self.log.debug(
                    f"  reset to {candidate.config}: {candidate.perf:.4f}ms "
                    f"(delta={delta:+.4f}ms, {delta_pct:+.1f}%) [{status}]"
                )

            # Collect all single-attribute resets that maintained performance
            good_candidates = [
                c
                for c in candidates[1:]
                if math.isfinite(c.perf) and c.perf <= current.perf
            ]

            if len(good_candidates) > 1:
                # Try combining all good single-attribute resets at once
                combined_flat = [*current.flat_values]
                for c in good_candidates:
                    for i in range(len(combined_flat)):
                        if c.flat_values[i] != current.flat_values[i]:
                            combined_flat[i] = c.flat_values[i]
                combined = self.make_unbenchmarked(combined_flat)
                if combined.config != current.config:
                    self._benchmark_members([combined])
                    self.rebenchmark([current, combined])
                    if math.isfinite(combined.perf) and combined.perf <= current.perf:
                        current = combined
                        simplified = True

            if not simplified and good_candidates:
                current = good_candidates[0]
                simplified = True

            if simplified:
                self.log(
                    f"Finishing round {round_num}: simplified to {current.config}, perf={current.perf:.4f}ms"
                )
            else:
                self.log(
                    f"Finishing round {round_num}: no simplification maintained performance, stopping early"
                )
                break

        # Minimize the final config by removing values that match defaults
        minimal_config = current.config.minimize(self.config_spec)
        current = PopulationMember(
            flat_values=current.flat_values,
            config=minimal_config,
            perfs=current.perfs,
            _result=current._result,
        )
        self.log(f"Finishing phase complete: final config={current.config}")
        return current


def population_statistics(population: list[PopulationMember]) -> str:
    """
    Create a summary of the population performance.

    Args:
        population: The population of configurations.

    Returns:
        A string summarizing the performance of the population.
    """
    population = sorted(population, key=performance)
    status_counts: collections.Counter[str] = collections.Counter()
    working: list[PopulationMember] = []
    for member in population:
        status = member.status
        if math.isfinite(member.perf):
            working.append(member)
            if status not in {"ok", "error", "timeout"}:
                status = "ok"
        else:
            if status not in {"error", "timeout"}:
                status = "error"
        if status == "timeout":
            status_counts["timeout"] += 1
        elif status == "error":
            status_counts["error"] += 1
        else:
            status_counts["ok"] += 1
    if len(working) == 0:
        raise exc.NoConfigFound
    parts: list[str] = []
    for label in ("error", "timeout", "ok"):
        count = status_counts.get(label, 0)
        if count:
            parts.append(f"{label}={count}")

    parts.extend(
        (
            f"min={working[0].perf:.4f}",
            f"mid={working[len(working) // 2].perf:.4f}",
            f"max={working[-1].perf:.4f}",
            f"best={pprint.pformat(dict(population[0].config), width=100, compact=True)}",
        )
    )
    return "\n" + "\n".join(parts)
