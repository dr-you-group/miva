"""CPU worker processes for a stage's independent units (site.yaml ``workers``).

A unit is one call of a top-level function on its own arguments — for the
evaluation stage, one slice's metrics and curves, or one arm pair's paired
bootstrap. :func:`ordered_map` runs the units in worker processes and hands
the results back in submission order, so the caller assembles exactly the
rows it would have built calling the units one after another.

Results do not depend on the worker count. That is a property of the units,
not of this module: every unit that resamples builds its own
``numpy.random.default_rng(seed)`` (``mival.metrics.bootstrap``), so no
random stream is shared between units or consumed in loop order. The worker
count is therefore an execution key (``mival.pipeline.study.EXECUTION_KEYS``)
and stays out of every config_hash.

Workers are always started with ``spawn``, the only start method Windows has,
so a site's Linux and Windows servers run the same code path, and no worker
inherits a forked copy of the parent's threads (pyarrow, BLAS).

Every unit runs under a one-thread BLAS limit (``_run_unit``), with
``workers: 1`` exactly as in a worker process. The parallelism is across
units, and ``workers`` processes each spawning a full BLAS pool would
oversubscribe the machine; and some BLAS builds (MKL) do not promise the same
last bits at different thread counts, so the serial and the parallel path
must not differ in it. The limit goes through threadpoolctl (a scikit-learn
dependency) and binds only libraries it can control: OpenBLAS, MKL, BLIS and
OpenMP, not macOS Accelerate. Code outside the units (loading, threshold and
update fits) is not limited, in either path.

Memory is bounded by submitting at most ``WINDOW_PER_WORKER * workers`` units
ahead of the one whose result is awaited: the units' arguments are built
lazily by the caller's generator, so only that many are held at once.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

#: Environment override of site.yaml ``workers`` (tests, one-off runs).
WORKERS_ENV = "MIVAL_WORKERS"
#: Without a setting: one worker per CPU, at most this many. Beyond about
#: eight the evaluation stage is bound by its largest units, not by cores.
DEFAULT_MAX_WORKERS = 8
#: Units submitted ahead of the oldest pending one, per worker.
WINDOW_PER_WORKER = 4
START_METHOD = "spawn"

def default_workers() -> int:
    return max(1, min(os.cpu_count() or 1, DEFAULT_MAX_WORKERS))


def resolve_workers(configured: Any = None) -> int:
    """The worker count: ``MIVAL_WORKERS``, else site.yaml ``workers``, else the default.

    ``1`` runs every unit in this process (no pool is started). A value that
    is not a positive integer is an error rather than a silent fallback: a
    mistyped setting would otherwise run on one core for hours unnoticed.
    """
    source, value = WORKERS_ENV, os.environ.get(WORKERS_ENV)
    if value is None or str(value).strip() == "":
        source, value = "site.yaml workers", configured
    if value is None:
        return default_workers()
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    try:
        number = int(str(value).strip())
    except ValueError:
        raise ValueError(f"{source} must be a positive integer, got {value!r}") from None
    if number < 1:
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    return number


#: BLAS threads a unit runs with, in every path (see the module docstring).
UNIT_BLAS_THREADS = 1


def _blas_limit():
    """A context manager holding BLAS at ``UNIT_BLAS_THREADS``, or a no-op without threadpoolctl."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        from contextlib import nullcontext

        return nullcontext()
    return threadpool_limits(limits=UNIT_BLAS_THREADS, user_api="blas")


def blas_threads() -> Optional[int]:
    """The largest thread count of the BLAS libraries threadpoolctl sees here, or None."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:  # pragma: no cover
        return None
    counts = [info["num_threads"] for info in threadpool_info() if info.get("user_api") == "blas"]
    return max(counts) if counts else None


def _run_unit(function: Callable[..., Any], args: Tuple[Any, ...]) -> Any:
    """``function(*args)`` under the BLAS limit — what both paths call for every unit."""
    with _blas_limit():
        return function(*args)


def _with_context(exc: BaseException, label: str) -> BaseException:
    """``exc`` with ``label`` in front of its message; type and traceback (remote too) are kept."""
    if label:
        args = exc.args
        if args and isinstance(args[0], str):
            exc.args = (f"{label}: {args[0]}",) + tuple(args[1:])
        else:
            exc.args = (label,) + tuple(args)
    return exc


def ordered_map(
    function: Callable[..., Any],
    items: Iterable[Tuple[Any, Tuple[Any, ...]]],
    workers: int,
    start_method: Optional[str] = None,
    describe: Optional[Callable[[Tuple[Any, ...]], str]] = None,
) -> Iterator[Tuple[Any, Any]]:
    """``(tag, function(*args))`` for each ``(tag, args)`` of ``items``, in input order.

    ``function`` must be a module-level function and ``args`` picklable
    (spawn pickles both). ``tag`` stays in this process. ``items`` is
    consumed lazily — with ``workers == 1`` one item per result, so a caller
    whose generator has side effects sees them interleaved with the results
    exactly as in a plain loop. A unit's exception is raised here, its
    message prefixed with ``describe(args)`` (which run_key, subgroup...) and
    its worker traceback kept as its cause; the units not yet started are
    cancelled.
    """
    def label(args) -> str:
        return describe(args) if describe is not None else ""

    if workers <= 1:
        for tag, args in items:
            try:
                result = _run_unit(function, args)
            except Exception as exc:
                raise _with_context(exc, label(args))
            yield tag, result
        return

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context(start_method or START_METHOD)
    window = max(1, WINDOW_PER_WORKER * int(workers))
    pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=context)
    pending: deque = deque()

    def collect():
        tag_done, unit_label, future = pending.popleft()
        try:
            return tag_done, future.result()
        except Exception as exc:
            raise _with_context(exc, unit_label)

    try:
        for tag, args in items:
            pending.append((tag, label(args), pool.submit(_run_unit, function, args)))
            if len(pending) >= window:
                yield collect()
        while pending:
            yield collect()
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
