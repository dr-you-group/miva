"""Parquet I/O, imported lazily.

Plan 1 established that ``mival`` core must import in both the Python 3.10
torch environment and the Python 3.9 Keras 2.7 environment. pandas and pyarrow
are heavier and more version-fragile than either, so every import of them
happens inside a function. ``import mival.pipeline`` must never require them:
a stage that writes no tables must still be able to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional, List, Mapping, Sequence, Union

_MISSING = (
    "reading and writing parquet requires pandas and pyarrow. Install them into "
    "this environment (see infra/aws/setup-model-envs.sh)."
)


def _pandas():
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(_MISSING) from exc
    return pandas


def write_table(
    rows: Iterable[Mapping[str, Any]],
    path: Union[str, Path],
    columns: Sequence[str],
) -> Path:
    """Write rows to parquet with an explicit, fixed column order.

    ``columns`` is required rather than inferred. An inferred schema silently
    changes when a stage happens to emit zero rows, or emits a row where an
    optional field is absent, and the resulting parquet files no longer
    concatenate.
    """
    pandas = _pandas()
    materialized = [dict(row) for row in rows]
    for index, row in enumerate(materialized):
        unknown = set(row) - set(columns)
        if unknown:
            raise ValueError(f"row {index} has columns not in the schema: {sorted(unknown)}")
    frame = pandas.DataFrame(materialized, columns=list(columns))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return target


def join_key(series):
    """Ids as text for joining tables that store them differently.

    Prediction files carry ids as strings, a table built from the CDM keeps
    its integer ids, and a column of integers with a gap comes back from
    parquet as float. An integer-valued number is written without a decimal
    part (``101``, ``101.0`` and ``"101"`` all become ``"101"``); every other
    value is ``str(value)`` unchanged, so ``"007"`` stays ``"007"``. A missing
    value is None.
    """
    import numbers

    pandas = _pandas()

    def text(value):
        if value is None or (not isinstance(value, str) and pandas.isna(value)):
            return None
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return str(int(value))
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and float(value).is_integer():
            return str(int(value))
        return str(value)

    return series.astype(object).map(text)


def read_table(path: Union[str, Path]):
    return _pandas().read_parquet(path)


def read_dataset(
    root: Union[str, Path],
    columns: Sequence[str],
    include: Optional[Callable[[Path], bool]] = None,
):
    """Concatenate every parquet part under ``root`` into one frame.

    Returns an empty frame with the given columns when no part files exist, so
    that callers can group and filter without a special case for "nothing was
    excluded yet". ``include`` keeps only the part files it accepts.
    """
    pandas = _pandas()
    parts: List[Path] = sorted(Path(root).rglob("*.parquet")) if Path(root).exists() else []
    if include is not None:
        parts = [part for part in parts if include(part)]
    if not parts:
        return pandas.DataFrame(columns=list(columns))
    return pandas.concat([pandas.read_parquet(part) for part in parts], ignore_index=True)
