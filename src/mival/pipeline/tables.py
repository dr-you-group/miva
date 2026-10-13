"""Parquet I/O, imported lazily.

Plan 1 established that ``mival`` core must import in both the Python 3.10
torch environment and the Python 3.9 Keras 2.7 environment. pandas and pyarrow
are heavier and more version-fragile than either, so every import of them
happens inside a function. ``import mival.pipeline`` must never require them:
a stage that writes no tables must still be able to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Union

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


def read_table(path: Union[str, Path]):
    return _pandas().read_parquet(path)


def read_dataset(root: Union[str, Path], columns: Sequence[str]):
    """Concatenate every parquet part under ``root`` into one frame.

    Returns an empty frame with the given columns when no part files exist, so
    that callers can group and filter without a special case for "nothing was
    excluded yet".
    """
    pandas = _pandas()
    parts: List[Path] = sorted(Path(root).rglob("*.parquet")) if Path(root).exists() else []
    if not parts:
        return pandas.DataFrame(columns=list(columns))
    return pandas.concat([pandas.read_parquet(part) for part in parts], ignore_index=True)
