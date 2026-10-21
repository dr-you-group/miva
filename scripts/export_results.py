"""The results zip a site sends back: results-<site>-<YYMMDD>.zip in $MIVAL_HOME.

    usage: export_results.sh [--allow-small-stage-totals] [study ...]     (Linux)
           export_results.cmd [--allow-small-stage-totals] [study ...]    (Windows)
           e.g. export_results.sh lvef ecg-age

With no study named, every study whose runs.json exists is bundled. Flags and
study names may come in any order.

    --allow-small-stage-totals
                export although a stage excluded 1 to min_cell - 1 records in
                total (that total can be recovered from the counts before and
                after the stage). Only after the site has decided that count
                may leave; the bundle's provenance/export_manifest.json records it.

    MIVAL_HOME  bundle root holding miva/, envs/, models/, code/, config/
                (default: the directory above this repository)
    MIVAL_RUNS  where run artifacts are (default: $MIVAL_HOME/runs)
    STUDY_ROOT  study directories (default: studies/site in mi-va, studies in the shared copy)

The bundle holds aggregate values only; small cells are masked and every
file is checked before the zip is written (see src/mival/export.py).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
# This file lives in scripts/site/ in mi-va and in scripts/ in the shared copy.
REPO = next(p for p in HERE.parents if (p / "src" / "mival").is_dir())
sys.path.insert(0, str(REPO / "src"))


def study_root() -> Path:
    if os.environ.get("STUDY_ROOT"):
        return Path(os.environ["STUDY_ROOT"]).resolve()
    site = REPO / "studies" / "site"
    return site if site.is_dir() else REPO / "studies"


def _tolerant_stdio() -> None:
    """Never fail on printing a line the console encoding (e.g. cp949) cannot show."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _tolerant_stdio()
    given = list(sys.argv[1:] if argv is None else argv)
    # Flags apart from names: ``--studies`` takes every word after it, so a
    # flag written after the names would otherwise be read as a study.
    flags = [a for a in given if a.startswith("-")]
    studies = [a for a in given if not a.startswith("-")]
    home = Path(os.environ.setdefault("MIVAL_HOME", str(REPO.parent))).resolve()
    runs = Path(os.environ.get("MIVAL_RUNS") or home / "runs").resolve()
    args = flags + [
        "--study-root", str(study_root()), "--runs-root", str(runs), "--home", str(home)] + (
        ["--studies"] + studies if studies else [])
    os.chdir(home)
    from mival import export

    try:
        export.main(args)
    except SystemExit as exc:
        if exc.code not in (0, None):
            print("export FAILED: no new zip was written (a zip already at that name is from an earlier run)")
        raise
    except Exception:
        traceback.print_exc()
        print("export FAILED: no new zip was written (a zip already at that name is from an earlier run)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
