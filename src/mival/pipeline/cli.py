"""Stage CLI (spec §3.1).

Each stage is an independent command. Input is the previous stage's artifact
plus a spec file; output is an artifact plus a run manifest.

Input artifacts are named explicitly with ``--input name=path``. Resolving
them automatically from the previous stage would mean guessing which
``config_hash`` of that stage was meant, and guessing wrong produces a result
that is silently built on the wrong upstream run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .layout import DEFAULT_RUNS_ROOT
from .stage import execute, get_stage, prepare, stage_names
from .study import load_study


def _parse_inputs(pairs: Sequence[str]) -> Dict[str, Path]:
    inputs: Dict[str, Path] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise SystemExit(f"--input must be given as name=path, got {pair!r}")
        path = Path(value)
        if not (path.is_file() or path.is_dir()):
            raise SystemExit(f"--input {name}: {path} is not a file or directory")
        inputs[name] = path
    return inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mival", description="MI-VAL pipeline stages")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stages", help="list the registered stages in pipeline order")

    run = sub.add_parser("run", help="run one stage")
    run.add_argument("stage", choices=stage_names())
    run.add_argument("--study", required=True, help="path to study.yaml or its directory")
    run.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="an input artifact, repeatable",
    )
    run.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    run.add_argument(
        "--force",
        action="store_true",
        help="re-run even when this config_hash already has a manifest",
    )
    run.add_argument("--seed", type=int, default=None, help="overrides the study seed")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "stages":
        for name in stage_names():
            print(name)
        return 0

    study = load_study(args.study)
    stage = get_stage(args.stage)

    inputs = _parse_inputs(args.input)
    missing = [name for name in stage.required_inputs() if name not in inputs]
    if missing:
        raise SystemExit(f"stage {args.stage!r} requires --input for: {', '.join(missing)}")

    ctx = prepare(
        stage=stage,
        study_id=study.study_id,
        site=study.site,
        spec=study.stage_spec(stage.spec_key()),
        inputs=inputs,
        runs_root=args.runs_root,
        seed=args.seed if args.seed is not None else study.seed,
    )
    summary = execute(stage, ctx, force=args.force)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
