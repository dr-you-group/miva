"""One study end to end: retrieve -> profile -> preprocess -> describe -> models -> evaluate.

    usage: run_study.sh <study>      (Linux)     e.g. run_study.sh lvef
           run_study.cmd <study>     (Windows)
           <mival-torch python> run_study.py <study>

The .sh and .cmd files only find the mival-torch interpreter and call this
file, so every site runs the same steps whatever its OS.

    MIVAL_HOME  bundle root holding miva/, envs/, models/, code/, config/
                (config/site.yaml and config/micdm.env are the site's own settings)
                (default: the directory above this repository)
    MIVAL_RUNS  where run artifacts go (default: $MIVAL_HOME/runs)
    STUDY_ROOT  study directories (default: studies/site in mi-va, studies in the shared copy)

Every stage is ``python -m mival.pipeline.cli run <stage>``. A finished stage
is not re-run: its config_hash already has a manifest, so a second invocation
after a failure resumes from the failed stage.

Written under ``$MIVAL_RUNS/<study_id>/_run/``:

- ``runs.json``: stage -> run_dirs of this invocation only (a skipped stage
  lists the run_dir it reused). Started empty on every invocation.
- ``train_log_all.jsonl``: every models run's ``train_log.jsonl``, in arm order.
- ``predictions_all/``: every models run's prediction files.
- ``run.log`` and one log per stage (one per arm for models).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_models  # noqa: E402  (also puts src/ on sys.path)
from run_models import REPO, child_env, last_run_dir, read_log, stage_command  # noqa: E402

from mival.pipeline.envs import env_python  # noqa: E402
from mival.pipeline.study import load_study  # noqa: E402

#: The order the steps run in; the same as the runnable part of STAGE_ORDER.
_SCRIPTS_FROM_REPO = HERE.relative_to(REPO)

STAGES = ("retrieve", "profile", "preprocess", "describe", "models", "evaluate")
CONTROLLER_ENV = "mival-torch"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_study_root() -> Path:
    if os.environ.get("STUDY_ROOT"):
        return Path(os.environ["STUDY_ROOT"]).resolve()
    site = REPO / "studies" / "site"
    return site if site.is_dir() else REPO / "studies"


class RunsRecord:
    """``runs.json`` of this invocation: rewritten from empty when created, then after every stage."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.runs: Dict[str, List[str]] = {}
        self._write()

    def add(self, stage: str, run_dirs: Sequence[str]) -> None:
        self.runs.setdefault(stage, []).extend(str(d) for d in run_dirs)
        self._write()

    def _write(self) -> None:
        self.path.write_bytes((json.dumps(self.runs, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def artifact(run_dir: str, name: str) -> Path:
    return Path(run_dir) / "artifacts" / name


class StudyRun:
    def __init__(self, name: str, study_dir: Path, home: Path, runs_root: Path, controller: Path):
        self.name = name
        self.study_dir = study_dir
        self.home = home
        self.runs_root = runs_root
        self.controller = controller
        self.study = load_study(study_dir)
        self.out = runs_root / self.study.study_id / "_run"
        self.out.mkdir(parents=True, exist_ok=True)
        self.runs = RunsRecord(self.out / "runs.json")
        self.paths: Dict[str, Path] = {}
        self.warnings: List[str] = []
        self.gpus = 0

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with open(self.out / "run.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def stage(self, name: str, inputs: Mapping[str, Any]) -> Optional[str]:
        """Run one stage with the controller env; its run_dir, or ``None`` after printing the log tail."""
        log_path = self.out / f"{name}.log"
        self.log(name)
        with open(log_path, "wb") as handle:
            code = _call(stage_command(self.controller, name, self.study_dir, self.runs_root, inputs),
                         handle)
        run_dir = last_run_dir(read_log(log_path))
        if code != 0 or run_dir is None:
            self.log(f"{name} FAILED (log: {log_path})")
            print("\n".join(read_log(log_path).splitlines()[-30:]))
            return None
        self.runs.add(name, [run_dir])
        return run_dir

    # -- the steps, in STAGES order ---------------------------------------

    def retrieve(self) -> bool:
        run_dir = self.stage("retrieve", {})
        if run_dir is None:
            return False
        self.paths["cohort_index"] = artifact(run_dir, "cohort_index.parquet")
        self.paths["ecg_attributes"] = artifact(run_dir, "ecg_attributes.parquet")
        return True

    def profile(self) -> bool:
        run_dir = self.stage("profile", {"cohort_index": self.paths["cohort_index"]})
        if run_dir is None:
            return False
        self.paths["cohort_split"] = artifact(run_dir, "cohort_split.parquet")
        return True

    def preprocess(self) -> bool:
        run_dir = self.stage("preprocess", {"cohort_index": self.paths["cohort_index"]})
        if run_dir is None:
            return False
        self.paths["preprocess_index"] = artifact(run_dir, "preprocess_index.parquet")
        self.paths["source_index"] = artifact(run_dir, "source_index.parquet")
        return True

    def describe(self) -> bool:
        run_dir = self.stage("describe", {
            "cohort_index": self.paths["cohort_index"],
            "cohort_split": self.paths["cohort_split"],
            "ecg_attributes": self.paths["ecg_attributes"],
            "source_index": self.paths["source_index"],
            "runs": self.runs.path,
        })
        if run_dir is None:
            return False
        self.paths["attributes"] = artifact(run_dir, "attributes.parquet")
        return True

    def models(self) -> bool:
        arms = run_models.plan_arms(self.study_dir, self.out)
        try:
            slots = run_models.slots_for(self.study.execution.get("gpus"), self.gpus)
        except ValueError as exc:
            self.log(f"models FAILED: {exc} (see config/site.yaml; check_bundle reports it too)")
            return False
        self.log(f"models: {len(arms)} arm(s), {len(slots)} at a time "
                 f"(slots {', '.join(s if s is not None else 'cpu' for s in slots)})")
        result = run_models.run_arms(arms, slots, self.runs_root, {
            "preprocess_index": self.paths["preprocess_index"],
            "cohort_split": self.paths["cohort_split"],
            "cohort_index": self.paths["cohort_index"],
        }, gpu_available=self.gpus > 0)
        for item in result.skipped:
            self.log(f"models [{item['arm']}] SKIPPED: {item['reason']}")
        for item in result.failed:
            self.log(f"models [{item['arm']}] FAILED: {item['reason']} (log: {item['log']})")
            print("\n".join(read_log(Path(item["log"])).splitlines()[-30:]))
        if result.failed or result.skipped:
            self.warnings.append(f"one or more model arms failed or were skipped; see {self.out}/models-*.log")
        run_dirs = [item["run_dir"] for item in result.runs]
        if not run_dirs:
            # evaluate would only fail on an empty predictions directory.
            self.log(f"models FAILED: no model arm finished; see {self.out}/models-*.log")
            return False
        self.runs.add("models", run_dirs)
        self.paths["predictions"] = collect_predictions(run_dirs, self.out / "predictions_all")
        self.paths["train_log"] = merge_train_logs(run_dirs, self.out / "train_log_all.jsonl")
        return True

    def evaluate(self) -> bool:
        run_dir = self.stage("evaluate", {
            "predictions": self.paths["predictions"],
            "cohort_split": self.paths["cohort_split"],
            "attributes": self.paths["attributes"],
            "train_log": self.paths["train_log"],
            "runs": self.runs.path,
        })
        if run_dir is None:
            return False
        self.log(f"done: {run_dir}")
        return True

    def run(self) -> int:
        self.gpus = run_models.gpu_count(self.controller)
        self.log(f"study {self.name}, MIVAL_HOME {self.home}, runs {self.runs_root}, GPU {self.gpus}")
        steps: Dict[str, Callable[[], bool]] = {name: getattr(self, name) for name in STAGES}
        for name in STAGES:
            if not steps[name]():
                self._warn()
                return 1
        self._warn()
        print(f"next: {export_hint()} {self.name}")
        return 0

    def _warn(self) -> None:
        for warning in self.warnings:
            self.log(f"WARNING: {warning}")


def _call(command: List[str], handle) -> int:
    return subprocess.call(command, cwd=str(run_models.REPO), env=child_env(), stdout=handle,
                           stderr=subprocess.STDOUT)


def collect_predictions(run_dirs: Sequence[str], target: Path) -> Path:
    """Every models run's prediction files in one fresh directory."""
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for run_dir in run_dirs:
        for path in sorted(artifact(run_dir, "predictions").glob("*.parquet")):
            shutil.copy2(path, target / path.name)
    return target


def merge_train_logs(run_dirs: Sequence[str], target: Path) -> Path:
    """Every models run's ``train_log.jsonl`` in one file, in run order."""
    lines: List[str] = []
    for run_dir in run_dirs:
        path = artifact(run_dir, "train_log.jsonl")
        if path.is_file():
            lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    target.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))
    return target


def export_hint() -> str:
    """Where the export script is, from the repository root, written the way this OS writes paths."""
    if os.name == "nt":
        return str(PureWindowsPath(_SCRIPTS_FROM_REPO, "export_results.cmd"))
    return (_SCRIPTS_FROM_REPO / "export_results.sh").as_posix()


def _tolerant_stdio() -> None:
    """Never fail on printing a child's log line the console encoding (e.g. cp949) cannot show."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _tolerant_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: run_study.py <study>      e.g. run_study.py lvef")
        return 2
    name = argv[0]
    home = Path(os.environ.setdefault("MIVAL_HOME", str(REPO.parent)))
    runs_root = Path(os.environ.get("MIVAL_RUNS") or home / "runs").resolve()
    study_dir = default_study_root() / name
    if not (study_dir / "study.yaml").is_file():
        print(f"no {study_dir / 'study.yaml'}")
        return 1
    try:
        controller = env_python(home / "envs" / CONTROLLER_ENV)
    except FileNotFoundError as exc:
        print(f"{exc} (see README: env 설치)")
        return 1
    if not (home / "config" / "site.yaml").is_file():
        print(f"no {home / 'config' / 'site.yaml'} (see README 4.2: 기관 설정)")
        return 1
    return StudyRun(name, study_dir, home, runs_root, controller).run()


if __name__ == "__main__":
    sys.exit(main())
