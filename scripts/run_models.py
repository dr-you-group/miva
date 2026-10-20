"""The models stage of one study, one arm per job, as many jobs at once as there are GPU slots.

    python run_models.py --study <study dir> --out <O> --runs-root <R> \\
        --pre <preprocess_index> --split <cohort_split> --ci <cohort_index>

run_study.py calls this for its models step; it can also be run on its own.

- Each arm gets its own copy of the study, ``<O>/study-<env>-<model_id>-<label_def>/study.yaml``
  with the arms list cut to that arm, and its own log
  ``<O>/models-<env>-<model_id>-<label_def>.log``. The env is the one the arm's
  card names in ``x-mival.runtime.env``.
- The slots are site.yaml ``gpus`` (one GPU per slot, ``CUDA_VISIBLE_DEVICES``
  set to that number). With no ``gpus``: ``[0]`` when a GPU is visible, else
  one CPU slot. A free slot takes the next arm.
- Extra variables a card's code needs at run time come from the card's
  ``x-mival.runtime.run_env``; no env or model is named here.
- The arm's env directories lead PATH, as activating the env would do, so a
  tool the env ships beside its interpreter (ninja for a CUDA extension) is found.
- An arm whose card says ``runtime.requires_gpu`` is skipped, with the reason,
  when no GPU is visible.
- A card's platform override (``runtime.windows`` on Windows, or wherever
  ``MIVAL_RUNTIME_PLATFORM`` names that platform) is applied when the card is
  loaded, so ``run_env`` and the rest of ``runtime`` are the overridden ones;
  the arm's entry in the printed JSON then carries ``runtime_override``.
- One failing arm does not stop the others. When all are done the result is
  printed as JSON: ``runs`` (arms that finished, with their run_dir),
  ``failed`` and ``skipped``, each in the study's arm order.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
# This file lives in scripts/site/ in mi-va and in scripts/ in the shared copy.
REPO = next(p for p in HERE.parents if (p / "src" / "mival").is_dir())
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mival.modelcard import load_registry  # noqa: E402
from mival.pipeline.envs import env_python  # noqa: E402
from mival.pipeline.study import load_study  # noqa: E402

#: Every stage runs as ``python -m <this> run <stage> ...``. A ``-c`` string
#: would break Windows multiprocessing, which re-imports the main module.
CLI_MODULE = "mival.pipeline.cli"
_RUN_DIR = re.compile(r'"run_dir":\s*("(?:[^"\\]|\\.)*")')


def stage_command(python, stage: str, study, runs_root, inputs: Mapping[str, Any]) -> List[str]:
    command = [str(python), "-m", CLI_MODULE, "run", stage, "--study", str(study),
               "--runs-root", str(runs_root)]
    for name, path in inputs.items():
        command += ["--input", f"{name}={path}"]
    return command


def child_env(extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """This process's environment with ``src`` first on PYTHONPATH and UTF-8 stdio."""
    env = dict(os.environ)
    parts = [str(REPO / "src")] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update({str(k): str(v) for k, v in (extra or {}).items()})
    return env


def read_log(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def last_run_dir(text: str) -> Optional[str]:
    """The run_dir of the stage summary the CLI printed last, if any."""
    found = _RUN_DIR.findall(text)
    return json.loads(found[-1]) if found else None


def gpu_count(python) -> int:
    """How many GPUs torch sees in the env of ``python``; 0 when it cannot tell."""
    try:
        result = subprocess.run(
            [str(python), "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True, text=True, timeout=300, env=child_env(),
        )
        return int(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else 0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def slots_for(gpus: Optional[Sequence[Any]], n_gpu: int,
              visible: Optional[str] = None) -> List[Optional[str]]:
    """One entry per concurrent job: the ``CUDA_VISIBLE_DEVICES`` of its GPU, or ``None`` for CPU.

    ``gpus`` numbers the GPUs this process sees, as torch (and check_bundle)
    count them. Each job is given ``CUDA_VISIBLE_DEVICES`` of its own, which
    replaces any mask the site set for the whole run, so with a parent mask
    (``CUDA_VISIBLE_DEVICES=4,5``) GPU 0 is the mask's first entry, 4, not
    the machine's GPU 0. ``visible`` defaults to this process's mask.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
    mask = [d.strip() for d in visible.split(",") if d.strip()] if visible and visible.strip() else None

    def device(index: int) -> str:
        if mask is None:
            return str(index)
        if not 0 <= index < len(mask):
            raise ValueError(f"site.yaml gpus lists {index}, but CUDA_VISIBLE_DEVICES={visible} "
                             f"leaves {len(mask)} GPU(s) (0..{len(mask) - 1})")
        return mask[index]

    if gpus:
        bad = [g for g in gpus if isinstance(g, bool) or not isinstance(g, int)]
        if bad:
            raise ValueError(f"site.yaml gpus must list GPU numbers such as [0, 1]; got {bad}")
        return [device(g) for g in gpus]
    return [device(0)] if n_gpu > 0 else [None]


@dataclass
class Arm:
    name: str
    arm: Dict[str, Any]
    env_dir: Path
    runtime: Dict[str, Any]
    study_dir: Path
    log: Path
    #: The name of the platform override the card was loaded with, if any.
    runtime_override: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        described = {"arm": self.name, "model_id": self.arm["model_id"],
                     "training_mode": self.arm.get("training_mode"),
                     "label_def": self.arm.get("label_def"), "log": str(self.log)}
        if self.runtime_override:
            described["runtime_override"] = self.runtime_override
        return described


@dataclass
class _Job:
    arm: Arm
    slot: Optional[str]
    process: Any
    handle: Any


@dataclass
class Result:
    runs: List[Dict[str, Any]] = field(default_factory=list)
    failed: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"runs": self.runs, "failed": self.failed, "skipped": self.skipped}


def _study_file(path: Path) -> Path:
    path = Path(path)
    return path / "study.yaml" if path.is_dir() else path


def _raw_study(path: Path) -> Dict[str, Any]:
    """The study as written, with relative references made absolute so a copy elsewhere still reads them."""
    import yaml

    target = _study_file(path)
    body = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    reference = body.get("site_config")
    if isinstance(reference, str) and "${" not in reference and not Path(reference).is_absolute():
        body["site_config"] = str((target.parent / reference).resolve())
    for stage, spec in list((body.get("stages") or {}).items()):
        if isinstance(spec, str) and not Path(spec).is_absolute():
            body["stages"][stage] = str((target.parent / spec).resolve())
    return body


def _registry(path: str) -> Path:
    # A relative registry is read from the repository root, where every stage runs.
    registry = Path(path)
    return registry if registry.is_absolute() else REPO / registry


def plan_arms(study_path: Path, out: Path) -> List[Arm]:
    """One study copy per arm under ``out``, in the study's arm order."""
    import yaml

    study = load_study(study_path)
    models = study.stage_spec("models")
    cards = load_registry(_registry(models["registry"]))
    raw = _raw_study(study_path)
    arms: List[Arm] = []
    used = set()
    for arm in raw["stages"]["models"]["arms"]:
        card = cards.get(arm["model_id"])
        if card is None:
            raise ValueError(f"arm {arm['model_id']}: no card in {models['registry']}")
        env_dir = Path(card.runtime["env"])
        name = f"{env_dir.name}-{arm['model_id']}-{arm.get('label_def')}"
        base, n = name, 2
        while name in used:  # the same model and label twice, e.g. two training modes
            name, n = f"{base}-{n}", n + 1
        used.add(name)
        study_dir = Path(out) / f"study-{name}"
        study_dir.mkdir(parents=True, exist_ok=True)
        copy = json.loads(json.dumps(raw))
        copy["stages"]["models"]["arms"] = [arm]
        (study_dir / "study.yaml").write_text(
            yaml.safe_dump(copy, sort_keys=False, allow_unicode=True), encoding="utf-8")
        arms.append(Arm(name=name, arm=dict(arm), env_dir=env_dir, runtime=dict(card.runtime),
                        study_dir=study_dir, log=Path(out) / f"models-{name}.log",
                        runtime_override=(card.runtime_override or {}).get("name")))
    return arms


def env_path(python: Path, current: str) -> str:
    """``current`` PATH behind the directories an activated env puts first.

    POSIX: the interpreter's ``bin``. A Windows conda env keeps ``python.exe``
    at its root and its tools in ``Library\\bin`` and ``Scripts``.
    """
    root = Path(python).parent
    dirs = [root] + [d for d in (root / "Library" / "bin", root / "Scripts") if d.is_dir()]
    return os.pathsep.join([str(d) for d in dirs] + [p for p in current.split(os.pathsep) if p])


def _start(arm: Arm, python: Path, slot: Optional[str], runs_root: Path,
           inputs: Mapping[str, Any]) -> _Job:
    extra = dict(arm.runtime.get("run_env") or {})
    if slot is not None:
        extra["CUDA_VISIBLE_DEVICES"] = slot
    extra["PATH"] = env_path(python, os.environ.get("PATH", ""))
    handle = open(arm.log, "wb")
    try:
        process = subprocess.Popen(
            stage_command(python, "models", arm.study_dir, runs_root, inputs),
            cwd=str(REPO), env=child_env(extra), stdout=handle, stderr=subprocess.STDOUT,
        )
    except BaseException:
        handle.close()
        raise
    return _Job(arm=arm, slot=slot, process=process, handle=handle)


def run_arms(arms: Sequence[Arm], slots: Sequence[Optional[str]], runs_root: Path,
             inputs: Mapping[str, Any], gpu_available: bool, poll_s: float = 0.1) -> Result:
    """Run every arm, at most ``len(slots)`` at once; the result lists arms in ``arms`` order."""
    order = {arm.name: i for i, arm in enumerate(arms)}
    result = Result()
    queue: List[Tuple[Arm, Path]] = []
    for arm in arms:
        if arm.runtime.get("requires_gpu") and not gpu_available:
            result.skipped.append(dict(arm.describe(), reason="its card requires a GPU and none is visible"))
            continue
        try:
            queue.append((arm, env_python(arm.env_dir)))
        except FileNotFoundError as exc:
            result.failed.append(dict(arm.describe(), reason=str(exc)))

    free = list(slots)
    running: List[_Job] = []
    while queue or running:
        while queue and free:
            arm, python = queue.pop(0)
            slot = free.pop(0)
            try:
                running.append(_start(arm, python, slot, runs_root, inputs))
            except OSError as exc:  # e.g. an env interpreter that cannot be executed
                free.append(slot)
                result.failed.append(dict(arm.describe(), slot=slot, reason=f"could not start: {exc}"))
        time.sleep(poll_s)
        for job in [j for j in running if j.process.poll() is not None]:
            running.remove(job)
            job.handle.close()
            free.append(job.slot)
            entry = dict(job.arm.describe(), slot=job.slot)
            run_dir = last_run_dir(read_log(job.arm.log))
            if job.process.returncode == 0 and run_dir:
                result.runs.append(dict(entry, run_dir=run_dir))
            else:
                result.failed.append(dict(entry, reason=f"exit code {job.process.returncode}"))
    for listing in (result.runs, result.failed, result.skipped):
        listing.sort(key=lambda item: order[item["arm"]])
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--pre", required=True, help="preprocess_index artifact")
    parser.add_argument("--split", required=True, help="cohort_split artifact")
    parser.add_argument("--ci", required=True, help="cohort_index artifact")
    args = parser.parse_args(argv)
    os.environ.setdefault("MIVAL_HOME", str(REPO.parent))
    study = load_study(args.study)
    n_gpu = gpu_count(sys.executable)
    arms = plan_arms(Path(args.study), Path(args.out))
    inputs = {"preprocess_index": args.pre, "cohort_split": args.split, "cohort_index": args.ci}
    result = run_arms(arms, slots_for(study.execution.get("gpus"), n_gpu), Path(args.runs_root),
                      inputs, gpu_available=n_gpu > 0)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
