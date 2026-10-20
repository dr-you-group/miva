"""Check the site config and every card a study uses (weights, code, env, GPU, CUDA).

    MIVAL_HOME=/path/to/bundle python check_bundle.py [study ...]

With no study named, every study under the study root is checked (studies/site
in mi-va, studies in the shared copy; STUDY_ROOT overrides). The study is
loaded the way the pipeline loads it, so a missing or conflicting
config/site.yaml shows up here. Weights are compared against the sha256 in
the card, so a truncated copy is caught here rather than in the models stage.

Also counted as problems:
- site.yaml still holding the template's ``site`` or ``models.cohort_sources``;
  the contamination flag compares the real dataset name with each model's
  pretraining corpora, so a placeholder makes it meaningless.
- a ``gpus`` number that torch in the mival-torch env cannot see.
- a module the mival-torch env cannot import although a stage it runs needs
  it: the DB driver of ``retrieve.dbms``, pydicom for the DICOM loader, and
  the table, metric and figure libraries.
- an env whose torch was built for a newer CUDA than the driver reports
  (``nvidia-smi`` "CUDA Version").
- on Windows, a card that compiles a CUDA extension (``runtime.compile_env``)
  but names no Windows way to run (``runtime.windows``). A card that does is
  loaded with that override applied, and its line says so (not a problem).
- an arm env (a card's ``runtime.env``) that cannot import what the models
  stage imports there (``ARM_MODULES``); and a card that ``requires_gpu`` on a
  machine where torch sees no GPU, since the run would skip that arm.
- on Windows without long paths (``LongPathsEnabled`` under
  ``HKLM/SYSTEM/CurrentControlSet/Control/FileSystem``), a run path that
  would pass the 260-character MAX_PATH. The longest file a run writes is a
  prediction file, ``<runs>/<study>/models/<config_hash>/artifacts/predictions/
  <run_key>.parquet``; its length is estimated from ``MIVAL_RUNS`` and the
  study's own site and arms, so the models stage does not
  fail on it after hours of preprocessing.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
# This file lives in scripts/site/ in mi-va and in scripts/ in the shared copy.
REPO = next(p for p in HERE.parents if (p / "src" / "mival").is_dir())
sys.path.insert(0, str(REPO / "src"))

from mival.modelcard import load_registry  # noqa: E402
from mival.pipeline.envs import env_python  # noqa: E402
from mival.pipeline.study import load_study  # noqa: E402
from mival.runtime_platform import PLATFORM_KEYS, STATE_DICT_TRANSFORM, state_dict_transform_spec  # noqa: E402

#: The values config/site.yaml.example ships with.
TEMPLATE_SITE = "my-site"
TEMPLATE_COHORT_SOURCES = ["my-site-ecg"]
CONTROLLER_ENV = "mival-torch"
#: Imported by the stages the controller env runs (retrieve .. evaluate).
CONTROLLER_MODULES = ("numpy", "scipy", "yaml", "pandas", "pyarrow", "sklearn", "matplotlib")
#: Imported by the models stage in each arm env (``python -m mival.pipeline.cli run models``).
ARM_MODULES = ("numpy", "yaml", "pandas", "pyarrow")
DB_DRIVERS = {"postgresql": "psycopg", "sqlserver": "pymssql"}
#: Windows MAX_PATH is 260 characters with the terminating NUL.
WINDOWS_MAX_PATH = 259
LONG_PATHS_KEY = r"SYSTEM\CurrentControlSet\Control\FileSystem"
LOADER_MODULES = {"dicom": "pydicom"}


def study_root() -> Path:
    if os.environ.get("STUDY_ROOT"):
        return Path(os.environ["STUDY_ROOT"])
    site = REPO / "studies" / "site"
    return site if site.is_dir() else REPO / "studies"


def runs_root() -> Path:
    """``MIVAL_RUNS``, else ``$MIVAL_HOME/runs``, as run_study resolves it."""
    home = Path(os.environ.get("MIVAL_HOME") or REPO.parent)
    return Path(os.environ.get("MIVAL_RUNS") or home / "runs").resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not text:
        return None
    found = re.match(r"\s*(\d+)(?:\.(\d+))?", str(text))
    return (int(found.group(1)), int(found.group(2) or 0)) if found else None


def driver_cuda_version() -> Optional[str]:
    """The highest CUDA version the installed driver supports, from ``nvidia-smi``; None without one."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r"CUDA Version:\s*([0-9.]+)", out)
    return found.group(1) if found else None


def _probe(python: Path, code: str) -> Optional[str]:
    try:
        result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    return lines[-1].strip() if result.returncode == 0 and lines else None


def torch_cuda_version(python: Path) -> Optional[str]:
    """``torch.version.cuda`` in the env of ``python``; None for an env without a CUDA torch."""
    found = _probe(python, "import torch; print(torch.version.cuda)")
    return None if found in (None, "None") else found


def torch_device_count(python: Path) -> int:
    found = _probe(python, "import torch; print(torch.cuda.device_count())")
    try:
        return int(found) if found is not None else 0
    except ValueError:
        return 0


def missing_modules(python: Path, modules: Sequence[str]) -> List[str]:
    """The ``modules`` the interpreter ``python`` cannot import; all of them when it cannot be run."""
    code = ("import importlib.util as u; "
            f"print('missing:' + ','.join(m for m in {list(modules)!r} if u.find_spec(m) is None))")
    found = _probe(python, code)
    if found is None or not found.startswith("missing:"):
        return list(modules)
    return [m for m in found[len("missing:"):].split(",") if m]


def controller_modules(retrieve: dict, preprocess: dict) -> List[str]:
    """What the controller env imports for this study: libraries, DB driver, record loader."""
    modules = list(CONTROLLER_MODULES)
    driver = DB_DRIVERS.get(retrieve.get("dbms", "postgresql"))
    if driver:
        modules.append(driver)
    loader = LOADER_MODULES.get(preprocess.get("loader", "npz"))
    if loader:
        modules.append(loader)
    return modules


def check_site(retrieve: dict) -> list:
    found = []
    if not Path(retrieve.get("dsn_env", "")).is_file():
        found.append(f"no DB connection file {retrieve.get('dsn_env')}")
    root = retrieve.get("local_path_root")
    if root is not None and not Path(root).is_dir():
        found.append(f"no DICOM root {root}")
    return found


def check_template_values(study) -> List[str]:
    found = []
    if study.site == TEMPLATE_SITE:
        found.append(f"site is still the template value {TEMPLATE_SITE!r}; set this site's identifier")
    sources = study.stage_spec("models").get("cohort_sources")
    if list(sources or []) == TEMPLATE_COHORT_SOURCES:
        found.append(f"models.cohort_sources is still the template value {TEMPLATE_COHORT_SOURCES}; "
                     "name this cohort's dataset so the pretraining overlap check means something")
    return found


def check_gpus(gpus: Optional[Sequence], device_count: int) -> List[str]:
    gpus = list(gpus or [])
    found = []
    bad = [g for g in gpus if isinstance(g, bool) or not isinstance(g, int) or not 0 <= g < device_count]
    if bad:
        found.append(f"gpus {bad} not visible to torch ({device_count} GPU(s): "
                     f"{list(range(device_count)) if device_count else 'none'})")
    twice = sorted({g for g in gpus if gpus.count(g) > 1}, key=str)
    if twice:
        found.append(f"gpus lists {twice} more than once; each number is one slot, so two arms would share a GPU")
    return found


def check_driver_cuda(env: str, torch_cuda: Optional[str], driver: Optional[str]) -> List[str]:
    """A problem when the env's torch needs a newer CUDA than the driver supports."""
    built, supported = _version(torch_cuda), _version(driver)
    if built is None or supported is None or built <= supported:
        return []
    return [f"env {env}: torch built for CUDA {torch_cuda} but the driver supports up to CUDA {driver}"]


def check_windows_runtime(card, platform: Optional[str] = None) -> List[str]:
    platform = sys.platform if platform is None else platform
    runtime = card.runtime or {}
    if platform != "win32" or not runtime.get("compile_env") or runtime.get("windows"):
        return []
    return [f"{card.model_id}: its code compiles a CUDA extension (runtime.compile_env) and the card "
            "names no Windows runtime (runtime.windows); this model cannot run on Windows yet"]


def check_state_dict_transform(card) -> List[str]:
    """The module each ``runtime.state_dict_transform`` names must be where it says.

    Both the one in effect here and the one every platform block declares
    (``runtime.windows``), so a missing wrapper is found when the bundle is
    checked on Linux, not first at the Windows site.
    """
    runtime = card.runtime or {}
    values = [runtime.get(STATE_DICT_TRANSFORM)]
    values += [(runtime.get(key) or {}).get(STATE_DICT_TRANSFORM) for key in PLATFORM_KEYS.values()
               if isinstance(runtime.get(key), dict)]
    found: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        name, own = state_dict_transform_spec(value, card.model_id)
        root = own or card.raw["x-mival"].get("code_path")
        if not root or (name, root) in seen:
            continue  # without a root it is imported from the env, which check_bundle does not open
        seen.add((name, root))
        top = name.partition(":")[0].partition(".")[0]
        if not ((Path(root) / f"{top}.py").is_file() or (Path(root) / top / "__init__.py").is_file()):
            found.append(f"{card.model_id}: runtime.{STATE_DICT_TRANSFORM} {name}: no {top}.py under {root}")
    return found


def runtime_override_info(card) -> List[str]:
    """An info line (not a problem) naming the platform override the card was loaded with."""
    override = getattr(card, "runtime_override", None)
    if not override:
        return []
    changes = "; ".join(override.get("changes") or ()) or "no changes"
    return [f"runtime.{override['name']} override applies on {override['platform']}: {changes}"]


def prebuilt_extension_info(card, platform: Optional[str] = None) -> List[str]:
    """An info line for a card with ``runtime.prebuilt_torch_extensions``: which built extensions it finds.

    Not a problem either way: without one the card's code compiles it at first
    load, which needs the toolchain its ``compile_env`` names.
    """
    runtime = card.runtime or {}
    if not runtime.get("prebuilt_torch_extensions"):
        return []
    root = (runtime.get("run_env") or {}).get("TORCH_EXTENSIONS_DIR")
    platform = sys.platform if platform is None else platform
    suffix = ".pyd" if platform == "win32" else ".so"
    names = sorted(p.parent.name for p in Path(root).glob(f"*/*{suffix}")
                   if p.stem == p.parent.name) if root else []
    if names:
        return [f"prebuilt torch extension(s) {', '.join(names)} in {root}"]
    if runtime.get("compile_env"):
        return [f"no prebuilt torch extension in {root}; its code compiles one at first load"]
    return []


def longest_run_path(runs_root: Path, study) -> int:
    """Characters in the longest path a run of ``study`` writes: one model's prediction file.

    ``<runs>/<study>/models/<config_hash>/artifacts/predictions/<run_key>.parquet``,
    with the run_key of each arm at its longest: a recipe id of full length,
    the perturbation id preprocess writes on every record, the test split and
    a fold of two characters (``na``; a dev fold number is shorter).
    """
    from mival.pipeline.hashing import CONFIG_HASH_LENGTH
    from mival.pipeline.layout import ARTIFACTS_DIR
    from mival.pipeline.runkey import RunKey
    from mival.stages._predictions import TEST_SPLIT
    from mival.stages.models import PREDICTIONS_DIR
    from mival.stages.preprocess import BASELINE_PERTURBATION_ID, RECIPE_ID_LENGTH

    names = [len(RunKey(site=study.site, model_id=str(arm["model_id"]),
                        training_mode=str(arm["training_mode"]), recipe_id="0" * RECIPE_ID_LENGTH,
                        perturbation_id=BASELINE_PERTURBATION_ID, label_def=str(arm["label_def"]),
                        split=TEST_SPLIT, fold=None).to_string())
             for arm in study.stage_spec("models").get("arms") or ()]
    if not names:
        return 0
    parts = [study.study_id, "models", "0" * CONFIG_HASH_LENGTH, ARTIFACTS_DIR, PREDICTIONS_DIR]
    return len(str(runs_root)) + sum(1 + len(p) for p in parts) + 1 + max(names) + len(".parquet")


def long_paths_enabled() -> Optional[bool]:
    """Windows ``LongPathsEnabled``; None when it cannot be read (not Windows, no key)."""
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, LONG_PATHS_KEY) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
    except OSError:
        return None
    return bool(value)


def check_path_length(runs_root: Path, study, platform: Optional[str] = None,
                      enabled: Optional[bool] = None) -> List[str]:
    """On Windows without long paths, a problem when a run path would pass MAX_PATH."""
    platform = sys.platform if platform is None else platform
    if platform != "win32":
        return []
    enabled = long_paths_enabled() if enabled is None else enabled
    if enabled:
        return []
    length = longest_run_path(runs_root, study)
    if length <= WINDOWS_MAX_PATH:
        return []
    return [f"run paths reach about {length} characters under MIVAL_RUNS={runs_root}, above "
            f"Windows' {WINDOWS_MAX_PATH + 1}-character limit (LongPathsEnabled is off): the models "
            "stage would fail writing predictions. Enable long paths (HKLM\\" + LONG_PATHS_KEY
            + " LongPathsEnabled=1, then reboot) or set MIVAL_RUNS to a short root such as "
            "D:\\mival\\runs"]


def check_arm_env(card, machine: "Machine") -> List[str]:
    """What the models stage needs from a card's env, and whether the arm will run here at all."""
    found = []
    runtime = card.runtime or {}
    env = runtime.get("env")
    if env:
        found.extend(machine.env_missing(env))
    if runtime.get("requires_gpu") and machine.device_count == 0:
        found.append(f"{card.model_id}: its card requires a GPU and torch sees none; the run "
                     "will skip its arms")
    return found


def check_card(card) -> List[str]:
    ext = card.raw["x-mival"]
    found = []
    for weight in card.weights:
        path = Path(weight["uri"])
        if not path.is_file():
            found.append(f"missing {path}")
        elif weight.get("sha256") and sha256(path) != weight["sha256"]:
            found.append(f"sha256 mismatch {path}")
    code = ext.get("code_path")
    if code and not Path(code).is_dir():
        found.append(f"missing code_path {code}")
    found.extend(check_state_dict_transform(card))
    env = card.runtime.get("env")
    if env:
        try:
            env_python(Path(env))
        except FileNotFoundError:
            found.append(f"missing env {env}")
    found.extend(check_windows_runtime(card))
    return found


class Machine:
    """What this machine offers, asked once: driver CUDA, GPUs, each env's torch CUDA build."""

    def __init__(self, home: Path):
        self.home = home
        self.driver = driver_cuda_version()
        self._torch_cuda: Dict[str, Optional[str]] = {}
        self._missing: Dict[Tuple[str, ...], List[str]] = {}
        self._env_missing: Dict[str, List[str]] = {}
        try:
            self.controller: Optional[Path] = env_python(home / "envs" / CONTROLLER_ENV)
        except FileNotFoundError:
            self.controller = None
        self.device_count = torch_device_count(self.controller) if self.controller else 0

    def controller_missing(self, modules: Sequence[str]) -> List[str]:
        """The problem with the controller env for these modules, if any; asked once per module set."""
        key = tuple(modules)
        if key not in self._missing:
            if self.controller is None:
                self._missing[key] = [f"no {CONTROLLER_ENV} env"]
            else:
                gone = missing_modules(self.controller, key)
                self._missing[key] = [f"{CONTROLLER_ENV} cannot import {', '.join(gone)}"] if gone else []
        return self._missing[key]

    def env_missing(self, env: str) -> List[str]:
        """The problem with an arm env for :data:`ARM_MODULES`, if any; asked once per env.

        An env that is not there is reported by :func:`check_card`.
        """
        if env not in self._env_missing:
            try:
                python = env_python(Path(env))
            except FileNotFoundError:
                self._env_missing[env] = []
            else:
                gone = missing_modules(python, ARM_MODULES)
                self._env_missing[env] = [f"env {env} cannot import {', '.join(gone)}"] if gone else []
        return self._env_missing[env]

    def torch_cuda(self, env: str) -> Optional[str]:
        if env not in self._torch_cuda:
            try:
                self._torch_cuda[env] = torch_cuda_version(env_python(Path(env)))
            except FileNotFoundError:
                self._torch_cuda[env] = None
        return self._torch_cuda[env]


def check_study(name: str, machine: Machine) -> int:
    try:
        study = load_study(study_root() / name)
    except ValueError as exc:
        print(f"  site config: {exc}")
        return 1
    retrieve = study.stage_spec("retrieve")
    site = check_site(retrieve) + check_template_values(study)
    site += check_gpus(study.execution.get("gpus"), machine.device_count)
    site += machine.controller_missing(controller_modules(retrieve, study.stage_spec("preprocess")))
    site += check_path_length(runs_root(), study)
    print(f"  site {study.site}, dbms {retrieve.get('dbms', 'postgresql')}, schema {retrieve.get('schema')}: "
          f"{'ok' if not site else '; '.join(site)}")
    problems = len(site)
    models = study.stage_spec("models")
    registry = Path(models["registry"])
    cards = load_registry(registry if registry.is_absolute() else REPO / registry)
    for model_id in sorted({arm["model_id"] for arm in models["arms"]}):
        card = cards.get(model_id)
        if card is None:
            print(f"  {model_id}: no card in {models['registry']}")
            problems += 1
            continue
        found = check_card(card) + check_arm_env(card, machine)
        env = card.runtime.get("env")
        if env:
            found.extend(check_driver_cuda(env, machine.torch_cuda(env), machine.driver))
        info = runtime_override_info(card) + prebuilt_extension_info(card)
        note = f" (info: {'; '.join(info)})" if info else ""
        print(f"  {model_id}: {'ok' if not found else '; '.join(found)}{note}")
        problems += len(found)
    return problems


def main(argv):
    os.environ.setdefault("MIVAL_HOME", str(REPO.parent))
    names = argv or sorted(p.parent.name for p in study_root().glob("*/study.yaml"))
    machine = Machine(Path(os.environ["MIVAL_HOME"]))
    print(f"driver CUDA {machine.driver or 'none'}, GPUs visible to {CONTROLLER_ENV}: {machine.device_count}")
    total = 0
    for name in names:
        print(name)
        total += check_study(name, machine)
    print("all ok" if not total else f"{total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
