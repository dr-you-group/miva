"""Check the site config and every card a study uses (weights, code, env).

    MIVAL_HOME=/path/to/bundle python scripts/check_bundle.py [study ...]

With no study named, every study under studies/ is checked. The study is
loaded the way the pipeline loads it, so a missing or conflicting
config/site.yaml shows up here. Weights are compared against the sha256 in
the card, so a truncated copy is caught here rather than in the models stage.
"""

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mival.modelcard import load_registry  # noqa: E402
from mival.pipeline.study import load_study  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_site(retrieve: dict) -> list:
    found = []
    if not Path(retrieve.get("dsn_env", "")).is_file():
        found.append(f"no DB connection file {retrieve.get('dsn_env')}")
    root = retrieve.get("local_path_root")
    if root is not None and not Path(root).is_dir():
        found.append(f"no DICOM root {root}")
    return found


def check_study(name: str) -> int:
    try:
        study = load_study(REPO / "studies" / name)
    except ValueError as exc:
        print(f"  site config: {exc}")
        return 1
    retrieve = study.stage_spec("retrieve")
    site = check_site(retrieve)
    print(f"  site {study.site}, dbms {retrieve.get('dbms', 'postgresql')}, schema {retrieve.get('schema')}: "
          f"{'ok' if not site else '; '.join(site)}")
    problems = len(site)
    models = study.stage_spec("models")
    cards = load_registry(REPO / models["registry"])
    for model_id in sorted({arm["model_id"] for arm in models["arms"]}):
        card = cards.get(model_id)
        if card is None:
            print(f"  {model_id}: no card in {models['registry']}")
            problems += 1
            continue
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
        env = card.runtime.get("env")
        if env and not (Path(env) / "bin" / "python").exists():
            found.append(f"missing env {env}")
        print(f"  {model_id}: {'ok' if not found else '; '.join(found)}")
        problems += len(found)
    return problems


def main(argv):
    names = argv or sorted(p.parent.name for p in (REPO / "studies").glob("*/study.yaml"))
    total = 0
    for name in names:
        print(name)
        total += check_study(name)
    print("all ok" if not total else f"{total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
