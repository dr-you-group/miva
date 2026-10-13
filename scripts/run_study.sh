#!/usr/bin/env bash
# One study end to end: retrieve -> profile -> preprocess -> models (one run per
# runtime env named by the arms' cards) -> evaluate.
#
#   usage: scripts/run_study.sh <study>        e.g. scripts/run_study.sh lvef
#
#   MIVAL_HOME  bundle root holding miva/, envs/, models/, code/, config/
#               (config/site.yaml and config/micdm.env are the site's own settings)
#               (default: the directory above this repository)
#   MIVAL_RUNS  where run artifacts go (default: $MIVAL_HOME/runs)
#
# A finished stage is not re-run: its config_hash already has a manifest, so a
# second invocation after a failure resumes from the failed stage.
set -uo pipefail

S=${1:?usage: scripts/run_study.sh <study>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export MIVAL_HOME=${MIVAL_HOME:-$(dirname "$REPO")}
R=${MIVAL_RUNS:-$MIVAL_HOME/runs}
CTRL=$MIVAL_HOME/envs/mival-torch/bin/python
O=$R/$S/_run
cd "$REPO"
[ -f "studies/$S/study.yaml" ] || { echo "no studies/$S/study.yaml"; exit 1; }
[ -x "$CTRL" ] || { echo "no $CTRL (see README: env 설치)"; exit 1; }
[ -f "$MIVAL_HOME/config/site.yaml" ] || { echo "no $MIVAL_HOME/config/site.yaml (see README 4.2: 기관 설정)"; exit 1; }
mkdir -p "$O"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/run.log"; }
cli() { local py=$1; shift; PYTHONPATH=src "$py" -c "import sys; from mival.pipeline.cli import main; sys.exit(main())" "$@"; }
run_dir() { grep -o '"run_dir": "[^"]*"' "$1" | tail -1 | sed 's/"run_dir": "\(.*\)"/\1/'; }
stage() {  # stage <name> <python> <log> args...
  local name=$1 py=$2 out=$3; shift 3
  log "$name"
  if ! cli "$py" run "$name" --study "${STUDY_DIR:-studies/$S}" --runs-root "$R" "$@" > "$out" 2>&1; then
    log "$name FAILED (log: $out)"; tail -30 "$out"; return 1
  fi
}

GPU=$("$CTRL" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
log "study $S, MIVAL_HOME $MIVAL_HOME, runs $R, GPU $GPU"

stage retrieve "$CTRL" "$O/retrieve.log" || exit 1
CI=$(run_dir "$O/retrieve.log")/artifacts/cohort_index.parquet
stage profile "$CTRL" "$O/profile.log" --input cohort_index="$CI" || exit 1
SPL=$(run_dir "$O/profile.log")/artifacts/cohort_split.parquet
stage preprocess "$CTRL" "$O/preprocess.log" --input cohort_index="$CI" || exit 1
PRE=$(run_dir "$O/preprocess.log")/artifacts/preprocess_index.parquet

# Split the arms by the env their card names; each group gets its own study copy.
ENVS=$("$CTRL" - "$S" "$O" "$GPU" <<'PY'
import json, os, pathlib, sys, yaml
study, out, gpu = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3] == "1"
st = yaml.safe_load(open(f"studies/{study}/study.yaml"))
m = st["stages"]["models"]
cards = {}
for p in pathlib.Path(m["registry"]).glob("*.json"):
    body = json.loads(p.read_text())
    cards[body["model_id"]] = body
groups = {}
for arm in m["arms"]:
    env = os.path.expandvars(cards[arm["model_id"]]["x-mival"]["runtime"]["env"])
    if pathlib.Path(env).name == "mival-xecg" and not gpu:
        print(f"skip {arm['model_id']} ({arm['label_def']}): needs a GPU", file=sys.stderr)
        continue
    groups.setdefault(env, []).append(arm)
for env, arms in groups.items():
    d = out / f"study-{pathlib.Path(env).name}"
    d.mkdir(exist_ok=True)
    copy = json.loads(json.dumps(st))
    copy["stages"]["models"]["arms"] = arms
    (d / "study.yaml").write_text(yaml.safe_dump(copy, sort_keys=False, allow_unicode=True))
    print(env)
PY
) || exit 1

rm -rf "$O/predictions_all"; mkdir -p "$O/predictions_all"
FAILED=0
for E in $ENVS; do
  name=$(basename "$E")
  [ -x "$E/bin/python" ] || { log "models [$name] SKIPPED: no $E"; FAILED=1; continue; }
  if [ "$name" = mival-xecg ]; then export TORCH_EXTENSIONS_DIR=$MIVAL_HOME/cache/torch_extensions; fi
  STUDY_DIR=$O/study-$name stage models "$E/bin/python" "$O/models-$name.log" \
    --input preprocess_index="$PRE" --input cohort_split="$SPL" --input cohort_index="$CI" || { FAILED=1; continue; }
  cp "$(run_dir "$O/models-$name.log")"/artifacts/predictions/*.parquet "$O/predictions_all/"
done

stage evaluate "$CTRL" "$O/evaluate.log" --input predictions="$O/predictions_all" --input cohort_split="$SPL" || exit 1
EV=$(run_dir "$O/evaluate.log")
log "done: $EV"
[ "$FAILED" = 0 ] || log "WARNING: one or more model envs failed or were skipped; see $O/models-*.log"
