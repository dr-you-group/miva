#!/usr/bin/env bash
# Studies end to end (see run_study.py next to this file, which does the work).
#
#   usage: run_study.sh [study ...]    e.g. run_study.sh (every study), run_study.sh lvef
#
# Run the same command again after a failure or an interruption to resume:
# finished stages and model arms are skipped.
#
#   MIVAL_HOME  bundle root holding miva/, envs/, models/, code/, config/
#               (default: the directory above this repository)
#   MIVAL_RUNS  where run artifacts go (default: $MIVAL_HOME/runs)
#
# This file only finds the mival-torch interpreter; Windows uses run_study.cmd
# the same way, so both run the same Python steps.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
# This script lives in scripts/site/ in mi-va and in scripts/ in the shared copy.
REPO=$(cd "$HERE" && while [ ! -d src/mival ] && [ "$PWD" != / ]; do cd ..; done; pwd)
[ -d "$REPO/src/mival" ] || { echo "no src/mival above $HERE"; exit 1; }
export MIVAL_HOME=${MIVAL_HOME:-$(dirname "$REPO")}
CTRL=$MIVAL_HOME/envs/mival-torch/bin/python
[ -x "$CTRL" ] || { echo "no $CTRL (see README: env 설치)"; exit 1; }
exec "$CTRL" "$HERE/run_study.py" "$@"
