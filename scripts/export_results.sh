#!/usr/bin/env bash
# The results zip a site sends back (see export_results.py next to this file).
#
#   usage: export_results.sh [--allow-small-stage-totals] [study ...]
#          e.g. export_results.sh lvef ecg-age
#
#   --allow-small-stage-totals  export although a stage excluded 1 to min_cell - 1
#                               records in total; only after the site decided that
#                               count may leave (recorded in the bundle)
#
#   MIVAL_HOME  bundle root (default: the directory above this repository)
#   MIVAL_RUNS  where run artifacts are (default: $MIVAL_HOME/runs)
#   STUDY_ROOT  study directories (default: studies/site in mi-va, studies in the shared copy)
#
# This file only finds the mival-torch interpreter; Windows uses
# export_results.cmd the same way.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
# This script lives in scripts/site/ in mi-va and in scripts/ in the shared copy.
REPO=$(cd "$HERE" && while [ ! -d src/mival ] && [ "$PWD" != / ]; do cd ..; done; pwd)
[ -d "$REPO/src/mival" ] || { echo "no src/mival above $HERE"; exit 1; }
export MIVAL_HOME=${MIVAL_HOME:-$(dirname "$REPO")}
CTRL=$MIVAL_HOME/envs/mival-torch/bin/python
[ -x "$CTRL" ] || { echo "no $CTRL (see README: env 설치)"; exit 1; }
exec "$CTRL" "$HERE/export_results.py" "$@"
