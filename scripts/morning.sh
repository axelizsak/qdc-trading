#!/usr/bin/env bash
# The automatic weekday morning: what every customer shares (graph.morning), then the day up to the orders review
# (graph.pipeline). The three reviews stay human: open the cockpit to answer them, then ship logs/.
#
# Installed as a launchd job by scripts/install_morning.sh; its output goes to ~/Library/Logs/qdc-morning.log.
# It runs the code that is checked out and writes to logs/, nothing else: no git command. Running it twice the same
# day costs nothing: the day's Macro/News is read back from logs/, and a day already run is not run again.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
if [ "$(date +%u)" -ge 6 ]; then
    echo "=== $(date '+%F %T') weekend: nothing to run"
    exit 0
fi

echo "=== $(date '+%F %T') morning run"
.venv/bin/python -m graph.morning || echo "morning run failed (exit $?)"
echo "=== $(date '+%F %T') the day, up to the orders review"
.venv/bin/python -m graph.pipeline || echo "day run failed (exit $?)"
echo "=== $(date '+%F %T') done: open the cockpit for the reviews"
