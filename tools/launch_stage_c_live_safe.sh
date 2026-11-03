#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/you/projects/kinetickick
PY=/home/you/conda_env/env_isaaclab/bin/python
NEW_PATTERN='^/home/you/conda_env/env_isaaclab/bin/python tools/show_paid_stage_c_physical_live.py'
OLD_PATTERN='^/home/you/conda_env/env_isaaclab/bin/python tools/show_paid_stage1_physical_live.py'

cd "$ROOT"
if pgrep -af "$NEW_PATTERN"; then
  echo "STAGE_C_ALREADY_RUNNING: one Isaac Sim window already exists, no duplicate started."
  exit 0
fi
if pgrep -af "$OLD_PATTERN"; then
  echo "BLOCKED: older Stage I Isaac Sim is still running. Close it before Stage C."
  exit 2
fi

avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
echo "MEMORY_AVAILABLE_KiB=$avail_kb"
if (( avail_kb < 17000000 )); then
  echo "BLOCKED: need at least 16.2 GiB available memory before GUI startup."
  exit 3
fi
: "${DISPLAY:=:0}"
: "${XAUTHORITY:=/run/user/1000/gdm/Xauthority}"
export DISPLAY XAUTHORITY PYTHONUNBUFFERED=1
echo "START_STAGE_C: DISPLAY=$DISPLAY, XAUTHORITY=$XAUTHORITY"
exec "$PY" tools/show_paid_stage_c_physical_live.py --ball-x 0.04 --ball-y -0.77 --camera WIDE
