#!/usr/bin/env bash
# Poll the VM until R5 (train.py) stops, then exit 0 so the harness re-invokes Claude.
# Read-only: never touches config or the chip. Polls every 15 min, hard cap 24 h.
set -u
SSH="gcloud alpha compute tpus tpu-vm ssh chmawa --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap"
# Remote one-liner: is a *python* train.py alive? + highest R5 checkpoint step.
REMOTE='if ps -u $(id -un) -o comm=,args= | awk "/train\.py/ && \$1 ~ /python/" | grep -q python; then s=RUN; else s=STOP; fi; step=$(find ~/grpo_runs/R5 -type d -regextype posix-extended -regex ".*/[0-9]+" 2>/dev/null | grep -oE "[0-9]+$" | sort -n | tail -1); echo "STATE $s ${step:-NA}"'

for i in $(seq 1 96); do
  line=$($SSH --command="$REMOTE" 2>/dev/null | grep '^STATE' | head -1)
  ts=$(date -u +%H:%MZ)
  echo "[poll $i $ts] ${line:-<no-ssh-response>}"
  case "$line" in
    *"STATE STOP"*) echo "RESULT: R5_STOPPED — ${line}"; exit 0;;
  esac
  sleep 900
done
echo "RESULT: TIMEOUT_24H — R5 still running, re-arm watcher"
exit 0
