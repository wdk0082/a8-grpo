#!/usr/bin/env bash
# Poll the VM until the n=1319 sweep finishes (marker) or dies (no marker), then exit
# so the harness re-invokes Claude. ROBUST: ssh keepalive bounds a hung poll to ~45s
# (the previous watcher hung forever because it had no keepalive). Read-only.
set -u
SSH=(gcloud alpha compute tpus tpu-vm ssh chmawa --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap
     --ssh-flag="-o ConnectTimeout=20" --ssh-flag="-o ServerAliveInterval=15" --ssh-flag="-o ServerAliveCountMax=3")
REMOTE='if [ -f ~/grpo_runs/SWEEP_N1319_DONE ]; then s=DONE; elif ps -u $(id -un) -o comm=,args= | awk "/evaluate\.py/ && \$1 ~ /python/" | grep -q python; then s=RUN; else s=STOPPED; fi; cnt=$(ls ~/grpo_runs/evals/eval_*_n1319.json 2>/dev/null | wc -l | tr -d " "); echo "STATE $s $cnt/9"'

for i in $(seq 1 72); do
  line=$("${SSH[@]}" --command="$REMOTE" 2>/dev/null | grep '^STATE' | head -1)
  ts=$(date -u +%H:%MZ)
  echo "[poll $i $ts] ${line:-<no-ssh-response>}"
  case "$line" in
    *"STATE DONE"*)    echo "RESULT: SWEEP_DONE — ${line}"; exit 0;;
    *"STATE STOPPED"*) echo "RESULT: SWEEP_STOPPED_NO_MARKER — ${line}"; exit 0;;
  esac
  sleep 480
done
echo "RESULT: TIMEOUT — sweep still running after 72 polls"
exit 0
