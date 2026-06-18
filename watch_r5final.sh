#!/usr/bin/env bash
# Poll until R5_final's n=1319 dump appears (or the re-run dies), then exit so Claude wakes.
set -u
SSH=(gcloud alpha compute tpus tpu-vm ssh chmawa --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap
     --ssh-flag="-o ConnectTimeout=20" --ssh-flag="-o ServerAliveInterval=15" --ssh-flag="-o ServerAliveCountMax=3")
for i in $(seq 1 40); do
  s=$("${SSH[@]}" --command='if [ -f ~/grpo_runs/evals/eval_R5_final_n1319.json ]; then grep -o "\"acc\": [0-9.]*" ~/grpo_runs/evals/eval_R5_final_n1319.json | head -1; else (tmux has-session -t r5f 2>/dev/null && echo RUNNING || echo GONE-NODUMP); fi' 2>/dev/null)
  echo "[poll $i $(date -u +%H:%MZ)] ${s:-<no-response>}"
  case "$s" in
    *acc*)        echo "RESULT: R5_FINAL_DONE ${s}"; exit 0;;
    *GONE-NODUMP*) echo "RESULT: R5_FINAL_FAILED_AGAIN"; exit 0;;
  esac
  sleep 60
done
echo "RESULT: TIMEOUT"; exit 0
