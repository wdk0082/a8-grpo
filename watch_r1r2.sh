#!/usr/bin/env bash
# Poll until R1/R2 n=1319 evals finish (4 dumps + marker).
set -u
SSH=(gcloud alpha compute tpus tpu-vm ssh chmawa --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap
     --ssh-flag="-o ConnectTimeout=20" --ssh-flag="-o ServerAliveInterval=15" --ssh-flag="-o ServerAliveCountMax=3")
for i in $(seq 1 70); do
  s=$("${SSH[@]}" --command='if [ -f ~/grpo_runs/R1R2_N1319_DONE ]; then echo DONE; else echo "PENDING $(ls ~/grpo_runs/evals/eval_R1_*_n1319.json ~/grpo_runs/evals/eval_R2_*_n1319.json 2>/dev/null | wc -l | tr -d " ")/4"; fi' 2>/dev/null)
  echo "[poll $i $(date -u +%H:%MZ)] ${s:-<no-response>}"
  case "$s" in *DONE*) echo "RESULT: R1R2_N1319_DONE"; exit 0;; esac
  sleep 90
done
echo "RESULT: TIMEOUT"; exit 0
