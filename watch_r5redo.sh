#!/usr/bin/env bash
# Poll until the R5 redo marker appears (R5_final from r5f, then R5_best from r5redo),
# reporting both dump sizes so we can confirm they wrote non-empty this time.
set -u
SSH=(gcloud alpha compute tpus tpu-vm ssh chmawa --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap
     --ssh-flag="-o ConnectTimeout=20" --ssh-flag="-o ServerAliveInterval=15" --ssh-flag="-o ServerAliveCountMax=3")
for i in $(seq 1 55); do
  s=$("${SSH[@]}" --command='if [ -f ~/grpo_runs/R5_REDO_DONE ]; then echo "DONE best=$(wc -c <~/grpo_runs/evals/eval_R5_best_n1319.json 2>/dev/null) final=$(wc -c <~/grpo_runs/evals/eval_R5_final_n1319.json 2>/dev/null)"; else echo PENDING; fi' 2>/dev/null)
  echo "[poll $i $(date -u +%H:%MZ)] ${s:-<no-response>}"
  case "$s" in *DONE*) echo "RESULT: $s"; exit 0;; esac
  sleep 60
done
echo "RESULT: TIMEOUT"; exit 0
