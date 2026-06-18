#!/usr/bin/env bash
# Full GSM8K test (n=1319) eval of the decisive checkpoints. Run detached on the VM.
# Writes eval_<tag>_n1319.json next to the n=64 dumps (does NOT overwrite them).
set -u
cd ~/tpu-2026/scripts
set -a; source ~/.env 2>/dev/null; set +a
source ~/venvs/tunix/bin/activate
export DATA_SOURCE=hf NUM_TEST_BATCHES=1319
EVALS=~/grpo_runs/evals
LOG=~/grpo_runs/sweep_n1319.log
rm -f ~/grpo_runs/SWEEP_N1319_DONE
echo "=== SWEEP START $(date -u +%FT%TZ)  NUM_TEST_BATCHES=$NUM_TEST_BATCHES ===" | tee "$LOG"

run_eval () {  # $1=tag  $2=ckptdir|BASE  $3=step
  local tag="$1" ckpt="$2" step="$3" t0 t1
  t0=$(date +%s)
  echo "--- [$tag] step=$step start $(date -u +%T)Z ---" | tee -a "$LOG"
  if [ "$ckpt" = "BASE" ]; then
    RUN_NAME=base python evaluate.py --step -1 \
      --dump "$EVALS/eval_${tag}_n1319.json" 2>&1 | grep -E '^FINAL|^Wrote|Error|Traceback|Exception' | tee -a "$LOG"
  else
    RUN_NAME="$tag" python evaluate.py --step "$step" --ckpt-dir "$ckpt" \
      --dump "$EVALS/eval_${tag}_n1319.json" 2>&1 | grep -E '^FINAL|^Wrote|Error|Traceback|Exception' | tee -a "$LOG"
  fi
  t1=$(date +%s)
  echo "    [$tag] took $((t1-t0))s" | tee -a "$LOG"
}

run_eval base      BASE                          -1
run_eval R0_best   ~/grpo_runs/R0/ckpts/actor    700
run_eval R0_final  ~/grpo_runs/R0/ckpts/actor    3364
run_eval R4_best   ~/grpo_runs/R4/ckpts/actor    1300
run_eval R4_final  ~/grpo_runs/R4/ckpts/actor    3364
run_eval R3b_best  ~/grpo_runs/R3b/ckpts/actor   1000
run_eval R3b_final ~/grpo_runs/R3b/ckpts/actor   3364
run_eval R5_best   ~/grpo_runs/R5/ckpts/actor    1300
run_eval R5_final  ~/grpo_runs/R5/ckpts/actor    3364

echo "=== SWEEP DONE $(date -u +%FT%TZ) ===" | tee -a "$LOG"
touch ~/grpo_runs/SWEEP_N1319_DONE
