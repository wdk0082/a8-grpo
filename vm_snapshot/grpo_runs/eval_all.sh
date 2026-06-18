#!/bin/bash
cd ~/tpu-2026/scripts
set -a; source ~/.env; set +a
source ~/venvs/tunix/bin/activate
OUT=~/grpo_runs/evals; mkdir -p "$OUT"
run_eval(){ python evaluate.py "$@" 2>&1 | grep -aE 'FINAL|Wrote|No checkpoint|Error|Traceback' | tail -3; }
echo "### base"; run_eval --step -1 --dump "$OUT/eval_base.json"
for spec in R0:700 R1:600 R2:400 R4:1300; do
  name=${spec%%:*}; step=${spec##*:}
  echo "### $name best ($step)"; run_eval --step "$step" --ckpt-dir ~/grpo_runs/$name/ckpts/actor --dump "$OUT/eval_${name}_best.json"
  echo "### $name final (3364)"; run_eval --step 3364 --ckpt-dir ~/grpo_runs/$name/ckpts/actor --dump "$OUT/eval_${name}_final.json"
done
echo "### ALL EVALS DONE"
