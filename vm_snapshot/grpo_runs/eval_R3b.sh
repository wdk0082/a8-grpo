#!/bin/bash
cd ~/tpu-2026/scripts
set -a; source ~/.env; set +a
source ~/venvs/tunix/bin/activate
OUT=~/grpo_runs/evals; mkdir -p "$OUT"
echo "### R3b best (1000)"; python evaluate.py --step 1000 --ckpt-dir ~/grpo_runs/R3b/ckpts/actor --dump "$OUT/eval_R3b_best.json" 2>&1 | grep -aE 'FINAL|Wrote|No checkpoint|Error|Traceback' | tail -2
echo "### R3b final (3364)"; python evaluate.py --step 3364 --ckpt-dir ~/grpo_runs/R3b/ckpts/actor --dump "$OUT/eval_R3b_final.json" 2>&1 | grep -aE 'FINAL|Wrote|No checkpoint|Error|Traceback' | tail -2
echo "### R3b EVALS DONE"
