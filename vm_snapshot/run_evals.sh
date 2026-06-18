#!/usr/bin/env bash
# Part I.1(ii) evals via the (patched) scripts/evaluate.py, on the EXACT TFDS
# split the rest of the team uses (pre-materialized jax-free into JSON).
source "$HOME/venvs/tunix/bin/activate"
set -a; source "$HOME/.env"; set +a
export TF_CPP_MIN_LOG_LEVEL=3
cd "$HOME/tpu-2026/scripts" || exit 1
CK="$HOME/ckpts_backup/actor"
TJ="$HOME/gsm8k_test_tfds.json"

echo "########## PREP: materialize TFDS test split (jax-free) ##########"
python -u prepare_test_tfds.py || { echo "PREP FAILED"; exit 1; }

echo "########## EVAL START $(date -u +%H:%M:%S) ##########"
echo "########## [1/4] BASE gemma-3-1b-it ##########"
python -u evaluate.py --test-json "$TJ" --preset greedy
echo "########## [2/4] FINETUNED step 2000 ##########"
python -u evaluate.py --test-json "$TJ" --preset greedy --step 2000 --ckpt-dir "$CK"
echo "########## [3/4] FINETUNED step 3000 ##########"
python -u evaluate.py --test-json "$TJ" --preset greedy --step 3000 --ckpt-dir "$CK"
echo "########## [4/4] FINETUNED step 3364 (resulting) ##########"
python -u evaluate.py --test-json "$TJ" --preset greedy --step 3364 --ckpt-dir "$CK"
echo "########## EVAL DONE $(date -u +%H:%M:%S) ##########"
