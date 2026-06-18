"""Materialize the EXACT shipped TFDS+grain GSM8K test split (the same one the
rest of the team evaluates on) into JSON, in a process that does NOT import jax
or config (so TFDS doesn't crash — see the protobuf-7.x baseline patch).

This reproduces build_train_val_test's test_ds line exactly:
    get_dataset(test_dir, "test", "tfds").batch(train_micro_batch_size)[:num_test_batches]
with the shipped defaults (TRAIN_MICRO_BATCH_SIZE=1, NUM_TEST_BATCHES=64), so the
grain shuffle(seed=42) ordering is identical to the shipped evaluate.py path.

Run from ~/tpu-2026/scripts:  python prepare_test_tfds.py
Then:  python evaluate.py --test-json ~/gsm8k_test_tfds.json [--step N --ckpt-dir ...]
"""
import json
import os
import sys

sys.path.insert(0, ".")
from data import get_dataset  # data.py imports no jax / no config

# Shipped config defaults (NOT imported from config.py — that pulls in jax).
TRAIN_MICRO_BATCH_SIZE = 1
NUM_TEST_BATCHES = 64
TEST_DATA_DIR = "./data/test"

test_ds = get_dataset(TEST_DATA_DIR, "test", "tfds").batch(TRAIN_MICRO_BATCH_SIZE)[:NUM_TEST_BATCHES]
batches = [
    {"question": list(b["question"]), "answer": list(b["answer"]), "prompts": list(b["prompts"])}
    for b in test_ds
]
out = os.path.expanduser("~/gsm8k_test_tfds.json")
with open(out, "w") as f:
    json.dump(batches, f)
print(f"wrote {len(batches)} batches -> {out}")
print("first question:", batches[0]["question"][0][:80])
