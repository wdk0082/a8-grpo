"""Compare the first-64 GSM8K test problems from TFDS vs HF (no jax import, so
TFDS works). Confirms whether the two data sources yield the same eval split."""
import sys
sys.path.insert(0, ".")  # run from ~/tpu-2026/scripts
from data import get_dataset

def first_qs(source, n=64):
    ds = get_dataset(f"./cmpdata/{source}", "test", source).batch(1)[:n]
    return [b["question"][0] for b in ds]

tfds_qs = first_qs("tfds")
hf_qs = first_qs("hf")
print("TFDS n:", len(tfds_qs), " HF n:", len(hf_qs))
print("IDENTICAL (order+content):", tfds_qs == hf_qs)
print("set overlap:", len(set(tfds_qs) & set(hf_qs)), "/ 64")
print("--- TFDS[0] ---", repr(tfds_qs[0][:110]))
print("--- HF[0]   ---", repr(hf_qs[0][:110]))
print("--- TFDS[1] ---", repr(tfds_qs[1][:110]))
print("--- HF[1]   ---", repr(hf_qs[1][:110]))
