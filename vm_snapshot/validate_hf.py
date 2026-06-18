"""Validate the HF token + gated Gemma access (reads HF_TOKEN from env; no download)."""
import os
from huggingface_hub import HfApi

t = os.environ.get("HF_TOKEN", "")
print("HF_TOKEN present:", bool(t), "len:", len(t))
api = HfApi()
try:
    who = api.whoami(token=t)
    print("whoami:", who.get("name"), "type:", who.get("type"))
except Exception as e:
    print("WHOAMI_FAILED:", repr(e)[:200]); raise SystemExit(1)
try:
    mi = api.model_info("google/gemma-3-1b-it", token=t)
    print("GEMMA_ACCESS_OK:", mi.id, "gated:", getattr(mi, "gated", None))
except Exception as e:
    print("GEMMA_ACCESS_FAILED:", repr(e)[:300]); raise SystemExit(2)
