"""Standalone evaluation of a (LoRA) policy on the GSM8K test set.

Reports three numbers:
  * accuracy           — exact numeric match
  * partial_accuracy   — answer within 10% of ground truth
  * format_accuracy    — fraction of completions whose template parses

Run as:
    python evaluate.py                                       # BASE model (no restore)
    python evaluate.py --step 3364 --ckpt-dir <dir>          # a finetuned checkpoint
    python evaluate.py --step 0    --ckpt-dir <dir>          # latest checkpoint

BASELINE PATCHES (documented in the report):
  1. Added Orbax LoRA restore (--step / --ckpt-dir), using the same
     CheckpointManager.maybe_restore(..., restore_only_lora_params=True) as
     chat.py. As-shipped this script built a fresh B=0 LoRA (== base) and never
     restored a checkpoint, so it could ONLY measure the base model and could
     not produce the I.1(ii) finetuned number. Default --step -1 keeps the
     original behaviour (BASE, no restore).
  2. Added --source hf (default) which loads GSM8K from HF `datasets`, because
     TFDS 4.9.9's gsm8k builder crashes under protobuf 7.x in the current venv.

Generation settings, NUM_TEST_BATCHES, the data split (seed 42) and the accuracy
metric are UNCHANGED, so the base model and every checkpoint are scored under
identical settings (a fair comparison).
"""
import argparse
import json
import os
import random

from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.sft.checkpoint_manager import CheckpointManager

from config import (
    CKPT_DIR,
    GENERATION_CONFIGS,
    MAX_PROMPT_LENGTH,
    NUM_TEST_BATCHES,
    TEST_DATA_DIR,
    TOTAL_GENERATION_STEPS,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MICRO_BATCH_SIZE,
    NUM_BATCHES,
    NUM_EPOCHS,
    DATA_SOURCE,
)
from data import SYSTEM_PROMPT, TEMPLATE, build_train_val_test
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer, model_config_for
from rewards import match_format, match_numbers


def generate(question, sampler, eos_tokens, temperature=0.7, top_k=50, top_p=0.95, seed=None):
    if isinstance(question, str):
        batch = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question)]
    else:
        batch = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q) for q in question]

    out = sampler(
        input_strings=batch,
        max_generation_steps=TOTAL_GENERATION_STEPS,
        temperature=temperature, top_k=top_k, top_p=top_p,
        echo=False, seed=seed, eos_tokens=eos_tokens,
    )
    return out.text[0] if isinstance(question, str) else out.text


def evaluate(dataset, sampler, eos_tokens, temperature=0.7, top_k=50, top_p=0.95, num_passes=1):
    corr = partially_corr = corr_format = total = 0
    flags = []  # per-example exact-correctness for bootstrap CI

    for batch in tqdm(dataset):
        answers = batch["answer"]
        questions = batch["question"]
        per_q = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = generate(questions, sampler, eos_tokens, temperature, top_k, top_p, seed=p)
            for i, r in enumerate(responses):
                per_q[i].append(r)

        for q, responses, ans in zip(questions, per_q, answers):
            got_corr = got_partial = got_format = False
            for r in responses:
                ext = guess.group(1) if (guess := match_numbers.search(r)) is not None else "-1e9"
                try:
                    if float(ext.strip()) == float(ans.strip()):
                        got_corr = True
                    ratio = float(ext.strip()) / float(ans.strip())
                    if 0.9 <= ratio <= 1.1:
                        got_partial = True
                except Exception:
                    pass
                if match_format.search(r) is not None:
                    got_format = True
                if got_corr and got_partial and got_format:
                    break

            corr += int(got_corr)
            partially_corr += int(got_partial)
            corr_format += int(got_format)
            flags.append(int(got_corr))
            total += 1
            if total % 10 == 0:
                print(f"===> corr={corr} total={total} acc={corr/total*100:.2f}% "
                      f"partial={partially_corr/total*100:.2f}% fmt={corr_format/total*100:.2f}%")

    return corr, total, corr/total*100, partially_corr/total*100, corr_format/total*100, flags


def bootstrap_ci(flags, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI for the mean of 0/1 flags. Returns (lo%, hi%)."""
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(100.0 * sum(flags[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return (means[int((alpha / 2) * n_boot)], means[int((1 - alpha / 2) * n_boot)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="greedy", choices=list(GENERATION_CONFIGS))
    ap.add_argument("--source", default="hf", choices=["tfds", "kaggle", "hf"],
                    help="GSM8K source. Default 'hf' (TFDS 4.9.9 crashes under protobuf 7.x here).")
    ap.add_argument("--ckpt-dir", default=os.path.join(CKPT_DIR, "actor"),
                    help="Directory of per-step Orbax checkpoint subdirs (used with --step).")
    ap.add_argument("--step", type=int, default=-1,
                    help="-1 (default) = BASE model, no restore; 0 = latest checkpoint; N = step N.")
    ap.add_argument("--test-json", default=None,
                    help="Pre-materialized test split (JSON list of size-1 batches) to use "
                         "instead of build_train_val_test. Produced jax-free by "
                         "prepare_test_tfds.py so the TFDS split matches the rest of the team.")
    ap.add_argument("--dump", default=None,
                    help="Write per-example results + bootstrap CI to this JSON (default eval_<tag>.json).")
    args = ap.parse_args()

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    tag = "BASE"
    if args.step >= 0:
        mgr = CheckpointManager(root_directory=args.ckpt_dir)
        n_restored, _ = mgr.maybe_restore(
            model=lora, step=(None if args.step == 0 else args.step),
            restore_only_lora_params=True,
        )
        if n_restored == 0:
            raise RuntimeError(f"No checkpoint found under {args.ckpt_dir} (step={args.step}).")
        tag = f"FINETUNED step {n_restored}"
        print(f"Restored LoRA params from step {n_restored}")
    else:
        print("No --step given: evaluating the BASE model (B=0 LoRA == base).")

    if args.test_json:
        test_ds = json.load(open(os.path.expanduser(args.test_json)))
        print(f"Loaded {len(test_ds)} pre-materialized test batches from {args.test_json}")
    else:
        _, _, test_ds = build_train_val_test(
            NUM_BATCHES, NUM_TEST_BATCHES, TRAIN_MICRO_BATCH_SIZE, TRAIN_FRACTION,
            NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR, source=args.source,
        )

    sampler = sampler_lib.Sampler(
        transformer=lora,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )
    data_label = (f"test_json={os.path.basename(args.test_json)}"
                  if args.test_json else f"source={args.source}")
    n, t, acc, pacc, facc, flags = evaluate(test_ds, sampler, eos_tokens, **GENERATION_CONFIGS[args.preset])
    lo, hi = bootstrap_ci(flags)
    print(f"\nFINAL [{tag}] preset={args.preset} {data_label} n_test={t}: "
          f"correct={n}/{t}  acc={acc:.2f}%  (95% CI {lo:.2f}-{hi:.2f})  "
          f"partial={pacc:.2f}%  format={facc:.2f}%")

    dump_path = args.dump or f"eval_{tag.replace(' ', '_')}.json"
    with open(dump_path, "w") as fh:
        json.dump({"tag": tag, "preset": args.preset, "data": data_label,
                   "n": t, "correct": n, "acc": acc, "ci95": [lo, hi],
                   "partial": pacc, "format": facc, "per_example_correct": flags},
                  fh, indent=2)
    print(f"Wrote {dump_path}")


if __name__ == "__main__":
    main()
