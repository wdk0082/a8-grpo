# Part I.1 — Reproducing the baseline (results reference)

Team **chmawa** (members sm3035, bc654, zw499). The single GRPO baseline run is
**team-shared** (launched by sm3035); experiments are jointly owned — state this
on the report title page. The eval numbers below **match the rest of the team
exactly** (same TFDS split).

## I.1(i) — the run
- **Repo / commit:** `borisbolliet/tpu-2026` @ `77c5a67034a7daa869066d1955b06cdca49f9887`
- **Config:** unmodified `scripts/config.py` (RANK 64, α 64, G=2, μ=1, β=0.08,
  ε=0.2, LR 3e-6, warmup 10%, cosine decay, MAX_GRAD_NORM 0.1, MAX_PROMPT 256,
  TOTAL_GENERATION_STEPS 768).
- **Hardware:** Google Cloud TPU **v6e-1** (single chip), Tunix/JAX, LoRA.
- **Steps:** **3364** = MAX_STEPS (full run). **Wall-clock:** 12:52:51 → 17:35:40
  UTC (2026-06-08) ≈ **4 h 43 m**.
- **Logs:** W&B `a8-grpo` / run `bdbugenj`
  <https://wandb.ai/sichengma0514-university-of-cambridge/a8-grpo/runs/bdbugenj> ;
  TensorBoard scalars (45 tags) → `tb_scalars_full.csv`.

## I.1(ii) — held-out GSM8K accuracy
Eval: **`scripts/evaluate.py`** (patched), greedy/deterministic, on the **first
NUM_TEST_BATCHES = 64** GSM8K **test** problems (shuffle seed 42) — the **TFDS**
split, the same 64 for every model and identical to the rest of the team.
(TFDS can't be imported alongside jax here, so the split is pre-materialized
jax-free by `prepare_test_tfds.py` into `~/gsm8k_test_tfds.json` and loaded via
`evaluate.py --test-json`.) Accuracy = exact numeric match of the extracted answer.

| Model | exact-match | within ±10% | format |
|---|---|---|---|
| Base `gemma-3-1b-it` | **51.56%** (33/64) | 53.12% | 6.25% |
| Finetuned step 2000 | **28.12%** (18/64) | 29.69% | 35.94% |
| Finetuned step 3000 | **6.25%** (4/64) | 7.81% | 12.50% |
| Finetuned step 3364 (resulting) | **3.12%** (2/64) | 6.25% | 12.50% |

Commands:
`python evaluate.py --test-json ~/gsm8k_test_tfds.json --preset greedy` (base);
add `--step {2000,3000,3364} --ckpt-dir ~/ckpts_backup/actor` for the checkpoints.
n=64 ⇒ ~±12% 95% CI; gaps (52→28→6→3) exceed it. For I.3 add a 2nd seed / bootstrap CI / larger n.

> Note: a `--source hf` path also exists (loads GSM8K from HF `datasets`), but HF
> and TFDS order the test set differently (only 3/64 overlap at seed 42), so it
> gives a *different* split and is NOT used for the team-consistent table above.

## I.1(iii) — training curves
- `i1_reward_kl.png` — mean reward `rewards/train/mean` and `actor/train/kl` vs step.
- `i1_diagnostics.png` — completion length and held-out eval reward vs step.
- `i1_eval_accuracy.png` — the I.1(ii) accuracy/format bars.

Mean reward + held-out eval reward **peak ≈ step 450** then decline; **KL** rises
0.16→~0.5 with late spikes (~27–40); **completion length** inflates (~200→~500 by
step ~1750) then collapses (~120). MAX_TO_KEEP=4 pruned every checkpoint before
2000; best surviving is step 2000.

## Honest diagnosis
The default config **over-optimizes**: GRPO pushes the format/shaping reward up
(peaks 35.9% at step 2000) while *true* correctness collapses (52%→28%→6%→3%).
Reward-hacking + policy drift (rising KL, length blow-up). This is the failure
mode I.2(c)/I.3(c) target and the motivation for the I.3 fix (length penalty,
higher β, lower LR, or early stopping near step ~450).

## Baseline patches (didn't run as-shipped on our hardware)
1. **`run_tmux.sh`** hardcodes Boris's home path (`/home/boris_bolliet_cmbagent_community/...`); ran under a generic tmux session.
2. **`config.py`** W&B entity defaults to a third party (`milindsarkaryt-iiser-mohali`); logged to the team's own W&B `a8-grpo`.
3. **`scripts/evaluate.py` never restored a checkpoint** — it builds a fresh B=0
   LoRA (≡ base), so as-shipped it can only score *base*. **Patched in place:**
   added `--step` / `--ckpt-dir` Orbax restore
   (`CheckpointManager.maybe_restore(..., restore_only_lora_params=True)`, as in
   `chat.py`); default `--step -1` preserves the original base-only behaviour.
4. **TFDS/protobuf:** a fresh venv pulls `protobuf 7.34.1`, which removed
   `FieldDescriptor.label` that TFDS 4.9.9's gsm8k builder needs — it crashes once
   jax/tunix have imported the proto backend. Fix: `prepare_test_tfds.py`
   materializes the **exact** shipped TFDS+grain test split (seed 42, first 64)
   in a jax-free process → JSON; `evaluate.py --test-json` loads it. (A `--source
   hf` fallback was also added to `data.py`, but it yields a *different* split.)

## Artifacts
- Local: `tb_scalars_full.csv`, `i1_reward_kl.png`, `i1_diagnostics.png`, `i1_eval_accuracy.png`
- Patched code: `tpu-2026/scripts/{evaluate.py, data.py, prepare_test_tfds.py}` (also on the VM)
- Eval split: VM `~/gsm8k_test_tfds.json`; checkpoints backed up → VM `~/ckpts_backup/actor` (2000/2500/3000/3364)
