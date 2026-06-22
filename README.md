# Cambridge A8 — Multi-Agent Systems & Agentic AI: GRPO on GSM8K (team chmawa)

GRPO LoRA finetuning of `google/gemma-3-1b-it` on GSM8K (TPU v6e-1, Tunix/JAX), coursework Parts I.1–I.4.
Team **chmawa** (sm3035, bc654, zw499).

## Headline result (I.3)
A **β×G 2×2** (KL-penalty weight × group size) over 6 full GRPO runs. **Group size G=8 is the winning
intervention**: R3b (G=8, β=0) reaches **57.2%** on the full GSM8K test (n=1319) vs base **47.4%** —
**+9.9 pp, paired bootstrap CI [+7.1, +12.7]**. The training collapse is confined to the **G=2 / β>0**
corner (R0, R1). Write-up: **`i3_results.md`**; figures **`figures/F4_2x2_GxB.pdf`**, **`figures/F2_accuracy.pdf`**.

## Layout
| Path | What |
|---|---|
| `report/` | **the deliverable** — `report_part1.pdf` (Parts I.1–I.4) + its self-contained LaTeX source and figures |
| `i3_results.md` · `i3_runs_plan.md` | write-ups: the I.3 controlled study (results + diagnosis) · the I.2–I.3 runs/metrics plan |
| `analysis/` | analysis tooling, **run from the repo root**: `plot_report.py` (figures) · `paired_ci.py` (paired bootstrap CIs) · `export_wandb.py` (W&B→CSV) |
| `training_logs/` | per-run W&B scalar exports `tb_scalars_*.csv` (source for the training-curve figures) |
| `figures/` | report figures F1–F4, S1–S3 (PNG+PDF) + `F2_accuracy_table.tex` |
| `evals/` | per-checkpoint eval dumps (per-example correctness + bootstrap CI) — n=64 and n=1319 |
| `scripts/` | VM orchestration: `run_sweep_n1319.sh` (full-n eval sweep) · `pull_all_ckpts.sh` (checkpoint-archive pull) |
| `tpu-2026/` · `tpu-2026_our_changes.patch` | training code + our changes as one diff (see *Upstream / our contribution* below) |
| `vm_snapshot/` | archive pulled off the TPU VM before deletion: patched code, per-run `launch_*.sh`, full `train.log`s |
| `ckpts_archive/MANIFEST.md` | manifest of the 210 LoRA checkpoints — **the checkpoints themselves are local-only, not in git** (46 GB) |

## Upstream / our contribution
`tpu-2026/` is a **detached** clone of [`borisbolliet/tpu-2026`](https://github.com/borisbolliet/tpu-2026)
@ `77c5a67` (its `.git` was removed — we don't track the reference repo). Our changes are applied in
`tpu-2026/scripts/` and also captured as a single diff in **`tpu-2026_our_changes.patch`**
(reproduce: clone upstream @ `77c5a67`, then `git apply tpu-2026_our_changes.patch`):
`config/data/evaluate/rewards/train.py` modified (+205/−22) and `prepare_test_tfds.py` added.

## W&B (project `a8-grpo`)
R0 `8c2785ut` · R1 `9p3kota8` · R2 `xt3d5b0e` · R4 `082vyug5` · R3b `sgjawrsw` · R5 `qu0uammy`
