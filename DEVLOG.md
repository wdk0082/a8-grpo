# Dev log

Running log of changes to the Part I report and supporting docs. Newest first.

## 2026-06-18

### Rename
- `report.tex` / `report.pdf` → **`report_part1.tex`** / **`report_part1.pdf`** (old `report.pdf` removed); recompiled with `tectonic`.

### Self-review round 1 — 11 fixes
Read-through of Part I against the data; numbers = review item. All in `report_part1.tex` unless noted.

**Substantive**
1. **I.3 motivation** — "lowers advantage-estimate variance" → "lowers the variance of the **gradient estimate** $\hat g$". The $O(1/K)$ result is $\mathrm{Var}(\hat g)$; the per-sample advantage variance is $1-1/K$ (≈flat), so the old wording mis-named the quantity and contradicted I.4 Q1.
2. **I.3(d)** — was "the $\exp(\log\text{-ratio})$ KL blow-up that destroys the small-$K$ runs" (conflated two failure modes). Now distinguishes: **R0** = reward-hacking / over-optimisation collapse; **R1** = the KL-estimator instability (only with $\beta>0$).
3. **"Baseline" disambiguated** — `bdbugenj` (I.1 shared run; the grey "baseline" line in Fig 1) vs **R0 = "baseline re-run"** (I.3 controlled run: same config + dense checkpoints). Relabelled R0 in the 2×2 list and Table 1.
4. **28% vs 46.7% reconciled** — added a sentence: R0's dense checkpointing recovers the best-val that `bdbugenj`'s `MAX_TO_KEEP=4` pruning hid; also tagged I.1's 28.1% as the *best surviving* (pruned) checkpoint.

**Clarity / notation**
5. **K ≡ G stated once** — "group size $G$ (the $K$ of I.2(b) and the theory)".
6. **I.2(d) "locate" made specific** — "Tunix's RL learner" → "Tunix's `GRPOLearner` (module `tunix.rl`), in its clipped policy-gradient loss". *(Still confirm the exact function against the installed Tunix before the viva.)*
7. **TFDS→HF split justified** — Table 1 caption now notes HF and TFDS differ only in seed-42 ordering, so at full $n{=}1319$ the I.1→I.3 source switch is immaterial.

**Minor / adjacent**
8. **`i3_results.md`** (not the report) — R1 peak KL "~21" → **"~32.5"** in two places (β-paradox + F3 frontier), verified from `tb_scalars_R1.csv` (peak 32.5 @ step 1648). Confirmed Fig 1's "peak KL≈41" annotation is correct: it's baseline `bdbugenj`'s peak (41.0), the overall max.
9. **Jointly-owned line** — "six GRPO runs R0–R5" → "seven shared GRPO runs (`bdbugenj` and R0–R5)".
10. **I.2(c)** — "becomes $0/0$" → "has no signal (an $\varepsilon$ in the denominator averts a literal $0/0$)".
11. **I.3(d)** — "DAPO … small-$K$/long-output regime" → "DAPO … for long-output RL" (DAPO is not small-$K$).

### Layout
- Shrank Fig 1 (`F1_reward_kl`) to `0.86\linewidth` to absorb the added text and keep **I.1–I.3 within the 3-page limit** (verified: I.4 still starts on p.4). Clean build: 0 overfull, 0 missing-character.

### Verified correct (no change needed)
- Every Table 1 number and paired-CI claim matches the data; I.4's boxed results spot-check as correct.

### Still open (flagged, not done here)
- **GitLab**: port the repo to GitLab and swap the report link (submission requires GitLab, not GitHub).
- **Part II** (adaptive planning, ≤2 pp) not yet written — the final submission is a single PDF covering Parts I *and* II.
- Confirm the exact Tunix file/function for the PPO clip (item 6) before the viva.
