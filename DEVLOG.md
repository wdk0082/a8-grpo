# Dev log

Running log of changes to the Part I report and supporting docs. Newest first.

## 2026-06-20

### I.4.1 Q1(i) — explicit steps + conditioning convention
- Per request, expanded only Q1(i) (both `report/report_part1.tex` and `i4_theory_revised.tex`): added the skipped intermediate steps for $\mathbb{E}[X_i]=h_i\mathbb{E}[\hat A_i]=0$ and $\mathbb{E}[\hat g]=\tfrac1K\sum\mathbb{E}[X_i]=0$, and made the explanation state the convention $\mathbb{E}[\,\cdot\,]\equiv\mathbb{E}[\,\cdot\mid a_1,\dots,a_K]$ (all expectations conditional on the fixed actions). The $\mathbb{E}[\hat A_i]$ line and (ii)/(iii) left untouched. Clean build, 9 pp.

### I.4.1 Q1(iii) replaced with the user-supplied trace-based version
- Swapped the Q1(iii) answer (effective sample size) in **both** `report/report_part1.tex` and `i4_theory_revised.tex` for the version the user supplied verbatim ("use exactly this"). Recompiled clean — still **9 pp**; I.1–I.3 untouched on pp.1–3; the new (iii) sits in I.4.1 on pp.4–5. No overfull boxes / missing chars.
- **What changed vs the previous (directional) (iii)**: it now adopts the **lecture scalar-variance convention** $\operatorname{Var}_{\mathrm{tr}}(Z):=\operatorname{tr}(\operatorname{Cov}(Z))$, giving a single *scalar* $K_{\mathrm{eff}}=K(K-1)\lVert h_1\rVert^2/\sum_i\lVert h_i-\bar h\rVert^2$ in place of the per-direction $K_{\mathrm{eff}}(u)$. Retains the full matrix $\operatorname{Cov}(\hat g)=K^{-2}\sum_i(h_i-\bar h)(h_i-\bar h)^\top$, the iid sanity-check ($\operatorname{Var}_{\mathrm{tr,iid}}=\tfrac1K\operatorname{Var}_{\mathrm{tr}}(X_1)$), the $\Theta(K)$ growth result, the collinear $h_i=c_iv$ case, and the $h_i=h\Rightarrow K_{\mathrm{eff}}=\infty$ degenerate case.
- This **reverses** the 2026-06-19 note below ("drops the redundant trace-based K_eff, keeps the directional one"): the user's newest version deliberately uses the trace/scalar form to match the lecture convention.
- The `Let $\bar h=\dots$` opening at the top of I.4.1 is kept; the duplicate "Let $\bar h$" that led the pasted block was dropped to avoid repetition.

## 2026-06-19 (later)

### Self-contained `report/` folder
- Moved `report_part1.{tex,pdf}` into **`report/`** and copied the 3 figures it uses (`F1_reward_kl.pdf`, `F4_2x2_GxB.pdf`, `S3_group_degeneracy.pdf`) in with their report names; dropped `\graphicspath{{figures/}}` so they resolve co-located. `report/` rebuilds standalone (`tectonic report_part1.tex`). The `report/` figures are snapshots of `figures/` — re-copy if the plots are regenerated.

### Dropped the first-time baseline run (`bdbugenj`); I.1 now uses R0
- Verified R0 satisfies **I.1(i–iii)** under the default config (β=0.08, G=2; confirmed from `launch_R0.sh` — no knob overrides). Rewrote I.1 around R0 (W&B `8c2785ut`, ~5 h 05 m, 3364 steps).
- I.1(ii) reports **both** the `config.py`-default $n{=}64$ (base 45.3%, R0 42.2%→0.0%) and the full $n{=}1319$ for completeness (base 47.4%, R0 46.7%→6.2%), HF split / seed 42; **I.3 standardises on the full $n{=}1319$**. TFDS / the 51.6% number are dropped (they lived only in `bdbugenj`); R0 was evaluated on HF, so n=64 base is 45.3%.
- Tidy-up: trimmed baseline-patch (4) to just the HF-fallback justification (dropped the unused `prepare_test_tfds.py` / TFDS-materialisation clause) and removed the now-stale Table 1 caption note about an "I.1→I.3 source switch" (both are HF now). Only one TFDS mention remains (patch 4).
- Added baseline-patch (5): raised checkpoint retention (`SAVE_INTERVAL` 500→100, `MAX_TO_KEEP` 4→40) — persistence, not training; it's what lets us report best-val (the pruning problem `bdbugenj` had).
- **Regenerated F1 without the `bdbugenj` line** (now R0–R5; peak-KL annotation auto-updated 41→≈33 = R1's max).
- Removed the now-moot "baseline re-run" labels and the I.1↔I.3 28%-vs-47% reconciliation. `bdbugenj` no longer appears in the report. Clean build, 9 pp (I.1–I.3 on pp.1–3).

## 2026-06-19

### I.4 swapped to the revised version; report is now a single self-contained `.tex`
- Replaced the original I.4 with **`i4_theory_revised.tex`** and **inlined** it into `report_part1.tex` (no more `\input` — one self-contained file). Added the top-level `\section*{I.4 …}` heading (the revised file starts directly at subsection I.4.1). Clean build; still **9 pp** (I.1–I.3 on 1–3, revised I.4 on 4–9).
- `i4_theory.tex` (original) and `i4_theory_revised.tex` remain in the repo (no longer referenced by the report; kept for history).
- Gotcha logged: a first build mangled `\end{document}` → `␛nd{document}` because **zsh `echo` turns `\e` into an ESC byte (0x1B)**; rebuilt with `printf '%s'` (no escape processing).

### Revised vs original I.4 — the revised is better
- **Structure**: numbered subsections I.4.1/2/3 (vs one section + "1./2./3." paragraphs).
- **Concision** (I.4 penalises verbosity): ~1.2 KB shorter; drops the redundant trace-based `K_eff` (keeps the directional one, which is what the question asks for).
- **More rigorous**: Q1(ii) cleaner `Cov` via bilinearity + sharper "sign" statement; Q1(iii) cleaner `K_eff(u)` ratio + iid sanity-check + Θ(K); Q2(i) adds the small-η first-order expansion (shows η as a step-size); Q2(iii) explicit clip case-form; **Q3(iii-b)** the big upgrade — proves a deterministic baseline shift is unbiased (`E_p[s_θ]=0`), explains why weights alone don't fix group normalisation, and adds the **literally-unbiased leave-one-out Horvitz–Thompson baseline**; Q3(iv) matrix-form variance difference + explicit "no universal ordering".
- **No correctness regressions** (spot-checked the new derivations).

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
