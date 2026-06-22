#!/usr/bin/env python3
"""Generate the Part I.1-I.3 report figures from exported scalar CSVs + eval JSONs.

Scalar CSV  = long format [tag, step, value], one file per run
              (training_logs/tb_scalars_<run>.csv, produced by export_wandb.py).
Eval JSON   = evaluate.py --dump output (per-example correctness + bootstrap CI).

Figures -> ./figures/ as PNG + PDF (PDF for \\includegraphics in the LaTeX report):
  F1  mean reward + reference-KL vs step, runs overlaid   [I.1(iii), I.3(b)]  MAIN
  F2  held-out accuracy bars + LaTeX table (eval JSONs)    [I.1(ii), I.3(a)]  MAIN
  F3  KL vs held-out eval-reward (over-optimisation)       [I.3(c)]           MAIN
  S1  reward decomposition (4 terms)                       [I.2(c)]           SUPP
  S2  completion length + entropy proxy vs step            [I.3(c)]           SUPP

Examples:
  # run from the repo root:
  ~/ENTER/bin/python analysis/plot_report.py                 # R0 only (default)
  python analysis/plot_report.py --runs R0=training_logs/tb_scalars_R0.csv R1=training_logs/tb_scalars_R1.csv R3b=training_logs/tb_scalars_R3b.csv
  python analysis/plot_report.py --runs ... --evals Base=evals/eval_BASE.json "R0@best=evals/eval_R0_best.json"
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_REWARD = "rewards/train/mean"
T_KL = "actor/train/kl"
E_REWARD = "rewards/eval/mean"
LENGTH = "completions/train/mean_length"
PERP = "actor/train/perplexity"
TERMS = [
    ("rewards/train/match_format_exactly", "format exact (shaping)"),
    ("rewards/train/match_format_approximately", "format approx (shaping)"),
    ("rewards/train/check_answer", "check_answer (correctness)"),
    ("rewards/train/check_numbers", "check_numbers (correctness)"),
]
COLORS = {"baseline": "0.45", "R0": "black", "R1": "tab:blue",
          "R2": "tab:green", "R3": "tab:orange", "R3b": "tab:purple", "R4": "tab:red",
          "R5": "tab:brown"}

PARAMS = {"R0": "G=2, β=.08", "R1": "G=2, β=.3", "R2": "G=2, +len",
          "R4": "G=2, β=0", "R3b": "G=8, β=0", "R5": "G=8, β=.08"}


def label_for(name):
    return f"{name} ({PARAMS[name]})" if name in PARAMS else name


def color_for(name, i):
    return COLORS.get(name, f"C{i % 10}")


def load(path):
    df = pd.read_csv(path)
    return df[["tag", "step", "value"]]


def series(df, tag):
    s = df[df["tag"] == tag].sort_values("step").drop_duplicates("step", keep="last")
    return s["step"].to_numpy(float), s["value"].to_numpy(float)


def ema(y, alpha=0.08):
    if len(y) == 0:
        return y
    out = np.empty(len(y))
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def save(fig, outdir, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir}/{name}.png/.pdf")


def f1(runs, outdir):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    allkl = []
    for i, (n, df) in enumerate(runs.items()):
        c = color_for(n, i)
        x, y = series(df, T_REWARD)
        if len(x):
            a1.plot(x, y, color=c, alpha=.16, lw=.8)
            a1.plot(x, ema(y), color=c, lw=1.7, label=label_for(n))
        xk, yk = series(df, T_KL)
        if len(xk):
            a2.plot(xk, yk, color=c, alpha=.16, lw=.8)
            a2.plot(xk, ema(yk), color=c, lw=1.7, label=label_for(n))
            allkl.append(yk)
    a1.set_ylabel(r"mean reward $\bar r$")
    a1.grid(alpha=.3)
    a1.legend(fontsize=8)
    a1.set_title("Training reward and reference-KL  (faint = raw, bold = EMA)")
    a2.set_ylabel(r"KL$(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$")
    a2.set_xlabel("GRPO step")
    a2.grid(alpha=.3)
    if allkl:
        cat = np.concatenate(allkl)
        hi, mx = np.percentile(cat, 97), cat.max()
        if mx > hi * 1.3:
            a2.set_ylim(min(0, cat.min()), hi * 1.15)
            a2.text(.99, .95, f"y clipped; peak KL$\\approx${mx:.0f}", transform=a2.transAxes,
                    ha="right", va="top", fontsize=7, color="0.3")
    save(fig, outdir, "F1_reward_kl")


def f3(runs, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = None
    for i, (n, df) in enumerate(runs.items()):
        sk, vk = series(df, T_KL)
        se, ve = series(df, E_REWARD)
        if len(sk) == 0 or len(se) == 0:
            continue
        m = pd.merge_asof(pd.DataFrame({"step": se, "eval": ve}).sort_values("step"),
                          pd.DataFrame({"step": sk, "kl": vk}).sort_values("step"),
                          on="step", direction="nearest")
        ax.plot(m["kl"], m["eval"], "-", color=color_for(n, i), alpha=.4, lw=1)
        sc = ax.scatter(m["kl"], m["eval"], c=m["step"], cmap="viridis", s=18, zorder=3,
                        edgecolor=color_for(n, i), linewidth=.6, label=n)
    if sc is None:
        plt.close(fig)
        print("  F3 skipped (need actor/train/kl + rewards/eval/mean)")
        return
    fig.colorbar(sc, ax=ax).set_label("GRPO step")
    ax.set_xlabel(r"KL$(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$")
    ax.set_ylabel("held-out eval reward")
    ax.set_title("Over-optimisation frontier: KL vs held-out reward")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    save(fig, outdir, "F3_kl_vs_eval")


def s1(runs, outdir):
    """Reward decomposition, one panel per run (2x2 for 4 runs). R2 also shows
    its length_penalty term where logged."""
    items = list(runs.items())
    n = len(items)
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 3.4 * nrow),
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    terms = TERMS + [("rewards/train/length_penalty", "length_penalty")]
    drew = False
    for i, (name, df) in enumerate(items):
        ax = axes[i]
        for tag, lab in terms:
            x, y = series(df, tag)
            if len(x):
                ax.plot(x, ema(y), lw=1.4, label=lab)
                drew = True
        ax.axhline(0, color="k", lw=.5, alpha=.4)
        ax.set_title(name)
        ax.grid(alpha=.3)
        if i % ncol == 0:
            ax.set_ylabel("mean reward component")
        if i >= n - ncol:
            ax.set_xlabel("GRPO step")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    if not drew:
        plt.close(fig)
        print("  S1 skipped (no reward-component tags)")
        return
    hl = {}
    for ax in axes[:n]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    fig.legend(hl.values(), hl.keys(), loc="lower center", ncol=len(hl),
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    save(fig, outdir, "S1_reward_decomp")


def s2(runs, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax2 = ax.twinx()
    drew = False
    for i, (n, df) in enumerate(runs.items()):
        c = color_for(n, i)
        x, y = series(df, LENGTH)
        if len(x):
            ax.plot(x, ema(y), color=c, lw=1.7, label=f"{n} length")
            drew = True
        xp, yp = series(df, PERP)
        if len(xp):
            ax2.plot(xp, ema(yp), color=c, lw=1.1, ls="--", alpha=.7)
    if not drew:
        plt.close(fig)
        print("  S2 skipped (no completion-length tag)")
        return
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("mean completion length (tok)")
    ax2.set_ylabel("perplexity (entropy proxy, dashed)")
    ax.set_title("Completion length (solid) + entropy proxy (dashed)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, outdir, "S2_length_entropy")


def s3(runs, outdir):
    """Group-degeneracy diagnostic (I.4 Q1) — only runs with diag/* metric_fns
    (e.g. R3b at G=8): degenerate-group fraction (left) + σ_r / adv-std (right)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax2 = ax.twinx()
    drew = False
    for n, df in runs.items():
        xd, yd = series(df, "diag/train/degenerate_frac")
        if len(xd):
            ax.plot(xd, ema(yd), color="tab:red", lw=1.8, label="degenerate frac")
            drew = True
        for tag, lab, ls in [("diag/train/reward_std_mean", "σ_r", "--"),
                             ("diag/train/adv_std", "adv std", ":")]:
            x, y = series(df, tag)
            if len(x):
                ax2.plot(x, ema(y), color="0.35", lw=1.3, ls=ls, label=lab)
    if not drew:
        plt.close(fig)
        print("  S3 skipped (no diag/* tags — only instrumented runs)")
        return
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("degenerate-group fraction", color="tab:red")
    ax.set_ylim(-0.02, 1.05)
    ax2.set_ylabel("within-group σ_r / adv std (dashed)")
    ax.grid(alpha=.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    save(fig, outdir, "S3_group_degeneracy")


def f2(evals, outdir):
    if not evals:
        print("  F2 skipped (no --evals)")
        return
    rows = []
    for label, path in evals.items():
        d = json.load(open(path))
        acc = d.get("acc")
        lo, hi = d.get("ci95", [acc, acc])
        rows.append((label, acc, lo, hi, d.get("partial", float("nan")),
                     d.get("format", float("nan")), d.get("n")))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(rows)), 4.5))
    xs = np.arange(len(rows))
    accs = [r[1] for r in rows]
    yerr = [[max(0, r[1] - r[2]) for r in rows], [max(0, r[3] - r[1]) for r in rows]]
    ax.bar(xs, accs, yerr=yerr, capsize=4, color="tab:blue", alpha=.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("held-out GSM8K accuracy (%)")
    ax.set_title("Accuracy (exact match) ± 95% bootstrap CI")
    ax.grid(alpha=.3, axis="y")
    for x, a in zip(xs, accs):
        ax.text(x, a + 1, f"{a:.1f}", ha="center", fontsize=8)
    save(fig, outdir, "F2_accuracy")
    tex = ["\\begin{tabular}{lrrr}", "\\toprule",
           "Model & $n$ & Acc.\\,(\\%) & 95\\% CI \\\\", "\\midrule"]
    for lab, acc, lo, hi, part, fmt, n in rows:
        tex.append(f"{lab} & {n} & {acc:.2f} & [{lo:.1f},\\,{hi:.1f}] \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(outdir, "F2_accuracy_table.tex"), "w") as fh:
        fh.write("\n".join(tex))
    print(f"  wrote {outdir}/F2_accuracy_table.tex")


def f4_2x2(outdir, eval_dir="evals", suffix=""):
    """G x beta 2x2 of held-out accuracy (the design completed by R5). Cells tinted
    stable(green)/collapsed(red) + annotated best-val & final acc with bootstrap CIs.
    Reads evals/eval_<run>_<tag><suffix>.json so it works at any eval-n (suffix=_n1319)."""
    CELLS = {(2, 0.0): "R4", (2, 0.08): "R0", (8, 0.0): "R3b", (8, 0.08): "R5"}

    def info(run, tag):
        p = os.path.join(eval_dir, f"eval_{run}_{tag}{suffix}.json")
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        lo, hi = d.get("ci95", [d.get("acc"), d.get("acc")])
        return d.get("acc"), lo, hi, d.get("n")

    Gs, Bs = [8, 2], [0.0, 0.08]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    nval = "?"
    for r, G in enumerate(Gs):
        for c, B in enumerate(Bs):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            run = CELLS[(G, B)]
            best, final = info(run, "best"), info(run, "final")
            if best is None or final is None:
                ax.text(0.5, 0.5, f"{run}\n(no eval{suffix})", ha="center", va="center", fontsize=12)
                continue
            nval = final[3] or nval
            collapsed = final[0] is not None and final[0] < 20.0  # below the ~47-57% cluster
            ax.set_facecolor("#f3c9c5" if collapsed else "#cde8cd")
            lines = [run,
                     f"best  {best[0]:.1f}%  [{best[1]:.0f}-{best[2]:.0f}]",
                     f"final {final[0]:.1f}%  [{final[1]:.0f}-{final[2]:.0f}]"]
            ax.text(0.5, 0.60, "\n".join(lines), ha="center", va="center",
                    fontsize=12.5, fontweight="bold")
            if collapsed:
                ax.text(0.5, 0.24, "COLLAPSE", ha="center", va="center",
                        fontsize=14, color="firebrick", fontweight="bold")
    for c, B in enumerate(Bs):
        axes[0][c].set_title(f"$\\beta$ = {B:g}", fontsize=14)
    for r, G in enumerate(Gs):
        axes[r][0].set_ylabel(f"G = {G}", fontsize=14)
    fig.suptitle(f"Held-out GSM8K accuracy:  group size $G$ × KL weight $\\beta$   (n={nval})\n"
                 "collapse only at G=2, $\\beta$>0", fontsize=13)
    save(fig, outdir, "F4_2x2_GxB")


def parse_kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", help="name=tb_scalars_<run>.csv ...")
    ap.add_argument("--evals", nargs="+", help="label=eval_<x>.json ...")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--eval-suffix", default="", help="suffix on eval JSON filenames (e.g. _n1319)")
    a = ap.parse_args()
    spec = parse_kv(a.runs) or {"R0": "training_logs/tb_scalars_R0.csv"}
    runs = {}
    for n, p in spec.items():
        if os.path.exists(p):
            runs[n] = load(p)
        else:
            print(f"  WARN missing {p}, skipping {n}")
    if not runs:
        print("No run CSVs found.")
        return
    os.makedirs(a.outdir, exist_ok=True)
    print(f"Runs: {list(runs)}  ->  {a.outdir}/")
    f1(runs, a.outdir)
    f3(runs, a.outdir)
    s1(runs, a.outdir)
    s2(runs, a.outdir)
    s3(runs, a.outdir)
    f2(parse_kv(a.evals), a.outdir)
    f4_2x2(a.outdir, suffix=a.eval_suffix)
    print("done.")


if __name__ == "__main__":
    main()
