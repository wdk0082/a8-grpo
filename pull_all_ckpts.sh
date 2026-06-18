#!/usr/bin/env bash
# Pull ALL checkpoints (every saved step, all 6 runs) via tar-stream. Enumerates steps
# remotely per run; skips any already-complete local dir (so it resumes / dedups the
# 12 best+final we already pulled). ~44 GB left at ~12.6 MB/s ≈ ~1 h.
set -u
SSH=(gcloud alpha compute tpus tpu-vm ssh chmawa --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap
     --ssh-flag="-o ConnectTimeout=20" --ssh-flag="-o ServerAliveInterval=15" --ssh-flag="-o ServerAliveCountMax=3")
DEST=/Users/wdk0082/Projects/cambridge/a8-cw/ckpts_archive
REMOTE=/home/ext_wdk0082_gmail_com/grpo_runs
mkdir -p "$DEST"; rm -f "$DEST"/.*.tar
RUNS="R0 R1 R2 R4 R3b R5"
ok=0; skip=0; fail=0
for R in $RUNS; do
  echo "### enumerating $R $(date -u +%T)Z ###"
  STEPS=$("${SSH[@]}" --command="find $REMOTE/$R/ckpts/actor -maxdepth 1 -regextype posix-extended -regex '.*/[0-9]+' 2>/dev/null | grep -oE '[0-9]+\$' | sort -n" 2>/dev/null)
  for S in $STEPS; do
    if [ -d "$DEST/$R/$S" ] && [ "$(du -sm "$DEST/$R/$S" 2>/dev/null | cut -f1)" -gt 100 ]; then
      skip=$((skip+1)); continue
    fi
    mkdir -p "$DEST/$R"; rm -rf "$DEST/$R/$S"
    TAR="$DEST/.${R}_${S}.tar"
    "${SSH[@]}" --command="tar cf - -C $REMOTE/$R/ckpts/actor $S" > "$TAR" 2>/dev/null
    if tar tf "$TAR" >/dev/null 2>&1; then
      tar xf "$TAR" -C "$DEST/$R" && rm -f "$TAR" && ok=$((ok+1))
    else
      echo "  FAILED $R/$S"; rm -f "$TAR"; fail=$((fail+1))
    fi
  done
  echo "### $R done: cumulative ok=$ok skip=$skip fail=$fail, total $(du -sh "$DEST" 2>/dev/null | cut -f1) ###"
done
echo "=== ALL DONE: ok=$ok skip=$skip fail=$fail, total $(du -sh "$DEST" 2>/dev/null | cut -f1) ==="
echo "RESULT: ALL_CKPT_PULL_DONE ok=$ok skip=$skip fail=$fail"
