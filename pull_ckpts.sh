#!/usr/bin/env bash
# Pull the 12 reported checkpoints via tar-stream over SSH: one continuous stream per
# checkpoint (no per-file round-trips, no VM temp space). scp --recurse was ~5 MB/min on
# Orbax's many tiny shards; this is raw-bandwidth. Validates each tar; skips complete dirs.
set -u
SSH=(gcloud alpha compute tpus tpu-vm ssh chmawa --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap
     --ssh-flag="-o ConnectTimeout=20" --ssh-flag="-o ServerAliveInterval=15" --ssh-flag="-o ServerAliveCountMax=3")
DEST=/Users/wdk0082/Projects/cambridge/a8-cw/ckpts_archive
REMOTE=/home/ext_wdk0082_gmail_com/grpo_runs
mkdir -p "$DEST"; rm -f "$DEST"/.*.tar
PAIRS="R0:700 R0:3364 R1:600 R1:3364 R2:400 R2:3364 R4:1300 R4:3364 R3b:1000 R3b:3364 R5:1300 R5:3364"
ok=0; fail=0
for p in $PAIRS; do
  R="${p%%:*}"; S="${p##*:}"
  if [ -d "$DEST/$R/$S" ] && [ "$(du -sm "$DEST/$R/$S" 2>/dev/null | cut -f1)" -gt 100 ]; then
    echo "=== $R/$S SKIP (already have $(du -sh "$DEST/$R/$S" | cut -f1)) ==="; ok=$((ok+1)); continue
  fi
  echo "=== $R/$S $(date -u +%T)Z ==="
  mkdir -p "$DEST/$R"; rm -rf "$DEST/$R/$S"
  TAR="$DEST/.${R}_${S}.tar"
  "${SSH[@]}" --command="tar cf - -C $REMOTE/$R/ckpts/actor $S" > "$TAR" 2>/dev/null
  if tar tf "$TAR" >/dev/null 2>&1; then
    tar xf "$TAR" -C "$DEST/$R" && rm -f "$TAR" && echo "  OK $(du -sh "$DEST/$R/$S" 2>/dev/null | cut -f1)" && ok=$((ok+1))
  else
    echo "  FAILED (bad/empty tar) $R/$S"; rm -f "$TAR"; fail=$((fail+1))
  fi
done
echo "=== done: ok=$ok fail=$fail, total $(du -sh "$DEST" 2>/dev/null | cut -f1) ==="
echo "RESULT: CKPT_TAR_PULL_DONE ok=$ok fail=$fail"
