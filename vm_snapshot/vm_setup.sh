#!/usr/bin/env bash
# One-shot environment setup for team member's own home on the chmawa TPU VM.
# Idempotent. Does NOT touch the TPU (pip/network only) so it is safe to run
# while a teammate's training is in flight.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
echo "[setup] START $(date -u +%H:%M:%S)  home=$HOME"

# 1. uv (provides a prebuilt python3.12 without sudo/PPA)
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version 2>&1)"

# 2. python3.12
uv python install 3.12
echo "[setup] python3.12 -> $(uv python find 3.12)"

# 3. clone the baseline repo into my own home (idempotent)
cd "$HOME"
if [ ! -d "$HOME/tpu-2026/.git" ]; then
  git clone https://github.com/borisbolliet/tpu-2026.git
fi
cd "$HOME/tpu-2026"
echo "[setup] repo commit -> $(git rev-parse HEAD)"
git log -1 --format='[setup] commit date/msg -> %ci  %s'

# 4. build the tunix venv (jax/tunix/flax/libtpu). ~15-20 min.
./bootstrap.sh

echo "[setup] DONE_OK $(date -u +%H:%M:%S)"
