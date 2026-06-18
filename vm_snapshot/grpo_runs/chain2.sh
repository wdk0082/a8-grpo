#!/bin/bash
log=~/grpo_runs/chain.log
wait_done(){ while kill -0 "$1" 2>/dev/null; do sleep 60; done; sleep 15; grep -q 'Training finished.' "$2" 2>/dev/null; }
launch(){ tmux kill-session -t "$1" 2>/dev/null; tmux new-session -d -s "$1" "bash ~/grpo_runs/$1/launch_$1.sh"; sleep 12; pgrep -u "$(id -un)" -f 'python -u train.py' | head -1; }
echo "=== chain2 up $(date -u): R1(375586) -> R2 -> R4 ===" >> "$log"
if wait_done 375586 ~/grpo_runs/R1/train.log; then
  echo "$(date -u): R1 finished cleanly -> launching R2" >> "$log"
  P=$(launch R2); echo "$(date -u): R2 pid=$P" >> "$log"
  if [ -n "$P" ] && wait_done "$P" ~/grpo_runs/R2/train.log; then
    echo "$(date -u): R2 finished cleanly -> launching R4" >> "$log"
    P=$(launch R4); echo "$(date -u): R4 pid=$P" >> "$log"
  else echo "$(date -u): R2 not clean -> NOT launching R4; investigate" >> "$log"; fi
else echo "$(date -u): R1 not clean -> NOT launching R2; investigate" >> "$log"; fi
echo "$(date -u): chain2 finished" >> "$log"
