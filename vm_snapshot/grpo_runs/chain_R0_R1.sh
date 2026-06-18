#!/bin/bash
log=~/grpo_runs/chain.log
R0_PID=352446
echo "=== supervisor v2 up $(date -u): waiting for R0 PID $R0_PID to exit ===" >> "$log"
while kill -0 "$R0_PID" 2>/dev/null; do sleep 60; done
echo "$(date -u): R0 PID gone; checking completion" >> "$log"
sleep 15
if grep -q 'Training finished.' ~/grpo_runs/R0/train.log 2>/dev/null; then
  echo "$(date -u): R0 finished cleanly -> launching R1 (BETA=0.3)" >> "$log"
  tmux kill-session -t R1 2>/dev/null
  tmux new-session -d -s R1 "bash ~/grpo_runs/R1/launch_R1.sh"
  sleep 8
  if pgrep -u "$(id -un)" -f 'python -u train.py' >/dev/null 2>&1; then
    echo "$(date -u): R1 launched OK (python alive)" >> "$log"
  else
    echo "$(date -u): WARN R1 python not detected yet" >> "$log"
  fi
else
  echo "$(date -u): R0 did NOT print 'Training finished.' -> NOT launching R1; investigate" >> "$log"
fi
