"""Read the shared GRPO TensorBoard event file (CPU only; no TPU init) and
dump all scalar series to CSV + print a per-tag summary (handy for I.1(iii))."""
import csv, glob, os
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGDIR = "/tmp/content/tmp/tensorboard/grpo"
OUT = os.path.expanduser("~/tb_scalars.csv")

files = sorted(glob.glob(os.path.join(LOGDIR, "events.out.tfevents.*")))
print("event files:", [os.path.basename(f) for f in files])

ea = EventAccumulator(LOGDIR, size_guidance={"scalars": 0})  # 0 => keep all points
ea.Reload()
tags = ea.Tags().get("scalars", [])
print("SCALAR TAGS:", tags)

rows, bytag = [], defaultdict(list)
for tag in tags:
    for s in ea.Scalars(tag):
        rows.append((tag, s.step, s.value))
        bytag[tag].append((s.step, s.value))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "step", "value"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")

print("--- per-tag summary (n, step range, last value) ---")
for tag in tags:
    pts = sorted(bytag[tag])
    print(f"  {tag:40s} n={len(pts):4d}  steps[{pts[0][0]}..{pts[-1][0]}]  last={pts[-1][1]:.5f}")
