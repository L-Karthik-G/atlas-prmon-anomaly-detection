#!/usr/bin/env python3
import psutil, time, json, subprocess, sys, os

outfile = sys.argv[1]
cmd = sys.argv[2:]

proc = subprocess.Popen(cmd)
records = []
print(f"Monitoring PID {proc.pid}...")
while proc.poll() is None:
    try:
        p = psutil.Process(proc.pid)
        children = p.children(recursive=True)
        all_procs = [p] + children
        rss = sum(cp.memory_info().rss for cp in all_procs) / 1e6
        vms = sum(cp.memory_info().vms for cp in all_procs) / 1e6
        nprocs = len(all_procs)
        records.append({"time": time.time(), "rss_mb": rss, "vms_mb": vms, "nprocs": nprocs})
        time.sleep(1)
    except:
        break
proc.wait()
with open(outfile, 'w') as f:
    json.dump(records, f, indent=2)
print(f"Saved {len(records)} records to {outfile}")
