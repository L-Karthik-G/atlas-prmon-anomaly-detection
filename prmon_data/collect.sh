#!/bin/bash
PRMON=/home/karthik_g/prmon/build/package/prmon
BURNER=/home/karthik_g/prmon/build/package/tests/mem-burner
cd /home/karthik_g/prmon_data

run_capture() {
    local outfile=$1
    shift
    $BURNER "$@" &
    local BPID=$!
    echo ">>> Monitoring PID $BPID -> $outfile"
    $PRMON --pid $BPID --interval 1 --output $outfile
    wait $BPID 2>/dev/null
    echo ">>> Done: $outfile"
}

run_capture normal_run1.json  --malloc 200 --writef 0.5 --procs 2  --sleep 60
run_capture normal_run2.json  --malloc 200 --writef 0.5 --procs 2  --sleep 60
run_capture normal_run3.json  --malloc 200 --writef 0.5 --procs 2  --sleep 60
run_capture anomaly_highmem.json   --malloc 800 --writef 0.9 --procs 2  --sleep 60
run_capture anomaly_highprocs.json --malloc 200 --writef 0.5 --procs 16 --sleep 60
run_capture anomaly_combined.json  --malloc 800 --writef 0.9 --procs 16 --sleep 60

echo "All done!"
ls -la /home/karthik_g/prmon_data/
