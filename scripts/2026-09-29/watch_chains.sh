#!/bin/bash
# Emits the chains' job lines (written by jobs.sh from sentinel files) as they appear; while a device job runs, a
# progress line every 10 minutes (log size and last line), so a hang is not silent. Exits on phase_suite.exit.
# Usage: watch_chains.sh [ref_lines_already_seen] [suite_lines_already_seen]
R=/home/user/data/cosyvoice2_runs/0929
declare -A n=([phase_ref]=${1:-0} [phase_suite]=${2:-0})
last_progress=$(date +%s)
while :; do
  for f in phase_ref phase_suite; do
    [ -f $R/$f.log ] || continue
    total=$(wc -l < $R/$f.log)
    if [ "$total" -gt "${n[$f]}" ]; then
      sed -n "$(( ${n[$f]} + 1 )),${total}p" $R/$f.log \
        | grep -E "started|exited|TIMEOUT|not starting|did not|refusing|binaries|passed|failed|FAILED|ERROR|done"
      n[$f]=$total
    fi
  done
  if [ -f $R/phase_suite.exit ]; then echo "phase_suite sentinel: $(cat $R/phase_suite.exit)"; break; fi
  now=$(date +%s)
  if [ $((now - last_progress)) -ge 600 ]; then
    for job in suite perf; do
      if [ -f $R/$job.log ] && [ ! -f $R/$job.exit ]; then
        echo "progress $job: $(stat -c %s $R/$job.log) bytes; last: $(tail -c 160 $R/$job.log | tr '\n\r' '  ' | tr -s ' ')"
      fi
    done
    last_progress=$now
  fi
  sleep 10
done
