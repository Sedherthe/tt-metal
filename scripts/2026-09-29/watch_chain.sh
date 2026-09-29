#!/bin/bash
# Emits a chain's job lines (written by jobs.sh from sentinel files) as they land; for the named jobs, a progress line
# every 10 minutes while they run (log size, last line), so a hang is not silent. Exits when the chain's own sentinel
# appears. Usage: watch_chain.sh CHAIN_NAME [JOB ...]
R=/home/user/data/cosyvoice2_runs/0929
chain=$1; shift
seen=0; last=$(date +%s)
while :; do
  if [ -f $R/$chain.log ]; then
    total=$(wc -l < $R/$chain.log)
    if [ "$total" -gt "$seen" ]; then
      sed -n "$((seen + 1)),${total}p" $R/$chain.log | grep -E "started|exited|TIMEOUT|not starting|did not|refusing|binaries|passed|failed|FAILED|ERROR|identical|diff|Distinct|done|Traceback"
      seen=$total
    fi
  fi
  if [ -f $R/$chain.exit ]; then echo "$chain sentinel: $(cat $R/$chain.exit)"; break; fi
  now=$(date +%s)
  if [ $((now - last)) -ge 600 ]; then
    for job in "$@"; do
      [ -f $R/$job.log ] && [ ! -f $R/$job.exit ] && \
        echo "progress $job: $(stat -c %s $R/$job.log) bytes; last: $(tail -c 160 $R/$job.log | tr '\n\r' '  ' | tr -s ' ')"
    done
    last=$now
  fi
  sleep 10
done
