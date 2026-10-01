#!/bin/bash
# detach.sh NAME cmd...: runs cmd in its own session (setsid), out of reach of the harness. On 10-01 a chain started
# from a harness background command was killed with it at the command's 30-minute limit, its device job included
# (B43). Writes $RUN_DIR/NAME.log and, as its last action, $RUN_DIR/NAME.exit (the exit code). Wait with wait_job.
: "${RUN_DIR:?set RUN_DIR}"
name=$1; shift
rm -f "$RUN_DIR/$name.exit" "$RUN_DIR/$name.exit.tmp"
setsid bash -c '"$@" > "$0.log" 2>&1; rc=$?; echo $rc > "$0.exit.tmp" && mv "$0.exit.tmp" "$0.exit"' \
  "$RUN_DIR/$name" "$@" < /dev/null > /dev/null 2>&1 &
echo "$(date '+%F %T') detached $name (log $RUN_DIR/$name.log)"
