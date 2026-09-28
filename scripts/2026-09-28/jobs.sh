# Sentinel-file job control. Source it, set RUN_DIR, then:
#   start_job NAME cmd args...   # runs in the background; its last action writes $RUN_DIR/NAME.exit (exit code)
#   wait_job NAME TIMEOUT_S      # waits for that file. On timeout it REPORTS and returns 124; it never kills.
# No process-state checks anywhere (pgrep / kill -0 / ps all misled us: zombies are never reaped here).

start_job() {
  local name=$1; shift
  local sentinel="$RUN_DIR/$name.exit"
  rm -f "$sentinel" "$sentinel.tmp"
  (
    "$@" >"$RUN_DIR/$name.log" 2>&1
    rc=$?
    echo "$rc" >"$sentinel.tmp" && mv "$sentinel.tmp" "$sentinel"
  ) &
  echo "$(date '+%F %T') started $name (log $RUN_DIR/$name.log)"
}

wait_job() {
  local name=$1 timeout_s=$2
  local sentinel="$RUN_DIR/$name.exit" start
  start=$(date +%s)
  while [ ! -f "$sentinel" ]; do
    if [ $(($(date +%s) - start)) -ge "$timeout_s" ]; then
      echo "$(date '+%F %T') TIMEOUT: no $sentinel after ${timeout_s}s. NOT killed; it may still be running." \
        "Last log line: $(tail -c 300 "$RUN_DIR/$name.log" | tr '\n' ' ')"
      return 124
    fi
    sleep 15
  done
  local rc
  rc=$(cat "$sentinel")
  echo "$(date '+%F %T') $name exited $rc"
  return "$rc"
}
