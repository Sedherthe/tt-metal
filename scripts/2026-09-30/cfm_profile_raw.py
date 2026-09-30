"""One CFM Euler step's ops from the device profiler's raw logs, when tracy's own ops report cannot be generated.

`python -m tracy -r` post-processing asserted ("Device data missing: Op ... not present in
cpp_device_perf_report.csv"): during the compile solve, before `cfm_step_profile.py --profile` first flushes the
profiler, the DRAM marker buffers filled and some ops lost their device data. The step itself runs after that flush.
This reads the same two files tracy's post-processing reads and keeps only the step's ops:
- `tracy_ops_data.csv` (host side): every op's global call count and host time, and the two signposts
  (`euler_step_start`, `euler_step_end`). The parsing is `tools/tracy/process_ops_logs.py:import_tracy_op_logs`'s
  message format, without its 1.6 GB times file.
- `cpp_device_perf_report.csv` (device side): per op, keyed by the global call count, its firmware and kernel
  start/end cycles and durations.

On the device timeline, the step spans from its first op's firmware start to its last op's firmware end. That span
is its ops' firmware time plus the gaps between them, where the device waits for the next program. Per op name:
count, kernel time and cores.

    python3 cfm_profile_raw.py <tracy -o dir>/.logs [--out <json>]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--out")
    args = ap.parse_args()
    ops, marks, cached = [], {}, {}
    with open(f"{args.logs}/tracy_ops_data.csv", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";", quotechar="`"):
            msg, t = row["MessageName"], int(row["total_ns"])
            if "TT_SIGNPOST" in msg:
                marks[msg.split("TT_SIGNPOST:")[-1].strip()] = t
                continue
            if not (("TT_DNN" in msg or "TT_METAL" in msg) and "OP" in msg):
                continue
            head, _, body = msg.partition(" ->\n")
            if body:  # an uncached op: its JSON carries the call count
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    continue
                name = head.split(":", 1)[-1].split(",")[0].strip().strip('"')
                if "op_hash" in data:
                    cached[int(data["op_hash"])] = name
                ops.append({"id": int(data["global_call_count"]), "t": t, "name": name, "cached": False})
            else:  # a cached op: name, hash, device, cache hit, call count
                f = msg.split(":", 1)[-1].split(",")
                ops.append({"id": int(f[4]), "t": t, "name": cached.get(int(f[1]), f[0].strip().strip('"')), "cached": True})
    t0, t1 = marks["euler_step_start"], marks["euler_step_end"]
    step = [o for o in ops if t0 < o["t"] < t1]
    dev = {}
    with open(f"{args.logs}/cpp_device_perf_report.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            dev[int(row["GLOBAL CALL COUNT"])] = row
    rows = [(o, dev.get(o["id"])) for o in step]
    missing = [o["id"] for o, d in rows if d is None]
    have = sorted(((o, d) for o, d in rows if d is not None), key=lambda x: int(x[1]["DEVICE FW START CYCLE"]))

    def f(d, k):
        return float(d[k]) if d[k] not in ("", None) else 0.0

    fw_ns = sum(f(d, "DEVICE FW DURATION [ns]") for _, d in have)
    kernel_ns = sum(f(d, "DEVICE KERNEL DURATION [ns]") for _, d in have)
    cycles = sum(f(d, "DEVICE FW END CYCLE") - f(d, "DEVICE FW START CYCLE") for _, d in have)
    ns_per_cycle = fw_ns / cycles if cycles else float("nan")
    span_cycles = f(have[-1][1], "DEVICE FW END CYCLE") - f(have[0][1], "DEVICE FW START CYCLE")
    summary = {
        "host_ms_between_signposts": round((t1 - t0) / 1e6, 2),
        "ops": len(step),
        "ops_with_device_data": len(have),
        "ops_missing_device_data": len(missing),
        "ops_cached": sum(o["cached"] for o in step),
        "device_span_ms": round(span_cycles * ns_per_cycle / 1e6, 2),
        "device_fw_ms": round(fw_ns / 1e6, 2),
        "device_kernel_ms": round(kernel_ns / 1e6, 2),
        "device_gaps_ms": round((span_cycles * ns_per_cycle - fw_ns) / 1e6, 2),
        "ns_per_cycle": round(ns_per_cycle, 4),
    }
    by = defaultdict(lambda: {"count": 0, "kernel_ms": 0.0, "fw_ms": 0.0, "cores": set()})
    for o, d in have:
        b = by[d["OP NAME"]]
        b["count"] += 1
        b["kernel_ms"] += f(d, "DEVICE KERNEL DURATION [ns]") / 1e6
        b["fw_ms"] += f(d, "DEVICE FW DURATION [ns]") / 1e6
        b["cores"].add(int(f(d, "CORE COUNT")))
    table = sorted(({"op": k, "count": v["count"], "kernel_ms": round(v["kernel_ms"], 3), "fw_ms": round(v["fw_ms"], 3),
                     "cores": sorted(v["cores"])} for k, v in by.items()), key=lambda x: -x["kernel_ms"])  # fmt: skip
    print(json.dumps(summary, indent=1))
    print("\n| op | count | device kernel ms | device firmware ms | cores |")
    print("|---|---|---|---|---|")
    for t in table:
        cores = t["cores"] if len(t["cores"]) <= 4 else f"{t['cores'][0]}-{t['cores'][-1]}"
        print(f"| {t['op']} | {t['count']} | {t['kernel_ms']:.3f} | {t['fw_ms']:.3f} | {cores} |")
    if args.out:
        json.dump({"summary": summary, "by_op": table}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
