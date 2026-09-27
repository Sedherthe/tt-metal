"""OSV audit of a `pip freeze` file: querybatch, then each vulnerability's details. Local version labels stripped."""
import json, sys, urllib.request
pkgs = []
for line in open(sys.argv[1]):
    line = line.strip()
    if "==" in line and not line.startswith("#"):
        n, v = line.split("==", 1)
        pkgs.append((n, v.split("+")[0]))
def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))
res = post("https://api.osv.dev/v1/querybatch", {"queries": [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in pkgs]})
print(f"{len(pkgs)} packages queried")
hits = [(n, v, x["id"]) for (n, v), r in zip(pkgs, res["results"]) for x in r.get("vulns", [])]
for n, v, vid in hits:
    d = json.load(urllib.request.urlopen(f"https://api.osv.dev/v1/vulns/{vid}", timeout=60))
    sev = d.get("database_specific", {}).get("severity") or ",".join(s.get("score", "")[:40] for s in d.get("severity", []))
    fixed = sorted({e["fixed"] for a in d.get("affected", []) if a.get("package", {}).get("name", "").lower() == n.lower()
                    for rg in a.get("ranges", []) for e in rg.get("events", []) if "fixed" in e})
    cves = [a for a in d.get("aliases", []) if a.startswith("CVE")]
    print(f"| {n} {v} | {vid} | {','.join(cves)} | {sev} | fixed {','.join(fixed) or '-'} | {d.get('summary', '')[:110]} |")
print(f"{len(hits)} advisories")
