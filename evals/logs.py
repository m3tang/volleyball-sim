"""Turn the public chat's question log into eval cases.

    python -m evals.logs pull [--days 30]   copy logged questions from Cloud Logging to evals/logs/asks.jsonl
    python -m evals.logs triage             write evals/logs/inbox.md: what to look at, worst first

The API (rallyagent/api.py) logs one JSON line per answered question. pull merges new ones into
evals/logs/asks.jsonl (kept out of git: these are visitors' words). triage sorts them into:

  withheld     failed the number check, so the visitor got "couldn't answer reliably"
  tool errors  a tool call failed (unknown team, bad arguments, SQL errors)
  declined     the answer says the data can't answer it; check it really can't
  follow-ups   asked with earlier messages, the hardest kind to get right
  new          everything else not already in the golden set

Promote the useful ones by adding them to evals/logged.py, then `python -m evals.golden`.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

from evals.grade import refused

HERE = Path(__file__).resolve().parent
DIR = HERE / "logs"
ASKS = DIR / "asks.jsonl"
INBOX = DIR / "inbox.md"
PROJECT = os.environ.get("RALLY_GCP_PROJECT", "rally-odds-agent")
SERVICE = os.environ.get("RALLY_SERVICE", "rally-agent")


def _norm(q):
    return " ".join("".join(c for c in q.lower() if c.isalnum() or c.isspace()).split())


def load():
    if not ASKS.exists():
        return []
    return [json.loads(l) for l in ASKS.read_text().splitlines() if l.strip()]


def pull(days):
    flt = (f'resource.type="cloud_run_revision" AND resource.labels.service_name="{SERVICE}" '
           f'AND jsonPayload.event="ask"')
    cmd = ["gcloud", "logging", "read", flt, f"--project={PROJECT}", f"--freshness={days}d", "--format=json", "--limit=5000"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("gcloud not found; install the Google Cloud SDK or put it on PATH")
    except subprocess.CalledProcessError as e:
        sys.exit(f"gcloud failed: {e.stderr.strip()[:500]}")
    have = {r["id"] for r in load()}
    new = []
    for e in json.loads(out or "[]"):
        if e["insertId"] in have:
            continue
        new.append({"id": e["insertId"], "ts": e["timestamp"], **e["jsonPayload"]})
    rows = sorted(load() + new, key=lambda r: r["ts"])
    DIR.mkdir(exist_ok=True)
    ASKS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(new)} new questions ({len(rows)} total) in {ASKS}")


def known_questions():
    """Questions already in the golden set or logged.py, normalized."""
    from evals.logged import LOGGED
    golden = HERE / "golden.jsonl"
    qs = {_norm(json.loads(l)["question"]) for l in golden.read_text().splitlines()} if golden.exists() else set()
    return qs | {_norm(e["question"]) for e in LOGGED}


def classify(r):
    flags = []
    if not r.get("shown", True):
        flags.append("withheld")
    if r.get("tool_errors"):
        flags.append("tool errors")
    if refused(r.get("answer") or ""):
        flags.append("declined")
    if r.get("history_turns"):
        flags.append("follow-up")
    if r.get("status") not in (None, "ok"):
        flags.append(f"status {r['status']}")
    return flags


def triage():
    rows = load()
    if not rows:
        sys.exit("no logged questions yet; run `python -m evals.logs pull` first")
    known = known_questions()
    groups = {"withheld": [], "tool errors": [], "declined": [], "follow-up": [], "new": []}
    seen = set()
    for r in rows:
        flags = classify(r)
        key = (_norm(r["question"]), json.dumps(r.get("history") or []))
        if key in seen:
            continue
        seen.add(key)
        first = next((f for f in ("withheld", "tool errors", "declined", "follow-up") if f in flags), None)
        if first:
            groups[first].append((r, flags))
        elif key[0] not in known:
            groups["new"].append((r, flags))

    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    tools = Counter(t["name"] for r in rows for t in r.get("tools") or [])
    n = len(rows)
    out = ["# Question log inbox", "",
           f"{n} questions from {rows[0]['ts'][:10]} to {rows[-1]['ts'][:10]}, {len(seen)} distinct. "
           f"Withheld {sum('withheld' in classify(r) for r in rows)}, declined {sum('declined' in classify(r) for r in rows)}, "
           f"follow-ups {sum(bool(r.get('history_turns')) for r in rows)}.",
           f"Cost {sum(costs):.3f} USD total, {statistics.median(costs):.4f} median per question. "
           f"Latency median {statistics.median(lat):.1f}s, slowest {max(lat):.1f}s." if costs and lat else "",
           "Tools used: " + ", ".join(f"{k} {v}" for k, v in tools.most_common()), "",
           "Promote a question by adding it to `evals/logged.py`, then run `python -m evals.golden`.", ""]
    titles = {"withheld": "Withheld: failed the number check", "tool errors": "Tool errors",
              "declined": "Declined: check the data really can't answer these",
              "follow-up": "Follow-ups", "new": "New questions, not in the golden set"}
    for g, items in groups.items():
        out += [f"## {titles[g]} ({len(items)})", ""]
        for r, flags in items:
            out.append(f"- **{r['question']}**  ")
            meta = [r["ts"][:16].replace("T", " ")] + [f for f in flags if f != g]
            if r.get("tools"):
                meta.append("tools: " + ", ".join(t["name"] for t in r["tools"]))
            out.append("  " + " · ".join(meta) + "  ")
            for m in r.get("history") or []:
                out.append(f"  > {m['role']}: {m['content'][:200]}  ")
            if r.get("tool_errors"):
                out.append("  errors: " + "; ".join(e["error"][:160] for e in r["tool_errors"]) + "  ")
            if r.get("ungrounded"):
                out.append("  unchecked numbers: " + ", ".join(map(str, r["ungrounded"])) + "  ")
            out.append("  answer: " + (r.get("answer") or "(empty)").replace("\n", " ")[:400])
        out.append("")
    DIR.mkdir(exist_ok=True)
    INBOX.write_text("\n".join(out))
    print(f"wrote {INBOX}: " + ", ".join(f"{g} {len(v)}" for g, v in groups.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--days", type=int, default=30)
    sub.add_parser("triage")
    a = ap.parse_args()
    pull(a.days) if a.cmd == "pull" else triage()


if __name__ == "__main__":
    main()
