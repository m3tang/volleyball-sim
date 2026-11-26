"""Run an answerer over the golden set, grade every case, and write results.

  python -m evals.run --selftest                       free: oracle/null/... through the full pipeline,
                                                       checks each lands where it must (exit 1 if not)
  python -m evals.run --answerer oracle                one built-in answerer, summary printed
  python -m evals.run --answerer rallyagent.agent:answer --variant baseline [--reps 2]
                                                       the real agent (step 3); writes the hillclimb
                                                       layout under .claude/hillclimb/ask-agent/<variant>/

Output per variant directory:
  results.jsonl   one row per (case, rep), written as each case finishes; reruns skip rows already there
  traces/<id>_rep<k>.json   full exchange (system, question, every tool call and result, answer)
  errors.jsonl    attempts that produced nothing to score (exception, timeout); never in results.jsonl

Paid answerers (anything not built in) are gated: the runner hashes the eval code and golden set and
refuses to run if they changed since the last approved run, until rerun with --approve-harness.
"""
import argparse
import concurrent.futures as cf
import hashlib
import importlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import answerers as A
from .grade import grade

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLDEN = HERE / "golden.jsonl"
FLOW = ROOT / ".claude" / "hillclimb" / "ask-agent"
HARNESS = [HERE / "run.py", HERE / "grade.py", HERE / "golden.jsonl", ROOT / "rallytools" / "grounding.py"]
from rallyagent.pricing import cost_usd  # noqa: E402  (model x usage -> $, shared with the public API)

METRICS = [
    {"id": "correct", "label": "Correct", "kind": "binary"},
    {"id": "grounded", "label": "Grounded", "kind": "float", "scale": 1},
    {"id": "tool_choice", "label": "Tool choice", "kind": "binary"},
    {"id": "recall", "label": "Recall", "kind": "float", "scale": 1},
    {"id": "refused", "label": "Refused", "kind": "binary"},
]


def load_cases(ids=None, category=None, need_db=None):
    """need_db=False drops the SQL cases (for CI, where data/vb.duckdb is absent)."""
    cases = [json.loads(l) for l in GOLDEN.read_text().splitlines()]
    if need_db is False:
        cases = [c for c in cases if not any(t["tool"] == "query_stats" for t in c["tools"][:1])]
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    if category:
        cases = [c for c in cases if c["category"] in category]
    return cases


def get_answerer(name):
    if name in A.BUILTIN:
        return A.BUILTIN[name], True
    mod, _, fn = name.partition(":")
    return getattr(importlib.import_module(mod), fn or "answer"), False


def trace(case, run):
    t = []
    if run.get("system"):
        t.append({"role": "system", "content": run["system"]})
    t += case.get("history") or []
    t.append({"role": "user", "content": case["question"]})
    for c in run.get("tool_calls") or []:
        t.append({"role": "tool_call", "name": c["name"], "content": json.dumps(c.get("args") or {}, indent=2)})
        body = c["result"] if c.get("error") is None else {"error": c["error"]}
        t.append({"role": "tool_result", "content": json.dumps(body, indent=1, default=str)})
    if run.get("research_notes"):
        t.append({"role": "assistant", "content": "[research notes, passed to the writer]\n" + run["research_notes"]})
    t.append({"role": "assistant", "content": run.get("answer") or ""})
    return t


def harness_sha():
    h = hashlib.sha256()
    for p in HARNESS:
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:16]


def gate(approve):
    FLOW.mkdir(parents=True, exist_ok=True)
    state_file = FLOW / "_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    sha = harness_sha()
    if approve:
        state["harness_sha"] = sha
    elif state.get("harness_sha") != sha:
        sys.exit(f"eval harness changed (sha {sha}, approved {state.get('harness_sha')}). Review the diff, "
                 "then rerun with --approve-harness.")
    state.update({"metrics": METRICS,
                  "perf_fields": [{"id": "latency_s", "label": "Latency", "unit": "s"},
                                  {"id": "tool_calls", "label": "Tool calls"},
                                  {"id": "in_tokens", "label": "In tok"}, {"id": "out_tokens", "label": "Out tok"},
                                  {"id": "cost_usd", "label": "Cost", "unit": "$"}]})
    state_file.write_text(json.dumps(state, indent=2))


def run_one(answer_fn, case, timeout_s):
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(answer_fn, case)
        try:
            run = fut.result(timeout=timeout_s)
        except cf.TimeoutError:
            return None, {"class": "timeout", "detail": f"no answer within {timeout_s}s"}
        except Exception as e:  # harness or serving failure: not a model failure
            return None, {"class": "harness_error", "detail": f"{type(e).__name__}: {e}"}
    run.setdefault("latency_s", round(time.monotonic() - t0, 3))
    return run, None


def run(answerer, cases, out_dir=None, reps=1, concurrency=8, timeout_s=180):
    """Returns the result rows. With out_dir, writes results.jsonl / traces / errors.jsonl and resumes."""
    answer_fn, _ = get_answerer(answerer)
    done = set()
    if out_dir:
        (out_dir / "traces").mkdir(parents=True, exist_ok=True)
        rf = out_dir / "results.jsonl"
        if rf.exists():
            done = {(r["prompt_id"], r["rep"]) for r in map(json.loads, rf.read_text().splitlines())}
    jobs = [(c, k) for k in range(reps) for c in cases if (c["id"], k) not in done]
    rows = []

    def work(job):
        case, k = job
        run_, err = run_one(answer_fn, case, timeout_s)
        if err:
            return case, k, None, err
        g = grade(case, run_)
        stop = run_.get("stop_reason")
        usage = run_.get("usage") or {}
        row = {"prompt_id": case["id"], "rep": k, "prompt": case["question"], "tags": case["tags"],
               "status": "truncated" if stop == "max_tokens" else "ok", "stop_reason": stop,
               "grade": g["grade"], "model": run_.get("model"), "usage": usage,
               "in_tokens": usage.get("input_tokens", 0), "out_tokens": usage.get("output_tokens", 0),
               "latency_s": run_["latency_s"], "tool_calls": len(run_.get("tool_calls") or []),
               "cost_usd": cost_usd(run_.get("usage_by_model")), "answer": run_.get("answer"),
               "meta": {**g["detail"], "model_calls": run_.get("model_calls"), "grounding": run_.get("grounding")}}
        return case, k, (row, run_), None

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for case, k, res, err in ex.map(work, jobs):
            if err:
                if out_dir:
                    with open(out_dir / "errors.jsonl", "a") as f:
                        f.write(json.dumps({"prompt_id": case["id"], "rep": k, **err}) + "\n")
                continue
            row, run_ = res
            rows.append(row)
            if out_dir:
                (out_dir / "traces" / f"{case['id']}_rep{k}.json").write_text(json.dumps(trace(case, run_), indent=1, default=str))
                with open(out_dir / "results.jsonl", "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
    return rows


def _ci(xs, n_boot=2000, seed=0):
    if not xs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    boots = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
    return m, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def summarize(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    out = {"cases": len(ok), "truncated": len(rows) - len(ok)}
    for m in METRICS:
        xs = [r["grade"][m["id"]] for r in ok if m["id"] in r["grade"]]
        out[m["id"]] = _ci(xs)
    checked = sum(r["meta"]["checked_numbers"] for r in ok)
    bad = sum(len(r["meta"]["ungrounded"]) for r in ok)
    out["hallucination_rate"] = bad / checked if checked else 0.0
    out["cost_usd"] = sum(r.get("cost_usd", 0) for r in ok)
    lat = sorted(r["latency_s"] for r in ok)
    out["latency_p50"] = lat[len(lat) // 2] if lat else 0.0
    out["latency_p90"] = lat[int(0.9 * (len(lat) - 1))] if lat else 0.0
    out["numbers_per_answer"] = checked / len(ok) if ok else 0.0
    by = defaultdict(list)
    for r in ok:
        by[r["tags"][0]].append(r["grade"]["correct"])
    out["by_category"] = {k: sum(v) / len(v) for k, v in by.items()}
    out["n_by_category"] = {k: len(v) for k, v in by.items()}
    return out


def print_summary(name, s):
    f = lambda t: f"{100 * t[0]:5.1f}% [{100 * t[1]:.0f}-{100 * t[2]:.0f}]"
    print(f"{name}: {s['cases']} cases | correct {f(s['correct'])} | grounded {f(s['grounded'])} | "
          f"tool choice {f(s['tool_choice'])} | hallucination rate {100 * s['hallucination_rate']:.1f}% "
          f"({s['numbers_per_answer']:.1f} numbers/answer) | refused {100 * s['refused'][0]:.0f}%")
    if s["cost_usd"]:
        print(f"   cost ${s['cost_usd']:.3f} total, ${s['cost_usd'] / max(s['cases'], 1):.4f}/case | "
              f"latency p50 {s['latency_p50']:.1f}s, p90 {s['latency_p90']:.1f}s")
    print("   by category (correct): " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in s["by_category"].items()))


# What each built-in answerer must score, if the harness and grader work. `free` = share of cases a
# wrong-numbers answer can still get right (no expected numbers: fact-only or refusal cases).
EXPECT = {
    "oracle": lambda s, free: s["correct"][0] == 1 and s["grounded"][0] == 1 and s["tool_choice"][0] == 1,
    "no_tools": lambda s, free: s["correct"][0] == 1 and s["tool_choice"][0] == 0 and s["hallucination_rate"] > 0.95,
    "perturbed": lambda s, free: abs(s["correct"][0] - free) < 1e-9 and s["hallucination_rate"] > 0.75,
    "refuse_all": lambda s, free: abs(s["correct"][0] - s["n_by_category"]["out_of_scope"] / s["cases"]) < 1e-9
                                  and s["by_category"]["out_of_scope"] == 1,
    "null": lambda s, free: s["correct"][0] == 0 and s["numbers_per_answer"] == 0,
}


def selftest(cases):
    free = sum(not any(alt["numbers"] for alt in c["expect"].get("alternatives") or [c["expect"]]) for c in cases) / len(cases)
    ok = True
    for name in EXPECT:
        rows = run(name, cases)
        s = summarize(rows)
        print_summary(name, s)
        if not EXPECT[name](s, free):
            ok = False
            print(f"   FAILED: {name} did not land where it must")
            if name == "oracle":
                for r in rows:
                    g = r["grade"]
                    if g["correct"] < 1 or g["grounded"] < 1 or g.get("tool_choice", 1) < 1:
                        print(f"     {r['prompt_id']}: {g} {r['meta']['ungrounded']} {r['meta']['numbers']} {r['meta']['facts']}")
    return ok


def regrade(variant_dir):
    """Re-score saved transcripts with the current grader (no model calls). For grader fixes."""
    cases = {c["id"]: c for c in load_cases()}
    rf = variant_dir / "results.jsonl"
    rows = [json.loads(l) for l in rf.read_text().splitlines()]
    for r in rows:
        t = json.loads((variant_dir / "traces" / f"{r['prompt_id']}_rep{r['rep']}.json").read_text())
        calls = []
        for turn, nxt in zip(t, t[1:]):
            if turn["role"] == "tool_call":
                body = json.loads(nxt["content"])
                err = body.get("error") if isinstance(body, dict) and set(body) == {"error"} else None
                calls.append({"name": turn["name"], "args": json.loads(turn["content"]),
                              "result": None if err else body, "error": err})
        g = grade(cases[r["prompt_id"]], {"answer": r["answer"], "tool_calls": calls})
        r["grade"] = g["grade"]
        r["meta"] = {**r["meta"], **g["detail"]}
    rf.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--answerer", default="oracle")
    ap.add_argument("--variant", default=None, help="baseline, v1, ... (writes the hillclimb layout)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--category", nargs="*")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=180)
    ap.add_argument("--approve-harness", action="store_true")
    ap.add_argument("--regrade", action="store_true", help="re-score --variant's saved transcripts; no model calls")
    a = ap.parse_args()
    if a.regrade:
        print_summary(a.variant, summarize(regrade(FLOW / a.variant)))
        return
    from vbsim.data import DB
    ids = [x for arg in (a.ids or []) for x in arg.replace(",", " ").split()]  # "a b" or a,b or a b
    cases = load_cases(ids, a.category, need_db=None if DB.exists() else False)
    if ids and len(cases) != len(set(ids)):
        sys.exit(f"unknown case ids: {', '.join(sorted(set(ids) - {c['id'] for c in cases}))}")
    if not cases:
        sys.exit("no cases match")
    if a.selftest:
        sys.exit(0 if selftest(cases) else 1)
    _, free = get_answerer(a.answerer)
    out = None
    if a.variant:
        if not free:
            gate(a.approve_harness)
        out = FLOW / a.variant
    rows = run(a.answerer, cases, out, a.reps, a.concurrency, a.timeout_s)
    if out:  # summarize everything on disk, including resumed rows
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    print_summary(a.answerer, summarize(rows))


if __name__ == "__main__":
    main()
