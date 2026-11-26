"""Check that every number in an answer came from a tool result.

    check(answer, tool_calls, question="") -> {"mentions": [...], "n": int, "ungrounded": [...]}

A mention is a number as written in prose: "87%", "87.3 percent", "about 1 in 250", ".312",
"1,834", "3-1", "12th". Each is matched against a pool built from the tool calls (results and
arguments) at the precision the answer wrote it: "87%" matches 0.8730 and 0.8661 but not 0.8649.
"about 1 in N" matches the site's own display text or 1/p rounded. Complements (1 - p) are not
derived: tools return both teams' win probabilities explicitly.

Measured limit (python -m evals.run --selftest, "perturbed"): numbers nudged by 7 points still
match some unrelated value in a large tool result part of the time, mostly whole-percent rates
against team_profile's 12 steps. Grounding is a backstop; correctness grading checks the numbers
that matter against their exact referents.

Not checked (exempt): years (1990-2035), numbers that appear in the question, and structural
numbers ("round 2", "best of 5", "set 3", "top 25", "3 sets"). Spelled-out numbers ("five sets")
are not extracted. Arithmetic on tool numbers (a difference of two rates) is not derived: it
counts as ungrounded, which is the point. Ask a tool, don't compute.
"""
import json
import re
from dataclasses import asdict, dataclass

from .core import one_in

YEAR = range(1990, 2036)
_PAT = re.compile(r"""
    (?P<one_in>\b(?:1|one)[\s-]+in[\s-]+(?P<oi>\d[\d,]*))
  | (?P<pct>(?P<pv>(?<![\w.])\d+(?:\.\d+)?|(?<![\w\d])\.\d+)\s*(?:%|\s?percent(?:age\s+points?)?\b|\s?pct\b|\s?points?\s+of\s+percentage))
  | (?P<pair>(?<![\d.,])(?P<pa>\d{1,3})\s?[-–]\s?(?P<pb>\d{1,3})(?![\d%]|\.\d))
  | (?P<ord>(?<![\w.])(?P<ov>\d+)(?:st|nd|rd|th)\b)
  | (?P<num>(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])|(?<![\w\d])\.\d+(?![\w]))
""", re.X | re.I)
_BOUND = re.compile(r"(?:(?P<lt><|under|less than|below|fewer than)|(?P<gt>>|over|more than|above|greater than))\s*(?:about\s+|roughly\s+)?$", re.I)
_STRUCT_BEFORE = re.compile(r"(?:\bround|\bset|\bgame|\bbest[- ]of[- ]?|\btop|\bweek|\bdivision|\bfirst|\bthe)\s*$", re.I)
_STRUCT_AFTER = re.compile(r"^\s*(?:-?\s*sets?\b|-?\s*setters?\b|seeds?\b)", re.I)


@dataclass
class Mention:
    text: str
    kind: str          # pct | one_in | pair | num
    value: float
    decimals: int
    bound: str = ""    # "<" / ">" when written as "under 1%" / ">99%"
    pair: tuple = ()
    start: int = 0


def _decimals(s):
    s = s.replace(",", "")
    return len(s.split(".")[1]) if "." in s else 0


def extract(text):
    out = []
    for m in _PAT.finditer(text or ""):
        before = text[max(0, m.start() - 20):m.start()]
        b = _BOUND.search(before)
        bound = "<" if b and b["lt"] else ">" if b and b["gt"] else ""
        if m["one_in"]:
            out.append(Mention(m["one_in"], "one_in", float(m["oi"].replace(",", "")), 0, start=m.start()))
        elif m["pct"]:
            out.append(Mention(m["pct"], "pct", float(m["pv"]), _decimals(m["pv"]), bound, start=m.start()))
        elif m["pair"]:
            out.append(Mention(m["pair"], "pair", 0.0, 0, pair=(int(m["pa"]), int(m["pb"])), start=m.start()))
        elif m["ord"]:
            out.append(Mention(m["ord"], "num", float(m["ov"]), 0, start=m.start()))
        else:
            s = m["num"]
            out.append(Mention(s, "num", float(s.replace(",", "")), _decimals(s), bound, start=m.start()))
    return out


def _exempt(m, text, question_nums):
    if m.kind == "num" and m.decimals == 0 and int(m.value) in YEAR and "," not in m.text:
        return "year"
    if m.kind in ("num", "pct") and m.value in question_nums:
        return "in question"
    if m.kind == "pair" and (m.pair[0] in question_nums and m.pair[1] in question_nums):
        return "in question"
    if m.kind == "num" and m.decimals == 0:
        if _STRUCT_BEFORE.search(text[max(0, m.start - 12):m.start]) or _STRUCT_AFTER.match(text[m.start + len(m.text):]):
            return "structural"
    return ""


class Pool:
    """Every number the tool calls made available."""

    def __init__(self):
        self.nums, self.probs, self.one_ins, self.pairs = set(), set(), set(), set()

    def add_value(self, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return
        v = float(v)
        self.nums.add(v)
        if 0 <= v <= 1:
            self.probs.add(v)

    def add_text(self, s):
        for m in extract(s):
            if m.kind == "pct":
                self.add_value(m.value / 100)
                self.nums.add(m.value)
            elif m.kind == "one_in":
                self.one_ins.add(int(m.value))
            elif m.kind == "pair":
                self.pairs.add(m.pair)
            else:
                self.add_value(m.value)

    def add(self, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                self.add_text(str(k)) if re.fullmatch(r"\d{1,3}-\d{1,3}", str(k)) else None
                self.add(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self.add(v)
        elif isinstance(obj, str):
            self.add_text(obj)
        else:
            self.add_value(obj)


def _tol(d):
    return 0.5 * 10 ** -d + 1e-9


def matches_value(m, v, scale="any"):
    """Does mention m state value v at the precision it was written? scale: prob | count | decimal | any."""
    if m.kind == "pct":
        if scale in ("count", "decimal"):
            return False
        x = 100 * v
        if m.bound == "<":
            return 0 < x < m.value + 1e-9
        if m.bound == ">":
            return m.value - 1e-9 < x < 100
        if scale == "prob" and m.decimals == 0 and (x < 0.5 or x > 99.5):
            return False  # "0%" or "100%" for a long shot is not a statement of the odds
        return abs(x - m.value) <= _tol(m.decimals)
    if m.kind == "one_in":
        return scale in ("prob", "any") and 0 < v < 0.5 and int(m.value) in (round(1 / v), int(one_in(v).split()[-1].replace(",", "")))
    if m.kind == "num":
        if scale == "count":
            return m.decimals == 0 and abs(v - m.value) < 1e-9
        if scale == "decimal":
            return m.decimals >= 3 and abs(v - m.value) <= _tol(m.decimals)
        if scale == "prob":
            return 0 < m.value < 1 and m.decimals >= 2 and abs(v - m.value) <= _tol(m.decimals)
        return abs(v - m.value) <= _tol(m.decimals)
    return False


def _grounded(m, pool):
    if m.kind == "pair":
        a, b = m.pair
        if (a, b) in pool.pairs or (b, a) in pool.pairs:
            return True
        return float(a) in pool.nums and float(b) in pool.nums
    if m.kind == "one_in":  # the site's own "about 1 in N" text, or N = 1/p exactly rounded
        return int(m.value) in pool.one_ins or any(0 < p < 0.5 and round(1 / p) == int(m.value) for p in pool.probs)
    if m.kind == "pct":
        return (any(matches_value(m, p, "any") for p in pool.probs)
                or any(abs(x - m.value) <= _tol(m.decimals) for x in pool.nums))
    return any(abs(x - m.value) <= _tol(m.decimals) for x in pool.nums)


def check(answer, tool_calls, question=""):
    """tool_calls: [{"name", "args", "result"}] (calls that errored may be included; their
    error text is not a source of numbers)."""
    pool = Pool()
    for c in tool_calls:
        pool.add(c.get("args") or {})
        if c.get("result") is not None:
            pool.add(c["result"] if not isinstance(c["result"], str) else _maybe_json(c["result"]))
    qnums = {m.value for m in extract(question) if m.kind != "pair"} | {float(x) for m in extract(question) if m.kind == "pair" for x in m.pair}
    out, prev = [], None
    for m in extract(answer):
        why = _exempt(m, answer, qnums)
        ok = True if why else _grounded(m, pool)
        # "1.4% (about 1 in 70)": a 1-in-N right after a grounded percentage restates it at the site's
        # rounding, so it is grounded when it agrees with that percentage.
        if (not ok and m.kind == "one_in" and prev is not None and prev[0].kind == "pct" and prev[1]
                and m.start - (prev[0].start + len(prev[0].text)) <= 12 and 0 < prev[0].value < 50
                and one_in(prev[0].value / 100) == f"about 1 in {int(m.value):,}"):
            ok, why = True, "restates previous"
        out.append({**asdict(m), "exempt": why, "grounded": ok})
        prev = (m, ok)
    checked = [x for x in out if not x["exempt"]]
    return {"mentions": out, "n": len(checked), "n_grounded": sum(x["grounded"] for x in checked),
            "ungrounded": [x["text"] for x in checked if not x["grounded"]]}


def _maybe_json(s):
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return s
