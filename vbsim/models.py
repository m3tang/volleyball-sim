"""The two rally models. Both answer one question for any matchup:
    sideout(R, S, home_R) = P(receiving team R wins the rally when S serves)
which vbsim.matchsim turns into set and match probabilities.

SideoutModel: one logistic regression. Each team has a receive strength and a serve strength.

TouchModel: the rally as a chain of possessions.
    serve (S vs R):               ace | service error | in play -> R's first-ball possession
    possession (T vs O, kind):    win | lose | to_opp (the ball goes to O: O's 'tr' possession)
Each step is its own team-vs-team regression. The chance of winning the rally is solved
exactly from the chain (no random simulation needed).
"""
import numpy as np

from . import fit as F

KINDS = ("fb", "tr")

# Settings tuned on 2024 by analysis/overconfidence.py: (lam, rho, form_sd).
#   lam: shrinkage toward the prior; rho: share of last season's strengths carried over;
#   form_sd: match-day form swing (logit scale) used when turning rally odds into match odds.
TUNED = {"sideout": (10, 1.0, 0.1), "touch": (10, 1.0, 0.1)}


def _np(x):
    return np.asarray(x, dtype=float)


class SideoutModel:
    name = "sideout"

    def __init__(self, n_teams, lam):
        self.n, self.lam = n_teams, lam
        self.p = None

    def fit(self, data, mask_by_table, idx, prior=None, warm=None, weights=None):
        d, mask = data["sideout"], mask_by_table["sideout"]
        Y = np.stack([_np(d["k"][mask]), _np(d["n"][mask] - d["k"][mask])], axis=1)
        if weights is not None:
            Y = Y * weights["sideout"][mask][:, None]
        pr = prior.p if prior is not None else None
        self.p = F.fit(Y, idx(d["t"][mask]), idx(d["o"][mask]), d["home"][mask], self.n, self.lam,
                       pr.off if pr else None, pr.def_ if pr else None,
                       init=warm.p if warm is not None else None)
        return self

    def scaled(self, rho):
        """A copy with team strengths multiplied by rho (used as next season's prior)."""
        m = SideoutModel(self.n, self.lam)
        m.p = F.Params(self.p.intercept, self.p.home, rho * self.p.off, rho * self.p.def_)
        return m

    def sideout(self, r_idx, s_idx, home_r):
        return self.p.probs(r_idx, s_idx, home_r)[:, 0]


class TouchModel:
    name = "touch"

    def __init__(self, n_teams, lam):
        self.n, self.lam = n_teams, lam
        self.parts = {}  # 'serve', 'fb', 'tr' -> Params

    def fit(self, data, mask_by_table, idx, prior=None, warm=None, weights=None):
        """weights: optional {table: per-row weight}, e.g. to count recent matches more."""
        specs = {"serve": ("serve", ["ace", "serr", "inplay"]),
                 "fb": ("fb", ["win", "lose", "to_opp"]),
                 "tr": ("tr", ["win", "lose", "to_opp"])}
        for part, (table, cols) in specs.items():
            d, m = data[table], mask_by_table[table]
            Y = np.stack([_np(d[c][m]) for c in cols], axis=1)
            if weights is not None:
                Y = Y * weights[table][m][:, None]
            pr = prior.parts[part] if prior is not None else None
            self.parts[part] = F.fit(Y, idx(d["t"][m]), idx(d["o"][m]), d["home"][m], self.n, self.lam,
                                     pr.off if pr else None, pr.def_ if pr else None,
                                     init=warm.parts[part] if warm is not None else None)
        return self

    def scaled(self, rho):
        m = TouchModel(self.n, self.lam)
        m.parts = {k: F.Params(p.intercept, p.home, rho * p.off, rho * p.def_) for k, p in self.parts.items()}
        return m

    def sideout(self, r_idx, s_idx, home_r):
        home_r = _np(home_r)
        srv = self.parts["serve"].probs(s_idx, r_idx, -home_r)        # ace, serr, inplay
        fb = self.parts["fb"].probs(r_idx, s_idx, home_r)             # R's first-ball possession
        tr_r = self.parts["tr"].probs(r_idx, s_idx, home_r)           # R's later possessions
        tr_s = self.parts["tr"].probs(s_idx, r_idx, -home_r)          # S's possessions
        # x = P(R wins | R has a transition possession), y = same for S:
        #   x = wR + tR*(1-y),   y = wS + tS*(1-x)
        wR, tR = tr_r[:, 0], tr_r[:, 2]
        wS, tS = tr_s[:, 0], tr_s[:, 2]
        x = (wR + tR * (1 - wS - tS)) / (1 - tR * tS)
        y = wS + tS * (1 - x)
        w_fb = fb[:, 0] + fb[:, 2] * (1 - y)
        return srv[:, 1] + srv[:, 2] * w_fb

    def explain(self, r_idx, s_idx, home_r):
        """Step probabilities for a matchup, for explanations later."""
        home_r = _np(home_r)
        return {"serve(ace, serr, inplay)": self.parts["serve"].probs(s_idx, r_idx, -home_r),
                "first ball(win, lose, to_opp)": self.parts["fb"].probs(r_idx, s_idx, home_r),
                "transition(win, lose, to_opp)": self.parts["tr"].probs(r_idx, s_idx, home_r)}
