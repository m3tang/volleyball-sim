"""Penalized multinomial logistic regression with team-vs-team effects.

For row i, where team T (the "offense", e.g. the receiving team) faces team O (the "defense"):
    eta[i, c] = intercept[c] + off[T, c] - def[O, c] + home_coef[c] * home[i]    for c < C-1
    eta[i, C-1] = 0                                                              (reference class)
    P(outcome c) = softmax(eta[i])[c]
Rows hold outcome counts (many rallies or possessions per row), so a season fits fast.

Strengths are pulled toward a prior (last season's values, or zero) with an L2 penalty `lam`.
That is the shrinkage: a team with few rallies stays close to its prior.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass
class Params:
    intercept: np.ndarray  # (K,)
    home: np.ndarray       # (K,)
    off: np.ndarray        # (n_teams, K)
    def_: np.ndarray       # (n_teams, K)

    def eta(self, t_idx, o_idx, home):
        return (self.intercept + self.off[t_idx] - self.def_[o_idx]
                + np.asarray(home, dtype=float)[:, None] * self.home)

    def probs(self, t_idx, o_idx, home):
        """(N, C) outcome probabilities."""
        eta = self.eta(np.asarray(t_idx), np.asarray(o_idx), home)
        full = np.concatenate([eta, np.zeros((eta.shape[0], 1))], axis=1)
        return np.exp(full - logsumexp(full, axis=1, keepdims=True))


def fit(Y, t_idx, o_idx, home, n_teams, lam, prior_off=None, prior_def=None, init=None):
    """Y: (N, C) outcome counts. Returns Params with K = C - 1 columns."""
    Y = np.asarray(Y, dtype=float)
    t_idx = np.asarray(t_idx)
    o_idx = np.asarray(o_idx)
    home = np.asarray(home, dtype=float)
    N, C = Y.shape
    K = C - 1
    prior_off = np.zeros((n_teams, K)) if prior_off is None else prior_off
    prior_def = np.zeros((n_teams, K)) if prior_def is None else prior_def
    n_rows = Y.sum(axis=1)
    total = max(n_rows.sum(), 1.0)

    def unpack(theta):
        b, h = theta[:K], theta[K:2 * K]
        off = theta[2 * K:2 * K + n_teams * K].reshape(n_teams, K)
        dfn = theta[2 * K + n_teams * K:].reshape(n_teams, K)
        return b, h, off, dfn

    def objective(theta):
        b, h, off, dfn = unpack(theta)
        eta = b + off[t_idx] - dfn[o_idx] + home[:, None] * h
        full = np.concatenate([eta, np.zeros((N, 1))], axis=1)
        logp = full - logsumexp(full, axis=1, keepdims=True)
        ll = (Y * logp).sum()
        R = (Y - n_rows[:, None] * np.exp(logp))[:, :K]  # d ll / d eta
        g_off = np.zeros((n_teams, K))
        g_def = np.zeros((n_teams, K))
        for k in range(K):
            g_off[:, k] = np.bincount(t_idx, weights=R[:, k], minlength=n_teams)
            g_def[:, k] = -np.bincount(o_idx, weights=R[:, k], minlength=n_teams)
        d_off, d_def = off - prior_off, dfn - prior_def
        obj = -ll + lam * ((d_off ** 2).sum() + (d_def ** 2).sum())
        grad = np.concatenate([
            -R.sum(axis=0),
            -(R * home[:, None]).sum(axis=0),
            (-g_off + 2 * lam * d_off).ravel(),
            (-g_def + 2 * lam * d_def).ravel(),
        ])
        return obj / total, grad / total

    if init is None:
        # Start at the overall outcome shares and the prior.
        share = (Y.sum(axis=0) + 1) / (Y.sum() + C)
        b0 = np.log(share[:K] / share[K])
        theta0 = np.concatenate([b0, np.zeros(K), prior_off.ravel(), prior_def.ravel()])
    else:
        theta0 = np.concatenate([init.intercept, init.home, init.off.ravel(), init.def_.ravel()])

    res = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-9, "ftol": 1e-12})
    b, h, off, dfn = unpack(res.x)
    return Params(b, h, off, dfn)
