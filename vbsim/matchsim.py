"""Exact set and match win probabilities from rally win probabilities.

Inputs are numpy arrays (one entry per matchup), so many matchups are computed at once.
  p_srv: P(A wins a rally when A serves)    = 1 - sideout rate of B against A
  p_rcv: P(A wins a rally when B serves)    = sideout rate of A against B
Rules: rally scoring, the rally winner serves next, sets to 25 (5th to 15), win by 2,
best of five. Rallies are treated as independent given who serves.
"""
import numpy as np


def _deuce(p_srv, p_rcv):
    """P(A wins the set) from a tie at target-1 or later, for A serving (x) and B serving (y).

    From a tie, whoever wins the next rally leads by 1 and serves:
      U = A up 1, A serving = p_srv + (1 - p_srv) * y
      D = B up 1, B serving = p_rcv * x
      x = p_srv * U + (1 - p_srv) * D
      y = p_rcv * U + (1 - p_rcv) * D
    """
    a11 = 1 - (1 - p_srv) * p_rcv
    a12 = -p_srv * (1 - p_srv)
    a21 = -(1 - p_rcv) * p_rcv
    a22 = 1 - p_rcv * (1 - p_srv)
    b1 = p_srv ** 2
    b2 = p_rcv * p_srv
    det = a11 * a22 - a12 * a21
    return (b1 * a22 - a12 * b2) / det, (a11 * b2 - a21 * b1) / det


def set_win_prob(p_srv, p_rcv, target=25, a_serves_first=True):
    p_srv = np.asarray(p_srv, dtype=float)
    p_rcv = np.asarray(p_rcv, dtype=float)
    t = target
    tie_a, tie_b = _deuce(p_srv, p_rcv)
    one = np.ones_like(p_srv)
    zero = np.zeros_like(p_srv)
    # V[(a, b, a_serving)] = P(A wins the set) from that score.
    V = {}

    def val(a, b, a_serving):
        if a >= t and a - b >= 2:
            return one
        if b >= t and b - a >= 2:
            return zero
        if a == b == t - 1:
            return tie_a if a_serving else tie_b
        return V[(a, b, a_serving)]

    for a in range(t - 1, -1, -1):
        for b in range(t - 1, -1, -1):
            if a == b == t - 1:
                continue
            for a_serving in (True, False):
                p = p_srv if a_serving else p_rcv
                V[(a, b, a_serving)] = p * val(a + 1, b, True) + (1 - p) * val(a, b + 1, False)
    return val(0, 0, a_serves_first)


def match_win_prob(p_srv, p_rcv, form_sd=0.0, n_nodes=21):
    """P(A wins a best-of-five match). First server of each set is taken as a coin flip.

    form_sd > 0 adds match-day form: for the whole match, A's sideout rate is shifted by +d
    and B's by -d on the logit scale, with d ~ Normal(0, form_sd^2). The result is averaged
    over d with Gauss-Hermite quadrature. This widens the range of outcomes a matchup can
    produce, which pulls lopsided odds toward 50%.
    """
    if form_sd > 0:
        p_srv = np.asarray(p_srv, dtype=float)
        p_rcv = np.asarray(p_rcv, dtype=float)
        x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
        w = w / w.sum()
        logit = lambda p: np.log(p) - np.log1p(-p)
        expit = lambda z: 1 / (1 + np.exp(-z))
        d = form_sd * x[:, None]                          # (nodes, 1)
        so_a = expit(logit(p_rcv)[None, :] + d)
        so_b = expit(logit(1 - p_srv)[None, :] - d)
        p = match_win_prob((1 - so_b).ravel(), so_a.ravel()).reshape(so_a.shape)
        # numpy 2.0 with Apple's Accelerate can raise false warnings in matmul; check instead.
        with np.errstate(all="ignore"):
            out = w @ p
        if not np.isfinite(out).all():
            raise ValueError("non-finite match probability")
        return out
    s = 0.5 * (set_win_prob(p_srv, p_rcv, 25, True) + set_win_prob(p_srv, p_rcv, 25, False))
    f = 0.5 * (set_win_prob(p_srv, p_rcv, 15, True) + set_win_prob(p_srv, p_rcv, 15, False))
    # 3-0, 3-1, or 2-2 then win the fifth.
    return s ** 3 + 3 * s ** 3 * (1 - s) + 6 * s ** 2 * (1 - s) ** 2 * f


def simulate_set(p_srv, p_rcv, target, a_serves_first, rng):
    """One simulated set (scalars). Returns (a_points, b_points). Used to test the exact math."""
    a = b = 0
    a_serving = a_serves_first
    while not ((a >= target or b >= target) and abs(a - b) >= 2):
        if rng.random() < (p_srv if a_serving else p_rcv):
            a, a_serving = a + 1, True
        else:
            b, a_serving = b + 1, False
    return a, b
