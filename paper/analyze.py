"""Analysis of the four executed ibm_fez runs."""
import json
import numpy as np

import os
_HW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
FILES = {6: os.path.join(_HW, 'qaoa_jobs_6assets_ibm_fez.json'),
         8: os.path.join(_HW, 'qaoa_jobs_8assets_ibm_fez.json'),
         10: os.path.join(_HW, 'qaoa_jobs_10assets_ibm_fez.json'),
         12: os.path.join(_HW, 'qaoa_jobs_12assets_ibm_fez.json')}


def ising_terms(r, C, lam, mu, B):
    n = len(r)
    Q = lam * C + mu * np.ones((n, n))
    L = r + 2 * mu * B * np.ones(n)
    offset = mu * B ** 2
    h = np.zeros(n)
    J = np.zeros((n, n))
    for i in range(n):
        offset += Q[i, i] / 2 - L[i] / 2
        h[i] += -Q[i, i] / 2 + L[i] / 2
    for i in range(n):
        for j in range(i + 1, n):
            c = 2 * Q[i, j]
            offset += c / 4
            h[i] -= c / 4
            h[j] -= c / 4
            J[i, j] += c / 4
    return h, J, float(offset)


class Run:
    def __init__(self, n, path):
        d = json.load(open(path))
        self.d = d
        self.n = n
        p = d['provenance']
        self.p = p['p']
        self.prov = p
        self.B = p['budget']
        pr = d['problem']
        self.names = pr['names']
        r = np.array(pr['expected_returns'])
        C = np.array(pr['covariance'])
        self.r, self.C = r, C
        N = 2 ** n
        self.N = N
        bits = (np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1
        self.bits = bits
        quad = np.einsum('ki,ij,kj->k', bits, C, bits)
        lin = bits @ r
        viol = (bits.sum(1) - self.B) ** 2
        self.cost = p['lam'] * quad - lin + p['mu'] * viol
        self.feas = bits.sum(1) == self.B
        self.C_opt = float(pr['C_opt'])
        self.opt_index = int(pr['opt_index'])
        self.C_worst = float(pr['C_worst_feas'])
        self.C_mean_feas = float(pr['C_mean_feas'])
        h, J, off = ising_terms(r, C, p['lam'], p['mu'], self.B)
        scale = max(np.abs(h).max(), np.abs(J).max())
        self.h, self.J, self.scale, self.offset = h / scale, J / scale, scale, off
        z = 1 - 2 * bits
        diag = z @ self.h
        for i in range(n):
            for j in range(i + 1, n):
                diag += self.J[i, j] * z[:, i] * z[:, j]
        self.diag = diag
        assert np.abs(self.cost - (scale * diag + off)).max() < 1e-6
        self.order = np.argsort(diag)
        self.e_sorted = diag[self.order]
        self.alpha = p['alpha']
        self.e_ro = p['readout_err']

    # ---- helpers
    def ratio(self, c):
        return (self.C_worst - c) / (self.C_worst - self.C_opt)

    def quality(self, probs):
        fm = float(probs[self.feas].sum())
        cond = probs * self.feas / max(fm, 1e-15)
        return dict(feas_mass=fm, mean_cost=float(self.cost @ probs),
                    mean_cost_feas=float(self.cost @ cond),
                    ratio=float(self.ratio(self.cost @ cond)),
                    p_opt=float(probs[self.opt_index]),
                    p_opt_feas=float(cond[self.opt_index]))

    def qaoa_probs(self, params, reps=None):
        reps = self.p if reps is None else reps
        b, g = params[:reps], params[reps:]
        psi = np.ones(self.N, dtype=complex) / np.sqrt(self.N)
        for k in range(reps):
            psi = psi * np.exp(-1j * g[k] * self.diag)
            psi = psi.reshape([2] * self.n)
            cb, sb = np.cos(b[k]), -1j * np.sin(b[k])
            for q in range(self.n):
                ax = self.n - 1 - q
                psi = np.moveaxis(psi, ax, 0)
                a_, b_ = psi[0].copy(), psi[1].copy()
                psi[0], psi[1] = cb * a_ + sb * b_, sb * a_ + cb * b_
                psi = np.moveaxis(psi, 0, ax)
            psi = psi.reshape(-1)
        return np.abs(psi) ** 2

    def cvar(self, probs, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        q = probs[self.order]
        cum = np.cumsum(q)
        k = min(int(np.searchsorted(cum, alpha)) + 1, q.size)
        w = q[:k].copy()
        w[-1] -= max(cum[k - 1] - alpha, 0.0)
        return float(self.e_sorted[:k] @ w / w.sum()) if w.sum() > 0 else float(self.e_sorted[0])

    def readout_mix(self, p, e):
        p = p.reshape([2] * self.n).copy()
        for q in range(self.n):
            ax = self.n - 1 - q
            p = np.moveaxis(p, ax, 0)
            a_, b_ = p[0].copy(), p[1].copy()
            p[0], p[1] = (1 - e) * a_ + e * b_, e * a_ + (1 - e) * b_
            p = np.moveaxis(p, 0, ax)
        return p.reshape(-1)

    def twin(self, p_ideal, eta):
        return self.readout_mix((1 - eta) * p_ideal + eta / self.N, self.e_ro)

    def fit_eta(self, measured, params, reps=None):
        """Effective depolarising strength that reproduces the measured mean energy."""
        p_id = self.qaoa_probs(np.asarray(params), reps)
        m = float(self.diag @ measured)
        lo, hi = 0.0, 1.0
        m0 = float(self.diag @ self.twin(p_id, 0.0))
        for _ in range(60):
            mid = (lo + hi) / 2
            v = float(self.diag @ self.twin(p_id, mid))
            if (v > m) == (m0 > m):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


RUNS = {n: Run(n, f) for n, f in FILES.items()}
