"""Emit every table and inline number of the paper from the measured data."""
import json
import os
import sys
from math import comb

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import RUNS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [6, 8, 10, 12]
MACROS = {}


_W = {"6": "six", "8": "eight", "10": "ten", "12": "twelve"}


def m(k, v):
    MACROS[k] = v
    return v


def mn(base, n, v):
    """Macro name with the size spelled out: LaTeX control sequences take no digits."""
    return m(base + _W[str(n)], v)


def w(name, text):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(text)
    print("wrote", name)


# ------------------------------------------------------------------ instances
def tab_instances():
    rows = []
    for n in SIZES:
        R = RUNS[n]
        p = R.d["provenance"]
        rows.append(rf"{n} & {n//2} & {p['lam']:.3f} & {p['mu']:.0f} & "
                    rf"{comb(n, n//2)} & {R.C_opt:.1f} & {R.C_worst:.1f} & "
                    rf"{R.ratio(R.C_mean_feas):.3f} \\")
    w("tab_instances.tex", r"""\begin{tabular}{cccccccc}
$n$ & $B$ & $\lambda$ & $\mu$ & $\binom{n}{B}$ & $C_{\min}$ & $C_{\max}^{\rm feas}$
 & $r$(uniform) \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}""")


# -------------------------------------------------------------------- circuit
def tab_circuit():
    rows = []
    for n in SIZES:
        c = RUNS[n].d["circuit"]
        rows.append(rf"{n} & {RUNS[n].p} & {c['n2q']} & {c['depth2q']} & "
                    rf"{c['generic_n2q']} & {c['generic_depth2q']} & "
                    rf"{c['generic_n2q']/c['n2q']:.2f} & "
                    rf"{c['generic_depth2q']/c['depth2q']:.2f} \\")
    m("swapgatemin", f"{min(RUNS[n].d['circuit']['generic_n2q']/RUNS[n].d['circuit']['n2q'] for n in SIZES):.2f}")
    m("swapgatemax", f"{max(RUNS[n].d['circuit']['generic_n2q']/RUNS[n].d['circuit']['n2q'] for n in SIZES):.2f}")
    m("swapdepthmin", f"{min(RUNS[n].d['circuit']['generic_depth2q']/RUNS[n].d['circuit']['depth2q'] for n in SIZES):.1f}")
    m("swapdepthmax", f"{max(RUNS[n].d['circuit']['generic_depth2q']/RUNS[n].d['circuit']['depth2q'] for n in SIZES):.1f}")
    w("tab_circuit.tex", r"""\begin{tabular}{cccccccc}
& & \multicolumn{2}{c}{swap network} & \multicolumn{2}{c}{preset, level 3}
 & \multicolumn{2}{c}{ratio} \\
$n$ & $p$ & $N_{2q}$ & $D_{2q}$ & $N_{2q}$ & $D_{2q}$ & gates & depth \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}""")


# ---------------------------------------------------------------- convergence
def tab_convergence():
    rows = []
    for n in SIZES:
        R = RUNS[n]
        b = np.maximum.accumulate(R.d["bps"]["trace"]["ideal_ratio"])
        s = np.maximum.accumulate(R.d["serial"]["trace"]["ideal_ratio"])
        hit = np.where(s >= b[0])[0]
        j_match = f"{int(hit[0]) + 1}" if hit.size else r"$>%d$" % len(s)
        nb = int(np.argmax(b >= 0.95 * b[-1])) + 1
        ns = int(np.argmax(s >= 0.95 * s[-1])) + 1
        rows.append(rf"{n} & {R.p} & {len(b)} & {b[0]:.3f} & {j_match} & "
                    rf"{nb} & {ns} & {b[-1]:.3f} & {s[-1]:.3f} \\")
        mn("bpsjobone", n, f"{b[0]:.3f}")
        mn("bpsfinal", n, f"{b[-1]:.3f}")
        mn("serfinal", n, f"{s[-1]:.3f}")
        mn("jmatch", n, j_match.replace("$", ""))
        mn("njobs", n, str(len(b)))
    w("tab_convergence.tex", r"""\begin{tabular}{ccccccccc}
& & & \multicolumn{2}{c}{after one BPS job} & \multicolumn{2}{c}{jobs to 95\%}
 & \multicolumn{2}{c}{best so far} \\
$n$ & $p$ & jobs/arm & $r_{\rm BPS}$ & COBYLA needs & BPS & COBYLA
 & $r_{\rm BPS}$ & $r_{\rm COB}$ \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}""")


# --------------------------------------------------------------- final quality
def tab_final():
    rows = []
    for n in SIZES:
        R = RUNS[n]
        f = R.d["final"]
        nf = comb(n, n // 2)
        pr = np.array(R.d["readouts"]["bps"])
        fi = np.where(R.feas)[0]
        rk = int(np.where(fi[np.argsort(-pr[fi])] == R.opt_index)[0][0]) + 1
        rho = spearmanr(pr[fi], R.cost[fi]).statistic
        rhoc = spearmanr(np.array(R.d["readouts"]["control"])[fi], R.cost[fi]).statistic
        rows.append(
            rf"{n} & {f['batched pattern search']['ratio']:.3f} & "
            rf"{f['serial baseline']['ratio']:.3f} & "
            rf"{f['gamma = 0 control']['ratio']:.3f} & "
            rf"{f['uniform |+>^n']['ratio']:.3f} & "
            rf"{f['batched pattern search']['feas_mass']:.3f} & "
            rf"{f['uniform |+>^n']['feas_mass']:.3f} & "
            rf"{f['batched pattern search']['p_opt_feas'] * nf:.1f} & "
            rf"{rk}/{nf} & ${rho:.2f}$ & ${rhoc:+.2f}$ \\")
        mn("ratio", n, f"{f['batched pattern search']['ratio']:.3f}")
        mn("ratioser", n, f"{f['serial baseline']['ratio']:.3f}")
        mn("ratioctrl", n, f"{f['gamma = 0 control']['ratio']:.3f}")
        mn("ratiounif", n, f"{f['uniform |+>^n']['ratio']:.3f}")
        mn("enh", n, f"{f['batched pattern search']['p_opt_feas'] * nf:.1f}")
        mn("rank", n, str(rk))
        mn("nfeas", n, str(nf))
        mn("rho", n, f"{rho:.2f}")
        mn("rhoctrl", n, f"{rhoc:+.2f}")
        mn("drift", n, f"{R.d['drift']['shift']:+.3f}")
        mn("ratioidealbps", n, f"{f['noiseless, BPS params']['ratio']:.3f}")
        mn("ratioidealser", n, f"{f['noiseless, serial params']['ratio']:.3f}")
    w("tab_final.tex", r"""\begin{tabular}{ccccccccccc}
& \multicolumn{4}{c}{approximation ratio $\mid$ feasible}
 & \multicolumn{2}{c}{feasible mass} & & & \multicolumn{2}{c}{rank corr.} \\
$n$ & BPS & COBYLA & $\gamma{=}0$ & uniform & BPS & uniform
 & $\frac{p({\rm opt}\mid{\rm feas})}{\rm uniform}$ & rank of opt.
 & $\rho_s$ & $\rho_s^{\gamma=0}$ \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}""")


# --------------------------------------------------------------------- ledger
def tab_ledger():
    rows = []
    tq = tw = tj = ts = 0
    for n in SIZES:
        R = RUNS[n]
        log = R.d["ledger"]["log"]
        q = R.d["ledger"]["qpu_seconds"]
        wl = sum(j["wall"] for j in log)
        tq += q
        tw += wl
        tj += len(log)
        ts += R.d["ledger"]["shots"]
        rows.append(rf"{n} & {R.p} & {R.d['provenance']['shots']} & "
                    rf"{R.d['provenance']['job_shots']:,}".replace(",", "{,}") + " & " +
                    rf"{len(log)} & " +
                    rf"{R.d['ledger']['shots']:,}".replace(",", "{,}") + " & " +
                    rf"{q:.0f} & {wl:.0f} & "
                    rf"{wl/q:.1f} & {np.median([j['wall'] for j in log]):.0f} \\")
    rows.append(r"\hline")
    rows.append(rf"total & & & & {tj} & " + f"{ts:,}".replace(",", "{,}") +
                rf" & {tq:.0f} & {tw:.0f} & {tw/tq:.1f} & \\")
    m("totjobs", str(tj))
    m("totshots", f"{ts:,}".replace(",", "\\,"))
    m("totqpu", f"{tq:.0f}")
    m("totqpumin", f"{tq/60:.1f}")
    m("totwall", f"{tw:.0f}")
    m("totwallmin", f"{tw/60:.0f}")
    m("wallratio", f"{tw/tq:.1f}")
    w("tab_ledger.tex", (r"""\begin{tabular}{cccccccccc}
$n$ & $p$ & shots/cand. & shots/job & jobs & total shots & QPU\,/\,s
 & wall\,/\,s & ratio & median wall\,/\,s \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}"""))


# ------------------------------------------------------------------ noise law
def noise_law():
    n2q, eta = [], []
    for n in SIZES:
        R = RUNS[n]
        eta.append(R.fit_eta(np.array(R.d["readouts"]["bps"]),
                             np.array(R.d["bps"]["best_x"])))
        n2q.append(R.d["circuit"]["n2q"])
        mn("eta", n, f"{eta[-1]:.3f}")
        mn("ntwoq", n, str(n2q[-1]))
    n2q, eta = np.array(n2q, float), np.array(eta)
    A = np.vstack([np.ones(4), n2q]).T
    (a, b), *_ = np.linalg.lstsq(A, -np.log(1 - eta), rcond=None)
    m("etaa", f"{a:.3f}")
    m("etab", f"{b*1e3:.3f}")
    m("etamaxres", f"{np.abs(eta - (1 - np.exp(-(a + b*n2q)))).max():.3f}")
    # cost model residual
    T = lambda d: 1e-6 * (352.6 + 0.2673 * d)
    res = []
    for n in SIZES:
        d2q = RUNS[n].d["circuit"]["depth2q"]
        for j in RUNS[n].d["ledger"]["log"]:
            res.append(abs(j["qpu_measured"] - (3.0 + T(d2q) * j["rows"] * j["shots"])))
    m("costres", f"{max(res)*1e3:.0f}")
    dr, df = [], []
    for n in SIZES:
        R = RUNS[n]
        rb = np.array(R.d["readouts"]["bps"])
        xb = np.array(R.d["bps"]["best_x"])
        pt = R.twin(R.qaoa_probs(xb), R.fit_eta(rb, xb))
        dr.append(abs(R.quality(pt)["ratio"] - R.quality(rb)["ratio"]))
        df.append(abs(R.quality(pt)["feas_mass"] - R.quality(rb)["feas_mass"]))
    m("twinratiodev", f"{np.ceil(max(dr)*100)/100:.2f}")
    m("twinfeasdev", f"{np.ceil(max(df)*100)/100:.2f}")
    m("czerr", f"{RUNS[6].d['circuit']['cz_error']*100:.3f}")


if __name__ == "__main__":
    tab_instances()
    tab_circuit()
    tab_convergence()
    tab_final()
    tab_ledger()
    noise_law()
    import tables_extra
    extra = []
    for pth, fn in (("../hw/qaoa_jobs_depth_scan_ibm_fez.json", tables_extra.tab_depth),
                    ("../hw/qaoa_jobs_replicates_8assets_ibm_fez.json",
                     tables_extra.tab_replicates)):
        q = os.path.join(HERE, pth)
        if os.path.exists(q):
            fn(q, RUNS, m, mn, w)
            extra.append(q)
    if extra:
        tables_extra.totals(extra, RUNS, m, w)
        tables_extra.wallstats(extra, RUNS, m)
    lines = [r"%% auto-generated by make_tables.py -- do not edit"]
    for k, v in MACROS.items():
        lines.append(rf"\newcommand{{\{k.replace('_','')}}}{{{v}}}")
    w("numbers.tex", "\n".join(lines) + "\n")
    print(f"\n{len(MACROS)} macros")
    for k, v in MACROS.items():
        print(f"  \\{k} = {v}")
