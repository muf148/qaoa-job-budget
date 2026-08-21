"""Tables and macros for the two follow-up experiments (depth scan, replicates)."""
import json
import os
from math import comb

import numpy as np

SIZES = [6, 8, 10, 12]
DEPTHS = [1, 2, 3, 4, 5]
T_SHOT = lambda d2q: 1e-6 * (352.6 + 0.2673 * d2q)


def load(path):
    return json.load(open(path))


def tab_depth(path, RUNS, m, mn, w):
    d = load(path)
    scan = d["scan"]
    rows = []
    for n in SIZES:
        e = scan[str(n)]["depths"]
        nf = comb(n, n // 2)
        meas = [e[str(p)]["meas"]["noiseless"]["measured"] for p in DEPTHS]
        ideal = [e[str(p)]["meas"]["noiseless"]["noiseless"] for p in DEPTHS]
        pstar = DEPTHS[int(np.argmax([q["ratio"] for q in meas]))]
        rows.append(str(n) + " & " + " & ".join(f"{q['ratio']:.3f}" for q in meas)
                    + f" & {pstar} & {RUNS[n].p} " + r"\\")
        rows.append(r"& " + " & ".join(r"\textit{%.3f}" % q["ratio"] for q in ideal)
                    + r" & & \\")
        rows.append(r"& " + " & ".join(r"\small{%.0f$\times$}" % (q["p_opt_feas"] * nf)
                                       for q in meas) + r" & & \\[3pt]")
        mn("pstar", n, str(pstar))
        mn("depthbest", n, f"{max(q['ratio'] for q in meas):.3f}")
        mn("depthrun", n, f"{meas[DEPTHS.index(RUNS[n].p)]['ratio']:.3f}")
        mn("depthone", n, f"{meas[0]['ratio']:.3f}")
        mn("enhbest", n, f"{max(q['p_opt_feas'] * nf for q in meas):.0f}")
        mn("enhone", n, f"{meas[0]['p_opt_feas'] * nf:.0f}")
        mn("idealfive", n, f"{ideal[-1]['ratio']:.3f}")

    d12 = scan["12"]["depths"]
    m("depthgaintwelve", "%.3f" % (d12["2"]["meas"]["noiseless"]["measured"]["ratio"]
                                   - d12["5"]["meas"]["noiseless"]["measured"]["ratio"]))
    na = [d12[str(p)]["meas"]["model"]["measured"]["ratio"]
          - d12[str(p)]["meas"]["noiseless"]["measured"]["ratio"] for p in DEPTHS]
    m("noiseawaremax", f"{max(abs(x) for x in na):.3f}")
    m("depthjobs", str(d["ledger"]["jobs"]))
    m("depthqpu", f"{d['ledger']['qpu_seconds']:.0f}")

    head = (r"\begin{tabular}{cccccccc}" "\n"
            r"$n$ & $p=1$ & $p=2$ & $p=3$ & $p=4$ & $p=5$ & $p^\star$ & $p$ run \\" "\n"
            r"\hline" "\n")
    w("tab_depth.tex", head + "\n".join(rows) + "\n" + r"\end{tabular}")


def tab_replicates(path, RUNS, m, mn, w):
    d = load(path)
    ref = RUNS[8]
    allst = {"0.50": dict(bps=ref.d["bps"]["trace"]["ideal_ratio"],
                          ser=ref.d["serial"]["trace"]["ideal_ratio"],
                          fb=ref.d["final"]["batched pattern search"],
                          fs=ref.d["final"]["serial baseline"])}
    for k, v in d["starts"].items():
        allst[k] = dict(bps=v["bps"]["trace"]["ideal_ratio"],
                        ser=v["serial"]["trace"]["ideal_ratio"],
                        fb=v["final"]["bps"], fs=v["final"]["serial"])

    rows, nb_, ns_, bf, sf, mb, ms = [], [], [], [], [], [], []
    for k in sorted(allst, key=float):
        st = allst[k]
        b = np.maximum.accumulate(st["bps"])
        s_ = np.maximum.accumulate(st["ser"])
        hit = np.where(s_ >= b[0])[0]
        jm = str(int(hit[0]) + 1) if hit.size else r"$>%d$" % len(s_)
        a = int(np.argmax(b >= 0.95 * b[-1])) + 1
        c = int(np.argmax(s_ >= 0.95 * s_[-1])) + 1
        nb_.append(a)
        ns_.append(c)
        bf.append(b[-1])
        sf.append(s_[-1])
        mb.append(st["fb"]["ratio"])
        ms.append(st["fs"]["ratio"])
        tag = r"$^{\dagger}$" if k == "0.50" else r"\phantom{$^\dagger$}"
        rows.append(f"{float(k):.2f}{tag} & {b[0]:.3f} & {s_[0]:.3f} & {jm} & {a} & {c} & "
                    f"{b[-1]:.3f} & {s_[-1]:.3f} & {st['fb']['ratio']:.3f} & "
                    f"{st['fs']['ratio']:.3f} " + r"\\")
    rows.append(r"\hline")
    rows.append(r"mean & %.3f & %.3f & & %.2f & %.2f & %.3f & %.3f & %.3f & %.3f \\"
                % (np.mean([allst[k]["bps"][0] for k in allst]),
                   np.mean([allst[k]["ser"][0] for k in allst]),
                   np.mean(nb_), np.mean(ns_), np.mean(bf), np.mean(sf),
                   np.mean(mb), np.mean(ms)))

    for name, val in (("repnbmean", f"{np.mean(nb_):.2f}"),
                      ("repnsmean", f"{np.mean(ns_):.2f}"),
                      ("repnbrange", f"{min(nb_)}--{max(nb_)}"),
                      ("repnsrange", f"{min(ns_)}--{max(ns_)}"),
                      ("repbpsfinal", f"{np.mean(bf):.3f}"),
                      ("repserfinal", f"{np.mean(sf):.3f}"),
                      ("repbpssd", f"{np.std(bf):.3f}"),
                      ("repsersd", f"{np.std(sf):.3f}"),
                      ("repbpsjobone", f"{np.mean([allst[k]['bps'][0] for k in allst]):.3f}"),
                      ("repserjobone", f"{np.mean([allst[k]['ser'][0] for k in allst]):.3f}"),
                      ("repmeasbps", f"{np.mean(mb):.3f}"),
                      ("repmeasser", f"{np.mean(ms):.3f}"),
                      ("repmeasbpssd", f"{np.std(mb):.3f}"),
                      ("repmeassersd", f"{np.std(ms):.3f}"),
                      ("repwins", str(sum(1 for a, b in zip(mb, ms) if a > b))),
                      ("repjobs", str(d["ledger"]["jobs"])),
                      ("repqpu", f"{d['ledger']['qpu_seconds']:.0f}")):
        m(name, val)

    head = (r"\begin{tabular}{cccccccccc}" "\n"
            r"& \multicolumn{2}{c}{after one job} & COBYLA "
            r"& \multicolumn{2}{c}{jobs to 95\%} & \multicolumn{2}{c}{best so far}"
            r" & \multicolumn{2}{c}{read-out} \\" "\n"
            r"$\Delta t$ & BPS & COB & matches at & BPS & COB & BPS & COB & BPS & COB \\" "\n"
            r"\hline" "\n")
    w("tab_replicates.tex", head + "\n".join(rows) + "\n" + r"\end{tabular}")


def wallstats(paths, RUNS, m):
    """Wall-clock statistics: the only genuinely measured timing in the study."""
    wall, shots = [], []
    for n in SIZES:
        for j in RUNS[n].d["ledger"]["log"]:
            wall.append(j["wall"])
            shots.append(j["rows"] * j["shots"])
    for p in paths:
        for j in load(p)["ledger"]["log"]:
            wall.append(j["wall"])
            shots.append(j["rows"] * j["shots"])
    wall, shots = np.array(wall), np.array(shots)
    m("medianwall", f"{np.median(wall):.1f}")
    m("minwall", f"{wall.min():.1f}")
    m("maxwall", f"{wall.max():.0f}")
    m("meanwall", f"{wall.mean():.0f}")
    lo = shots <= np.percentile(shots, 25)
    m("minshots", f"{shots.min():,}".replace(",", "\\,"))
    m("medianwallsmall", f"{np.median(wall[lo]):.1f}")
    m("wallnjobs", str(len(wall)))


def totals(paths, RUNS, m, w):
    """Grand totals and the out-of-sample test of the cost model."""
    res, jobs, qpu, wall, shots = [], 0, 0.0, 0.0, 0
    for p in paths:
        d = load(p)
        for j in d["ledger"]["log"]:
            res.append(abs(j["qpu_measured"]
                           - (3.0 + T_SHOT(j["d2q"]) * j["rows"] * j["shots"])))
            wall += j["wall"]
        jobs += d["ledger"]["jobs"]
        qpu += d["ledger"]["qpu_seconds"]
        shots += d["ledger"]["shots"]
    m("oosjobs", str(jobs))
    m("oosres", f"{max(res) * 1e3:.1f}")
    m("oosqpu", f"{qpu:.0f}")
    mainq = sum(RUNS[n].d["ledger"]["qpu_seconds"] for n in SIZES)
    mainw = sum(j["wall"] for n in SIZES for j in RUNS[n].d["ledger"]["log"])
    mainj = sum(len(RUNS[n].d["ledger"]["log"]) for n in SIZES)
    mains = sum(RUNS[n].d["ledger"]["shots"] for n in SIZES)
    m("alljobs", str(mainj + jobs))
    m("allqpu", f"{mainq + qpu:.0f}")
    m("allqpumin", f"{(mainq + qpu) / 60:.0f}")
    m("allwall", f"{mainw + wall:.0f}")
    m("allwallmin", f"{(mainw + wall) / 60:.0f}")
    m("allwallratio", f"{(mainw + wall) / (mainq + qpu):.1f}")
    m("allshots", f"{mains + shots:,}".replace(",", "\\,"))
