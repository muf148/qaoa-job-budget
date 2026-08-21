"""Figures for 'QAOA under a hardware-job budget'. Run from paper/."""
import json
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import RUNS  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito, colour-blind safe; every series also carries a distinct marker/linestyle
BPS_C, SER_C, THIRD_C, FOURTH_C = "#0072B2", "#D55E00", "#009E73", "#E69F00"
GREY, LGREY = "#4d4d4d", "#b0b0b0"
SIZES = [6, 8, 10, 12]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1: concept
def fig_concept():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1))
    for ax, (title, npts, color) in zip(
            axes, [("batched pattern search\n10 points $\\times$ 512 shots", 10, BPS_C),
                   ("serial optimiser\n1 point $\\times$ 5120 shots", 1, SER_C)]):
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="none", ec=GREY, lw=0.8))
        for i in range(npts):
            w = 1.0 / npts
            ax.add_patch(plt.Rectangle((i * w + 0.006, 0.04), w - 0.012, 0.92,
                                       fc=color, ec="white", lw=0.8, alpha=0.85))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, -0.22, "one job = one PUB = 5120 shots", ha="center",
                va="top", transform=ax.transAxes, color=GREY, fontsize=8)
    fig.suptitle("How one job's shot budget is spent", y=1.06)
    save(fig, "fig_concept")


# ------------------------------------------------------- fig 2: circuit synthesis
def fig_circuit():
    n2q = [RUNS[n].d["circuit"]["n2q"] for n in SIZES]
    g2q = [RUNS[n].d["circuit"]["generic_n2q"] for n in SIZES]
    d2q = [RUNS[n].d["circuit"]["depth2q"] for n in SIZES]
    gd2q = [RUNS[n].d["circuit"]["generic_depth2q"] for n in SIZES]
    lab = [f"$n={n}$\n$p={RUNS[n].p}$" for n in SIZES]
    x = np.arange(4)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax, (a, b, ylab) in zip((a1, a2),
                                [(n2q, g2q, "two-qubit gates"),
                                 (d2q, gd2q, "two-qubit depth")]):
        ax.bar(x - 0.19, a, 0.36, color=BPS_C, label="SWAP network", zorder=3)
        ax.bar(x + 0.19, b, 0.36, color=LGREY, label="preset pass manager, L3", zorder=3)
        for xi, (u, v) in enumerate(zip(a, b)):
            ax.text(xi, max(u, v) * 1.04, f"{v/u:.2f}$\\times$", ha="center",
                    fontsize=7.5, color=GREY)
        ax.set_xticks(x)
        ax.set_xticklabels(lab)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", zorder=0)
        ax.set_ylim(0, max(max(a), max(b)) * 1.18)
    a1.legend(frameon=False, loc="upper left")
    save(fig, "fig_circuit")


# ------------------------------------------------------- fig 3: the main result
def fig_convergence():
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.2), sharex="col")
    for j, n in enumerate(SIZES):
        R = RUNS[n]
        b, s = R.d["bps"]["trace"], R.d["serial"]["trace"]
        jb = np.arange(1, len(b["ideal_ratio"]) + 1)
        js = np.arange(1, len(s["ideal_ratio"]) + 1)

        ax = axes[0, j]
        ax.plot(jb, np.maximum.accumulate(b["ideal_ratio"]), "o-", color=BPS_C,
                label="batched pattern search", zorder=3)
        ax.plot(js, np.maximum.accumulate(s["ideal_ratio"]), "s--", color=SER_C,
                label="COBYLA (serial)", zorder=3)
        ax.set_title(f"$n={n}$, $p={R.p}$")
        ax.grid(zorder=0)
        ax.set_ylim(0.55, 0.99)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.text(0.97, 0.05, f"uniform: {R.ratio(R.C_mean_feas):.3f}", fontsize=6.8,
                color=GREY, ha="right", va="bottom", transform=ax.transAxes)
        if j == 0:
            ax.set_ylabel("noiseless ratio\nof the incumbent")

        ax = axes[1, j]
        ax.plot(jb, b["best_cvar"], "o-", color=BPS_C, zorder=3)
        ax.plot(js, s["best"], "s--", color=SER_C, zorder=3)
        ax.axhline(R.cvar(np.full(R.N, 1 / R.N)), color=GREY, ls=":", lw=1.0,
                   label=r"uniform $|+\rangle^{\otimes n}$")
        ax.grid(zorder=0)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("hardware jobs")
        if j == 0:
            ax.set_ylabel(r"best measured CVaR$_{0.5}$")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2[-1:], l1 + l2[-1:], frameon=False, ncol=3, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout()
    save(fig, "fig_convergence")


# ------------------------------------------------- fig 4: final read-out quality
def fig_final():
    keys = ["batched pattern search", "serial baseline", "gamma = 0 control",
            "uniform |+>^n", "noiseless, BPS params"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    x = np.arange(4)
    series = [("batched pattern search", BPS_C, "o-", "batched pattern search"),
              ("serial baseline", SER_C, "s--", "COBYLA (serial)"),
              ("noiseless, BPS params", THIRD_C, "^:", "noiseless, BPS parameters"),
              ("gamma = 0 control", GREY, "v-.", r"$\gamma=0$ control")]

    for ax, metric, ylab in zip(
            axes, ["ratio", "feas_mass", None],
            ["approx. ratio $|$ feasible", "feasible mass",
             r"$p(\mathrm{opt}|\mathrm{feas})\,/\,$uniform"]):
        for key, c, st, lab in series:
            if metric is None:
                from math import comb
                y, ub = [], []
                for n in SIZES:
                    v = RUNS[n].d["final"][key]["p_opt_feas"] * comb(n, n // 2)
                    if v == 0:                       # never observed: plot the upper bound
                        fm = RUNS[n].d["final"][key]["feas_mass"]
                        v = comb(n, n // 2) / (8192 * fm)
                        ub.append((SIZES.index(n), v))
                    y.append(v)
                for xi, v in ub:
                    ax.annotate("", xy=(xi, v * 0.45), xytext=(xi, v),
                                arrowprops=dict(arrowstyle="->", color=c, lw=1.0))
                ax.set_yscale("log")
            else:
                y = [RUNS[n].d["final"][key][metric] for n in SIZES]
            ax.plot(x, y, st, color=c, label=lab, zorder=3)
        if metric is None:
            ax.axhline(1.0, color=LGREY, lw=1.0, ls="-", zorder=2)
        else:
            ax.plot(x, [RUNS[n].d["final"]["uniform |+>^n"][metric] for n in SIZES],
                    "-", color=LGREY, lw=1.2, zorder=2, label=r"uniform $|+\rangle^{\otimes n}$")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}" for n in SIZES])
        ax.set_xlabel("assets $n$")
        ax.set_ylabel(ylab)
        ax.grid(zorder=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    save(fig, "fig_final")


# ------------------------------------------------------- fig 5: device ranking
def fig_ranking():
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.2))
    for ax, n in zip(axes, SIZES):
        R = RUNS[n]
        pr = np.array(R.d["readouts"]["bps"])
        fi = np.where(R.feas)[0]
        ax.scatter(R.cost[fi], pr[fi], s=7, color=BPS_C, alpha=0.55,
                   edgecolors="none", zorder=3, label="feasible portfolios")
        ax.scatter([R.cost[R.opt_index]], [pr[R.opt_index]], s=42, marker="*",
                   color=SER_C, zorder=4, label="true optimum")
        ax.axhline(1 / len(fi), color=GREY, ls=":", lw=1.0,
                   label="uniform over feasible")
        ax.set_yscale("log")
        ax.set_xlabel("exact portfolio cost")
        ax.set_title(f"$n={n}$")
        ax.grid(zorder=0)
        if n == 6:
            ax.set_ylabel("measured probability")
        rk = int(np.where(fi[np.argsort(-pr[fi])] == R.opt_index)[0][0]) + 1
        from scipy.stats import spearmanr
        rho = spearmanr(pr[fi], R.cost[fi]).statistic
        ax.text(0.96, 0.06, f"optimum ranked {rk} of {len(fi)}\n"
                            rf"$\rho_s = {rho:.2f}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=GREY)
        ax.margins(x=0.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_ranking")


# ------------------------------------------------------- fig 6: noise model
def fig_noise():
    n2q, eta = [], []
    for n in SIZES:
        R = RUNS[n]
        eta.append(R.fit_eta(np.array(R.d["readouts"]["bps"]), np.array(R.d["bps"]["best_x"])))
        n2q.append(R.d["circuit"]["n2q"])
    n2q, eta = np.array(n2q, float), np.array(eta)
    A = np.vstack([np.ones(4), n2q]).T
    (a, b), *_ = np.linalg.lstsq(A, -np.log(1 - eta), rcond=None)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    xx = np.linspace(0, 1100, 200)
    a1.plot(xx, 1 - np.exp(-(a + b * xx)), "-", color=GREY, lw=1.2,
            label=fr"$1-e^{{-({a:.3f}+{b*1e3:.3f}\times10^{{-3}}N_{{2q}})}}$")
    a1.plot(n2q, eta, "o", color=BPS_C, ms=6, zorder=3, label="fitted on hardware")
    for x_, y_, n in zip(n2q, eta, SIZES):
        a1.annotate(f"$n={n}$", (x_, y_), textcoords="offset points", xytext=(6, -9),
                    fontsize=7, color=GREY)
    a1.set_xlabel("two-qubit gates $N_{2q}$")
    a1.set_ylabel(r"effective depolarising strength $\eta$")
    a1.legend(frameon=False, loc="lower right", fontsize=7)
    a1.grid(zorder=0)
    a1.set_ylim(0, 1)

    labels = ["ratio $|$ feas", "feasible mass", r"$p(\mathrm{opt}|\mathrm{feas})$"]
    mk = ["o", "s", "^"]
    for i, key in enumerate(["ratio", "feas_mass", "p_opt_feas"]):
        xs, ys = [], []
        for n in SIZES:
            R = RUNS[n]
            meas = np.array(R.d["readouts"]["bps"])
            x0 = np.array(R.d["bps"]["best_x"])
            et = R.fit_eta(meas, x0)
            xs.append(R.quality(R.twin(R.qaoa_probs(x0), et))[key])
            ys.append(R.quality(meas)[key])
        a2.plot(xs, ys, mk[i], color=[BPS_C, SER_C, THIRD_C][i], ms=5, ls="none",
                label=labels[i], zorder=3)
    lim = [0, 0.95]
    a2.plot(lim, lim, "-", color=LGREY, lw=1.0, zorder=2)
    a2.set_xlabel("depolarising model")
    a2.set_ylabel("measured on ibm_fez")
    a2.set_xlim(*lim)
    a2.set_ylim(*lim)
    a2.legend(frameon=False, fontsize=7, loc="upper left")
    a2.grid(zorder=0)
    fig.tight_layout()
    save(fig, "fig_noise")
    return a, b


# ------------------------------------------------------- fig 7: the cost of a job
def fig_cost():
    T = lambda d: 1e-6 * (352.6 + 0.2673 * d)
    wall, size, shots, qmod = [], [], [], []
    for n in SIZES:
        d2q = RUNS[n].d["circuit"]["depth2q"]
        for j in RUNS[n].d["ledger"]["log"]:
            wall.append(j["wall"])
            size.append(n)
            shots.append(j["rows"] * j["shots"])
            qmod.append(3.0 + T(d2q) * j["rows"] * j["shots"])
    for p in ("../hw/qaoa_jobs_depth_scan_ibm_fez.json",
              "../hw/qaoa_jobs_replicates_8assets_ibm_fez.json"):
        q = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
        if not os.path.exists(q):
            continue
        for j in json.load(open(q))["ledger"]["log"]:
            wall.append(j["wall"])
            shots.append(j["rows"] * j["shots"])
            qmod.append(3.0 + T(j["d2q"]) * j["rows"] * j["shots"])
            size.append({36: 6, 72: 8, 120: 10, 180: 12}.get(j["d2q"], 0)
                        or 6 + 2 * ((j["d2q"] // 3 - 1) // 2))
    wall, size = np.array(wall), np.array(size)
    shots, qmod = np.array(shots, float), np.array(qmod)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7))
    for n, c, mk in zip(SIZES, [BPS_C, SER_C, THIRD_C, FOURTH_C], "os^v"):
        k = size == n
        if k.sum():
            a1.plot(shots[k] / 1e3, wall[k], mk, color=c, ms=3.6, ls="none", alpha=0.6,
                    label=f"$n={n}$", zorder=3)
    o = np.argsort(shots)
    a1.plot(shots[o] / 1e3, qmod[o], "-", color=GREY, lw=1.1, zorder=4,
            label="modelled QPU time")
    a1.axhline(wall.min(), color="k", ls="--", lw=0.9, zorder=5,
               label=f"fastest round trip, {wall.min():.1f} s")
    a1.set_yscale("log")
    a1.set_xlabel("shots in the job / $10^3$")
    a1.set_ylabel("wall-clock time / s")
    a1.legend(frameon=False, fontsize=6.6, loc="upper left", ncol=2)
    a1.grid(zorder=0)

    for n, c, mk in zip(SIZES, [BPS_C, SER_C, THIRD_C, FOURTH_C], "os^v"):
        v = np.sort(wall[size == n])
        if v.size:
            a2.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=c,
                    label=f"$n={n}$", zorder=3)
    a2.set_xscale("log")
    a2.set_xlabel("wall-clock time per job / s")
    a2.set_ylabel("empirical CDF")
    a2.grid(zorder=0)
    a2.legend(frameon=False, fontsize=7, loc="lower right")
    a2.text(0.03, 0.99, f"{len(wall)} jobs\nmedian {np.median(wall):.1f} s\n"
                        f"max {wall.max():.0f} s", transform=a2.transAxes,
            fontsize=7, va="top", color=GREY,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.tight_layout()
    save(fig, "fig_cost")
    print(f"   wall: n={len(wall)} median {np.median(wall):.1f}s min {wall.min():.1f}s "
          f"max {wall.max():.0f}s total {wall.sum():.0f}s")


# ------------------------------------------------------- optional: depth scan
DEPTHS = [1, 2, 3, 4, 5]


def fig_depth(path):
    from math import comb
    scan = json.load(open(path))["scan"]
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.0), sharex="col")
    for j, n in enumerate(SIZES):
        e = scan[str(n)]["depths"]
        nf = comb(n, n // 2)
        meas = [e[str(p)]["meas"]["noiseless"]["measured"] for p in DEPTHS]
        ideal = [e[str(p)]["meas"]["noiseless"]["noiseless"] for p in DEPTHS]
        mod = [e[str(p)]["meas"]["model"]["measured"] if "model" in e[str(p)]["meas"]
               else None for p in DEPTHS]
        pstar = DEPTHS[int(np.argmax([q["ratio"] for q in meas]))]

        ax = axes[0, j]
        ax.axvline(pstar, color=BPS_C, lw=7, alpha=0.12, zorder=1)
        ax.plot(DEPTHS, [q["ratio"] for q in ideal], "^:", color=THIRD_C,
                label="noiseless", zorder=3)
        ax.plot(DEPTHS, [q["ratio"] for q in meas], "o-", color=BPS_C,
                label="measured on ibm_fez", zorder=4)
        if any(mod):
            ax.plot(DEPTHS, [q["ratio"] if q else np.nan for q in mod], "s--", color=SER_C,
                    label="measured, noise-aware params", zorder=3)
        ax.axhline(RUNS[n].ratio(RUNS[n].C_mean_feas), color=GREY, ls=":", lw=1.0,
                   label="uniform over feasible")
        ax.plot([RUNS[n].p], [meas[DEPTHS.index(RUNS[n].p)]["ratio"]], "o", mfc="none",
                mec="k", ms=11, mew=1.2, zorder=5)
        ax.set_title(f"$n={n}$,  $p^\\star={pstar}$")
        ax.set_ylim(0.42, 1.0)
        ax.grid(zorder=0)
        if j == 0:
            ax.set_ylabel("approx. ratio\n$|$ feasible")

        ax = axes[1, j]
        ax.axvline(pstar, color=BPS_C, lw=7, alpha=0.12, zorder=1)
        ax.plot(DEPTHS, [q["p_opt_feas"] * nf for q in meas], "o-", color=BPS_C, zorder=4)
        if any(mod):
            ax.plot(DEPTHS, [q["p_opt_feas"] * nf if q else np.nan for q in mod], "s--",
                    color=SER_C, zorder=3)
        ax.axhline(1.0, color=GREY, ls=":", lw=1.0)
        ax.set_xlabel("QAOA layers $p$")
        ax.set_xticks(DEPTHS)
        ax.set_yscale("log")
        ax.set_ylim(0.7, 60)
        ax.grid(zorder=0)
        if j == 0:
            ax.set_ylabel(r"$p(\mathrm{opt}|\mathrm{feas})$" "\n" r"$/\,$uniform")
    hh, ll = [], []
    for ax in axes[0]:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in ll:
                hh.append(h_)
                ll.append(l_)
    fig.legend(hh, ll, frameon=False, ncol=4, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fig_depth")


# ------------------------------------------------------- optional: replicates
def fig_replicates(path):
    d = json.load(open(path))
    ref = RUNS[8]
    allst = {"0.50": dict(bps=ref.d["bps"]["trace"]["ideal_ratio"],
                          ser=ref.d["serial"]["trace"]["ideal_ratio"],
                          fb=ref.d["final"]["batched pattern search"]["ratio"],
                          fs=ref.d["final"]["serial baseline"]["ratio"])}
    for k, v in d["starts"].items():
        allst[k] = dict(bps=v["bps"]["trace"]["ideal_ratio"],
                        ser=v["serial"]["trace"]["ideal_ratio"],
                        fb=v["final"]["bps"]["ratio"], fs=v["final"]["serial"]["ratio"])
    dts = sorted(allst, key=float)
    x = np.arange(len(dts))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.1, 2.5))
    for key, c, mk, lab in (("bps", BPS_C, "o-", "batched pattern search"),
                            ("ser", SER_C, "s--", "COBYLA (serial)")):
        curves = [np.maximum.accumulate(allst[k][key]) for k in dts]
        L = min(len(v) for v in curves)
        arr = np.array([v[:L] for v in curves])
        jj = np.arange(1, L + 1)
        for row in arr:
            a1.plot(jj, row, "-", color=c, lw=0.7, alpha=0.35, zorder=2)
        a1.plot(jj, arr.mean(0), mk, color=c, label=lab, zorder=4)
        a1.fill_between(jj, arr.min(0), arr.max(0), color=c, alpha=0.15, lw=0, zorder=1)
        n95 = [int(np.argmax(v >= 0.95 * v[-1])) + 1 for v in arr]
        a2.plot(x, n95, mk, color=c, zorder=3)
        print(f"   {key}: n95 = {n95}, mean {np.mean(n95):.2f}")
    a1.axhline(ref.ratio(ref.C_mean_feas), color=GREY, ls=":", lw=1.0,
               label="uniform over feasible")
    a1.set_xlabel("hardware jobs")
    a1.set_ylabel("noiseless ratio\nof the incumbent")
    a1.set_xticks(range(1, 11, 3))
    a1.grid(zorder=0)

    a2.set_xlabel(r"initialisation $\Delta t$")
    a2.set_ylabel("jobs to 95% of own final")
    a2.set_xticks(x)
    a2.set_xticklabels([f"{float(k):.2f}" for k in dts])
    a2.set_yticks(range(0, 10, 2))
    a2.grid(zorder=0)

    a3.plot(x, [allst[k]["fb"] for k in dts], "o-", color=BPS_C, zorder=3)
    a3.plot(x, [allst[k]["fs"] for k in dts], "s--", color=SER_C, zorder=3)
    a3.axhline(ref.d["final"]["gamma = 0 control"]["ratio"], color=GREY, ls="-.", lw=1.0,
               label=r"$\gamma=0$ control")
    a3.set_xlabel(r"initialisation $\Delta t$")
    a3.set_ylabel("measured ratio $|$ feasible")
    a3.set_xticks(x)
    a3.set_xticklabels([f"{float(k):.2f}" for k in dts])
    a3.grid(zorder=0)
    a3.legend(frameon=False, fontsize=7, loc="center right")

    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    save(fig, "fig_replicates")


if __name__ == "__main__":
    fig_concept()
    fig_circuit()
    fig_convergence()
    fig_final()
    fig_ranking()
    a, b = fig_noise()
    print(f"noise law: eta = 1 - exp(-({a:.4f} + {b:.6f} N2q))")
    fig_cost()
    for p, fn in (("../hw/qaoa_jobs_depth_scan_ibm_fez.json", fig_depth),
                  ("../hw/qaoa_jobs_replicates_8assets_ibm_fez.json", fig_replicates)):
        q = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
        if os.path.exists(q):
            fn(q)
        else:
            print("skipping", os.path.basename(p), "(not yet available)")
