# Batched pattern search for QAOA parameter optimization on
cloud-accessed quantum hardware

Data, notebooks and analysis for *"Batched pattern search for QAOA parameter optimization on
cloud-accessed quantum hardware"* (M. Faryad).

Everything here reproduces the paper end to end: run the notebooks to regenerate the data, or
run the analysis scripts on the released data to regenerate every figure, table and number in
the manuscript.

---

## What the study does

Four cardinality-constrained portfolio problems — choose $n/2$ of $n$ assets, for
$n = 6, 8, 10, 12$ — solved on IBM's 156-qubit Heron r2 processor `ibm_fez`, with the
**submitted job** as the unit of cost.

Two optimisers run head to head under an identical budget of jobs *and* shots:

| arm | what one job buys |
|---|---|
| batched pattern search | $4p+2$ parameter vectors, $S_{\rm job}/(4p+2)$ shots each |
| COBYLA (serial control) | 1 parameter vector, all $S_{\rm job}$ shots |

Every instance is small enough to enumerate exhaustively, so the exact optimum and the exact
ranking of every feasible portfolio are known and no claim rests on a heuristic reference.

Headline result: the batched search reaches after **one** job a parameter quality that COBYLA
needs 3, 4, 5 and 12 jobs to match, at $n = 6, 8, 10, 12$ respectively. Across four
initialisations at $n = 8$, the batched arm reaches 95% of its final value in 1.5 jobs on
average against 5.75 for COBYLA, with the ordering holding at every start.

Two follow-up experiments: a 25-job depth scan at classically optimal parameters, which locates
the device's optimal circuit depth at $p = 2$–$3$; and the replicate study above.

---

## Layout

```
notebooks/    one self-contained notebook per instance size, plus two follow-ups
data/         the JSON each notebook wrote, including every IBM job id
paper/        LaTeX source, figures, and the scripts that generate them
```

### `notebooks/`

| file | what it runs | metered QPU |
|---|---|---|
| `NB16a_qaoa_jobs_6_assets.ipynb` | 6 assets, budget 3 | ~100 s |
| `NB16b_qaoa_jobs_8_assets.ipynb` | 8 assets, budget 4 | ~193 s |
| `NB16c_qaoa_jobs_10_assets.ipynb` | 10 assets, budget 5 | ~448 s |
| `NB16d_qaoa_jobs_12_assets.ipynb` | 12 assets, budget 6 | ~587 s |
| `NB17_replicates_8_assets.ipynb` | three further initialisations at 8 assets | 536 s |
| `NB18_depth_scan.ipynb` | depth $p = 1\ldots5$ at classically optimal parameters | 152 s |

All six were executed on `ibm_fez`; the notebooks in this repository carry their output. Total
across the study: 193 jobs, 1,617 s of metered QPU time.

Each notebook is standalone — no imports from this repository — and each has a `DRY_RUN`
switch that runs the whole experiment against a calibrated noise surrogate in about a minute
at zero cost. **Start there.** Set `DRY_RUN = False`, check `ACCOUNT_NAME` and
`BACKEND_NAME`, and run top to bottom.

A ledger enforces the QPU budget at run time: before every submission it predicts the metered
cost, switches to the device's own reported usage after two jobs, and refuses to submit a job
that would exceed the cap.

### `data/`

`qaoa_jobs_<n>assets_ibm_fez.json` — one file per run, containing

* `provenance` — every hyperparameter, the backend, the date
* `circuit` — gate counts, two-qubit depth, the physical chain, the calibration CZ error
* `problem` — returns, covariance, exact optimum, exact baselines
* `ledger` — per job: IBM job id, rows, shots, predicted and measured QPU seconds, wall
  clock, **and the raw integer counts of every parameter row**
* `bps` / `serial` — full optimisation traces, including the exact noiseless quality of the
  incumbent after every job
* `readouts` — the deep read-out distributions and the $\gamma = 0$ control
* `drift` — the repeat of the first job, submitted last
* `final` — the metrics table

`qaoa_jobs_depth_scan_ibm_fez.json` and `qaoa_jobs_replicates_8assets_ibm_fez.json` hold the
two follow-up runs, in the same format.

Because the raw counts are included, every quantity in the paper can be recomputed from these
files alone — including ones we did not report, such as CVaR at a different $\alpha$.

### `paper/`

```
qaoa_job_budget.tex   manuscript (revtex4-2)
refs.bib              bibliography
analyze.py            loads the JSON, rebuilds the Hamiltonians, verifies the mapping
make_figures.py       every figure
make_tables.py        every table, and numbers.tex
figures/              generated PDFs and PNGs
```

`numbers.tex` is a file of LaTeX macros emitted by `make_tables.py`. **No number in the
manuscript is typed by hand** — each one is a macro that resolves to a value computed from the
released data, so the text cannot drift out of step with the results.

---

## Reproducing

```bash
pip install -r requirements.txt
cd paper
python3 make_figures.py      # regenerates figures/
python3 make_tables.py       # regenerates tab_*.tex and numbers.tex
latexmk -pdf qaoa_job_budget.tex
```

To reproduce the data rather than the analysis, run the notebooks. You will need an IBM
Quantum account with access to a Heron device; the free Open Plan is sufficient for the
$n = 6$ and $n = 8$ runs.

---



## Licence

Code (notebooks and scripts): MIT.
Data (`data/`) and figures: CC-BY-4.0.

## Citation

```bibtex
@misc{faryad2026jobs,
  author = {Faryad, Muhammad},
  title  = {Batched pattern search for QAOA parameter optimization on
cloud-accessed quantum hardware},
  year   = {2026},
  eprint = {<arXiv id>},
  archivePrefix = {arXiv}
}
```



## Acknowledgements

Hardware access was provided under the IBM Quantum Open Plan promotion of March 2026
(180 minutes of QPU time). The views expressed are the author's own.
