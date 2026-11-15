#!/usr/bin/env python3
"""
Bootstrap return periods of drought DURATION and INTENSITY (per-event values, not means)
for the KS-significant grid cells: Stationary vs Non-stationary SPEI-12.

Input : csv/events_stationary_sig_cells.csv, csv/events_nonstationary_sig_cells.csv
        (written by SPEI12_KS_test_sig_cells.py; one row per event per cell)
Output: csv/event_values_<scenario>_sig_cells.csv      every event value of every cell
        csv/return_period_duration.csv / return_period_intensity.csv   curves + 95 % CI
        csv/significance_sta_vs_nsta_{v}.csv                Sta vs NSta tests
        figures/return_period_*.png, difference_*.png

    python SPEI12_return_period_sig_cells.py
Needs: numpy pandas scipy matplotlib
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from scipy import stats


# =============================== CONFIG ====================================
class cfg:
    BASE = "./SPEI-12_KS_significant_cells"
    EV_STA = BASE + "/csv/events_stationary_sig_cells_{var}.csv"
    EV_NSTA = BASE + "/csv/events_nonstationary_sig_cells_{var}.csv"
    CSV_DIR = BASE + "/csv"
    FIG_DIR = BASE + "/figures"

    START_YEAR, END_YEAR = 1930, 2025   # events whose start year is in this range
    N_BOOT, CI, SEED = 1000, 95, 42
    N_GRID = 300                        # number of return periods on the curve

    # What is resampled in the bootstrap:
    #  "cell"  : whole grid cells are resampled (keeps the spatial/temporal clustering of events;
    #            neighbouring cells share the same droughts, so this gives honest CIs)  [recommended]
    #  "event" : single events are resampled (plain i.i.d. bootstrap, as in the notebook;
    #            with ~4e5 events the CI becomes extremely narrow)
    BOOT_UNIT = "cell"

    # Return-period axis:
    #  "events": T = (n+1)/rank, exactly as in the notebook (T counted in events)
    #  "years" : T in YEARS at a typical grid cell, T = (cell-years) / rank
    RP_MODE = "events"

    DUR_MONTHS_TO_YEARS = True          # notebook plots Duration/12
    ALPHA = 0.05
# ===========================================================================


# ------------------------------ core bootstrap ------------------------------
class Sample:
    """Events of one scenario, pre-sorted so rank 1 = most extreme."""
    def __init__(self, df, col, ascending, cell_index):
        v = df[col].to_numpy(float)
        ok = np.isfinite(v)
        v, cid = v[ok], cell_index[df["cell_id"].to_numpy()[ok]]
        o = np.argsort(v if ascending else -v, kind="stable")
        self.v, self.cid, self.n = v[o], cid[o], len(v)


def rank_for(T, n_b, cell_years):
    if cfg.RP_MODE == "events":
        return (n_b + 1.0) / T
    return (n_b + 1.0) / n_b * cell_years / T


def curve(S, w, T, cell_years):
    """Value at every return period T for event weights w (aligned with S.v)."""
    cw = np.cumsum(w)
    nb = cw[-1]
    r = np.ceil(np.clip(rank_for(T, nb, cell_years), 1, nb))
    idx = np.searchsorted(cw, r, side="left")
    return S.v[np.minimum(idx, S.n - 1)]


def make_grid(Ss, cell_years):
    if cfg.RP_MODE == "events":
        n = max(S.n for S in Ss)
        return np.logspace(0, np.log10(n + 1), cfg.N_GRID)
    n = max(S.n for S in Ss)
    return np.logspace(np.log10(cell_years / n), np.log10(cell_years / 10.0), cfg.N_GRID)


def run_bootstrap(Sa, Sb, ncell, T, cell_years):
    rng = np.random.default_rng(cfg.SEED)
    pa = curve(Sa, np.ones(Sa.n), T, cell_years)
    pb = curve(Sb, np.ones(Sb.n), T, cell_years)
    A = np.empty((cfg.N_BOOT, len(T))); B = np.empty_like(A)
    for k in range(cfg.N_BOOT):
        if cfg.BOOT_UNIT == "cell":                        # same cell draw for both scenarios
            m = np.bincount(rng.integers(0, ncell, ncell), minlength=ncell)
            wa, wb = m[Sa.cid], m[Sb.cid]
        else:
            wa = np.bincount(rng.integers(0, Sa.n, Sa.n), minlength=Sa.n)
            wb = np.bincount(rng.integers(0, Sb.n, Sb.n), minlength=Sb.n)
        A[k], B[k] = curve(Sa, wa, T, cell_years), curve(Sb, wb, T, cell_years)
    q = [(100 - cfg.CI) / 2, 100 - (100 - cfg.CI) / 2]
    return pa, pb, np.percentile(A, q, 0), np.percentile(B, q, 0), np.percentile(A - B, q, 0)


# ---------------------------------- plots -----------------------------------
def box(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("black"); sp.set_linewidth(2)


def xlabel():
    return "Return Period (events)" if cfg.RP_MODE == "events" else "Return Period (years)"


def plot_curves(T, pa, pb, ca, cb, ylabel, fname, loc, integer_ticks):
    plt.figure(figsize=(10, 6))
    plt.plot(T, pa, "-", marker="o", ms=3, color="blue", label="Stationary SPEI 12")
    plt.plot(T, pb, "-", marker="o", ms=3, color="red", label="Non-Stationary SPEI 12")
    plt.fill_between(T, ca[0], ca[1], color="grey", alpha=0.2, label=f"{cfg.CI}% Bootstrap CI")
    plt.fill_between(T, cb[0], cb[1], color="grey", alpha=0.2)
    plt.xlabel(xlabel(), fontsize=20, fontweight="bold")
    plt.ylabel(ylabel, fontsize=20, fontweight="bold")
    plt.xticks(fontsize=22, fontweight="bold"); plt.yticks(fontsize=22, fontweight="bold")
    plt.xscale("log"); plt.grid(True, which="both", ls="--", lw=0.5)
    ax = plt.gca()
    if integer_ticks:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    lg = plt.legend(loc=loc, fontsize=16, frameon=True)
    lg.get_frame().set_edgecolor("black"); lg.get_frame().set_linewidth(2)
    box(ax); plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()


def plot_diff(T, d_pt, d_ci, ylabel, fname):
    plt.figure(figsize=(10, 4.5))
    plt.fill_between(T, d_ci[0], d_ci[1], color="grey", alpha=0.3, label=f"{cfg.CI}% bootstrap CI")
    plt.plot(T, d_pt, color="black", lw=2, label="Stationary - Non-stationary")
    plt.axhline(0, color="red", lw=1, ls="--")
    plt.xscale("log"); plt.grid(True, which="both", ls="--", lw=0.5)
    plt.xlabel(xlabel(), fontsize=16, fontweight="bold")
    plt.ylabel("Difference in " + ylabel.lower(), fontsize=16, fontweight="bold")
    plt.legend(fontsize=12); box(plt.gca()); plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()


# ----------------------------------- main -----------------------------------
def load_events(path, label):
    df = pd.read_csv(path)
    y = df["start_year"] if "start_year" in df.columns else df["year"]
    df = df[y.between(cfg.START_YEAR, cfg.END_YEAR)].copy()
    df["Duration_years"] = df["Duration"] / 12.0
    print(f"[{label}] {len(df):,} events in {df.cell_id.nunique():,} cells "
          f"({cfg.START_YEAR}-{cfg.END_YEAR})")
    return df


def run_one(v):
    """Return-period analysis of ONE variable (v = "duration" or "intensity") on the
    grid cells that are significant for that variable only."""
    for d in (cfg.CSV_DIR, cfg.FIG_DIR):
        os.makedirs(d, exist_ok=True)
    da, db = load_events(cfg.EV_STA.format(var=v), "Sta"), load_events(cfg.EV_NSTA.format(var=v), "NSta")

    # ---- store every event value of every cell (no means) ------------------
    keep = ["cell_id", "lat", "lon", "event_id", "start", "end", "peak_date", "start_year",
            "Duration", "Duration_years", "Intensity", "I_abs"]
    for name, d in (("stationary", da), ("nonstationary", db)):
        d[[c for c in keep if c in d.columns]].sort_values(["cell_id", "event_id"]).to_csv(
            f"{cfg.CSV_DIR}/event_values_{name}_sig_cells_{v}.csv", index=False)

    cells = np.union1d(da.cell_id.unique(), db.cell_id.unique())
    cell_index = np.full(int(cells.max()) + 1, -1, int)
    cell_index[cells] = np.arange(len(cells))
    ncell = len(cells)
    n_years = (cfg.END_YEAR - cfg.START_YEAR + 1)
    cell_years = ncell * n_years
    print(f"Cells in resampling frame: {ncell:,}; boot unit = {cfg.BOOT_UNIT}; "
          f"RP mode = {cfg.RP_MODE}")

    summary = []
    for var, col, lab, loc, integer in (("duration", "Duration_years" if cfg.DUR_MONTHS_TO_YEARS
                                         else "Duration", "Duration (years)" if cfg.DUR_MONTHS_TO_YEARS else "Duration (months)", "upper left", True),
                                        ("intensity", "Intensity", "Intensity", "upper right", False)):
        if var != v:
            continue
        # duration: largest = rank 1.  intensity: signed peak SPEI (negative) -> most negative = rank 1
        asc = (var == "intensity") and bool(np.nanmedian(da[col]) < 0)
        Sa, Sb = Sample(da, col, asc, cell_index), Sample(db, col, asc, cell_index)
        T = make_grid([Sa, Sb], cell_years)
        print(f"[{var}] bootstrapping ({cfg.N_BOOT} replicates) ...")
        pa, pb, ca, cb, cd = run_bootstrap(Sa, Sb, ncell, T, cell_years)
        diff = pa - pb
        sig_T = (cd[0] > 0) | (cd[1] < 0)

        pd.DataFrame({"return_period": T,
                      "sta": pa, "sta_lo": ca[0], "sta_hi": ca[1],
                      "nsta": pb, "nsta_lo": cb[0], "nsta_hi": cb[1],
                      "diff_sta_minus_nsta": diff, "diff_lo": cd[0], "diff_hi": cd[1],
                      "diff_significant": sig_T}).to_csv(
            f"{cfg.CSV_DIR}/return_period_{var}.csv", index=False)
        plot_curves(T, pa, pb, ca, cb, lab, f"{cfg.FIG_DIR}/return_period_{var}.png", loc, integer)
        plot_diff(T, diff, cd, lab, f"{cfg.FIG_DIR}/difference_{var}.png")

        # ---- tests Sta vs NSta ----
        a, b = Sa.v, Sb.v
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        ks = stats.ks_2samp(a, b, method="asymp")
        # paired test on cells present in both (per-cell median of event values)
        ma = da.groupby("cell_id")[col].median()
        mb = db.groupby("cell_id")[col].median()
        pr = pd.concat([ma, mb], axis=1, join="inner").dropna()
        wp = stats.wilcoxon(pr.iloc[:, 0], pr.iloc[:, 1]).pvalue if len(pr) > 5 else np.nan
        summary.append(dict(
            variable=var, n_events_sta=len(a), n_events_nsta=len(b),
            median_sta=np.median(a), median_nsta=np.median(b),
            mannwhitney_p_pooled_events=mw.pvalue, ks_p_pooled_events=ks.pvalue,
            n_paired_cells=len(pr), wilcoxon_p_paired_cell_medians=wp,
            frac_return_periods_diff_significant=float(sig_T.mean()),
            significantly_different=bool(sig_T.any() and wp < cfg.ALPHA)))

    res = pd.DataFrame(summary)
    res.to_csv(f"{cfg.CSV_DIR}/significance_sta_vs_nsta_{v}.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print("\n=== Stationary vs Non-stationary ===\n", res.T)
    print("\nNote: pooled-event Mann-Whitney / KS p-values ignore spatial dependence and are "
          "almost always ~0 with this many events; rely on the paired-cell Wilcoxon and the "
          "cell-bootstrap CI of the difference.")


def main(variables=("duration", "intensity")):
    for v in variables:
        print("\n" + "=" * 70 + f"\nReturn periods: {v.upper()} (cells significant for {v} only)\n" + "=" * 70)
        run_one(v)


if __name__ == "__main__":
    main()
