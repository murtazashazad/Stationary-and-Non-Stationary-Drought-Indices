"""
Two-sample Kolmogorov-Smirnov test, ECDFs and Robinson maps
STATIONARY vs NON-STATIONARY SPEI-12
=========================================================================
Compares the monthly SPEI-12 series from the stationary method with the series
from the non-stationary (30-yr trailing window) method, over the months where
both exist.

Tests
    * every grid cell                          -> maps
    * every AR6 land region (area-weighted     -> ECDF panels, D, n, verdict
      mean series)
    * all selected land (area-weighted mean)   -> global ECDF figure

Per-test statistics
    D          KS statistic (largest vertical gap between the two ECDFs)
    D_signed   D with the DIRECTION of the gap
               positive (red)  : non-stationary SPEI tends to be HIGHER (wetter)
               negative (blue) : non-stationary SPEI tends to be LOWER  (drier)
    p          KS p-value, months treated as independent
    p_neff     KS p-value using the effective sample size (lag-1 autocorrelation)
    n, n_eff   number of months, effective two-sample size
    "Significant" in every label = p_neff < ALPHA

Figures (OUTPUT_DIR/figures, PNG + PDF)  -- Robinson projection, Antarctica removed
    KS_ECDF_global              ECDFs of the two datasets, D, n, n_eff, p, verdict
    KS_ECDF_regions             one ECDF panel per AR6 region with the same labels
    KS_map_signed_D             signed D per cell, blue gradient (negative) to red
                                gradient (positive), continuous, no classes
    KS_map_signed_D_significant same, only where the difference is significant
                                (grey = tested, not significant)
    KS_map_area_pct             % of each region's area that is significantly
                                negative (blue) / positive (red), continuous 0-100 %
    KS_map_evidence             -log10(p_neff) per cell, continuous

Notes
    * "Negative / positive" is the direction of the distribution shift between the
      two SPEI versions, NOT a Mann-Kendall / Sen's-slope trend.
    * The two series come from the same water balance, so they are strongly
      dependent; a two-sample KS test asks whether the two marginal distributions
      differ, not whether the series differ month by month.
    * The only place alpha enters the maps is the per-cell test that feeds the
      area percentages and the "significant only" map.

References
    Kolmogorov, A. (1933) Giornale dell'Istituto Italiano degli Attuari 4, 83-91.
    Smirnov, N. (1948) Annals of Mathematical Statistics 19, 279-281.
    Stephens, M.A. (1970) J. R. Stat. Soc. B 32, 115-122.
    Zwiers, F.W. & von Storch, H. (1995) J. Climate 8, 336-351.
    Bretherton, C.S. et al. (1999) J. Climate 12, 1990-2007.
    Robinson, A.H. (1974) International Yearbook of Cartography 14, 145-155.
    Harrower, M. & Brewer, C.A. (2003) The Cartographic Journal 40, 27-37.
    Crameri, F. et al. (2020) Nature Communications 11, 5444.
    Wilks, D.S. (2016) Bull. Amer. Meteor. Soc. 97, 2263-2273.
    Iturbide, M. et al. (2020) Earth Syst. Sci. Data 12, 2959-2970.

Requirements
    pip install xarray numpy pandas scipy regionmask matplotlib cartopy netCDF4

Usage
    python KS_test_ECDF_and_maps.py
    (optional) python KS_test_ECDF_and_maps.py st.nc ns.nc out_dir
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ks_2samp
from scipy.special import kolmogorov
import regionmask
import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS -- EDIT THESE
# ============================================================================
TREND_DIR = "/home/murtaza13/paper2/Grid_based_analysis/Trends/SPEI/data"
F_SPEI_ST = f"{TREND_DIR}/SPEI12_stationary.nc"
F_SPEI_NS = f"{TREND_DIR}/SPEI12_nonstationary_30yr.nc"
VAR_NAME = "SPEI12"
OUTPUT_DIR = f"{TREND_DIR}/KS_grid_trend"
EXCLUDED_REGIONS = ["EAN", "WAN", "GIC", "SAH", "ARP", "ECA"]

if len(sys.argv) >= 3:
    F_SPEI_ST, F_SPEI_NS = sys.argv[1], sys.argv[2]
if len(sys.argv) >= 4:
    OUTPUT_DIR = sys.argv[3]

FIG_DIR = f"{OUTPUT_DIR}/figures"
CSV_DIR = f"{OUTPUT_DIR}/csv"
F_OUT_NC = f"{OUTPUT_DIR}/KS_grid_results.nc"

ALPHA = 0.05
MIN_MONTHS = 120            # minimum months valid in both series to test a cell / region
LAT_BLOCK = 30              # latitude rows processed at a time (limits memory)
ACCUM_MONTHS = 12
LAT_MIN_MAP = -58.0         # everything south of this (Antarctica) is removed
LAT_MAX_MAP = 85.0
SIG_KEY = "sig_neff"        # significance field used for maps / area %: "sig" or "sig_neff"
REGION_COLS = 5             # columns in the regional ECDF figure

for d in (OUTPUT_DIR, FIG_DIR, CSV_DIR):
    os.makedirs(d, exist_ok=True)

# ============================================================================
# PLOT STYLE
# ============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.5,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "legend.fontsize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

PC = ccrs.PlateCarree()              # projection of the data
PROJ = ccrs.Robinson()               # projection of the maps

CMAP_DIVERGING = "RdBu_r"            # blue gradient = negative, red gradient = positive
CMAP_NEG = "Blues"                   # 0-100 % area, significantly negative
CMAP_POS = "Reds"                    # 0-100 % area, significantly positive
CMAP_EVIDENCE = "Reds"
COL_ST = "#1A1A1A"                   # ECDF stationary      (Okabe-Ito, colour-blind safe)
COL_NS = "#E69F00"                   # ECDF non-stationary
COL_GAP = "#CC3311"                  # marker of the D gap
COL_SIG = "#B2182B"
COL_NSIG = "#4D4D4D"
GREY = "#E3E3E3"


# ============================================================================
# STATISTICS
# ============================================================================
def lag1(x):
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else 0.0


def n_eff_of(x):
    """Effective sample size from lag-1 autocorrelation (capped at n, floored at 2)."""
    n = len(x)
    r1 = lag1(x)
    ne = n * (1.0 - r1) / (1.0 + r1) if r1 > -0.99 else float(n)
    return float(np.clip(ne, 2.0, n)), r1


def ks_p_from_neff(D, ne_a, ne_b):
    """KS p-value with effective sample sizes (Stephens 1970 small-sample adjustment)."""
    en = ne_a * ne_b / (ne_a + ne_b)
    s = np.sqrt(en)
    return float(kolmogorov((s + 0.12 + 0.11 / s) * D)), float(en)


def ecdf_gap(a, b):
    """
    Largest vertical gap between the ECDFs of a (stationary) and b (non-stationary).
    Returns D, location x of the gap, F_a(x), F_b(x) and the sign
    (+1: F_a > F_b there, i.e. non-stationary values tend to be HIGHER).
    """
    sa, sb = np.sort(a), np.sort(b)
    grid = np.concatenate([sa, sb])
    fa = np.searchsorted(sa, grid, side="right") / len(sa)
    fb = np.searchsorted(sb, grid, side="right") / len(sb)
    diff = fa - fb
    k = int(np.argmax(np.abs(diff)))
    sign = 1.0 if diff[k] >= 0 else -1.0
    return float(abs(diff[k])), float(grid[k]), float(fa[k]), float(fb[k]), sign


def ks_stats(a, b):
    """Full KS summary for two series (used for regions and the global series)."""
    r = ks_2samp(a, b, method="asymp")
    D, x_at, fa, fb, sign = ecdf_gap(a, b)
    ne_a, r1a = n_eff_of(a)
    ne_b, r1b = n_eff_of(b)
    p_ne, en = ks_p_from_neff(D, ne_a, ne_b)
    return {"D": D, "D_signed": sign * D, "p": float(r.pvalue), "p_neff": p_ne,
            "n": len(a), "n_eff": en, "x_at": x_at, "Fa": fa, "Fb": fb,
            "r1_stationary": r1a, "r1_nonstationary": r1b,
            "significant": bool(p_ne < ALPHA)}


# ============================================================================
# DATA + GRID-CELL TEST
# ============================================================================
def wrap(ds):
    """Longitudes -> -180..180, ascending."""
    return ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")


def get_regions():
    all_regions = regionmask.defined_regions.ar6.land
    keep = [i for i, ab in enumerate(all_regions.abbrevs) if ab not in EXCLUDED_REGIONS]
    return all_regions[keep]


def run_grid_ks(regions):
    st = wrap(xr.open_dataset(F_SPEI_ST))
    ns = wrap(xr.open_dataset(F_SPEI_NS))
    a_da, b_da = st[VAR_NAME], ns[VAR_NAME]
    if (a_da.shape != b_da.shape or not np.allclose(st.lat.values, ns.lat.values)
            or not np.allclose(st.lon.values, ns.lon.values)
            or not np.array_equal(st.time.values, ns.time.values)):
        raise ValueError("The two SPEI files must share the same time, lat and lon axes.")

    n_time, n_lat, n_lon = a_da.shape
    shp = (n_lat, n_lon)
    rm = regions.mask(st.lon, st.lat).values               # region number per cell, NaN outside
    w_lat = np.cos(np.deg2rad(st.lat.values))

    names = ["D", "D_signed", "p", "p_neff", "n", "n_eff", "r1_stationary", "r1_nonstationary"]
    out = {k: np.full(shp, np.nan, dtype=np.float64) for k in names}

    # area-weighted regional sums (for the regional / global ECDFs)
    acc = {int(r): {"W": np.zeros(n_time), "A": np.zeros(n_time), "B": np.zeros(n_time)}
           for r in regions.numbers}
    acc_all = {"W": np.zeros(n_time), "A": np.zeros(n_time), "B": np.zeros(n_time)}

    print(f"Grid {n_lat} x {n_lon}, {n_time} months; testing cells with >= {MIN_MONTHS} common months")
    t0 = time.time()
    n_done = 0
    for i0 in range(0, n_lat, LAT_BLOCK):
        i1 = min(i0 + LAT_BLOCK, n_lat)
        A = a_da.isel(lat=slice(i0, i1)).values.reshape(n_time, -1).astype(np.float64)
        B = b_da.isel(lat=slice(i0, i1)).values.reshape(n_time, -1).astype(np.float64)
        both = np.isfinite(A) & np.isfinite(B)
        ok_cells = np.where(both.sum(axis=0) >= MIN_MONTHS)[0]
        blk = {k: np.full(A.shape[1], np.nan) for k in names}
        for j in ok_cells:
            ok = both[:, j]
            a, b = A[ok, j], B[ok, j]
            r = ks_2samp(a, b, method="asymp")
            D, _, _, _, sign = ecdf_gap(a, b)
            ne_a, r1a = n_eff_of(a)
            ne_b, r1b = n_eff_of(b)
            p_ne, en = ks_p_from_neff(D, ne_a, ne_b)
            blk["D"][j], blk["D_signed"][j] = D, sign * D
            blk["p"][j], blk["p_neff"][j] = r.pvalue, p_ne
            blk["n"][j], blk["n_eff"][j] = len(a), en
            blk["r1_stationary"][j], blk["r1_nonstationary"][j] = r1a, r1b
        for k in names:
            out[k][i0:i1] = blk[k].reshape(i1 - i0, n_lon)

        # area-weighted regional sums for this block of latitude rows
        ids = rm[i0:i1].ravel()
        wcell = np.repeat(w_lat[i0:i1], n_lon)
        for rid in np.unique(ids[np.isfinite(ids)]):
            cols = ids == rid
            bc = both[:, cols]
            wc = wcell[cols]
            wv = bc * wc
            sa = (np.where(bc, A[:, cols], 0.0) * wc).sum(axis=1)
            sb = (np.where(bc, B[:, cols], 0.0) * wc).sum(axis=1)
            for tgt in (acc[int(rid)], acc_all):
                tgt["W"] += wv.sum(axis=1)
                tgt["A"] += sa
                tgt["B"] += sb

        n_done += len(ok_cells)
        print(f"  lat rows {i0:3d}-{i1:3d} done | cells tested so far: {n_done:,} | {time.time() - t0:.0f}s")

    valid = np.isfinite(out["p"])
    # -log10(p) in float64 BEFORE the float32 cast (tiny p-values underflow to 0 in float32)
    mlog_p = -np.log10(np.clip(out["p"], 1e-300, 1.0))
    mlog_p_neff = -np.log10(np.clip(out["p_neff"], 1e-300, 1.0))
    sig = np.where(valid, (out["p"] < ALPHA).astype(float), np.nan)
    sig_neff = np.where(valid, (out["p_neff"] < ALPHA).astype(float), np.nan)

    data_vars = {k: (("lat", "lon"), v.astype("float32")) for k, v in out.items()}
    data_vars.update({
        "mlog10_p": (("lat", "lon"), mlog_p.astype("float32")),
        "mlog10_p_neff": (("lat", "lon"), mlog_p_neff.astype("float32")),
        "sig": (("lat", "lon"), sig.astype("float32")),
        "sig_neff": (("lat", "lon"), sig_neff.astype("float32")),
        "region": (("lat", "lon"), rm.astype("float32")),
    })
    ds = xr.Dataset(data_vars, coords={"lat": st.lat.values, "lon": st.lon.values})
    desc = {
        "D": "KS statistic (stationary vs non-stationary SPEI-12)",
        "D_signed": "KS statistic with sign: positive = non-stationary SPEI tends higher (wetter)",
        "p": "KS p-value assuming independent months",
        "p_neff": "KS p-value using effective sample size (lag-1 autocorrelation)",
        "mlog10_p": "-log10(p), months independent (capped at 300)",
        "mlog10_p_neff": "-log10(p_neff) (capped at 300)",
        "sig": "1 = significant difference (p < alpha)",
        "sig_neff": "1 = significant difference (p_neff < alpha)",
        "region": "AR6 region number",
        "n": "number of common months", "n_eff": "two-sample effective sample size",
        "r1_stationary": "lag-1 autocorrelation, stationary SPEI",
        "r1_nonstationary": "lag-1 autocorrelation, non-stationary SPEI",
    }
    for k, v in desc.items():
        ds[k].attrs["long_name"] = v
    ds.attrs = {
        "title": f"Grid-cell KS test: stationary vs non-stationary SPEI-{ACCUM_MONTHS}",
        "alpha": ALPHA, "min_months": MIN_MONTHS,
        "stationary_file": os.path.basename(F_SPEI_ST),
        "nonstationary_file": os.path.basename(F_SPEI_NS),
        "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)}
           for v in ds.data_vars}
    ds.to_netcdf(F_OUT_NC, encoding=enc)
    print(f"\nSaved: {F_OUT_NC}")

    # regional and global series
    def series(t):
        ok = t["W"] > 0
        return (np.where(ok, t["A"] / np.where(ok, t["W"], 1), np.nan),
                np.where(ok, t["B"] / np.where(ok, t["W"], 1), np.nan))

    reg_series = {int(r): series(acc[int(r)]) for r in regions.numbers}
    glob_series = series(acc_all)
    st.close()
    ns.close()
    return ds, rm, reg_series, glob_series


def test_series(sa, sb):
    ok = np.isfinite(sa) & np.isfinite(sb)
    if ok.sum() < MIN_MONTHS:
        return None, None, None
    a, b = sa[ok], sb[ok]
    return ks_stats(a, b), a, b


def regional_tables(ds, regions, rm, reg_series):
    """Per-region KS on the area-weighted mean series + area-weighted % significant cells."""
    sig = ds[SIG_KEY].values
    Ds = ds["D_signed"].values
    w = np.cos(np.deg2rad(ds.lat.values))[:, None] * np.ones((1, ds.sizes["lon"]))
    pct_neg = np.full(sig.shape, np.nan)
    pct_pos = np.full(sig.shape, np.nan)
    rows, ecdf_data = [], {}
    for num, ab, nm in zip(regions.numbers, regions.abbrevs, regions.names):
        m = (rm == num) & np.isfinite(Ds)
        row = {"region": ab, "name": nm, "n_cells": int(m.sum())}
        if m.any():
            wt = w[m].sum()
            neg = 100.0 * w[m & (sig == 1) & (Ds < 0)].sum() / wt
            pos = 100.0 * w[m & (sig == 1) & (Ds > 0)].sum() / wt
            pct_neg[rm == num], pct_pos[rm == num] = neg, pos
            row.update({"area_pct_sig_negative": neg, "area_pct_sig_positive": pos,
                        "area_pct_sig_total": neg + pos})
        stats, a, b = test_series(*reg_series[int(num)])
        if stats is not None:
            row.update({"series_" + k: v for k, v in stats.items()
                        if k in ("D", "D_signed", "p", "p_neff", "n", "n_eff", "significant")})
            ecdf_data[ab] = (stats, a, b, nm)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{CSV_DIR}/ks_regional_summary.csv", index=False)
    print("\n" + df[["region", "n_cells", "area_pct_sig_negative", "area_pct_sig_positive"]]
          .round(1).to_string(index=False))
    coords = {"lat": ds.lat.values, "lon": ds.lon.values}
    return (xr.DataArray(pct_neg, coords=coords, dims=("lat", "lon")),
            xr.DataArray(pct_pos, coords=coords, dims=("lat", "lon")), ecdf_data)


# ============================================================================
# ECDF FIGURES
# ============================================================================
def verdict_text(s):
    return "Significant difference" if s["significant"] else "No significant difference"


def draw_ecdf(ax, a, b, s, compact=False, legend=False):
    lw = 1.8 if compact else 2.6
    fs = 10 if compact else 15
    xa, xb = np.sort(a), np.sort(b)
    ya = np.arange(1, len(xa) + 1) / len(xa)
    yb = np.arange(1, len(xb) + 1) / len(xb)
    ax.step(xa, ya, where="post", color=COL_ST, lw=lw, label="Stationary", zorder=3)
    ax.step(xb, yb, where="post", color=COL_NS, lw=lw, label="Non-stationary (30-yr)",
            zorder=4)
    # the KS distance D: vertical gap between the ECDFs where they differ most
    ax.plot([s["x_at"], s["x_at"]], [s["Fa"], s["Fb"]], color=COL_GAP, lw=lw + 0.8,
            solid_capstyle="butt", zorder=5)
    ax.plot([s["x_at"]], [s["Fa"]], "o", color=COL_GAP, ms=4 if compact else 7, zorder=6)
    ax.plot([s["x_at"]], [s["Fb"]], "o", color=COL_GAP, ms=4 if compact else 7, zorder=6)
    txt = (f"D = {s['D']:.3f}\n"
           f"n = {s['n']:,}   n_eff = {s['n_eff']:.0f}\n"
           f"p = {s['p_neff']:.3g}")
    ax.text(0.97, 0.30, txt, transform=ax.transAxes, ha="right", va="top", fontsize=fs,
            fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.8, alpha=0.9),
            zorder=7)
    ax.text(0.97, 0.04, verdict_text(s), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fs, fontweight="bold", color=COL_SIG if s["significant"] else COL_NSIG,
            zorder=7)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="0.85", lw=0.7, zorder=0)
    ax.tick_params(labelsize=9 if compact else 14, width=1.2)
    if legend:
        ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.7")


def plot_global_ecdf(glob_series):
    s, a, b = test_series(*glob_series)
    if s is None:
        print("  (global ECDF skipped: too few common months)")
        return None
    fig, ax = plt.subplots(figsize=(10, 7.2))
    draw_ecdf(ax, a, b, s)
    ax.set_xlabel(f"SPEI-{ACCUM_MONTHS} (area-weighted mean of selected land)")
    ax.set_ylabel("Empirical cumulative distribution")
    ax.set_title(f"Two-sample KS test: stationary vs non-stationary SPEI-{ACCUM_MONTHS}",
                 pad=12)
    handles = [Line2D([0], [0], color=COL_ST, lw=2.6, label="Stationary"),
               Line2D([0], [0], color=COL_NS, lw=2.6, label="Non-stationary (30-yr)"),
               Line2D([0], [0], color=COL_GAP, lw=3.4, label="KS distance D")]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="0.7")
    savefig(fig, "KS_ECDF_global")
    pd.DataFrame([{k: v for k, v in s.items()}]).to_csv(f"{CSV_DIR}/ks_global_summary.csv",
                                                       index=False)
    return s


def plot_region_ecdfs(ecdf_data):
    keys = list(ecdf_data.keys())
    if not keys:
        print("  (regional ECDFs skipped: no region with enough months)")
        return
    ncol = REGION_COLS
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.9 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for k, ab in enumerate(keys):
        ax = axes[k // ncol, k % ncol]
        ax.set_visible(True)
        s, a, b, nm = ecdf_data[ab]
        draw_ecdf(ax, a, b, s, compact=True)
        ax.set_title(f"{ab}  {nm}", fontsize=11, loc="left", pad=4)
        if k % ncol == 0:
            ax.set_ylabel("ECDF", fontsize=12)
        if k // ncol == nrow - 1 or k + ncol >= len(keys):
            ax.set_xlabel("SPEI-12", fontsize=12)
    handles = [Line2D([0], [0], color=COL_ST, lw=2.6, label="Stationary"),
               Line2D([0], [0], color=COL_NS, lw=2.6, label="Non-stationary (30-yr)"),
               Line2D([0], [0], color=COL_GAP, lw=3.4, label="KS distance D")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               prop={"weight": "bold", "size": 15}, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"ECDFs of stationary vs non-stationary SPEI-{ACCUM_MONTHS} by AR6 region "
                 f"(area-weighted regional mean)", fontsize=22, fontweight="bold", y=0.998)
    fig.subplots_adjust(hspace=0.45, wspace=0.22, top=0.965, bottom=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/KS_ECDF_regions.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {FIG_DIR}/KS_ECDF_regions.png / .pdf")


# ============================================================================
# MAP FIGURES (Robinson, Antarctica removed)
# ============================================================================
def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {FIG_DIR}/{name}.png / .pdf")


def prep(da):
    """Drop Antarctica (south of LAT_MIN_MAP)."""
    return da.where(da.lat > LAT_MIN_MAP)


def setup_map(ax, regions):
    ax.set_extent((-180, 180, LAT_MIN_MAP, LAT_MAX_MAP), crs=PC)
    ax.add_geometries(list(regions.polygons), crs=PC, facecolor="none",
                      edgecolor="black", linewidth=0.6, zorder=3)
    ax.coastlines(linewidth=0.7, color="black", zorder=4)
    ax.gridlines(draw_labels=False, linewidth=0.4, color="grey", alpha=0.5,
                 linestyle=":", xlocs=np.arange(-180, 181, 60),
                 ylocs=np.arange(-60, 91, 30), zorder=2)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
        sp.set_edgecolor("black")


def add_colorbar(fig, ax, im, label, extend="neither", ticks=None):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.045,
                      aspect=40, extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.set_label(label, fontsize=16, fontweight="bold")
    cb.ax.tick_params(labelsize=14, width=1.5, length=6)
    cb.outline.set_linewidth(1.5)
    for t in cb.ax.get_xticklabels():
        t.set_fontweight("bold")
    return cb


def draw(ax, da, **kw):
    return ax.pcolormesh(da.lon, da.lat, da.values, transform=PC, shading="auto",
                         rasterized=True, zorder=1, **kw)


SIGNED_LABEL = ("Signed KS statistic D\n(negative: non-stationary drier  |  "
                "positive: non-stationary wetter)")


def d_limit(ds):
    return float(np.nanpercentile(np.abs(ds["D_signed"].values), 99))


def plot_signed_D(ds, regions):
    """Signed D for every tested cell: blue gradient (negative) to red gradient (positive)."""
    lim = d_limit(ds)
    fig = plt.figure(figsize=(14, 7.6))
    ax = plt.axes(projection=PROJ)
    im = draw(ax, prep(ds["D_signed"]), cmap=CMAP_DIVERGING, norm=Normalize(-lim, lim))
    setup_map(ax, regions)
    ax.set_title(f"KS test: stationary vs non-stationary SPEI-{ACCUM_MONTHS}",
                 fontsize=18, fontweight="bold", pad=12)
    add_colorbar(fig, ax, im, SIGNED_LABEL, extend="both")
    savefig(fig, "KS_map_signed_D")


def plot_signed_D_significant(ds, regions):
    """Signed D only where the difference is significant; grey = tested, not significant."""
    D, sig = ds["D_signed"], ds[SIG_KEY]
    lim = d_limit(ds)
    tested = np.isfinite(D.values)
    pct = 100.0 * np.mean(sig.values[tested] == 1)
    fig = plt.figure(figsize=(14, 7.6))
    ax = plt.axes(projection=PROJ)
    grey = prep(xr.where(np.isfinite(D) & (sig != 1), 1.0, np.nan))
    ax.pcolormesh(grey.lon, grey.lat, grey.values, cmap=ListedColormap([GREY]),
                  vmin=0, vmax=1, transform=PC, shading="auto", rasterized=True, zorder=1)
    im = draw(ax, prep(D.where(sig == 1)), cmap=CMAP_DIVERGING, norm=Normalize(-lim, lim))
    setup_map(ax, regions)
    ax.set_title(f"Significant differences, stationary vs non-stationary SPEI-{ACCUM_MONTHS}"
                 + "\n" + f"{pct:.1f}% of tested cells significant", fontsize=17,
                 fontweight="bold", pad=12)
    ax.legend(handles=[Patch(facecolor=GREY, edgecolor="black", label="Not significant")],
              loc="lower left", frameon=False, prop={"weight": "bold", "size": 14})
    add_colorbar(fig, ax, im, SIGNED_LABEL, extend="both")
    savefig(fig, "KS_map_signed_D_significant")


def plot_area_pct(pct_neg, pct_pos, regions):
    """Share of each region's area with a significant negative / positive difference."""
    panels = [(pct_neg, CMAP_NEG, "(a) Significantly negative (non-stationary drier)"),
              (pct_pos, CMAP_POS, "(b) Significantly positive (non-stationary wetter)")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 15.6), subplot_kw={"projection": PROJ})
    for ax, (da, cmap, title) in zip(axes, panels):
        im = draw(ax, prep(da), cmap=cmap, vmin=0, vmax=100)
        setup_map(ax, regions)
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=10)
        add_colorbar(fig, ax, im, "Area-weighted share of region (%)",
                     ticks=np.arange(0, 101, 20))
    fig.suptitle(f"Area of each AR6 region with a significant stationary vs non-stationary "
                 f"SPEI-{ACCUM_MONTHS} difference", fontsize=19, fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.30, top=0.95)
    savefig(fig, "KS_map_area_pct")


def plot_evidence(ds, regions):
    """Continuous strength of evidence -log10(p_neff) per cell."""
    da = prep(ds["mlog10_p_neff"].astype("float64"))
    vals = da.values[np.isfinite(da.values)]
    vmax = float(np.percentile(vals, 99)) if vals.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    fig = plt.figure(figsize=(14, 7.6))
    ax = plt.axes(projection=PROJ)
    im = draw(ax, da, cmap=CMAP_EVIDENCE, vmin=0, vmax=vmax)
    setup_map(ax, regions)
    ax.set_title(f"Strength of evidence, autocorrelation-corrected KS test (SPEI-{ACCUM_MONTHS})",
                 fontsize=17, fontweight="bold", pad=12)
    add_colorbar(fig, ax, im, r"$-\log_{10}(p_{n_{eff}})$", extend="max")
    savefig(fig, "KS_map_evidence")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("KS TEST: STATIONARY vs NON-STATIONARY SPEI-12")
    print(f"  stationary     : {F_SPEI_ST}")
    print(f"  non-stationary : {F_SPEI_NS}")
    print("=" * 80)
    regions = get_regions()
    ds, rm, reg_series, glob_series = run_grid_ks(regions)

    v = ds[SIG_KEY].values
    v = v[np.isfinite(v)]
    print(f"\nCells tested: {len(v):,} | significant ({SIG_KEY}, alpha={ALPHA}): "
          f"{100.0 * (v == 1).mean():.1f}%")

    pct_neg, pct_pos, ecdf_data = regional_tables(ds, regions, rm, reg_series)

    print("\nGenerating figures ...")
    plot_global_ecdf(glob_series)
    plot_region_ecdfs(ecdf_data)
    plot_signed_D(ds, regions)
    plot_signed_D_significant(ds, regions)
    plot_area_pct(pct_neg, pct_pos, regions)
    plot_evidence(ds, regions)
    print("\nDONE. Outputs in:", os.path.abspath(OUTPUT_DIR))
