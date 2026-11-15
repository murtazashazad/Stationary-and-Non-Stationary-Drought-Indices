"""
GRID-CELL two-sample KS test: STATIONARY vs NON-STATIONARY SPI-12
=========================================================================
For EVERY grid cell the monthly SPI-12 series obtained with the stationary
method is compared with the series obtained with the non-stationary (30-yr
trailing window) method, over the months where both exist (1930 onward).
Months are treated as independent (classical two-sample KS test, no
autocorrelation / effective-sample-size correction).

Inputs
    .../Trends/data/SPI12_stationary.nc
    .../Trends/data/SPI12_nonstationary_30yr.nc

Per-cell results (saved to NetCDF)
    D        KS statistic
    p        KS p-value (months treated as independent)
    p_fdr    Benjamini-Hochberg adjusted p across all tested cells
    sig      1 if p < ALPHA
    sig_fdr  1 if p_fdr < ALPHA  (field-significance control)
    n        number of common months

Figures (./KS_grid_test_SPI12/figures, PNG + PDF, 600 dpi)
    KS_grid_significance_map : RED = significant difference, BLUE = no significant difference
    KS_grid_D_map            : KS statistic D per grid cell
    KS_grid_area_barchart    : % of land AREA (cos-latitude weighted) significant / not significant
    Maps are plain longitude-latitude (equirectangular) axes, only cells inside the kept
    IPCC AR6 land regions are tested and drawn, with region borders and labels.

Caveat
    The two series come from the SAME water balance, so a two-sample KS asks only
    whether the two marginal distributions differ, not whether the series differ
    month by month. Testing ~57,000 cells gives chance false positives; sig_fdr controls this.

Requirements:
    pip install xarray numpy pandas scipy regionmask matplotlib shapely netCDF4

Usage
    python KS_test_grid_SPI12_stationary_vs_nonstationary.py
    (optional) python KS_test_grid_SPI12_stationary_vs_nonstationary.py st.nc ns.nc out_dir
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import ks_2samp
import regionmask

warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS -- EDIT THESE
# ============================================================================
TREND_DIR = "/home/murtaza13/paper2/Grid_based_analysis/Trends/SPI/data"
F_SPI_ST = f"{TREND_DIR}/SPI12_stationary.nc"
F_SPI_NS = f"{TREND_DIR}/SPI12_nonstationary_30yr.nc"
VAR_NAME = "SPI12"
OUTPUT_DIR = f"{TREND_DIR}/KS_grid_test_SPI12"

if len(sys.argv) >= 3:
    F_SPI_ST, F_SPI_NS = sys.argv[1], sys.argv[2]
if len(sys.argv) >= 4:
    OUTPUT_DIR = sys.argv[3]

FIG_DIR = f"{OUTPUT_DIR}/figures"
CSV_DIR = f"{OUTPUT_DIR}/csv"
F_OUT_NC = f"{OUTPUT_DIR}/KS_grid_results.nc"

ALPHA = 0.05
MIN_MONTHS = 120            # minimum months valid in both series to test a cell
LAT_BLOCK = 30              # latitude rows processed at a time (limits memory)
EXCLUDED_REGIONS = ["GIC", "SAH", "ARP", "ECA"]     # same as stationary_SPI12.py
ACCUM_MONTHS = 12

for d in (OUTPUT_DIR, FIG_DIR, CSV_DIR):
    os.makedirs(d, exist_ok=True)

# ============================================================================
# PLOT STYLE
# ============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.6,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COL_SIG = "#B2182B"     # red  = significant difference
COL_NS = "#2166AC"      # blue = no significant difference
COL_LAND = "#efefef"    # land background
COL_BORDER = "#7a7a7a"  # country borders
COUNTRY_RES = 110       # Natural Earth country borders (regionmask; downloaded once)


# ============================================================================
# STATISTICS
# ============================================================================
def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def get_regions():
    all_regions = regionmask.defined_regions.ar6.land
    keep = [i for i, ab in enumerate(all_regions.abbrevs) if ab not in EXCLUDED_REGIONS]
    return all_regions[keep]


def get_region_mask(lon, lat, regions):
    """Boolean (lat, lon): True for cells inside a kept AR6 land region."""
    return np.isfinite(regions.mask(lon, lat).values)


def run_grid_ks():
    st = xr.open_dataset(F_SPI_ST)
    ns = xr.open_dataset(F_SPI_NS)
    a_da, b_da = st[VAR_NAME], ns[VAR_NAME]
    if (a_da.shape != b_da.shape or not np.allclose(st.lat.values, ns.lat.values)
            or not np.allclose(st.lon.values, ns.lon.values)
            or not np.array_equal(st.time.values, ns.time.values)):
        raise ValueError("The two SPI files must share the same time, lat and lon axes.")

    n_time, n_lat, n_lon = a_da.shape
    shp = (n_lat, n_lon)
    regions = get_regions()
    in_reg = get_region_mask(st.lon.values, st.lat.values, regions)   # only kept AR6 land cells are tested
    print(f"Cells inside the {len(regions.abbrevs)} kept AR6 land regions: {int(in_reg.sum()):,}")
    names = ["D", "p", "n"]
    out = {k: np.full(shp, np.nan, dtype=np.float64) for k in names}

    print(f"Grid {n_lat} x {n_lon}, {n_time} months; testing cells with >= {MIN_MONTHS} common months")
    t0 = time.time()
    n_done = 0
    for i0 in range(0, n_lat, LAT_BLOCK):
        i1 = min(i0 + LAT_BLOCK, n_lat)
        A = a_da.isel(lat=slice(i0, i1)).values.reshape(n_time, -1).astype(np.float64)
        B = b_da.isel(lat=slice(i0, i1)).values.reshape(n_time, -1).astype(np.float64)
        enough = (np.isfinite(A) & np.isfinite(B)).sum(axis=0) >= MIN_MONTHS
        ok_cells = np.where(enough & in_reg[i0:i1].ravel())[0]
        blk = {k: np.full(A.shape[1], np.nan) for k in names}
        for j in ok_cells:
            ok = np.isfinite(A[:, j]) & np.isfinite(B[:, j])
            a, b = A[ok, j], B[ok, j]
            r = ks_2samp(a, b, method="asymp")
            blk["D"][j], blk["p"][j], blk["n"][j] = r.statistic, r.pvalue, len(a)
        for k in names:
            out[k][i0:i1] = blk[k].reshape(i1 - i0, n_lon)
        n_done += len(ok_cells)
        print(f"  lat rows {i0:3d}-{i1:3d} done | cells tested so far: {n_done:,} | {time.time() - t0:.0f}s")

    valid = np.isfinite(out["p"])
    sig = np.where(valid, (out["p"] < ALPHA).astype(float), np.nan)
    p_fdr = bh_fdr(out["p"].ravel()).reshape(shp)
    sig_fdr = np.where(valid, (p_fdr < ALPHA).astype(float), np.nan)

    data_vars = {k: (("lat", "lon"), v.astype("float32")) for k, v in out.items()}
    data_vars.update({
        "p_fdr": (("lat", "lon"), p_fdr.astype("float32")),
        "sig": (("lat", "lon"), sig.astype("float32")),
        "sig_fdr": (("lat", "lon"), sig_fdr.astype("float32")),
        "in_region": (("lat", "lon"), in_reg.astype("float32")),
    })
    ds = xr.Dataset(data_vars, coords={"lat": st.lat.values, "lon": st.lon.values})
    desc = {
        "D": "KS statistic (stationary vs non-stationary SPI-12)",
        "p": "KS p-value, months treated as independent",
        "p_fdr": "Benjamini-Hochberg adjusted p across all tested grid cells",
        "sig": "1 = significant difference (p < alpha)",
        "sig_fdr": "1 = significant difference (FDR-adjusted p < alpha)",
        "n": "number of common months",
        "in_region": "1 = cell inside the kept IPCC AR6 land regions",
    }
    for k, v in desc.items():
        ds[k].attrs["long_name"] = v
    ds.attrs = {
        "title": f"Grid-cell KS test: stationary vs non-stationary SPI-{ACCUM_MONTHS}",
        "alpha": ALPHA, "min_months": MIN_MONTHS,
        "stationary_file": os.path.basename(F_SPI_ST),
        "nonstationary_file": os.path.basename(F_SPI_NS),
        "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)}
           for v in ds.data_vars}
    ds.to_netcdf(F_OUT_NC, encoding=enc)
    print(f"\nSaved: {F_OUT_NC}")
    st.close()
    ns.close()
    return ds


def area_weights(ds):
    """cos(latitude) cell-area weights on the (lat, lon) grid."""
    return np.cos(np.deg2rad(ds.lat.values))[:, None] * np.ones((1, ds.lon.size))


def area_pct(ds, key):
    """Area-weighted % of tested land area that is significant / not significant."""
    v = ds[key].values
    w = area_weights(ds)
    ok = np.isfinite(v)
    tot = w[ok].sum()
    return 100.0 * w[ok & (v == 1)].sum() / tot, 100.0 * w[ok & (v == 0)].sum() / tot


VARIANTS = [("sig", "KS test (p < 0.05)"),
            ("sig_fdr", "KS test + FDR")]


def summarise(ds):
    rows = []
    for key, lab in VARIANTS:
        v = ds[key].values
        v = v[np.isfinite(v)]
        a_sig, a_ns = area_pct(ds, key)
        rows.append({"test": lab, "n_cells": len(v), "n_significant": int((v == 1).sum()),
                     "pct_cells_significant": 100.0 * (v == 1).mean(),
                     "pct_cells_not_significant": 100.0 * (v == 0).mean(),
                     "pct_area_significant": a_sig, "pct_area_not_significant": a_ns})
    df = pd.DataFrame(rows).round(2)
    df.to_csv(f"{CSV_DIR}/ks_grid_summary.csv", index=False)
    print("\n" + df.to_string(index=False))
    return df


# ============================================================================
# FIGURES  (plain longitude-latitude axes, like a standard equirectangular map)
# ============================================================================
def get_countries():
    err = None
    for ver in ("natural_earth_v5_1_2", "natural_earth_v5_0_0"):
        try:
            return getattr(getattr(regionmask.defined_regions, ver), f"countries_{COUNTRY_RES}")
        except Exception as e:
            err = e
    print(f"[note] country borders unavailable ({type(err).__name__}); drawing without them")
    return None


def _rings(poly):
    for p in getattr(poly, "geoms", [poly]):
        yield p.exterior.xy
        for h in p.interiors:
            yield h.xy


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/{name}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {FIG_DIR}/{name}.png / .pdf")


def setup_map(ax, regions, countries, label_regions=True):
    ax.set_facecolor("white")
    if countries is not None:
        for poly in countries.polygons:
            for x, y in _rings(poly):
                ax.fill(x, y, color=COL_LAND, lw=0, zorder=1)
                ax.plot(x, y, color=COL_BORDER, lw=0.45, zorder=3)
    for poly, ab in zip(regions.polygons, regions.abbrevs):
        for x, y in _rings(poly):
            ax.plot(x, y, color="black", lw=1.0, solid_joinstyle="round", zorder=4)
        if label_regions:
            big = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)
            c = big.representative_point()
            ax.text(c.x, c.y, ab, ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.70))
    ax.set_xlim(-180, 180); ax.set_ylim(-58, 85); ax.set_aspect("equal", adjustable="box")
    xt, yt = np.arange(-180, 181, 60), np.arange(-60, 81, 30)
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([f"{abs(v)}\u00b0{'W' if v < 0 else 'E' if v > 0 else ''}" for v in xt])
    ax.set_yticklabels([f"{abs(v)}\u00b0{'S' if v < 0 else 'N' if v > 0 else ''}" for v in yt])
    ax.tick_params(labelsize=15, width=1.6, length=6)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.set_xlabel("Longitude", fontsize=16, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=16, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_linewidth(1.6); sp.set_edgecolor("black")


def plot_significance_map(ds, regions, countries):
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.pcolormesh(ds.lon, ds.lat, np.ma.masked_invalid(ds["sig"].values), cmap=ListedColormap([COL_NS, COL_SIG]),
                  vmin=0, vmax=1, shading="auto", rasterized=True, zorder=2, antialiased=False)
    setup_map(ax, regions, countries)
    a_sig, a_ns = area_pct(ds, "sig")
    ax.set_title(f"Grid-cell KS test: stationary vs non-stationary SPI-{ACCUM_MONTHS}",
                 loc="left", fontsize=19, fontweight="bold", pad=10)
    handles = [Patch(facecolor=COL_SIG, edgecolor="black", linewidth=1.5,
                     label=f"Significant difference (p < {ALPHA}): {a_sig:.1f}% of land area"),
               Patch(facecolor=COL_NS, edgecolor="black", linewidth=1.5,
                     label=f"No significant difference (p \u2265 {ALPHA}): {a_ns:.1f}% of land area")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True,
              edgecolor="black", fancybox=False, prop={"weight": "bold", "size": 15},
              handlelength=2.4, handleheight=1.5, columnspacing=2.5)
    savefig(fig, "KS_grid_significance_map")


def plot_D_map(ds, regions, countries):
    fig, ax = plt.subplots(figsize=(14, 7.6))
    vmax = float(np.nanpercentile(ds["D"].values, 99))
    im = ax.pcolormesh(ds.lon, ds.lat, np.ma.masked_invalid(ds["D"].values), cmap="Reds", vmin=0,
                       vmax=vmax, shading="auto", rasterized=True, zorder=2, antialiased=False)
    setup_map(ax, regions, countries)
    ax.set_title(f"KS statistic D between stationary and non-stationary SPI-{ACCUM_MONTHS}",
                 loc="left", fontsize=19, fontweight="bold", pad=10)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, fraction=0.05,
                      aspect=40, shrink=0.7, extend="max")
    cb.set_label("KS statistic D", fontsize=16, fontweight="bold")
    cb.ax.tick_params(labelsize=14, width=1.6, length=6)
    cb.outline.set_linewidth(1.6)
    for t in cb.ax.get_xticklabels():
        t.set_fontweight("bold")
    savefig(fig, "KS_grid_D_map")


def plot_area_barchart(ds):
    """Grouped bars: % of land area with significant (red) / non-significant (blue) difference."""
    labels = [lab for _, lab in VARIANTS]
    sig = np.array([area_pct(ds, k)[0] for k, _ in VARIANTS])
    ns = np.array([area_pct(ds, k)[1] for k, _ in VARIANTS])
    x = np.arange(len(VARIANTS)); w = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 7))
    b1 = ax.bar(x - w / 2, sig, w, color=COL_SIG, edgecolor="black", linewidth=1.5,
                label=f"Significant difference (p < {ALPHA})", zorder=3)
    b2 = ax.bar(x + w / 2, ns, w, color=COL_NS, edgecolor="black", linewidth=1.5,
                label=f"No significant difference (p \u2265 {ALPHA})", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=15, fontweight="bold")
    ax.set_xlim(-0.6, len(VARIANTS) - 0.4)
    ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Share of land area (%)", fontsize=17, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15, width=1.6, length=6)
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.yaxis.grid(True, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1, frameon=True, edgecolor="black",
              fancybox=False, prop={"weight": "bold", "size": 15})
    ax.set_title(f"Area share of significant differences\nstationary vs non-stationary SPI-{ACCUM_MONTHS}",
                 fontsize=17, fontweight="bold", pad=14)
    savefig(fig, "KS_grid_area_barchart")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("GRID-CELL KS TEST: STATIONARY vs NON-STATIONARY SPI-12")
    print(f"  stationary     : {F_SPI_ST}")
    print(f"  non-stationary : {F_SPI_NS}")
    print("=" * 80)
    ds = run_grid_ks()
    summarise(ds)
    print("\nGenerating figures ...")
    regions = get_regions()
    countries = get_countries()
    plot_significance_map(ds, regions, countries)
    plot_D_map(ds, regions, countries)
    plot_area_barchart(ds)
    print("\nDONE. Outputs in:", os.path.abspath(OUTPUT_DIR))
