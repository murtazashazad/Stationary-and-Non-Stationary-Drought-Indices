#!/usr/bin/env python3
"""
KS TEST: STATIONARY vs NON-STATIONARY (rolling 30-yr) SPI-12 TRENDS, GRID-CELL BASED
================================================================================
Needs: numpy pandas scipy xarray netCDF4 regionmask matplotlib shapely

Inputs (netCDF)
  STATIONARY_NC     SPEI12_stationary_trend.nc
  NONSTATIONARY_NC  SPEI12_nonstationary_30yr_trend.nc

The script auto-detects the trend variable (e.g. Sen's slope) and chooses the test design:

  MODE "temporal"  both files are (time/window, lat, lon):
        at every cell the two trend series are cut to their common steps and compared with the
        two-sample KS test (H0: same distribution).  p_neff additionally corrects for serial
        correlation (n_eff = n (1-r)/(1+r), Wilks 2011).
  MODE "spatial"   both files are (lat, lon) i.e. ONE trend value per cell (a single value per cell
        cannot be KS-tested on its own):
        the KS test is run on the values inside a (2*WIN+1)x(2*WIN+1) cell neighbourhood centred
        on every cell (local KS), stationary vs non-stationary.  Significance is optionally
        FDR-controlled because neighbouring windows overlap (field significance).

Map: RED = significant difference (p < ALPHA), BLUE = not significant.
Outputs (OUTPUT_DIR): csv/KS_trend_cell_results.csv, csv/KS_trend_region_summary.csv,
  netcdf/KS_trend_results.nc, figures/Fig_KS_trend_map, Fig_KS_trend_D_map, Fig_KS_trend_panel,
  Fig_KS_trend_global_ecdf (global ECDFs of tau and OLS slope, stationary vs non-stationary)
  (.pdf vector + .png 600 dpi)
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
from scipy.stats import ks_2samp, kstwobign
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# =============================================================================
# CONFIG  (edit here only)
# =============================================================================
class cfg:
    NONSTATIONARY_NC = "/home/murtaza13/paper2/Grid_based_analysis/Trends/SPI/data/SPI12_nonstationary_30yr_trend.nc"
    STATIONARY_NC = "/home/murtaza13/paper2/Grid_based_analysis/Trends/SPI/data/SPI12_stationary_trend.nc"
    VAR_STATIONARY = None       # e.g. "sen_slope"; None = auto-detect
    VAR_NONSTATIONARY = None
    ALPHA = 0.05
    FDR = None                  # None = auto (True in spatial mode, False in temporal); or True/False
    ONLY_KEPT_REGIONS = True    # keep only cells in the 40 AR6 land regions
    EXCLUDED_REGIONS = ["EAN", "WAN", "GIC", "SAH", "ARP", "ECA"]
    MIN_COMMON = 20             # temporal mode: min. common valid steps per cell
    WIN = 2                     # spatial mode: half-width of neighbourhood (2 -> 5x5 = 25 cells)
    MIN_WINDOW_VALID = 12       # spatial mode: min. valid pairs inside the window
    SLOPE_XLABEL = "SPI-12 slope (units per decade)"   # x-label of the OLS-slope ECDF panel
    FIG_FORMATS = ("pdf", "png")
    FIG_DPI = 600
    OUTPUT_DIR = "./KS_stationary_vs_nonstationary_trend"
    CSV_DIR = OUTPUT_DIR + "/csv"
    NC_DIR = OUTPUT_DIR + "/netcdf"
    FIG_DIR = OUTPUT_DIR + "/figures"


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 9.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})
COUNTRY_RES = 110
LAND, COUNTRY_LINE, REGION_LINE = "#ececec", "#8a8a8a", "black"
BLUE, RED = "#3b4cc0", "#b40426"


# =============================================================================
# MAP HELPERS
# =============================================================================
def get_regions():
    allr = regionmask.defined_regions.ar6.land
    keep = [i for i, ab in enumerate(allr.abbrevs) if ab not in cfg.EXCLUDED_REGIONS]
    return allr[keep]


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


def to_grid(df, col):
    """Values on the FULL regular lat/lon grid (NaN where no cell), so pcolormesh never stretches cells."""
    def axis(v):
        u = np.sort(v.unique())
        d = float(np.round(np.diff(u).min(), 6)) if len(u) > 1 else 0.5
        n = int(round((u[-1] - u[0]) / d)) + 1
        return u[0] + d * np.arange(n), d
    lons, dx = axis(df["lon"]); lats, dy = axis(df["lat"])
    Z = np.full((len(lats), len(lons)), np.nan)
    Z[np.round((df["lat"].values - lats[0]) / dy).astype(int),
      np.round((df["lon"].values - lons[0]) / dx).astype(int)] = df[col].values
    return lons, lats, Z


def edges(c):
    d = np.diff(c).mean() if len(c) > 1 else 0.5
    return np.r_[c - d / 2, c[-1] + d / 2]


def draw_regions(ax, regions, label=True):
    for poly, ab in zip(regions.polygons, regions.abbrevs):
        for x, y in _rings(poly):
            ax.plot(x, y, color=REGION_LINE, lw=0.5, solid_joinstyle="round", zorder=4)
        if label:
            big = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)
            c = big.representative_point()
            ax.text(c.x, c.y, ab, ha="center", va="center", fontsize=5.5, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.6))


def save(fig, name):
    for fmt in cfg.FIG_FORMATS:
        p = f"{cfg.FIG_DIR}/{name}.{fmt}"
        fig.savefig(p, dpi=cfg.FIG_DPI, bbox_inches="tight", facecolor="white")
        print("  saved", p)
    plt.close(fig)


def draw_map(ax, cells, values, cmap, vmin, vmax, regions, countries):
    df = cells[["lat", "lon"]].copy(); df["v"] = values
    ax.set_facecolor("white")
    if countries is not None:
        for poly in countries.polygons:
            for x, y in _rings(poly):
                ax.fill(x, y, color=LAND, lw=0, zorder=1)
                ax.plot(x, y, color=COUNTRY_LINE, lw=0.3, zorder=3)
    lons, lats, Z = to_grid(df, "v")
    m = ax.pcolormesh(edges(lons), edges(lats), np.ma.masked_invalid(Z), cmap=cmap, vmin=vmin, vmax=vmax,
                      rasterized=True, zorder=2, antialiased=False)
    draw_regions(ax, regions)
    ax.set_xlim(-180, 180); ax.set_ylim(-58, 85); ax.set_aspect("equal", adjustable="box")
    ax.set_yticks([-45, 0, 45, 80]); ax.set_yticklabels(["45°S", "0°", "45°N", "80°N"])
    ax.set_xticks(range(-180, 181, 60))
    ax.set_xticklabels([f"{abs(v)}°{'W' if v < 0 else 'E' if v > 0 else ''}" for v in range(-180, 181, 60)])
    ax.tick_params(width=1.0, length=4.5, labelsize=11)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    return m


# =============================================================================
# DATA
# =============================================================================
_SKIP = ("p_val", "pval", "p-val", "pvalue", "sig", "signif", "tau", "z_", "zstat", "intercept",
         "lower", "upper", "ci_", "_ci", "count", "nobs", "n_", "mask", "bnds", "bounds")
_PREF = ("slope", "sen", "trend", "beta", "change")


def open_trend(path, var=None):
    """Open a trend netCDF -> DataArray ('lat','lon' or ('step','lat','lon'))."""
    ds = xr.open_dataset(path)
    if var is None:
        cand = [v for v in ds.data_vars if ds[v].ndim >= 2 and
                {"lat", "lon", "latitude", "longitude", "y", "x"} & set(ds[v].dims)]
        good = [v for v in cand if not any(s in v.lower() for s in _SKIP)] or cand
        pref = [v for v in good if any(s in v.lower() for s in _PREF)]
        pick = pref or good
        if not pick:
            raise ValueError(f"no gridded variable in {path}: {list(ds.data_vars)}")
        if len(pick) > 1:
            print(f"  [note] several candidate variables {pick}; using '{pick[0]}' "
                  f"(set cfg.VAR_* to choose another)")
        var = pick[0]
    da = ds[var].squeeze(drop=True)
    ren = {a: b for a, b in (("latitude", "lat"), ("longitude", "lon"), ("y", "lat"), ("x", "lon"))
           if a in da.dims and b not in da.dims}
    da = da.rename(ren)
    extra = [d for d in da.dims if d not in ("lat", "lon")]
    if len(extra) > 1:
        raise ValueError(f"{var} has more than one extra dimension: {extra}")
    da = da.transpose(*extra, "lat", "lon")
    if da.lat.size > 1 and da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    if da.lon.size > 1 and da.lon.values.max() > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    print(f"  {os.path.basename(path)}: '{var}' dims={dict(da.sizes)} ")
    return da


def align(sta, non):
    """Put both on the same grid (and same steps if a step dimension exists)."""
    non = non.reindex(lat=sta.lat, lon=sta.lon, method="nearest", tolerance=0.01)
    ex_s = [d for d in sta.dims if d not in ("lat", "lon")]
    ex_n = [d for d in non.dims if d not in ("lat", "lon")]
    if ex_s != ex_n:
        raise ValueError(f"dimension mismatch: stationary {sta.dims} vs non-stationary {non.dims}")
    if ex_s:
        d = ex_s[0]
        common = np.intersect1d(sta[d].values, non[d].values)
        if len(common) == 0 and sta.sizes[d] == non.sizes[d]:
            non = non.assign_coords({d: sta[d].values}); common = sta[d].values
        if len(common) == 0:
            raise ValueError(f"no common '{d}' values")
        sta, non = sta.sel({d: common}), non.sel({d: common})
        print(f"  common steps along '{d}': {len(common)}")
    return sta, non


# =============================================================================
# STATISTICS
# =============================================================================
def lag1(x):
    x = x - x.mean(); d = np.sum(x * x)
    return float(np.sum(x[1:] * x[:-1]) / d) if d > 0 else 0.0


def bh_fdr(p, q):
    ok = np.isfinite(p); out = np.zeros(p.shape, bool)
    pv = p[ok]; order = np.argsort(pv); m = len(pv)
    below = np.where(pv[order] <= q * np.arange(1, m + 1) / m)[0]
    if len(below):
        rej = np.zeros(m, bool); rej[order[:below.max() + 1]] = True
        out[ok] = rej
    return out


def ks_temporal(sta, non):
    A, B = np.asarray(sta.values, float), np.asarray(non.values, float)
    lat, lon = sta.lat.values, sta.lon.values
    both = np.isfinite(A) & np.isfinite(B)
    ii, jj = np.where(both.sum(0) >= cfg.MIN_COMMON)
    print(f"  cells with >= {cfg.MIN_COMMON} common steps: {len(ii)}")
    rows = []
    for k, (i, j) in enumerate(zip(ii, jj)):
        ok = both[:, i, j]; a, b = A[ok, i, j], B[ok, i, j]
        r = ks_2samp(a, b, method="asymp"); n = len(a)
        na = max(n * (1 - lag1(a)) / (1 + lag1(a)), 2.0); nb = max(n * (1 - lag1(b)) / (1 + lag1(b)), 2.0)
        p_neff = float(kstwobign.sf(np.sqrt(na * nb / (na + nb)) * r.statistic))
        rows.append((i, j, lat[i], lon[j], n, r.statistic, r.pvalue, p_neff, a.mean(), b.mean()))
    return pd.DataFrame(rows, columns=["ilat", "ilon", "lat", "lon", "n", "D", "p", "p_neff",
                                       "mean_stationary", "mean_nonstationary"]), lat, lon


def ks_spatial(sta, non):
    A, B = np.asarray(sta.values, float), np.asarray(non.values, float)
    lat, lon = sta.lat.values, sta.lon.values
    both = np.isfinite(A) & np.isfinite(B)
    h = cfg.WIN; ny, nx = A.shape
    ii, jj = np.where(both)
    print(f"  cells with valid trends in both files: {len(ii)}")
    rows = []
    for k, (i, j) in enumerate(zip(ii, jj)):
        sl = (slice(max(i - h, 0), i + h + 1), slice(max(j - h, 0), j + h + 1))
        ok = both[sl]
        if ok.sum() < cfg.MIN_WINDOW_VALID:
            continue
        a, b = A[sl][ok], B[sl][ok]
        r = ks_2samp(a, b, method="auto")
        rows.append((i, j, lat[i], lon[j], len(a), r.statistic, r.pvalue, np.nan, a.mean(), b.mean()))
        if (k + 1) % 10000 == 0:
            print(f"    {k + 1}/{len(ii)} cells")
    return pd.DataFrame(rows, columns=["ilat", "ilon", "lat", "lon", "n", "D", "p", "p_neff",
                                       "mean_stationary", "mean_nonstationary"]), lat, lon


def label_regions(df, regions, lat, lon):
    m = regions.mask(lon, lat).values
    num2ab = dict(zip(regions.numbers, regions.abbrevs))
    ri = m[df["ilat"].values, df["ilon"].values]
    df["region"] = [num2ab.get(int(r), "none") if np.isfinite(r) else "none" for r in ri]
    return df


# =============================================================================
# FIGURES
# =============================================================================
def _sig_legend(ax, pct, fdr):
    lab = f"p < {cfg.ALPHA}" + (" (FDR-controlled)" if fdr else "")
    ax.legend(handles=[Patch(fc=RED, label=f"Significant difference ({lab}): {pct:.1f} % of cells"),
                       Patch(fc=BLUE, label=f"Not significant: {100 - pct:.1f} % of cells")],
              loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)


def sig_map(df, regions, countries, title, fdr):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    draw_map(ax, df, np.where(df["sig"].values, 1.0, 0.0), ListedColormap([BLUE, RED]), -0.5, 1.5,
             regions, countries)
    ax.set_title(title, loc="left", pad=4)
    _sig_legend(ax, 100 * df["sig"].mean(), fdr)
    save(fig, "Fig_KS_trend_map")


def d_map(df, regions, countries, title):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    hi = float(np.nanpercentile(df["D"], 98))
    m = draw_map(ax, df, df["D"].values, "YlOrRd", 0.0, hi, regions, countries)
    ax.set_title(title, loc="left", pad=4)
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.045, pad=0.08, extend="max",
                      aspect=45, shrink=0.55)
    cb.set_label("KS statistic D (maximum distance between the two cumulative distributions)")
    cb.outline.set_linewidth(0.8); cb.ax.tick_params(width=0.8, length=3, labelsize=10)
    save(fig, "Fig_KS_trend_D_map")


def panel(df, regions, countries, fdr):
    fig, axs = plt.subplots(2, 1, figsize=(8.4, 8.6))
    draw_map(axs[0], df, np.where(df["sig"].values, 1.0, 0.0), ListedColormap([BLUE, RED]), -0.5, 1.5,
             regions, countries)
    axs[0].set_title("a  Significance of the difference (KS test)", loc="left", pad=4, fontweight="bold")
    _sig_legend(axs[0], 100 * df["sig"].mean(), fdr)
    hi = float(np.nanpercentile(df["D"], 98))
    m = draw_map(axs[1], df, df["D"].values, "YlOrRd", 0.0, hi, regions, countries)
    axs[1].set_title("b  KS statistic D", loc="left", pad=4, fontweight="bold")
    cb = fig.colorbar(m, ax=axs[1], orientation="horizontal", fraction=0.045, pad=0.10, extend="max",
                      aspect=45, shrink=0.55)
    cb.set_label("D"); cb.outline.set_linewidth(0.8); cb.ax.tick_params(width=0.8, length=3, labelsize=10)
    fig.subplots_adjust(hspace=0.35)
    save(fig, "Fig_KS_trend_panel")



def find_var(path, keys, prefer=()):
    """First gridded variable whose name contains one of `keys` (p-value / CI variables skipped)."""
    ds = xr.open_dataset(path)
    bad = ("p_val", "pval", "pvalue", "p-val", "sig", "signif", "lower", "upper", "ci_")
    cand = [v for v in ds.data_vars if ds[v].ndim >= 2 and any(k in v.lower() for k in keys)
            and not any(b in v.lower() for b in bad)]
    pref = [v for v in cand if any(k in v.lower() for k in prefer)]
    pick = pref or cand
    return pick[0] if pick else None


def _fmt_p(p):
    return "p < 1e-300" if p < 1e-300 else f"p = {p:.2e}"


def ecdf_figure():
    """Global ECDFs of the stationary vs non-stationary trend values (all cells) + global KS test."""
    specs = [("Kendall's \u03c4", ("tau", "kendall"), (), "Kendall's \u03c4 of SPI-12"),
             ("OLS slope", ("slope", "ols"), ("ols",), cfg.SLOPE_XLABEL)]
    panels = []
    for title, keys, prefer, xlabel in specs:
        va = find_var(cfg.STATIONARY_NC, keys, prefer); vb = find_var(cfg.NONSTATIONARY_NC, keys, prefer)
        if va is None or vb is None:
            print(f"  [note] ECDF panel '{title}' skipped (variable not found: {va}, {vb})")
            continue
        sta, non = align(open_trend(cfg.STATIONARY_NC, va), open_trend(cfg.NONSTATIONARY_NC, vb))
        a, b = np.asarray(sta.values, float).ravel(), np.asarray(non.values, float).ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        r = ks_2samp(a, b)
        panels.append((title, xlabel, a, b, r))
    if not panels:
        print("  [note] no tau / slope variables found -> ECDF figure not drawn"); return
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(8.5 * n, 7), squeeze=False)
    rows = []
    for ax, (title, xlabel, a, b, r), letter in zip(axs[0], panels, "abcdef"):
        for v, lab, ls, col in ((a, "Stationary", "-", "black"),
                                (b, "Non-stationary (30-yr window)", "--", "0.5")):
            x = np.sort(v); y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, ls=ls, color=col, lw=3.2, label=lab)
        ax.axvline(0, color="black", lw=1.2,alpha=0.6, zorder=0)
        ax.grid(alpha=0.3, lw=0.5)
        sig = r.pvalue < cfg.ALPHA
        ax.text(0.97, 0.05, f"KS D = {r.statistic:.3f}\n{_fmt_p(r.pvalue)}\nn = {len(a):,} cells\n"
                f"{'Significant difference' if sig else 'No significant difference'}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=15, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RED if sig else BLUE, lw=3))
        ax.legend(loc="upper left", frameon=True, edgecolor="0.2", fontsize=12, prop=dict(weight="bold", size=12))
        ax.set_title(f"({letter}) {title}", loc="left", fontsize=18, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
        ax.set_ylabel("Cumulative probability", fontsize=16, fontweight="bold")
        ax.tick_params(width=1.5, length=6, labelsize=14)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)
        rows.append((title, len(a), r.statistic, r.pvalue, bool(sig)))
    fig.suptitle("Global ECDFs of SPI-12 trends: stationary vs non-stationary",
                 fontsize=20, fontweight="bold", y=1.02)
    fig.tight_layout()
    pd.DataFrame(rows, columns=["variable", "n_cells", "KS_D", "p", "significant"]
                 ).to_csv(f"{cfg.CSV_DIR}/KS_trend_global_ecdf_stats.csv", index=False)
    save(fig, "Fig_KS_trend_global_ecdf")

# =============================================================================
def main():
    for d in (cfg.OUTPUT_DIR, cfg.CSV_DIR, cfg.NC_DIR, cfg.FIG_DIR):
        os.makedirs(d, exist_ok=True)
    print("Global ECDF figure ...")
    ecdf_figure()
    print("Reading data ...")
    sta = open_trend(cfg.STATIONARY_NC, cfg.VAR_STATIONARY)
    non = open_trend(cfg.NONSTATIONARY_NC, cfg.VAR_NONSTATIONARY)
    sta, non = align(sta, non)

    temporal = sta.ndim == 3
    mode = "temporal" if temporal else "spatial"
    fdr = cfg.FDR if cfg.FDR is not None else (not temporal)
    print(f"KS test per grid cell, mode = {mode}")
    df, lat, lon = ks_temporal(sta, non) if temporal else ks_spatial(sta, non)
    if df.empty:
        raise RuntimeError("no cell could be tested - check data / MIN_* settings")

    regions = get_regions()
    df = label_regions(df, regions, lat, lon)
    if cfg.ONLY_KEPT_REGIONS:
        n0 = len(df)
        df = df[df["region"] != "none"].reset_index(drop=True)
        print(f"  cells outside the {len(regions.abbrevs)} kept regions removed: {n0 - len(df)} (kept {len(df)})")

    df["sig"] = bh_fdr(df["p"].values, 2 * cfg.ALPHA) if fdr else df["p"] < cfg.ALPHA
    if temporal:
        df["sig_neff"] = bh_fdr(df["p_neff"].values, 2 * cfg.ALPHA) if fdr else df["p_neff"] < cfg.ALPHA
    df.to_csv(f"{cfg.CSV_DIR}/KS_trend_cell_results.csv", index=False)

    summ = df.groupby("region").agg(n_cells=("sig", "size"),
                                    pct_significant=("sig", lambda s: 100 * s.mean()),
                                    median_D=("D", "median")).round(2)
    summ.loc["ALL"] = [len(df), round(100 * df["sig"].mean(), 2), round(df["D"].median(), 3)]
    summ.to_csv(f"{cfg.CSV_DIR}/KS_trend_region_summary.csv")
    print(f"\n{len(df)} cells | significant: {100 * df['sig'].mean():.1f} % | median D = {df['D'].median():.3f}")

    keys = ["D", "p", "sig"] + (["p_neff", "sig_neff"] if temporal else [])
    cube = {k: np.full((len(lat), len(lon)), np.nan) for k in keys}
    for k in keys:
        cube[k][df["ilat"].values, df["ilon"].values] = df[k].astype(float).values
    xr.Dataset({k: (("lat", "lon"), v) for k, v in cube.items()},
               coords={"lat": lat, "lon": lon}).to_netcdf(f"{cfg.NC_DIR}/KS_trend_results.nc")

    countries = get_countries()
    what = "rolling 30-year (time-varying) trends" if temporal else "trend fields (local 5×5 cell windows)"
    sig_map(df, regions, countries, f"KS test: stationary vs non-stationary SPI-12 {what}", fdr)
    d_map(df, regions, countries, "KS statistic D: stationary vs non-stationary SPI-12 trends")
    panel(df, regions, countries, fdr)
    print("done ->", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
