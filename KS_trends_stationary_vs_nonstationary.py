#!/usr/bin/env python3
"""
KS TEST: STATIONARY vs NON-STATIONARY (rolling 30-yr) SPEI-12 TRENDS, GRID-CELL BASED
================================================================================
Needs: numpy pandas scipy xarray netCDF4 regionmask matplotlib shapely cartopy

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

Figures (Cartopy, Robinson projection, continuous colour scales, no classes / cut-offs):
  Fig_KS_trend_trend_maps               cell trends: blue gradient = negative, red gradient = positive
                                        (symmetric range = largest |trend| in the data)
  Fig_KS_trend_area_percent             per IPCC region, % of area with negative (blue) / positive (red)
                                        trends, stationary vs non-stationary, continuous 0-100 %
  Fig_KS_trend_sig_map                  red = significant, blue = not significant (same style as the
                                        drought duration/intensity KS maps); *_neff = serial-correlation corrected
  Fig_KS_trend_map                      KS MAP: per IPCC region, % of area with a significant difference,
                                        continuous 0-100 % (blue = 0 %, red = 100 %, as in the KS maps)
  Fig_KS_trend_map_cell                 same colours at cell level: % of significant cells in a moving window
  Fig_KS_trend_D_map                    KS statistic D, continuous 0 -> max(D)
  Fig_KS_trend_global_ecdf              global ECDFs of tau and OLS slope
Area percentages are cos(latitude)-weighted.
Outputs (OUTPUT_DIR): csv/KS_trend_cell_results.csv, csv/KS_trend_region_summary.csv,
  csv/KS_trend_region_area_percent.csv, netcdf/KS_trend_results.nc, figures/*.pdf + *.png (600 dpi)
"""
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
from scipy.stats import ks_2samp, kstwobign
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# CONFIG  (edit here only)
# =============================================================================
class cfg:
    NONSTATIONARY_NC = "/home/murtaza13/paper2/Grid_based_analysis/Trends/data/SPEI12_nonstationary_30yr_trend.nc"
    STATIONARY_NC = "/home/murtaza13/paper2/Grid_based_analysis/Trends/data/SPEI12_stationary_trend.nc"
    VAR_STATIONARY = None       # e.g. "sen_slope"; None = auto-detect
    VAR_NONSTATIONARY = None
    ALPHA = 0.05
    FDR = None                  # None = auto (True in spatial mode, False in temporal); or True/False
    ONLY_KEPT_REGIONS = True    # keep only cells in the 40 AR6 land regions
    EXCLUDED_REGIONS = ["EAN", "WAN", "GIC", "SAH", "ARP", "ECA"]
    MIN_COMMON = 20             # temporal mode: min. common valid steps per cell
    WIN = 2                     # spatial mode: half-width of neighbourhood (2 -> 5x5 = 25 cells)
    MIN_WINDOW_VALID = 12       # spatial mode: min. valid pairs inside the window
    SLOPE_XLABEL = "SPEI-12 slope (units per decade)"   # x-label of the OLS-slope ECDF panel
    TREND_VMAX = None           # None = symmetric colour range from the data (no arbitrary cut-off)
    PCT_WIN = 2                 # cell-level % map: half-width of the window (2 -> 5x5 cells)
    LABEL_REGIONS = False       # True = print IPCC region codes on the maps
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


def save(fig, name):
    for fmt in cfg.FIG_FORMATS:
        p = f"{cfg.FIG_DIR}/{name}.{fmt}"
        fig.savefig(p, dpi=cfg.FIG_DPI, bbox_inches="tight", facecolor="white")
        print("  saved", p)
    plt.close(fig)



# =============================================================================
# CARTOPY / ROBINSON HELPERS
# =============================================================================
PC = ccrs.PlateCarree()


def new_map(fig, spec):
    ax = fig.add_subplot(spec, projection=ccrs.Robinson())
    ax.set_global()                                   # full globe, Antarctica included
    ax.set_facecolor("white")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.65", alpha=0.6, linestyle=":",
                 xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30), zorder=1.5)
    ax.spines["geo"].set_linewidth(1.0)
    return ax


def draw_base(ax, countries):
    """Light-grey land with country borders (regionmask polygons, no download needed)."""
    if countries is not None:
        ax.add_geometries(list(countries.polygons), PC, facecolor=LAND, edgecolor=COUNTRY_LINE,
                          linewidth=0.3, zorder=1)


def draw_outlines(ax, regions):
    """IPCC AR6 region boundaries (analysed regions only)."""
    ax.add_geometries(list(regions.polygons), PC, facecolor="none", edgecolor=REGION_LINE,
                      linewidth=0.6, zorder=4)
    if cfg.LABEL_REGIONS:
        for poly, ab in zip(regions.polygons, regions.abbrevs):
            big = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)
            c = big.representative_point()
            ax.text(c.x, c.y, ab, transform=PC, ha="center", va="center", fontsize=6, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.6))


def draw_cells(ax, lons, lats, Z, cmap, norm):
    return ax.pcolormesh(edges(lons), edges(lats), np.ma.masked_invalid(Z), cmap=cmap, norm=norm,
                         transform=PC, shading="flat", rasterized=True, zorder=2)


def fill_regions(ax, regions, values, cmap, norm):
    """Fill each IPCC region with a continuous colour for its value (white = no data)."""
    for poly, ab in zip(regions.polygons, regions.abbrevs):
        v = values.get(ab, np.nan)
        fc = "white" if not np.isfinite(v) else cmap(norm(v))
        ax.add_geometries([poly], PC, facecolor=fc, edgecolor="none", zorder=2)


def bold_cbar(fig, mappable, axes, label, ticks=None, fs=13, ts=12, shrink=0.8, aspect=40, pad=0.05):
    cb = fig.colorbar(mappable, ax=axes, orientation="horizontal", fraction=0.05, pad=pad,
                      aspect=aspect, shrink=shrink, ticks=ticks)
    cb.set_label(label, fontsize=fs, fontweight="bold")
    cb.ax.tick_params(labelsize=ts, width=1.0, length=3.5)
    plt.setp(cb.ax.get_xticklabels(), fontweight="bold")
    cb.outline.set_linewidth(1.0)
    return cb


def _title(ax, text, fs=14):
    ax.set_title(text, loc="left", fontsize=fs, fontweight="bold", pad=6)


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
# TREND GRIDS + AREA PERCENTAGES
# =============================================================================
def trend_grids(sta, non, regions):
    """Per-cell trend of each dataset on the full grid (temporal mode: mean of the trend series)."""
    lon, lat = sta.lon.values, sta.lat.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        TS = np.nanmean(sta.values.astype(float), axis=0) if sta.ndim == 3 else sta.values.astype(float)
        TN = np.nanmean(non.values.astype(float), axis=0) if non.ndim == 3 else non.values.astype(float)
    ok = np.isfinite(TS) & np.isfinite(TN)
    if cfg.ONLY_KEPT_REGIONS:
        ok &= np.isfinite(regions.mask(lon, lat).values)
    return lon, lat, np.where(ok, TS, np.nan), np.where(ok, TN, np.nan)


def region_area_stats(regions, lat, lon, TS, TN, df):
    """Area-weighted (cos latitude) % of each region's area: negative trend, positive trend,
    and significantly different (KS).  No classes or magnitude thresholds are used."""
    rm = regions.mask(lon, lat).values
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    sg = np.full(rm.shape, np.nan)
    sg[df["ilat"].values, df["ilon"].values] = df["sig"].astype(float).values

    def pct(sel, cond):
        tot = w[sel].sum()
        return 100.0 * w[sel & cond].sum() / tot if tot > 0 else np.nan

    rows = []
    items = list(zip(regions.numbers, regions.abbrevs)) + [(None, "ALL")]
    for num, ab in items:
        m = np.isfinite(rm) if num is None else (rm == num)
        row = {"region": ab, "n_cells": int(m.sum())}
        for tag, T in (("sta", TS), ("non", TN)):
            v = m & np.isfinite(T)
            row[f"pct_negative_{tag}"] = pct(v, T < 0)
            row[f"pct_positive_{tag}"] = pct(v, T > 0)
        s = m & np.isfinite(sg)
        row["pct_significant_difference"] = pct(s, sg == 1)
        rows.append(row)
    return pd.DataFrame(rows).round(2)


# =============================================================================
# FIGURES (Robinson projection, continuous colour scales, no classes)
# =============================================================================
def trend_maps(lon, lat, TS, TN, regions, countries):
    """Cell-level trends: blue gradient = negative, red gradient = positive, white = zero.
    Colour range = +/- the largest absolute trend in the data (same for both panels)."""
    vmax = cfg.TREND_VMAX or float(max(np.nanmax(np.abs(TS)), np.nanmax(np.abs(TN))))
    norm = Normalize(-vmax, vmax)
    fig = plt.figure(figsize=(9.5, 9.6), facecolor="white")
    gs = fig.add_gridspec(2, 1, hspace=0.10)
    axes = []
    for k, (T, name) in enumerate(((TS, "Stationary"), (TN, "Non-stationary (30-yr window)"))):
        ax = new_map(fig, gs[k]); draw_base(ax, countries)
        draw_cells(ax, lon, lat, T, "RdBu_r", norm)
        draw_outlines(ax, regions)
        _title(ax, f"({'ab'[k]}) {name}")
        axes.append(ax)
    bold_cbar(fig, cm.ScalarMappable(norm=norm, cmap="RdBu_r"), axes, cfg.SLOPE_XLABEL)
    save(fig, "Fig_KS_trend_trend_maps")


def area_pct_figure(regions, area, countries):
    """Per IPCC region: % of area with negative (blue) / positive (red) trends, continuous 0-100 %."""
    reg = area[area["region"] != "ALL"]
    norm = Normalize(0, 100)
    fig = plt.figure(figsize=(15, 8.8), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.04)
    cols = [[], []]
    for r, (tag, name) in enumerate((("sta", "Stationary"), ("non", "Non-stationary"))):
        for c, (kind, cmap) in enumerate((("negative", "Blues"), ("positive", "Reds"))):
            ax = new_map(fig, gs[r, c]); draw_base(ax, countries)
            fill_regions(ax, regions, dict(zip(reg["region"], reg[f"pct_{kind}_{tag}"])),
                         plt.get_cmap(cmap), norm)
            draw_outlines(ax, regions)
            _title(ax, f"({'abcd'[2 * r + c]}) {name}: {kind} trends")
            cols[c].append(ax)
    ticks = np.arange(0, 101, 20)
    bold_cbar(fig, cm.ScalarMappable(norm=norm, cmap="Blues"), cols[0],
              "Area with negative trend (%)", ticks)
    bold_cbar(fig, cm.ScalarMappable(norm=norm, cmap="Reds"), cols[1],
              "Area with positive trend (%)", ticks)
    save(fig, "Fig_KS_trend_area_percent")


# Continuous version of your two-colour KS palette (ColorBrewer RdYlBu, 5-class endpoints):
#   0 % significant = blue #2c7bb6   ...   100 % significant = red #d7191c
# The pale-yellow middle keeps the gradient clean (no muddy purple) and clearly different
# from the grey land.
SIG_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
SIG_CMAP = LinearSegmentedColormap.from_list("KS_blue_red", SIG_COLORS, N=256)
JOURNAL_FIG = (7.2, 4.5)        # inches (double-column width); fonts below are sized for this


def local_pct_significant(df, lat, lon):
    """Cell-level area percentage: % of tested cells that are significant inside a moving
    (2*PCT_WIN+1)^2 window.  Continuous 0-100 %; NaN where the cell itself was not tested."""
    S = np.zeros((len(lat), len(lon))); M = np.zeros_like(S)
    S[df["ilat"].values, df["ilon"].values] = df["sig"].astype(float).values
    M[df["ilat"].values, df["ilon"].values] = 1.0
    size = 2 * cfg.PCT_WIN + 1
    num = uniform_filter(S, size=size, mode="constant")
    den = uniform_filter(M, size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(M > 0, 100.0 * num / den, np.nan)


def _sig_label(fdr):
    return f"Area with significant difference (%),  p < {cfg.ALPHA}" + (" (FDR)" if fdr else "")


def _global_tag(ax, value):
    ax.text(0.012, 0.045, f"Global: {value:.1f} %", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3", lw=0.6))


def sigdiff_map(regions, area, countries, title, fdr):
    """KS map: each IPCC region coloured by the % of its area with a significant difference."""
    reg = area[area["region"] != "ALL"]
    norm = Normalize(0, 100)
    fig = plt.figure(figsize=JOURNAL_FIG, facecolor="white")
    ax = new_map(fig, fig.add_gridspec(1, 1)[0]); draw_base(ax, countries)
    fill_regions(ax, regions, dict(zip(reg["region"], reg["pct_significant_difference"])),
                 plt.get_cmap(SIG_CMAP), norm)
    draw_outlines(ax, regions)
    _title(ax, title, fs=9.5)
    _global_tag(ax, float(area.loc[area["region"] == "ALL", "pct_significant_difference"].iloc[0]))
    bold_cbar(fig, cm.ScalarMappable(norm=norm, cmap=SIG_CMAP), ax, _sig_label(fdr),
              np.arange(0, 101, 20), fs=9, ts=8.5, shrink=0.55, aspect=45, pad=0.03)
    save(fig, "Fig_KS_trend_map")


def cell_pct_map(lon, lat, P, regions, countries, title, fdr, global_pct=None):
    """Same colours at grid-cell level (moving-window area percentage)."""
    size = 2 * cfg.PCT_WIN + 1
    norm = Normalize(0, 100)
    fig = plt.figure(figsize=JOURNAL_FIG, facecolor="white")
    ax = new_map(fig, fig.add_gridspec(1, 1)[0]); draw_base(ax, countries)
    m = draw_cells(ax, lon, lat, P, SIG_CMAP, norm)
    draw_outlines(ax, regions)
    _title(ax, title, fs=9.5)
    if global_pct is not None:
        _global_tag(ax, global_pct)
    bold_cbar(fig, m, ax, f"Significant cells in {size}\u00d7{size} window (%),  p < {cfg.ALPHA}"
              + (" (FDR)" if fdr else ""), np.arange(0, 101, 20),
              fs=9, ts=8.5, shrink=0.55, aspect=45, pad=0.03)
    save(fig, "Fig_KS_trend_map_cell")


SIG_RED, SIG_BLUE, NT_GREY = "#d7191c", "#2c7bb6", "#d9d9d9"   # same colours as the KS event maps


def sig_category_grid(df, col, TS, TN):
    """0 = not tested, 1 = not significant, 2 = significant (NaN = no data / outside regions)."""
    c = np.full(TS.shape, np.nan, np.float32)
    c[np.isfinite(TS) & np.isfinite(TN)] = 0
    c[df["ilat"].values, df["ilon"].values] = 1.0 + df[col].astype(float).values
    return c


def sig_map_categorical(cat, lon, lat, regions, title, fname, pct, fdr):
    """Red = significant, blue = not significant. Same layout as make_cat_map() of
    SPEI12_KS_test_sig_cells.py: Robinson, coastlines incl. Antarctica, black IPCC lines
    (excluded regions in grey), bold title and legend."""
    allr = regionmask.defined_regions.ar6.land
    excl = [poly for ab, poly in zip(allr.abbrevs, allr.polygons) if ab in cfg.EXCLUDED_REGIONS]
    cmap = ListedColormap([NT_GREY, SIG_BLUE, SIG_RED])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

    fig = plt.figure(figsize=(11, 6.6), facecolor="white")
    ax = fig.add_axes([0.03, 0.14, 0.94, 0.77], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor("white")
    ax.pcolormesh(edges(lon), edges(lat), np.ma.masked_invalid(cat), cmap=cmap, norm=norm,
                  transform=PC, shading="flat", rasterized=True, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6, edgecolor="0.35", zorder=2)
    ax.add_geometries(excl, PC, facecolor="none", edgecolor="0.65", linewidth=0.7, zorder=3)
    ax.add_geometries(list(regions.polygons), PC, facecolor="none", edgecolor="black",
                      linewidth=1.1, zorder=4)
    ax.spines["geo"].set_linewidth(1.2)
    ax.set_title(title, fontsize=21, fontweight="bold", pad=12)

    tag = " (FDR)" if fdr else ""
    handles = [Patch(facecolor=SIG_RED, edgecolor="black",
                     label=f"Significant (p < {cfg.ALPHA}{tag}): {pct:.1f} %"),
               Patch(facecolor=SIG_BLUE, edgecolor="black",
                     label=f"Not significant: {100 - pct:.1f} %")]
    if np.any(cat == 0):
        handles.append(Patch(facecolor=NT_GREY, edgecolor="black", label="Not tested"))
    lg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True,
                    prop={"weight": "bold", "size": 15}, bbox_to_anchor=(0.5, 0.02))
    lg.get_frame().set_edgecolor("black")
    lg.get_frame().set_linewidth(1.5)
    save(fig, fname)


def d_map(df, regions, countries):
    """Cell-level KS statistic D on a continuous scale from 0 to its maximum."""
    lons, lats, Z = to_grid(df, "D")
    norm = Normalize(0.0, float(np.nanmax(df["D"])))
    fig = plt.figure(figsize=(9.5, 5.6), facecolor="white")
    ax = new_map(fig, fig.add_gridspec(1, 1)[0]); draw_base(ax, countries)
    m = draw_cells(ax, lons, lats, Z, "Reds", norm)
    draw_outlines(ax, regions)
    _title(ax, "KS statistic D: stationary vs non-stationary trends")
    bold_cbar(fig, m, ax, "KS statistic D")
    save(fig, "Fig_KS_trend_D_map")


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
    specs = [("Kendall's \u03c4", ("tau", "kendall"), (), "Kendall's \u03c4 of SPEI-12"),
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
    fig, axs = plt.subplots(1, n, figsize=(7.6 * n, 5.6), squeeze=False)
    rows = []
    for ax, (title, xlabel, a, b, r), letter in zip(axs[0], panels, "abcdef"):
        for v, lab, ls, col in ((a, "Stationary", "-", "black"),
                                (b, "Non-stationary (30-yr window)", "--", "0.5")):
            x = np.sort(v); y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, ls=ls, color=col, lw=2.6, label=lab)
        ax.axvline(0, color="0.35", lw=0.9, zorder=0)
        ax.grid(alpha=0.3, lw=0.5)
        sig = r.pvalue < cfg.ALPHA
        ax.text(0.97, 0.04, f"KS D = {r.statistic:.3f}\n{_fmt_p(r.pvalue)}\nn = {len(a):,} cells\n"
                f"{'Significant difference' if sig else 'No significant difference'}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RED if sig else BLUE, lw=2.2))
        ax.legend(loc="upper left", frameon=True, edgecolor="0.2", fontsize=11, prop=dict(weight="bold", size=11))
        ax.set_title(f"({letter}) {title}", loc="left", fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative probability", fontsize=13, fontweight="bold")
        ax.tick_params(width=1.2, length=5, labelsize=12)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)
        rows.append((title, len(a), r.statistic, r.pvalue, bool(sig)))
    fig.suptitle("Global ECDFs of SPEI-12 trends: stationary vs non-stationary",
                 fontsize=16, fontweight="bold", y=1.02)
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

    # per-cell trends on the full grid + area percentages per IPCC region
    lon_g, lat_g, TS, TN = trend_grids(sta, non, regions)
    area = region_area_stats(regions, lat_g, lon_g, TS, TN, df)
    area.to_csv(f"{cfg.CSV_DIR}/KS_trend_region_area_percent.csv", index=False)
    print(area.to_string(index=False))

    keys = ["D", "p", "sig"] + (["p_neff", "sig_neff"] if temporal else [])
    cube = {k: np.full((len(lat), len(lon)), np.nan) for k in keys}
    for k in keys:
        cube[k][df["ilat"].values, df["ilon"].values] = df[k].astype(float).values
    cube["trend_stationary"], cube["trend_nonstationary"] = TS, TN
    cube["pct_significant_local"] = local_pct_significant(df, lat, lon)
    xr.Dataset({k: (("lat", "lon"), v) for k, v in cube.items()},
               coords={"lat": lat, "lon": lon}).to_netcdf(f"{cfg.NC_DIR}/KS_trend_results.nc")

    countries = get_countries()
    trend_maps(lon_g, lat_g, TS, TN, regions, countries)
    area_pct_figure(regions, area, countries)
    what = ("rolling 30-year (time-varying) trends" if temporal else
            f"trend fields (local {2 * cfg.WIN + 1}\u00d7{2 * cfg.WIN + 1} cell windows)")
    title = f"KS test: stationary vs non-stationary SPEI-12 {what}"
    sigdiff_map(regions, area, countries, title, fdr)
    cell_pct_map(lon, lat, cube["pct_significant_local"], regions, countries, title, fdr,
                 global_pct=100 * df["sig"].mean())
    d_map(df, regions, countries)
    sig_map_categorical(sig_category_grid(df, "sig", TS, TN), lon_g, lat_g, regions,
                        "KS test: Stationary vs Non-stationary\nSPEI-12 Trends",
                        "Fig_KS_trend_sig_map", 100 * df["sig"].mean(), fdr)
    if temporal:
        sig_map_categorical(sig_category_grid(df, "sig_neff", TS, TN), lon_g, lat_g, regions,
                            "KS test: Stationary vs Non-stationary\nSPEI-12 Trends (autocorrelation-corrected)",
                            "Fig_KS_trend_sig_map_neff", 100 * df["sig_neff"].mean(), fdr)
    print("done ->", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
