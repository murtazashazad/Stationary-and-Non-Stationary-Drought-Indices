"""
Grid-based SPI-12: STATIONARY vs NON-STATIONARY (30-yr trailing window)
=========================================================================
Computes, for every grid cell of the CRU TS4.10 grid:
  1. Stationary SPI-12           (one gamma fit per calendar month over the
                                   full record 1901-2025)
  2. Non-stationary SPI-12       (gamma refit every year on the trailing
                                   30-year window of the same calendar month)
  3. Grid-cell Mann-Kendall trend (Kendall tau, p-value, category, OLS slope)
     for both SPI versions

Outputs (all inside ./data, relative to where the script is run):
  data/SPI12_stationary.nc
  data/SPI12_nonstationary_30yr.nc
  data/SPI12_stationary_trend.nc
  data/SPI12_nonstationary_30yr_trend.nc
  data/figures/SPI12_*.png (and .pdf)
  data/csv/SPI12_*.csv

Method notes
  * 12-month precipitation totals; gamma distribution fitted with Thom's (1958)
    closed-form estimator, with the mixed-distribution treatment of zero totals
    (probability of zero q), as in the reference SPI scripts.
  * Same land mask / excluded regions (EAN, WAN, GIC, SAH, ARP, ECA) as the SPEI script.
  * Trend test: scipy.stats.kendalltau on the monthly SPI-12 series.  SPI-12 is strongly
    autocorrelated (11/12 overlap), so the p-values over-state significance.
  * TREND_COMMON_PERIOD=True runs both trend tests over the SAME months (1930 onward).
  * The reference non-stationary SPI script had WINDOW_YEARS = 40 although its text
    said 30; here WINDOW_YEARS = 30 to match the SPEI analysis.  Change below if needed.

Requirements:
    pip install xarray numpy pandas scipy regionmask matplotlib cartopy netCDF4
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.stats import norm, gamma, kendalltau
import regionmask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS -- EDIT THESE
# ============================================================================
CRU_PRE_FILE = "/home/Data/CRU/ts4.10/cru_ts4.10.1901.2025.pre.dat.nc"

ACCUM_MONTHS = 12
START_YEAR = 1901
END_YEAR = 2025
WINDOW_YEARS = 30                 # trailing window for the non-stationary SPI
EXCLUDED_REGIONS = ["EAN", "WAN", "GIC", "SAH", "ARP", "ECA"]
USE_ALL_DATA_CELLS = True         # keep coastal/island cells outside coarse AR6 polygons

ALPHA = 0.05                      # Mann-Kendall significance level
MIN_VALID_YEARS = 10              # min valid values to fit the distribution
MIN_NONZERO = 5                   # min non-zero totals needed for the gamma fit
MIN_TREND_MONTHS = 120            # min valid months to run the trend test
TREND_COMMON_PERIOD = True        # trend of both SPIs over the same months

OUTPUT_DIR = "./data"
FIG_DIR = f"{OUTPUT_DIR}/figures"
CSV_DIR = f"{OUTPUT_DIR}/csv"

F_SPI_ST = f"{OUTPUT_DIR}/SPI{ACCUM_MONTHS}_stationary.nc"
F_SPI_NS = f"{OUTPUT_DIR}/SPI{ACCUM_MONTHS}_nonstationary_{WINDOW_YEARS}yr.nc"
F_TREND_ST = f"{OUTPUT_DIR}/SPI{ACCUM_MONTHS}_stationary_trend.nc"
F_TREND_NS = f"{OUTPUT_DIR}/SPI{ACCUM_MONTHS}_nonstationary_{WINDOW_YEARS}yr_trend.nc"

# Snapshot maps of SPI (year, month) shown in the SPI figures
SNAPSHOT_DATES = [(1950, 12), (1990, 12), (2020, 12)]

RUN_COMPUTE = True                # set False to only re-make figures from existing .nc
RUN_FIGURES = True

for d in (OUTPUT_DIR, FIG_DIR, CSV_DIR):
    os.makedirs(d, exist_ok=True)

# ============================================================================
# PLOT STYLE (high-impact journal: large, bold text; red/blue; black boundaries)
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

COL_DRY = "#B2182B"     # red
COL_WET = "#2166AC"     # blue
COL_NONE = "#D9D9D9"    # light grey (non-significant)
PC = ccrs.PlateCarree()


# ============================================================================
# STEP 1: DATA, REGIONS, LAND MASK
# ============================================================================
def load_precipitation():
    """CRU monthly precipitation (mm/month)."""
    print("Loading CRU precipitation ...")
    ds = xr.open_dataset(CRU_PRE_FILE)
    name = "pre" if "pre" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[name]
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"
    if lat_name != "lat":
        da = da.rename({lat_name: "lat", lon_name: "lon"})
    pre = da.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")).astype("float32")
    pre.name = "precipitation"
    print(f"  Precipitation shape: {pre.shape}; "
          f"time {str(pre.time.values[0])[:10]} to {str(pre.time.values[-1])[:10]}; "
          f"units: {pre.attrs.get('units', 'mm/month (assumed)')}")
    return pre


def get_ipcc_regions():
    all_regions = regionmask.defined_regions.ar6.land
    keep = [i for i, ab in enumerate(all_regions.abbrevs) if ab not in EXCLUDED_REGIONS]
    return all_regions[keep]


def get_land_mask(pre, regions):
    mask_3D = regions.mask_3D(pre.lon, pre.lat)
    land_mask = mask_3D.any(dim="region")
    if USE_ALL_DATA_CELLS:
        all_regions = regionmask.defined_regions.ar6.land
        excl_idx = [i for i, ab in enumerate(all_regions.abbrevs) if ab in EXCLUDED_REGIONS]
        excl_mask = all_regions[excl_idx].mask_3D(pre.lon, pre.lat).any(dim="region")
        has_data = pre.notnull().any(dim="time")
        land_mask = land_mask | (has_data & ~excl_mask)
    land_mask = land_mask.transpose("lat", "lon")
    print(f"  Land cells kept: {int(land_mask.sum())} / {land_mask.size}")
    return land_mask


# ============================================================================
# STEP 2: SPI CORE (THOM GAMMA + ZERO PROBABILITY, VECTORIZED)
# ============================================================================
def rolling_sum(x, w):
    """Rolling sum over axis 0; NaN unless the whole window is valid."""
    n = x.shape[0]
    isn = np.isnan(x)
    cs = np.cumsum(np.where(isn, 0.0, x), axis=0)
    cn = np.cumsum(isn, axis=0)
    s = cs[w - 1:].copy()
    c = cn[w - 1:].copy()
    s[1:] -= cs[:n - w]
    c[1:] -= cn[:n - w]
    s[c > 0] = np.nan
    out = np.full_like(x, np.nan)
    out[w - 1:] = s
    return out


def gamma_params(values_2d):
    """
    Thom (1958) closed-form gamma estimator, vectorized across cells (axis 1).
    Zeros are handled as a point mass q = P(total = 0); the gamma is fitted to the
    positive totals.  Returns shape, scale, q (each (n_cells,)), NaN where no valid fit.

    The logarithm is taken on a safely-masked copy (NaNs replaced by 1 before log and
    excluded from the sum), so no uninitialised values can enter the mean of ln(x).
    """
    n_cells = values_2d.shape[1]
    shape_a = np.full(n_cells, np.nan)
    scale_a = np.full(n_cells, np.nan)
    q_a = np.full(n_cells, np.nan)

    valid = np.isfinite(values_2d)
    n_valid = valid.sum(axis=0)
    good = n_valid >= MIN_VALID_YEARS
    if not np.any(good):
        return shape_a, scale_a, q_a

    vals = values_2d[:, good]
    v = valid[:, good]
    nv = n_valid[good].astype(float)

    q = ((vals == 0) & v).sum(axis=0) / nv
    pos = v & (vals > 0)
    n_pos = pos.sum(axis=0).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = np.where(pos, vals, 0.0).sum(axis=0) / n_pos
        mean_lnx = np.where(pos, np.log(np.where(pos, vals, 1.0)), 0.0).sum(axis=0) / n_pos
        A = np.log(mean_x) - mean_lnx
        shape = (1.0 + np.sqrt(1.0 + 4.0 * A / 3.0)) / (4.0 * A)
        scale = mean_x / shape

    ok = ((n_pos >= MIN_NONZERO) & (A > 0) & np.isfinite(shape) & np.isfinite(scale)
          & (shape > 0) & (scale > 0))
    shape_a[good] = np.where(ok, shape, np.nan)
    scale_a[good] = np.where(ok, scale, np.nan)
    q_a[good] = np.where(ok, q, np.nan)
    return shape_a, scale_a, q_a


def to_spi(x, shape, scale, q):
    """Mixed gamma CDF -> standard normal quantile."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cdf = q + (1.0 - q) * gamma.cdf(x, shape, loc=0, scale=scale)
        cdf = np.where(x == 0, q, cdf)
    return norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))


def spi_stationary(accum, months):
    """One distribution per calendar month fitted on the FULL record."""
    print("\nComputing STATIONARY SPI ...")
    out = np.full_like(accum, np.nan)
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        vals = accum[idx]
        sh, sc, q = gamma_params(vals)
        out[idx] = to_spi(vals, sh[None, :], sc[None, :], q[None, :])
        print(f"  Month {m:2d}: gamma fitted at {int(np.sum(~np.isnan(sh)))}/{accum.shape[1]} cells")
    return out


def spi_nonstationary(accum, years, months, window_years):
    """
    Trailing moving-window climatology: for target year Y and calendar month m the
    gamma is fitted on years [Y-window+1, Y] of month m only, then the value of
    (Y, m) is transformed with that fit.  First (window-1) years are NaN.
    """
    print(f"\nComputing NON-STATIONARY SPI (trailing {window_years}-yr window) ...")
    out = np.full_like(accum, np.nan)
    y0, y1 = int(years.min()), int(years.max())
    first_target = y0 + window_years - 1
    targets = np.arange(first_target, y1 + 1)
    print(f"  Defined for {first_target}-{y1} ({len(targets)} years)")

    t0 = time.time()
    for c, ty in enumerate(targets):
        in_win = (years >= ty - window_years + 1) & (years <= ty)
        for m in range(1, 13):
            fit_idx = np.where(in_win & (months == m))[0]
            tgt = np.where((years == ty) & (months == m))[0]
            if len(fit_idx) == 0 or len(tgt) == 0:
                continue
            sh, sc, q = gamma_params(accum[fit_idx])
            out[tgt[0]] = to_spi(accum[tgt[0]], sh, sc, q)
        if (c + 1) % 10 == 0 or c + 1 == len(targets):
            print(f"    target year {ty} ({c + 1}/{len(targets)})  [{time.time() - t0:.0f}s]")
    return out


# ============================================================================
# STEP 3: GRID-CELL TREND (MANN-KENDALL / KENDALL TAU)
# ============================================================================
def compute_trend(spi_c, t_years):
    """Kendall tau + p-value + category + OLS slope (per decade) at each cell."""
    n_cells = spi_c.shape[1]
    tau = np.full(n_cells, np.nan)
    p = np.full(n_cells, np.nan)
    slope = np.full(n_cells, np.nan)
    for j in range(n_cells):
        s = spi_c[:, j]
        ok = np.isfinite(s)
        if ok.sum() < MIN_TREND_MONTHS:
            continue
        tt, ss = t_years[ok], s[ok]
        tau[j], p[j] = kendalltau(tt, ss)
        tc = tt - tt.mean()
        slope[j] = (tc * (ss - ss.mean())).sum() / (tc ** 2).sum() * 10.0   # per decade
        if (j + 1) % 10000 == 0:
            print(f"    {j + 1}/{n_cells} cells")
    cat = np.where(np.isnan(p), np.nan, np.where(p < ALPHA, np.sign(tau), 0.0))
    return tau, p, cat, slope


# ============================================================================
# STEP 4: NETCDF I/O
# ============================================================================
def save_nc(ds, path):
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32",
               "_FillValue": np.float32(np.nan)} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"  Saved: {path}")


def to_grid(vec, valid_idx, n_lat, n_lon):
    full = np.full(n_lat * n_lon, np.nan, dtype=np.float32)
    full[valid_idx] = vec
    return full.reshape(n_lat, n_lon)


def common_attrs(extra):
    a = {
        "source": "CRU TS4.10 precipitation",
        "method": "12-month precipitation totals; gamma distribution (Thom 1958) with "
                  "zero-precipitation probability (McKee et al. 1993; Edwards & McKee 1997)",
        "excluded_regions": ",".join(EXCLUDED_REGIONS),
        "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    a.update(extra)
    return a


def run_analysis():
    pre = load_precipitation()
    regions = get_ipcc_regions()
    land_mask = get_land_mask(pre, regions)
    pre = pre.where(land_mask)

    n_time, n_lat, n_lon = pre.shape
    flat = pre.values.reshape(n_time, n_lat * n_lon)
    valid_idx = np.where(np.isfinite(flat).any(axis=0))[0]          # compact: land cells only
    print(f"  Cells with data: {len(valid_idx)} / {n_lat * n_lon}")
    P = flat[:, valid_idx].astype(np.float64)
    del flat

    times = pre["time"]
    years = times.dt.year.values
    months = times.dt.month.values

    print("\nAccumulating precipitation over", ACCUM_MONTHS, "months ...")
    accum = rolling_sum(P, ACCUM_MONTHS)
    del P

    spi_st = spi_stationary(accum, months)
    spi_ns = spi_nonstationary(accum, years, months, WINDOW_YEARS)
    del accum

    coords = {"time": times.values, "lat": pre.lat.values, "lon": pre.lon.values}
    results = {}
    for tag, arr, fname, desc in [
        ("stationary", spi_st, F_SPI_ST,
         f"Stationary SPI-{ACCUM_MONTHS}: one distribution per calendar month fitted on {START_YEAR}-{END_YEAR}"),
        ("nonstationary", spi_ns, F_SPI_NS,
         f"Non-stationary SPI-{ACCUM_MONTHS}: distribution refitted on trailing {WINDOW_YEARS}-year window"),
    ]:
        full = np.full((n_time, n_lat * n_lon), np.nan, dtype=np.float32)
        full[:, valid_idx] = arr
        da = xr.DataArray(full.reshape(n_time, n_lat, n_lon), dims=("time", "lat", "lon"),
                          coords=coords, name=f"SPI{ACCUM_MONTHS}",
                          attrs={"long_name": f"Standardized Precipitation Index ({ACCUM_MONTHS}-month)",
                                 "units": "1"})
        ds = da.to_dataset()
        ds.attrs = common_attrs({"title": desc})
        print(f"\nWriting {fname}")
        save_nc(ds, fname)
        del full, da, ds
        results[tag] = arr

    # ---- trends ----
    first_target = START_YEAR + WINDOW_YEARS - 1
    if TREND_COMMON_PERIOD:
        tmask = years >= first_target
        period = f"{first_target}-{END_YEAR}"
    else:
        tmask = np.ones(n_time, dtype=bool)
        period = f"{START_YEAR}-{END_YEAR} (stationary) / {first_target}-{END_YEAR} (non-stationary)"
    t_years = (np.arange(n_time) / 12.0)

    for tag, fname in [("stationary", F_TREND_ST), ("nonstationary", F_TREND_NS)]:
        m_ = tmask if (TREND_COMMON_PERIOD or tag == "nonstationary") else np.ones(n_time, bool)
        print(f"\nMann-Kendall trend on {tag} SPI ({int(m_.sum())} months) ...")
        tau, p, cat, slope = compute_trend(results[tag][m_], t_years[m_])
        ds = xr.Dataset(
            {
                "tau": (("lat", "lon"), to_grid(tau, valid_idx, n_lat, n_lon),
                        {"long_name": f"Kendall tau (monthly SPI-{ACCUM_MONTHS} vs time)"}),
                "p_value": (("lat", "lon"), to_grid(p, valid_idx, n_lat, n_lon),
                            {"long_name": "Two-sided p-value of Kendall tau"}),
                "trend_category": (("lat", "lon"), to_grid(cat, valid_idx, n_lat, n_lon),
                                   {"long_name": "-1 significant drying, 0 no significant trend, +1 significant wetting",
                                    "alpha": ALPHA}),
                "slope_per_decade": (("lat", "lon"), to_grid(slope, valid_idx, n_lat, n_lon),
                                     {"long_name": f"OLS slope of SPI-{ACCUM_MONTHS}",
                                      "units": "SPI units per decade"}),
            },
            coords={"lat": pre.lat.values, "lon": pre.lon.values},
        )
        ds.attrs = common_attrs({
            "title": f"Grid-cell trend of {tag} SPI-{ACCUM_MONTHS}",
            "trend_period": period, "alpha": ALPHA,
            "note": "Kendall tau on monthly series; autocorrelation NOT corrected.",
        })
        save_nc(ds, fname)

    del results
    print("\nComputation finished.")


# ============================================================================
# STEP 5: FIGURES
# ============================================================================
def setup_map(ax, regions, extent=(-180, 180, -58, 85)):
    ax.set_extent(extent, crs=PC)
    ax.add_geometries(list(regions.polygons), crs=PC, facecolor="none",
                      edgecolor="black", linewidth=0.7, zorder=3)
    ax.coastlines(linewidth=0.8, color="black", zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="black", zorder=4)
    ax.set_xticks(np.arange(-180, 181, 60), crs=PC)
    ax.set_yticks(np.arange(-60, 81, 30), crs=PC)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=14, width=1.5, length=6)
    ax.set_xlabel("Longitude", fontsize=16, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=16, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
        sp.set_edgecolor("black")


def style_cbar(cb, label):
    cb.set_label(label, fontsize=16, fontweight="bold")
    cb.ax.tick_params(labelsize=14, width=1.5, length=6)
    cb.outline.set_linewidth(1.5)
    for t in cb.ax.get_xticklabels() + cb.ax.get_yticklabels():
        t.set_fontweight("bold")


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {FIG_DIR}/{name}.png / .pdf")


def plot_spi_snapshots(path, regions, label, name):
    ds = xr.open_dataset(path)
    da = ds[f"SPI{ACCUM_MONTHS}"]
    n = len(SNAPSHOT_DATES)
    fig, axes = plt.subplots(n, 1, figsize=(13, 5.6 * n), subplot_kw={"projection": PC})
    axes = np.atleast_1d(axes)
    im = None
    for i, (ax, (y, m)) in enumerate(zip(axes, SNAPSHOT_DATES)):
        snap = da.sel(time=slice(f"{y}-{m:02d}", f"{y}-{m:02d}")).isel(time=0)
        im = ax.pcolormesh(da.lon, da.lat, snap.values, cmap="RdBu", vmin=-3, vmax=3,
                           transform=PC, shading="auto", rasterized=True, zorder=1)
        setup_map(ax, regions)
        ax.set_title(f"({chr(97 + i)}) {pd.Timestamp(year=y, month=m, day=1):%B %Y}",
                     loc="left", fontsize=18, fontweight="bold")
    fig.subplots_adjust(hspace=0.3, top=0.94)
    cb = fig.colorbar(im, ax=list(axes), orientation="horizontal", fraction=0.035,
                      pad=0.07, aspect=40, extend="both", ticks=np.arange(-3, 4, 1))
    style_cbar(cb, f"SPI-{ACCUM_MONTHS}  (red = dry, blue = wet)")
    fig.suptitle(label, fontsize=20, fontweight="bold", y=0.985)
    savefig(fig, name)
    ds.close()


def plot_global_mean_timeseries():
    fig, ax = plt.subplots(figsize=(14, 6))
    for path, col, lab in [(F_SPI_ST, COL_DRY, "Stationary"),
                           (F_SPI_NS, COL_WET, f"Non-stationary ({WINDOW_YEARS}-yr window)")]:
        ds = xr.open_dataset(path)
        da = ds[f"SPI{ACCUM_MONTHS}"]
        w = np.cos(np.deg2rad(da.lat))
        gm = da.weighted(w).mean(dim=("lat", "lon")).to_series()
        sm = gm.rolling(60, min_periods=36, center=True).mean()
        ax.plot(gm.index, gm.values, color=col, lw=0.8, alpha=0.3)
        ax.plot(sm.index, sm.values, color=col, lw=3.0, label=f"{lab} (60-month mean)")
        gm.to_csv(f"{CSV_DIR}/global_mean_SPI{ACCUM_MONTHS}_{lab.split()[0].lower()}.csv",
                  header=["global_mean_SPI"])
        ds.close()
    ax.axhline(0, color="black", lw=1.5)
    ax.set_xlabel("Year", fontsize=16, fontweight="bold")
    ax.set_ylabel(f"Area-weighted global mean SPI-{ACCUM_MONTHS}", fontsize=16, fontweight="bold")
    ax.set_title(f"Global-mean SPI-{ACCUM_MONTHS}: stationary vs non-stationary", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14, width=1.5, length=6)
    ax.grid(alpha=0.3)
    leg = ax.legend(loc="lower left", frameon=True, edgecolor="black", prop={"weight": "bold", "size": 14})
    leg.get_frame().set_linewidth(1.5)
    savefig(fig, f"SPI{ACCUM_MONTHS}_global_mean_timeseries")


def plot_trend_tau(path, regions, label, name):
    ds = xr.open_dataset(path)
    tau, p = ds["tau"].values, ds["p_value"].values
    sig = np.isfinite(p) & (p < ALPHA)
    nonsig = np.where(np.isfinite(tau) & ~sig, 1.0, np.nan)

    fig = plt.figure(figsize=(14, 7.5))
    ax = plt.axes(projection=PC)
    ax.pcolormesh(ds.lon, ds.lat, nonsig, cmap=ListedColormap([COL_NONE]), vmin=0, vmax=1,
                  transform=PC, shading="auto", rasterized=True, zorder=1)
    im = ax.pcolormesh(ds.lon, ds.lat, np.where(sig, tau, np.nan), cmap="RdBu", vmin=-0.4, vmax=0.4,
                       transform=PC, shading="auto", rasterized=True, zorder=2)
    setup_map(ax, regions)
    ax.set_title(label, fontsize=18, fontweight="bold", pad=12)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, fraction=0.05, aspect=40,
                      extend="both", ticks=np.arange(-0.4, 0.41, 0.2))
    style_cbar(cb, f"Kendall's τ of SPI-{ACCUM_MONTHS} (significant cells, p < {ALPHA};  red = drying, blue = wetting)")
    ax.legend(handles=[Patch(facecolor=COL_NONE, edgecolor="black", label=f"Not significant (p ≥ {ALPHA})")],
              loc="lower left", frameon=True, edgecolor="black", prop={"weight": "bold", "size": 13})
    savefig(fig, name)
    ds.close()


def category_percentages(cat):
    v = cat[np.isfinite(cat)]
    n = len(v)
    return np.array([(v == -1).sum(), (v == 0).sum(), (v == 1).sum()]) / n * 100.0, n


def plot_trend_category(path, regions, label, name):
    ds = xr.open_dataset(path)
    cat = ds["trend_category"].values
    pct, n = category_percentages(cat)
    cmap = ListedColormap([COL_DRY, COL_NONE, COL_WET])
    norm_c = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig = plt.figure(figsize=(14, 7.5))
    ax = plt.axes(projection=PC)
    ax.pcolormesh(ds.lon, ds.lat, cat, cmap=cmap, norm=norm_c, transform=PC,
                  shading="auto", rasterized=True, zorder=1)
    setup_map(ax, regions)
    ax.set_title(label, fontsize=18, fontweight="bold", pad=12)
    handles = [Patch(facecolor=COL_DRY, edgecolor="black", label=f"Significant drying ({pct[0]:.1f}%)"),
               Patch(facecolor=COL_NONE, edgecolor="black", label=f"No significant trend ({pct[1]:.1f}%)"),
               Patch(facecolor=COL_WET, edgecolor="black", label=f"Significant wetting ({pct[2]:.1f}%)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              frameon=False, prop={"weight": "bold", "size": 15}, handlelength=2.2, handleheight=1.4)
    savefig(fig, name)
    ds.close()


def plot_trend_summary_bar():
    rows = []
    for tag, f in [("Stationary", F_TREND_ST), (f"Non-stationary\n({WINDOW_YEARS}-yr window)", F_TREND_NS)]:
        ds = xr.open_dataset(f)
        pct, n = category_percentages(ds["trend_category"].values)
        rows.append((tag, pct, n))
        ds.close()
    pd.DataFrame({r[0].replace("\n", " "): r[1] for r in rows},
                 index=["drying_%", "no_trend_%", "wetting_%"]).to_csv(
        f"{CSV_DIR}/SPI{ACCUM_MONTHS}_trend_summary_percentages.csv")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(rows))
    wd = 0.25
    for j, (lab, col) in enumerate([("Significant drying", COL_DRY),
                                    ("No significant trend", COL_NONE),
                                    ("Significant wetting", COL_WET)]):
        vals = [r[1][j] for r in rows]
        bars = ax.bar(x + (j - 1) * wd, vals, wd, color=col, edgecolor="black", linewidth=1.5, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=15, fontweight="bold")
    ax.set_ylabel("Percentage of grid cells (%)", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(max(r[1]) for r in rows) * 1.2)
    ax.set_title(f"SPI-{ACCUM_MONTHS} trend categories (Mann-Kendall, α = {ALPHA})",
                 fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14, width=1.5, length=6)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
              prop={"weight": "bold", "size": 14})
    savefig(fig, f"SPI{ACCUM_MONTHS}_trend_summary_bar")


def plot_tau_difference(regions):
    a, b = xr.open_dataset(F_TREND_ST), xr.open_dataset(F_TREND_NS)
    diff = b["tau"].values - a["tau"].values
    fig = plt.figure(figsize=(14, 7.5))
    ax = plt.axes(projection=PC)
    im = ax.pcolormesh(a.lon, a.lat, diff, cmap="RdBu", vmin=-0.4, vmax=0.4, transform=PC,
                       shading="auto", rasterized=True, zorder=1)
    setup_map(ax, regions)
    ax.set_title(f"Trend difference in SPI-{ACCUM_MONTHS}: non-stationary minus stationary",
                 fontsize=18, fontweight="bold", pad=12)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, fraction=0.05, aspect=40,
                      extend="both", ticks=np.arange(-0.4, 0.41, 0.2))
    style_cbar(cb, "Δτ (non-stationary − stationary;  blue = wetter trend, red = drier trend)")
    savefig(fig, f"SPI{ACCUM_MONTHS}_trend_tau_difference")
    a.close()
    b.close()


def make_all_figures():
    print("\nGenerating figures from the saved NetCDF files ...")
    regions = get_ipcc_regions()
    plot_spi_snapshots(F_SPI_ST, regions, f"Stationary SPI-{ACCUM_MONTHS}",
                       f"SPI{ACCUM_MONTHS}_stationary_snapshots")
    plot_spi_snapshots(F_SPI_NS, regions, f"Non-stationary SPI-{ACCUM_MONTHS} ({WINDOW_YEARS}-yr trailing window)",
                       f"SPI{ACCUM_MONTHS}_nonstationary_snapshots")
    plot_global_mean_timeseries()
    plot_trend_tau(F_TREND_ST, regions, f"Trend in stationary SPI-{ACCUM_MONTHS}",
                   f"SPI{ACCUM_MONTHS}_stationary_trend_tau")
    plot_trend_tau(F_TREND_NS, regions, f"Trend in non-stationary SPI-{ACCUM_MONTHS} ({WINDOW_YEARS}-yr window)",
                   f"SPI{ACCUM_MONTHS}_nonstationary_trend_tau")
    plot_trend_category(F_TREND_ST, regions, f"Significant trend categories: stationary SPI-{ACCUM_MONTHS}",
                        f"SPI{ACCUM_MONTHS}_stationary_trend_category")
    plot_trend_category(F_TREND_NS, regions, f"Significant trend categories: non-stationary SPI-{ACCUM_MONTHS}",
                        f"SPI{ACCUM_MONTHS}_nonstationary_trend_category")
    plot_trend_summary_bar()
    plot_tau_difference(regions)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(f"SPI-{ACCUM_MONTHS}: STATIONARY vs NON-STATIONARY ({WINDOW_YEARS}-yr window), GRID-BASED")
    print(f"Period {START_YEAR}-{END_YEAR} | excluded regions: {EXCLUDED_REGIONS}")
    print("=" * 80)
    if RUN_COMPUTE:
        run_analysis()
    if RUN_FIGURES:
        make_all_figures()
    print("\nDONE. Outputs in:", os.path.abspath(OUTPUT_DIR))
