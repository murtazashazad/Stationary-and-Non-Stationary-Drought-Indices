"""
Grid-based SPEI-12: STATIONARY vs NON-STATIONARY (30-yr trailing window)
=========================================================================
Computes, for every grid cell of the CRU TS4.10 grid:
  1. Stationary SPEI-12           (one log-logistic fit per calendar month over the
                                    full record 1901-2025)
  2. Non-stationary SPEI-12       (log-logistic refit every year on the trailing
                                    30-year window of the same calendar month)
  3. Grid-cell Mann-Kendall trend (Kendall tau, p-value, category, OLS slope)
     for both SPEI versions

Outputs (all inside ./data, relative to where the script is run):
  data/SPEI12_stationary.nc
  data/SPEI12_nonstationary_30yr.nc
  data/SPEI12_stationary_trend.nc
  data/SPEI12_nonstationary_30yr_trend.nc
  data/figures/*.png (and .pdf)   <- publication figures made from the .nc files
  data/csv/trend_summary_percentages.csv

Method notes
  * Water balance D = P - PET, PET converted from mm/day to mm/month.
  * Distribution: 3-parameter generalized logistic (log-logistic family) fitted by
    L-moments (Hosking 1990), valid for either skew sign (as in the reference codes).
  * Same land mask / excluded regions (EAN, WAN, GIC, SAH, ARP, ECA) as the reference codes.
  * Trend test: scipy.stats.kendalltau on the monthly SPEI-12 series (as in the
    reference codes).  NOTE: SPEI-12 is strongly autocorrelated (11/12 overlap), so
    this test over-states significance.  Treat the p-values as indicative or
    replace by a modified MK (Hamed & Rao 1998) / effective-sample-size correction.
  * TREND_COMMON_PERIOD=True runs both trend tests over the SAME months
    (from the first non-stationary SPEI value, i.e. 1930 onward) so the two
    results are directly comparable.

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
from scipy.stats import norm, kendalltau
import regionmask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS -- EDIT THESE
# ============================================================================
CRU_PRE_FILE = "/home/Data/CRU/ts4.10/cru_ts4.10.1901.2025.pre.dat.nc"
CRU_PET_FILE = "/home/Data/CRU/ts4.10/cru_ts4.10.1901.2025.pet.dat.nc"

ACCUM_MONTHS = 12
START_YEAR = 1901
END_YEAR = 2025
WINDOW_YEARS = 30                 # trailing window for the non-stationary SPEI
EXCLUDED_REGIONS = ["EAN", "WAN", "GIC", "SAH", "ARP", "ECA"]
USE_ALL_DATA_CELLS = True         # keep coastal/island cells outside coarse AR6 polygons

ALPHA = 0.05                      # Mann-Kendall significance level
MIN_VALID_YEARS = 10              # min valid values to fit the distribution
MIN_TREND_MONTHS = 120            # min valid months to run the trend test
TREND_COMMON_PERIOD = True        # trend of both SPEIs over the same months

OUTPUT_DIR = "./data"
FIG_DIR = f"{OUTPUT_DIR}/figures"
CSV_DIR = f"{OUTPUT_DIR}/csv"

F_SPEI_ST = f"{OUTPUT_DIR}/SPEI{ACCUM_MONTHS}_stationary.nc"
F_SPEI_NS = f"{OUTPUT_DIR}/SPEI{ACCUM_MONTHS}_nonstationary_{WINDOW_YEARS}yr.nc"
F_TREND_ST = f"{OUTPUT_DIR}/SPEI{ACCUM_MONTHS}_stationary_trend.nc"
F_TREND_NS = f"{OUTPUT_DIR}/SPEI{ACCUM_MONTHS}_nonstationary_{WINDOW_YEARS}yr_trend.nc"

# Snapshot maps of SPEI (year, month) shown in the SPEI figures
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
# STEP 1: DATA, WATER BALANCE, REGIONS, LAND MASK
# ============================================================================
def _load_cru_variable(filepath, preferred_name):
    ds = xr.open_dataset(filepath)
    name = preferred_name if preferred_name in ds.data_vars else list(ds.data_vars)[0]
    da = ds[name]
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"
    if lat_name != "lat":
        da = da.rename({lat_name: "lat", lon_name: "lon"})
    return da.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))


def load_water_balance():
    """D = P - PET in mm/month (CRU 'pet' is mm/day -> multiplied by days in month)."""
    print("Loading CRU precipitation and PET ...")
    pre = _load_cru_variable(CRU_PRE_FILE, "pre")
    pet = _load_cru_variable(CRU_PET_FILE, "pet")
    pre, pet = xr.align(pre, pet, join="inner")

    units = str(pet.attrs.get("units", "")).lower()
    pet_daily = ("day" in units) or ("/d" in units)
    if not units:
        pet_daily = float(pet.max()) < 30
        print("  WARNING: PET has no units attribute; guessed from magnitude.")
    if pet_daily:
        print("  PET is mm/day -> converting to mm/month.")
        pet = pet * pet["time"].dt.days_in_month

    wb = (pre - pet).astype("float32")
    wb.name = "water_balance"
    print(f"  Water balance shape: {wb.shape}; "
          f"time {str(wb.time.values[0])[:10]} to {str(wb.time.values[-1])[:10]}")
    return wb


def get_ipcc_regions():
    all_regions = regionmask.defined_regions.ar6.land
    keep = [i for i, ab in enumerate(all_regions.abbrevs) if ab not in EXCLUDED_REGIONS]
    return all_regions[keep]


def get_land_mask(wb, regions):
    mask_3D = regions.mask_3D(wb.lon, wb.lat)
    land_mask = mask_3D.any(dim="region")
    if USE_ALL_DATA_CELLS:
        all_regions = regionmask.defined_regions.ar6.land
        excl_idx = [i for i, ab in enumerate(all_regions.abbrevs) if ab in EXCLUDED_REGIONS]
        excl_mask = all_regions[excl_idx].mask_3D(wb.lon, wb.lat).any(dim="region")
        has_data = wb.notnull().any(dim="time")
        land_mask = land_mask | (has_data & ~excl_mask)
    land_mask = land_mask.transpose("lat", "lon")
    print(f"  Land cells kept: {int(land_mask.sum())} / {land_mask.size}")
    return land_mask


# ============================================================================
# STEP 2: SPEI CORE (L-MOMENT GENERALIZED LOGISTIC, VECTORIZED)
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


def glo_params(values_2d):
    """
    3-parameter generalized logistic (log-logistic family) by L-moments
    (Hosking 1990), vectorized across cells (axis 1). Handles NaN.
    Returns xi (location), alpha (scale), k (shape), each (n_cells,).
    """
    n_cells = values_2d.shape[1]
    xi_a = np.full(n_cells, np.nan)
    al_a = np.full(n_cells, np.nan)
    k_a = np.full(n_cells, np.nan)

    valid = ~np.isnan(values_2d)
    n_valid = valid.sum(axis=0)
    good = n_valid >= MIN_VALID_YEARS
    if not np.any(good):
        return xi_a, al_a, k_a

    vals = values_2d[:, good]
    n = n_valid[good].astype(float)[None, :]
    N = vals.shape[0]

    xs = np.sort(vals, axis=0)                       # NaN pushed to the end
    rank = np.arange(1, N + 1).reshape(N, 1).astype(float)
    ok = rank <= n
    x = np.where(ok, xs, 0.0)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        b0 = x.sum(axis=0) / n[0]
        b1 = (np.where(ok, (rank - 1) / (n - 1), 0.0) * x).sum(axis=0) / n[0]
        b2 = (np.where(ok, (rank - 1) * (rank - 2) / ((n - 1) * (n - 2)), 0.0) * x).sum(axis=0) / n[0]

        l1 = b0
        l2 = 2 * b1 - b0
        l3 = 6 * b2 - 6 * b1 + b0
        k = -l3 / l2

        small = np.abs(k) < 1e-6
        ks = np.where(small, 1.0, k)
        alpha = np.where(small, l2, l2 * np.sin(ks * np.pi) / (ks * np.pi))
        xi = np.where(small, l1, l1 - alpha * (1.0 / ks - np.pi / np.sin(ks * np.pi)))
        k = np.where(small, 0.0, k)

    fit_ok = (np.isfinite(alpha) & np.isfinite(xi) & np.isfinite(k)
              & (alpha > 0) & (l2 > 0) & (np.abs(k) < 1))

    xi_a[good] = np.where(fit_ok, xi, np.nan)
    al_a[good] = np.where(fit_ok, alpha, np.nan)
    k_a[good] = np.where(fit_ok, k, np.nan)
    return xi_a, al_a, k_a


def glo_cdf(x, xi, alpha, k):
    """CDF of the generalized logistic distribution (any sign of k)."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = (x - xi) / alpha
        arg = 1.0 - k * z
        small = np.abs(k) < 1e-6
        y = np.where(small, z,
                     -np.log(np.where(arg > 0, arg, np.nan)) / np.where(small, 1.0, k))
        cdf = 1.0 / (1.0 + np.exp(-y))
        outside = (arg <= 0) & ~small
        cdf = np.where(outside, np.where(k > 0, 1 - 1e-6, 1e-6), cdf)
    return cdf


def spei_stationary(accum, months):
    """One distribution per calendar month fitted on the FULL record."""
    print("\nComputing STATIONARY SPEI ...")
    out = np.full_like(accum, np.nan)
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        vals = accum[idx]
        xi, al, k = glo_params(vals)
        cdf = glo_cdf(vals, xi[None, :], al[None, :], k[None, :])
        out[idx] = norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
        print(f"  Month {m:2d}: distribution fitted at {int(np.sum(~np.isnan(al)))}/{accum.shape[1]} cells")
    return out


def spei_nonstationary(accum, years, months, window_years):
    """
    Trailing moving-window climatology: for target year Y and calendar month m the
    distribution is fitted on years [Y-window+1, Y] of month m only, then the
    value of (Y, m) is transformed with that fit.  First (window-1) years are NaN.
    """
    print(f"\nComputing NON-STATIONARY SPEI (trailing {window_years}-yr window) ...")
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
            xi, al, k = glo_params(accum[fit_idx])
            cdf = glo_cdf(accum[tgt[0]], xi, al, k)
            out[tgt[0]] = norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
        if (c + 1) % 10 == 0 or c + 1 == len(targets):
            print(f"    target year {ty} ({c + 1}/{len(targets)})  [{time.time() - t0:.0f}s]")
    return out


# ============================================================================
# STEP 3: GRID-CELL TREND (MANN-KENDALL / KENDALL TAU)
# ============================================================================
def compute_trend(spei_c, t_years):
    """Kendall tau + p-value + category + OLS slope (per decade) at each cell."""
    n_cells = spei_c.shape[1]
    tau = np.full(n_cells, np.nan)
    p = np.full(n_cells, np.nan)
    slope = np.full(n_cells, np.nan)
    for j in range(n_cells):
        s = spei_c[:, j]
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
        "source": "CRU TS4.10 precipitation and potential evapotranspiration",
        "method": "Water balance D = P - PET; 3-parameter log-logistic (generalized "
                  "logistic) fitted by L-moments (Hosking 1990; Vicente-Serrano et al. 2010)",
        "excluded_regions": ",".join(EXCLUDED_REGIONS),
        "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    a.update(extra)
    return a


def run_analysis():
    wb = load_water_balance()
    regions = get_ipcc_regions()
    land_mask = get_land_mask(wb, regions)
    wb = wb.where(land_mask)

    n_time, n_lat, n_lon = wb.shape
    flat = wb.values.reshape(n_time, n_lat * n_lon)
    valid_idx = np.where(np.isfinite(flat).any(axis=0))[0]          # compact: land cells only
    print(f"  Cells with data: {len(valid_idx)} / {n_lat * n_lon}")
    D = flat[:, valid_idx].astype(np.float64)
    del flat

    times = wb["time"]
    years = times.dt.year.values
    months = times.dt.month.values

    print("\nAccumulating water balance over", ACCUM_MONTHS, "months ...")
    accum = rolling_sum(D, ACCUM_MONTHS)
    del D

    spei_st = spei_stationary(accum, months)
    spei_ns = spei_nonstationary(accum, years, months, WINDOW_YEARS)
    del accum

    coords = {"time": times.values, "lat": wb.lat.values, "lon": wb.lon.values}
    results = {}
    for tag, arr, fname, desc in [
        ("stationary", spei_st, F_SPEI_ST,
         f"Stationary SPEI-{ACCUM_MONTHS}: one distribution per calendar month fitted on {START_YEAR}-{END_YEAR}"),
        ("nonstationary", spei_ns, F_SPEI_NS,
         f"Non-stationary SPEI-{ACCUM_MONTHS}: distribution refitted on trailing {WINDOW_YEARS}-year window"),
    ]:
        full = np.full((n_time, n_lat * n_lon), np.nan, dtype=np.float32)
        full[:, valid_idx] = arr
        da = xr.DataArray(full.reshape(n_time, n_lat, n_lon), dims=("time", "lat", "lon"),
                          coords=coords, name=f"SPEI{ACCUM_MONTHS}",
                          attrs={"long_name": f"Standardized Precipitation Evapotranspiration Index ({ACCUM_MONTHS}-month)",
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
        print(f"\nMann-Kendall trend on {tag} SPEI ({int(m_.sum())} months) ...")
        tau, p, cat, slope = compute_trend(results[tag][m_], t_years[m_])
        ds = xr.Dataset(
            {
                "tau": (("lat", "lon"), to_grid(tau, valid_idx, n_lat, n_lon),
                        {"long_name": "Kendall tau (monthly SPEI-12 vs time)"}),
                "p_value": (("lat", "lon"), to_grid(p, valid_idx, n_lat, n_lon),
                            {"long_name": "Two-sided p-value of Kendall tau"}),
                "trend_category": (("lat", "lon"), to_grid(cat, valid_idx, n_lat, n_lon),
                                   {"long_name": "-1 significant drying, 0 no significant trend, +1 significant wetting",
                                    "alpha": ALPHA}),
                "slope_per_decade": (("lat", "lon"), to_grid(slope, valid_idx, n_lat, n_lon),
                                     {"long_name": "OLS slope of SPEI-12", "units": "SPEI units per decade"}),
            },
            coords={"lat": wb.lat.values, "lon": wb.lon.values},
        )
        ds.attrs = common_attrs({
            "title": f"Grid-cell trend of {tag} SPEI-{ACCUM_MONTHS}",
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


def plot_spei_snapshots(path, regions, label, name):
    ds = xr.open_dataset(path)
    da = ds[f"SPEI{ACCUM_MONTHS}"]
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
    style_cbar(cb, f"SPEI-{ACCUM_MONTHS}  (red = dry, blue = wet)")
    fig.suptitle(label, fontsize=20, fontweight="bold", y=0.985)
    savefig(fig, name)
    ds.close()


def plot_global_mean_timeseries():
    fig, ax = plt.subplots(figsize=(14, 6))
    for path, col, lab in [(F_SPEI_ST, COL_DRY, "Stationary"),
                           (F_SPEI_NS, COL_WET, f"Non-stationary ({WINDOW_YEARS}-yr window)")]:
        ds = xr.open_dataset(path)
        da = ds[f"SPEI{ACCUM_MONTHS}"]
        w = np.cos(np.deg2rad(da.lat))
        gm = da.weighted(w).mean(dim=("lat", "lon")).to_series()
        sm = gm.rolling(60, min_periods=36, center=True).mean()
        ax.plot(gm.index, gm.values, color=col, lw=0.8, alpha=0.3)
        ax.plot(sm.index, sm.values, color=col, lw=3.0, label=f"{lab} (60-month mean)")
        gm.to_csv(f"{CSV_DIR}/global_mean_SPEI{ACCUM_MONTHS}_{lab.split()[0].lower()}.csv", header=["global_mean_SPEI"])
        ds.close()
    ax.axhline(0, color="black", lw=1.5)
    ax.set_xlabel("Year", fontsize=16, fontweight="bold")
    ax.set_ylabel(f"Area-weighted global mean SPEI-{ACCUM_MONTHS}", fontsize=16, fontweight="bold")
    ax.set_title(f"Global-mean SPEI-{ACCUM_MONTHS}: stationary vs non-stationary", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14, width=1.5, length=6)
    ax.grid(alpha=0.3)
    leg = ax.legend(loc="lower left", frameon=True, edgecolor="black", prop={"weight": "bold", "size": 14})
    leg.get_frame().set_linewidth(1.5)
    savefig(fig, f"SPEI{ACCUM_MONTHS}_global_mean_timeseries")


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
    style_cbar(cb, f"Kendall's \u03c4 of SPEI-{ACCUM_MONTHS} (significant cells, p < {ALPHA};  red = drying, blue = wetting)")
    ax.legend(handles=[Patch(facecolor=COL_NONE, edgecolor="black", label=f"Not significant (p \u2265 {ALPHA})")],
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
                 index=["drying_%", "no_trend_%", "wetting_%"]).to_csv(f"{CSV_DIR}/trend_summary_percentages.csv")

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
    ax.set_title(f"SPEI-{ACCUM_MONTHS} trend categories (Mann-Kendall, \u03b1 = {ALPHA})",
                 fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14, width=1.5, length=6)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
                    prop={"weight": "bold", "size": 14})
    savefig(fig, f"SPEI{ACCUM_MONTHS}_trend_summary_bar")


def plot_tau_difference(regions):
    a, b = xr.open_dataset(F_TREND_ST), xr.open_dataset(F_TREND_NS)
    diff = b["tau"].values - a["tau"].values
    fig = plt.figure(figsize=(14, 7.5))
    ax = plt.axes(projection=PC)
    im = ax.pcolormesh(a.lon, a.lat, diff, cmap="RdBu", vmin=-0.4, vmax=0.4, transform=PC,
                       shading="auto", rasterized=True, zorder=1)
    setup_map(ax, regions)
    ax.set_title(f"Trend difference in SPEI-{ACCUM_MONTHS}: non-stationary minus stationary",
                 fontsize=18, fontweight="bold", pad=12)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, fraction=0.05, aspect=40,
                      extend="both", ticks=np.arange(-0.4, 0.41, 0.2))
    style_cbar(cb, "\u0394\u03c4 (non-stationary \u2212 stationary;  blue = wetter trend, red = drier trend)")
    savefig(fig, f"SPEI{ACCUM_MONTHS}_trend_tau_difference")
    a.close()
    b.close()


def make_all_figures():
    print("\nGenerating figures from the saved NetCDF files ...")
    regions = get_ipcc_regions()
    plot_spei_snapshots(F_SPEI_ST, regions, f"Stationary SPEI-{ACCUM_MONTHS}",
                        f"SPEI{ACCUM_MONTHS}_stationary_snapshots")
    plot_spei_snapshots(F_SPEI_NS, regions, f"Non-stationary SPEI-{ACCUM_MONTHS} ({WINDOW_YEARS}-yr trailing window)",
                        f"SPEI{ACCUM_MONTHS}_nonstationary_snapshots")
    plot_global_mean_timeseries()
    plot_trend_tau(F_TREND_ST, regions, f"Trend in stationary SPEI-{ACCUM_MONTHS}",
                   f"SPEI{ACCUM_MONTHS}_stationary_trend_tau")
    plot_trend_tau(F_TREND_NS, regions, f"Trend in non-stationary SPEI-{ACCUM_MONTHS} ({WINDOW_YEARS}-yr window)",
                   f"SPEI{ACCUM_MONTHS}_nonstationary_trend_tau")
    plot_trend_category(F_TREND_ST, regions, f"Significant trend categories: stationary SPEI-{ACCUM_MONTHS}",
                        f"SPEI{ACCUM_MONTHS}_stationary_trend_category")
    plot_trend_category(F_TREND_NS, regions, f"Significant trend categories: non-stationary SPEI-{ACCUM_MONTHS}",
                        f"SPEI{ACCUM_MONTHS}_nonstationary_trend_category")
    plot_trend_summary_bar()
    plot_tau_difference(regions)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(f"SPEI-{ACCUM_MONTHS}: STATIONARY vs NON-STATIONARY ({WINDOW_YEARS}-yr window), GRID-BASED")
    print(f"Period {START_YEAR}-{END_YEAR} | excluded regions: {EXCLUDED_REGIONS}")
    print("=" * 80)
    if RUN_COMPUTE:
        run_analysis()
    if RUN_FIGURES:
        make_all_figures()
    print("\nDONE. Outputs in:", os.path.abspath(OUTPUT_DIR))
