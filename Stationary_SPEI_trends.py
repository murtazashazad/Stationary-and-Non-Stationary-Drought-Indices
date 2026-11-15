"""
Grid-Point SPEI-12 Mann-Kendall Trend Analysis for IPCC AR6 Land Regions
============================================================================
- Water balance D = Precipitation - Potential Evapotranspiration (PET),
  computed at EVERY grid cell, with PET converted from mm/day to mm/month
  before subtraction (CRU stores 'pre' in mm/month but 'pet' in mm/day).
- SPEI-12 computed at every grid cell by fitting a 3-parameter LOG-LOGISTIC
  distribution to the accumulated water balance via L-moments (Ahmad,
  Sinclair & Werritty 1988; Vicente-Serrano, Begueria & Lopez-Moreno 2010),
  the standard published SPEI method. Unlike SPI's gamma distribution,
  log-logistic supports the negative values water balance can take.
- Excludes: EAN, WAN, GIC, SAH, ARP, ECA (analysis run on remaining regions)
- Grid-point Mann-Kendall trend test on the MONTHLY SPEI-12 series
- Regional trend percentages (% wetting / % no-trend / % drying per region)
- Figures (identical structure to the SPI-3 script):
    1. Grid-point trend map (red=drying, blue=wetting, grey=no trend) + IPCC boundaries
    2. Six continent-grouped trend area-fraction bar charts (y-axis 0-1)
    3. Regional trend map: grid-point dots + region boundary lines colored
       red/blue/black by dominant trend category (by grid-cell count)
    4. Regional trend map (dominant-category choropleth, discrete colors)
- Period: 1901-2023

Requirements:
    pip install xarray numpy pandas scipy regionmask matplotlib cartopy netCDF4
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.stats import norm, kendalltau
from scipy.special import gamma as gammafn
import regionmask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PLOT STYLE (Publication Quality)
# ============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.linewidth": 0.8,
})

# ============================================================================
# PARAMETERS -- EDIT THESE
# ============================================================================
CRU_PRE_FILE = "/home/Data/CRU/ts4.10/cru_ts4.10.1901.2025.pre.dat.nc"
CRU_PET_FILE = "/home/Data/CRU/ts4.10/cru_ts4.10.1901.2025.pet.dat.nc"
OUTPUT_DIR = "./SPEI-12_trend_results"
ACCUM_MONTHS = 12
START_YEAR = 1901
END_YEAR = 2025

# Regions to EXCLUDE (analysis runs on the remaining 40 IPCC AR6 land regions)
EXCLUDED_REGIONS = ["EAN", "WAN", "GIC", "SAH", "ARP", "ECA"]

# ----------------------------------------------------------------------
# IPCC AR6 land-region abbreviation -> continent grouping (6 continents).
# Central America (NCA, SCA) and the Caribbean (CAR) are grouped with
# North America geographically. The Mediterranean (MED) is grouped with
# Europe, matching the standard IPCC AR6 continental grouping. This
# covers all 40 regions remaining after EXCLUDED_REGIONS is applied.
# ----------------------------------------------------------------------
CONTINENT_MAP = {
    # North America (8)
    "NWN": "North America", "NEN": "North America", "WNA": "North America",
    "CNA": "North America", "ENA": "North America", "NCA": "North America",
    "SCA": "North America", "CAR": "North America",
    # South America (7)
    "NWS": "South America", "NSA": "South America", "NES": "South America",
    "SAM": "South America", "SWS": "South America", "SES": "South America",
    "SSA": "South America",
    # Europe (4)
    "NEU": "Europe", "WCE": "Europe", "EEU": "Europe", "MED": "Europe",
    # Africa (7)
    "WAF": "Africa", "CAF": "Africa", "NEAF": "Africa", "SEAF": "Africa",
    "WSAF": "Africa", "ESAF": "Africa", "MDG": "Africa",
    # Asia (9)
    "RAR": "Asia", "WSB": "Asia", "ESB": "Asia", "RFE": "Asia",
    "WCA": "Asia", "TIB": "Asia", "EAS": "Asia", "SAS": "Asia", "SEA": "Asia",
    # Oceania (5)
    "NAU": "Oceania", "CAU": "Oceania", "EAU": "Oceania", "SAU": "Oceania",
    "NZ": "Oceania",
}
CONTINENT_ORDER = ["North America", "South America", "Europe", "Africa", "Asia", "Oceania"]

ALPHA = 0.05                # significance level for Mann-Kendall test
MIN_VALID_YEARS = 10        # minimum valid years to fit log-logistic at a grid cell/month
USE_ALL_DATA_CELLS = True   # run trend test on all cells with data (except excluded regions)
MIN_TREND_MONTHS = 120      # minimum valid monthly points to run MK test at a cell

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/csv", exist_ok=True)


# ============================================================================
# STEP 1: LOAD CRU PRECIPITATION + PET, COMPUTE WATER BALANCE
# ============================================================================
def load_cru_precipitation(filepath, start_year=START_YEAR, end_year=END_YEAR):
    print("Loading CRU precipitation data...")
    ds = xr.open_dataset(filepath)

    var_name = "pre" if "pre" in ds.data_vars else list(ds.data_vars)[0]
    pre = ds[var_name]

    lat_name = "lat" if "lat" in pre.dims else "latitude"
    lon_name = "lon" if "lon" in pre.dims else "longitude"
    if lat_name != "lat":
        pre = pre.rename({lat_name: "lat", lon_name: "lon"})

    pre = pre.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    print(f"  Loaded precipitation: {pre.shape}")
    print(f"  Time range: {pre.time.values[0]} to {pre.time.values[-1]}")
    print(f"  Units: {pre.attrs.get('units', 'mm/month (assumed)')}")

    return pre


def load_cru_pet(filepath, start_year=START_YEAR, end_year=END_YEAR):
    """
    Load CRU PET. NOTE: CRU stores 'pet' in mm/day (a daily rate), NOT
    mm/month like 'pre'. This function returns the raw mm/day field;
    conversion to a monthly total happens in compute_water_balance().
    """
    print("Loading CRU PET data...")
    ds = xr.open_dataset(filepath)

    var_name = "pet" if "pet" in ds.data_vars else list(ds.data_vars)[0]
    pet = ds[var_name]

    lat_name = "lat" if "lat" in pet.dims else "latitude"
    lon_name = "lon" if "lon" in pet.dims else "longitude"
    if lat_name != "lat":
        pet = pet.rename({lat_name: "lat", lon_name: "lon"})

    pet = pet.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    print(f"  Loaded PET: {pet.shape}")
    print(f"  Time range: {pet.time.values[0]} to {pet.time.values[-1]}")
    print(f"  Units: {pet.attrs.get('units', 'mm/day (assumed)')}")

    return pet


def compute_water_balance(pre, pet):
    """
    Water balance D = Precipitation - PET, both converted to mm/month.
    CRU 'pre' is already mm/month; CRU 'pet' is mm/day and is multiplied
    by the number of days in each month before subtracting.
    Cells where either input is NaN (e.g. mismatched land masks between
    the pre and pet products) are naturally NaN in the result.
    """
    print("\nComputing water balance (P - PET, both in mm/month)...")
    days_in_month = pre["time"].dt.days_in_month
    pet_mm_month = pet * days_in_month

    water_balance = pre - pet_mm_month
    water_balance.name = "water_balance"
    water_balance.attrs["units"] = "mm/month"
    water_balance.attrs["long_name"] = "Precipitation minus PET (water balance)"

    n_total = water_balance.size
    n_nan = int(np.isnan(water_balance.values).sum())
    print(f"  Water balance computed: {water_balance.shape}")
    print(f"  NaN cells: {n_nan}/{n_total} ({100*n_nan/n_total:.2f}%) "
          f"(ocean + any pre/pet mask mismatch)")

    return water_balance


# ============================================================================
# STEP 2: IPCC AR6 LAND REGIONS (WITH EXCLUSION)
# ============================================================================
def get_ipcc_regions(excluded_regions=EXCLUDED_REGIONS):
    """Load IPCC AR6 land regions and remove excluded regions."""
    print("\nLoading IPCC AR6 land regions...")
    all_regions = regionmask.defined_regions.ar6.land
    print(f"  Total land regions available: {len(all_regions.abbrevs)}")

    if excluded_regions:
        print(f"  Excluding regions: {excluded_regions}")

        keep_indices = [
            i for i, ab in enumerate(all_regions.abbrevs)
            if ab not in excluded_regions
        ]

        found_excluded = [ab for ab in all_regions.abbrevs if ab in excluded_regions]
        missing = set(excluded_regions) - set(found_excluded)
        if missing:
            print(f"  WARNING: These excluded region codes were not found: {missing}")

        regions = all_regions[keep_indices]
        print(f"  Regions remaining after exclusion: {len(regions.abbrevs)}")
    else:
        regions = all_regions

    return regions


def get_land_mask(data, regions):
    """
    Returns:
      land_mask - boolean (lat, lon), True where grid cell is in ANY kept region
      mask_3D   - boolean (region, lat, lon), per-region masks
    """
    print("\nBuilding land mask from kept IPCC regions...")
    mask_3D = regions.mask_3D(data.lon, data.lat)
    land_mask = mask_3D.any(dim="region")
    if USE_ALL_DATA_CELLS:
        # Cell centres of coastal/island cells often fall just outside the coarse
        # AR6 polygons and were silently dropped (white gaps). Keep every cell
        # that has data, except cells inside the deliberately EXCLUDED regions.
        all_regions = regionmask.defined_regions.ar6.land
        excl_idx = [i for i, ab in enumerate(all_regions.abbrevs) if ab in EXCLUDED_REGIONS]
        if excl_idx:
            excl_mask = all_regions[excl_idx].mask_3D(data.lon, data.lat).any(dim="region")
        else:
            excl_mask = xr.zeros_like(land_mask)
        has_data = data.notnull().any(dim="time")
        land_mask = (land_mask | (has_data & ~excl_mask)).transpose("lat", "lon")
    n_land = int(land_mask.sum().values)
    n_total = land_mask.size
    print(f"  Land grid cells in kept regions: {n_land} / {n_total} ({n_land/n_total*100:.1f}%)")
    return land_mask, mask_3D


# ============================================================================
# STEP 3: GRID-POINT SPEI-12 (VECTORIZED, LOG-LOGISTIC VIA L-MOMENTS)
# ============================================================================
def glo_lmom_params(values_2d):
    """
    Fit the 3-parameter generalized logistic (= log-logistic family) via
    L-moments (Hosking 1990), vectorized over grid cells' calendar-month
    samples.

    WHY THIS REPLACES THE OLD FIT: the old closed form
    (alpha, beta, gamma with Gamma(1 +/- 1/beta)) is only valid for
    POSITIVELY skewed samples. For negatively skewed samples beta comes out
    <= 0 and the old code did `continue`, leaving NaN. Tested on synthetic
    data: ~50% of normal-like samples and 100% of negatively skewed samples
    were rejected, so many cells lost most/all calendar months, ended with
    < MIN_TREND_MONTHS valid SPEI values and appeared as white gaps.
    The Hosking parameterisation below is the SAME distribution family but
    is valid for either skew sign (shape k > 0 or k < 0).

    Returns xi (location), alpha (scale), k (shape), each shape (n_grid,).
    """
    n_grid = values_2d.shape[1]
    xi_arr = np.full(n_grid, np.nan)
    al_arr = np.full(n_grid, np.nan)
    k_arr = np.full(n_grid, np.nan)

    valid_mask = ~np.isnan(values_2d)
    n_valid = valid_mask.sum(axis=0)
    good_idx = np.where(n_valid >= MIN_VALID_YEARS)[0]

    for col in good_idx:
        x = np.sort(values_2d[valid_mask[:, col], col])
        n = len(x)
        i = np.arange(1, n + 1)

        # unbiased ascending probability-weighted moments
        b0 = x.mean()
        b1 = np.sum((i - 1) / (n - 1) * x) / n
        b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n

        l1 = b0
        l2 = 2 * b1 - b0
        l3 = 6 * b2 - 6 * b1 + b0
        if not np.isfinite(l2) or l2 <= 0:      # constant / degenerate sample
            continue
        t3 = l3 / l2
        k = -t3
        if abs(k) >= 1:
            continue

        if abs(k) < 1e-6:                       # logistic limit
            alpha = l2
            xi = l1
            k = 0.0
        else:
            alpha = l2 * np.sin(k * np.pi) / (k * np.pi)
            xi = l1 - alpha * (1.0 / k - np.pi / np.sin(k * np.pi))
        if not (np.isfinite(alpha) and np.isfinite(xi)) or alpha <= 0:
            continue

        xi_arr[col], al_arr[col], k_arr[col] = xi, alpha, k

    return xi_arr, al_arr, k_arr


def glo_cdf(x, xi, alpha, k):
    """CDF of the generalized logistic distribution (any sign of k)."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = (x - xi) / alpha
        arg = 1.0 - k * z
        small_k = np.abs(k) < 1e-6
        y = np.where(small_k, z, -np.log(np.where(arg > 0, arg, np.nan)) / np.where(small_k, 1.0, k))
        cdf = 1.0 / (1.0 + np.exp(-y))
        # outside the support: k>0 -> above upper bound (CDF=1); k<0 -> below lower bound (CDF=0)
        outside = (arg <= 0) & ~small_k
        cdf = np.where(outside, np.where(k > 0, 1 - 1e-6, 1e-6), cdf)
    return cdf


def compute_spei_gridded(water_balance, accum_months=ACCUM_MONTHS):
    """Compute SPEI at EVERY grid cell (vectorized across space, looped over 12 calendar months)."""
    print(f"\nComputing grid-point SPEI-{accum_months}...")

    accum = water_balance.rolling(time=accum_months, min_periods=accum_months).sum(skipna=True)

    n_time, n_lat, n_lon = accum.shape
    n_grid = n_lat * n_lon

    accum_flat = accum.values.astype(np.float64).reshape(n_time, n_grid)
    spei_flat = np.full_like(accum_flat, np.nan)

    months = accum["time"].dt.month.values

    for m in range(1, 13):
        month_time_idx = np.where(months == m)[0]
        vals = accum_flat[month_time_idx, :]

        xi_arr, al_arr, k_arr = glo_lmom_params(vals)
        cdf = glo_cdf(vals, xi_arr[None, :], al_arr[None, :], k_arr[None, :])
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spei_flat[month_time_idx, :] = norm.ppf(cdf)

        n_fit = np.sum(~np.isnan(al_arr))
        print(f"  Month {m:2d}: fitted distribution at {n_fit}/{n_grid} grid cells")

    spei_array = spei_flat.reshape(n_time, n_lat, n_lon)

    spei_da = xr.DataArray(
        spei_array,
        coords={"time": accum.time, "lat": accum.lat, "lon": accum.lon},
        dims=["time", "lat", "lon"],
        name=f"SPEI{accum_months}"
    )
    print(f"  Grid-point SPEI-{accum_months} computation complete.")
    return spei_da


# ============================================================================
# STEP 4: GRID-POINT MANN-KENDALL TREND TEST (MONTHLY SERIES)
# ============================================================================
def compute_grid_trend_monthly(spei_gridded, land_mask, alpha=ALPHA):
    """
    Mann-Kendall trend test at EVERY grid cell, computed on the full
    MONTHLY SPEI-12 series (not annual mean).

    Returns
    -------
    trend_da : xr.DataArray (lat, lon)
        1  = significant increasing (wetting)
       -1  = significant decreasing (drying)
        0  = no significant trend
       NaN = insufficient data / not in a kept region
    p_da, tau_da : xr.DataArray (lat, lon)
    """
    print("\nComputing grid-point Mann-Kendall trend on MONTHLY SPEI-12 series...")

    data = spei_gridded.values  # (n_time, n_lat, n_lon)
    n_time, n_lat, n_lon = data.shape
    t = np.arange(n_time)

    land_flat = land_mask.values.reshape(-1)
    data_flat = data.reshape(n_time, n_lat * n_lon)
    grid_indices = np.where(land_flat)[0]

    trend_flat = np.full(n_lat * n_lon, np.nan)
    p_flat = np.full(n_lat * n_lon, np.nan)
    tau_flat = np.full(n_lat * n_lon, np.nan)

    total = len(grid_indices)
    print(f"  Running MK test on {total} land grid cells (monthly series, n={n_time} each)...")

    for count, idx in enumerate(grid_indices):
        series = data_flat[:, idx]
        valid = ~np.isnan(series)
        if valid.sum() < MIN_TREND_MONTHS:
            continue

        tau, p = kendalltau(t[valid], series[valid])
        p_flat[idx] = p
        tau_flat[idx] = tau

        if p < alpha:
            trend_flat[idx] = 1.0 if tau > 0 else -1.0
        else:
            trend_flat[idx] = 0.0

        if (count + 1) % 5000 == 0:
            print(f"    Processed {count+1}/{total} grid cells...")

    coords = {"lat": spei_gridded.lat, "lon": spei_gridded.lon}
    trend_da = xr.DataArray(trend_flat.reshape(n_lat, n_lon), coords=coords,
                             dims=["lat", "lon"], name="trend_category")
    p_da = xr.DataArray(p_flat.reshape(n_lat, n_lon), coords=coords,
                         dims=["lat", "lon"], name="p_value")
    tau_da = xr.DataArray(tau_flat.reshape(n_lat, n_lon), coords=coords,
                           dims=["lat", "lon"], name="tau")

    print("  Grid-point trend test complete.")

    valid_trend = trend_flat[~np.isnan(trend_flat)]
    n_valid = len(valid_trend)
    if n_valid > 0:
        pct_wet = np.sum(valid_trend == 1) / n_valid * 100
        pct_dry = np.sum(valid_trend == -1) / n_valid * 100
        pct_none = np.sum(valid_trend == 0) / n_valid * 100
        print(f"\n  Global summary ({n_valid} grid cells with valid trend result):")
        print(f"    Significant wetting : {pct_wet:5.1f}%")
        print(f"    No significant trend : {pct_none:5.1f}%")
        print(f"    Significant drying  : {pct_dry:5.1f}%")

    return trend_da, p_da, tau_da


def save_trend_to_csv(trend_da, p_da, tau_da, output_dir=OUTPUT_DIR):
    """Save gridded trend results to CSV (long format: lat, lon, trend, p, tau)."""
    print("\nSaving grid-point trend results to CSV...")

    lon2d, lat2d = np.meshgrid(trend_da.lon.values, trend_da.lat.values)
    df = pd.DataFrame({
        "lat": lat2d.flatten(),
        "lon": lon2d.flatten(),
        "trend_category": trend_da.values.flatten(),
        "p_value": p_da.values.flatten(),
        "tau": tau_da.values.flatten(),
    }).dropna(subset=["trend_category"])

    path = f"{output_dir}/csv/SPEI-12_gridpoint_trend.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(df)} grid cells)")
    return df


# ============================================================================
# STEP 5: REGIONAL TREND PERCENTAGES (FROM GRID-POINT RESULTS)
# ============================================================================
def compute_regional_trend_percentage(trend_da, regions, mask_3D):
    """
    For each IPCC region, compute the percentage of its grid cells
    falling into each trend category (significant wetting / no trend /
    significant drying).
    """
    print("\nComputing regional trend percentages from grid-point trends...")

    results = []
    for r, abbrev in enumerate(regions.abbrevs):
        region_mask = mask_3D.isel(region=r).values
        cell_trend = trend_da.values[region_mask]
        cell_trend = cell_trend[~np.isnan(cell_trend)]
        total = len(cell_trend)

        if total == 0:
            pct_inc = pct_no = pct_dec = np.nan
            n_inc = n_no = n_dec = 0
        else:
            n_inc = int(np.sum(cell_trend == 1))
            n_no = int(np.sum(cell_trend == 0))
            n_dec = int(np.sum(cell_trend == -1))
            pct_inc = n_inc / total * 100
            pct_no = n_no / total * 100
            pct_dec = n_dec / total * 100

        results.append({
            "region_abbrev": abbrev,
            "region_name": regions.names[r],
            "continent": CONTINENT_MAP.get(abbrev, "Other"),
            "n_grid_cells": total,
            "n_wet": n_inc,
            "n_no_trend": n_no,
            "n_dry": n_dec,
            "pct_increasing": pct_inc,
            "pct_no_trend": pct_no,
            "pct_decreasing": pct_dec
        })
        print(f"  {abbrev:6s}: n={total:5d}  wetting={pct_inc:5.1f}%  "
              f"no-trend={pct_no:5.1f}%  drying={pct_dec:5.1f}%")

    df = pd.DataFrame(results)
    path = f"{OUTPUT_DIR}/csv/SPEI-12_regional_trend_percentages.csv"
    df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")

    return df


# ============================================================================
# STEP 6: FIGURES  (identical structure/logic to the SPI-3 script,
#                    only labels/filenames renamed to SPEI-12)
# ============================================================================
def plot_trend_map(trend_da, regions, output_dir=OUTPUT_DIR):
    """Grid-resolution map: red=drying, blue=wetting, grey=no trend, + IPCC boundaries."""
    print("\nPlotting grid-point trend map...")

    cmap = ListedColormap(["#B2182B", "#BDBDBD", "#2166AC"])  # dry, none, wet
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm_c = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    im = ax.pcolormesh(trend_da.lon, trend_da.lat, trend_da.values,
                        cmap=cmap, norm=norm_c, transform=ccrs.PlateCarree(),
                        shading="auto")

    for region_poly in regions.polygons:
        ax.add_geometries([region_poly], crs=ccrs.PlateCarree(),
                           facecolor="none", edgecolor="black", linewidth=0.6)

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05,
                         shrink=0.5, ticks=[-1, 0, 1])
    cbar.ax.set_xticklabels(["Drying", "No trend", "Wetting"])
    cbar.set_label("SPEI-12 Monthly Trend (Mann-Kendall)", fontsize=11, fontweight="bold")

    ax.set_title(f"Grid-Point SPEI-12 Trend, {START_YEAR}-{END_YEAR}\n"
                 f"(Mann-Kendall test on monthly series, α={ALPHA})",
                 fontsize=13, fontweight="bold")

    save_path = f"{output_dir}/figures/SPEI-12_gridpoint_trend_map.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_total_trend_percentage_bar(trend_da, output_dir=OUTPUT_DIR):
    """
    Bar chart of the TOTAL percentage of analysed grid cells (all kept land
    cells with a valid Mann-Kendall result) that show significant wetting,
    no significant trend, and significant drying. Also saves a small CSV.
    """
    print("\nPlotting total grid-cell trend percentage bar chart...")

    vals = trend_da.values
    vals = vals[~np.isnan(vals)]
    total = len(vals)
    n_dry = int(np.sum(vals == -1))
    n_no = int(np.sum(vals == 0))
    n_wet = int(np.sum(vals == 1))
    pct = np.array([n_dry, n_no, n_wet]) / total * 100
    counts = [n_dry, n_no, n_wet]

    labels = ["Drying", "No trend", "Wetting"]
    colors = ["#B2182B", "#BDBDBD", "#2166AC"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, pct, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    for bar, p_, n_ in zip(bars, pct, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, p_ + 1,
                f"{p_:.1f}%\n(n={n_:,})", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylim(0, max(pct) * 1.2)
    ax.set_ylabel("Percentage of grid cells (%)", fontweight="bold")
    ax.set_title(f"SPEI-12 Trend: Total Grid-Cell Percentage\n"
                 f"(Mann-Kendall, \u03b1={ALPHA}; {START_YEAR}\u2013{END_YEAR}; "
                 f"N={total:,} cells)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    save_path = f"{output_dir}/figures/SPEI-12_total_trend_percentage_bar.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    pd.DataFrame({"category": labels, "n_grid_cells": counts, "percentage": pct}) \
      .to_csv(f"{output_dir}/csv/SPEI-12_total_trend_percentage.csv", index=False)
    print(f"  Drying {pct[0]:.1f}% | No trend {pct[1]:.1f}% | Wetting {pct[2]:.1f}%")


def plot_regional_trend_percentage_bar_by_continent(trend_pct_df, output_dir=OUTPUT_DIR):
    """
    Six stacked bar charts (one per continent: North America, South America,
    Europe, Africa, Asia, Oceania), each showing the AREA FRACTION (0-1,
    not percent) of grid cells per IPCC AR6 region falling into each trend
    category (wetting / no-trend / drying).
    """
    print("\nCreating 6 continent-grouped trend area-fraction bar charts...")

    df = trend_pct_df.dropna(subset=["pct_increasing"]).copy()

    df["frac_increasing"] = df["pct_increasing"] / 100.0
    df["frac_no_trend"] = df["pct_no_trend"] / 100.0
    df["frac_decreasing"] = df["pct_decreasing"] / 100.0

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    for ax, continent in zip(axes, CONTINENT_ORDER):
        sub = df[df["continent"] == continent].copy()
        sub = sub.sort_values("frac_decreasing", ascending=False).reset_index(drop=True)

        x = np.arange(len(sub))

        if len(sub) == 0:
            ax.set_title(f"{continent} (no regions)", fontsize=14, fontweight="bold")
            ax.axis("off")
            continue

        ax.bar(x, sub["frac_increasing"], color="#2166AC", edgecolor="black",
               linewidth=0.3, label="Significant increasing (wetting)")
        ax.bar(x, sub["frac_no_trend"], bottom=sub["frac_increasing"],
               color="#BABABA", edgecolor="black", linewidth=0.3, label="No significant trend")
        ax.bar(x, sub["frac_decreasing"],
               bottom=sub["frac_increasing"] + sub["frac_no_trend"],
               color="#B2182B", edgecolor="black", linewidth=0.3, label="Significant decreasing (drying)")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["region_abbrev"], rotation=90, fontsize=12)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Area Fraction", fontsize=22, fontweight="bold")
        ax.set_title(continent, fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.4)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize=22, frameon=False)

    fig.suptitle(f"Grid-Point SPEI-12 Trend by IPCC Region, Grouped by Continent\n"
                 f"(Mann-Kendall test, α={ALPHA}, {START_YEAR}–{END_YEAR})",
                 fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_path = f"{output_dir}/figures/SPEI-12_regional_trend_percentage_bar_by_continent.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_regional_trend_map(trend_pct_df, regions, trend_da, output_dir=OUTPUT_DIR):
    """
    Combines two layers on one map:
      1. Individual GRID-POINT trend markers (small dots), colored by each
         cell's own dry/wet/no-trend category, using a palette distinct
         from the boundary lines below (light orange / light blue / light
         grey) so the two layers stay visually separable.
      2. IPCC region polygons with NO fill -- each region's BOUNDARY LINE
         is colored (bold red/blue/black) by its dominant trend category
         (by grid-cell count, same logic as plot_regional_trend_map_extremes).
    Labels still show the exact wetting/drying percentages per region.
    """
    print("\nPlotting regional trend map (grid points + boundary lines colored by category)...")

    df = trend_pct_df.set_index("region_abbrev")

    boundary_colors = {
        "wet": "#2166AC",   # blue boundary = wetting dominant
        "none": "black",    # black boundary = no-trend dominant
        "dry": "#B2182B",   # red boundary = drying dominant
    }

    point_colors = {
        1: "#6BAED6",    # light blue  = wetting grid cell
        0: "#BDBDBD",    # light grey  = no-trend grid cell
        -1: "#FD8D3C",   # light orange = drying grid cell
    }

    fig = plt.figure(figsize=(13, 6.5))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)

    lon2d, lat2d = np.meshgrid(trend_da.lon.values, trend_da.lat.values)
    vals = trend_da.values
    for category, color in point_colors.items():
        sel = vals == category
        if not np.any(sel):
            continue
        ax.scatter(lon2d[sel], lat2d[sel], color=color, s=1.5, marker="s",
                   transform=ccrs.PlateCarree(), zorder=1, linewidths=0)

    for i, region_poly in enumerate(regions.polygons):
        abbrev = regions.abbrevs[i]

        if abbrev not in df.index or pd.isna(df.loc[abbrev, "n_wet"]):
            edge_color = "lightgray"
        else:
            row = df.loc[abbrev]
            counts = {"wet": row["n_wet"], "none": row["n_no_trend"], "dry": row["n_dry"]}
            dominant = max(counts, key=counts.get)
            edge_color = boundary_colors[dominant]

        ax.add_geometries([region_poly], crs=ccrs.PlateCarree(),
                           facecolor="none", edgecolor=edge_color, linewidth=2.2,
                           zorder=3)

        cx, cy = region_poly.centroid.x, region_poly.centroid.y
        if abbrev in df.index and not pd.isna(df.loc[abbrev, "pct_increasing"]):
            label = f"{abbrev}\n+{df.loc[abbrev,'pct_increasing']:.0f}/-{df.loc[abbrev,'pct_decreasing']:.0f}%"
        else:
            label = abbrev

        ax.text(cx, cy, label, transform=ccrs.PlateCarree(),
                fontsize=5.5, fontweight="bold", ha="center", va="center",
                color="black", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.75))

    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.4, zorder=2)

    boundary_legend = [
        Patch(facecolor="none", edgecolor=boundary_colors["wet"], linewidth=2.2,
              label="Boundary: wetting dominant"),
        Patch(facecolor="none", edgecolor=boundary_colors["none"], linewidth=2.2,
              label="Boundary: no-trend dominant"),
        Patch(facecolor="none", edgecolor=boundary_colors["dry"], linewidth=2.2,
              label="Boundary: drying dominant"),
    ]
    point_legend = [
        Patch(facecolor=point_colors[1], edgecolor="none", label="Grid cell: wetting"),
        Patch(facecolor=point_colors[0], edgecolor="none", label="Grid cell: no trend"),
        Patch(facecolor=point_colors[-1], edgecolor="none", label="Grid cell: drying"),
    ]
    ax.legend(handles=boundary_legend + point_legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=12, frameon=False)

    ax.set_title(f"Regional SPEI-12 Trend Summary by IPCC AR6 Region\n"
                 f"(Mann-Kendall test on monthly series, α={ALPHA}; "
                 f"{START_YEAR}–{END_YEAR})\n"
                 f"Dots = individual grid-cell trend; boundary color = dominant "
                 f"category by grid-cell count; labels show % wetting / % drying",
                 fontsize=12, fontweight="bold", pad=15)

    save_path = f"{output_dir}/figures/SPEI-12_regional_trend_map.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_regional_trend_map_extremes(trend_pct_df, regions, output_dir=OUTPUT_DIR):
    """
    IPCC region polygons colored by which trend category has the HIGHEST
    NUMBER OF GRID CELLS within that region (not percentage -- raw counts):
        RED   = drying trend has the most grid cells in that region
        BLUE  = wetting trend has the most grid cells in that region
        BLACK = no-trend has the most grid cells in that region
    """
    print("\nPlotting regional trend map (red=dry / blue=wet / black=no-trend, by grid-cell count)...")

    df = trend_pct_df.set_index("region_abbrev")

    category_colors = {
        "wet": "#2166AC",
        "none": "grey",
        "dry": "#B2182B",
    }

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    for i, region_poly in enumerate(regions.polygons):
        abbrev = regions.abbrevs[i]

        if abbrev not in df.index or pd.isna(df.loc[abbrev, "n_wet"]):
            color = "white"
            label = abbrev
        else:
            row = df.loc[abbrev]
            counts = {"wet": row["n_wet"], "none": row["n_no_trend"], "dry": row["n_dry"]}
            dominant = max(counts, key=counts.get)
            color = category_colors[dominant]
            label = abbrev

        ax.add_geometries([region_poly], crs=ccrs.PlateCarree(),
                           facecolor=color, edgecolor="black", linewidth=0.8)

        cx, cy = region_poly.centroid.x, region_poly.centroid.y
        ax.text(cx, cy, label, transform=ccrs.PlateCarree(),
                fontsize=11, fontweight="bold", ha="center", va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.75))

    ax.coastlines(linewidth=0.8, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.spines["geo"].set_edgecolor("black")
    ax.spines["geo"].set_linewidth(1.2)

    ax.set_title("IPCC Region Colored by Dominant SPEI-12 Trend Category\n"
                 "(by number of grid cells, not percentage)\n"
                 f"(Mann-Kendall test on monthly series, α={ALPHA}; {START_YEAR}–{END_YEAR})",
                 fontsize=20, fontweight="bold", pad=20)

    legend_elements = [
        Patch(facecolor=category_colors["wet"], edgecolor="black",
              label="Highest count: wetting trend grid cells"),
        Patch(facecolor=category_colors["none"], edgecolor="black",
              label="Highest count: no-trend grid cells"),
        Patch(facecolor=category_colors["dry"], edgecolor="black",
              label="Highest count: drying trend grid cells"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               bbox_to_anchor=(0.5, -0.2), ncol=3,
               fontsize=18, frameon=False,
               handlelength=2.5, handleheight=1.8,
               columnspacing=2.0)

    plt.tight_layout()

    save_path = f"{output_dir}/figures/SPEI-12_regional_trend_map_extremes.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":

    print("=" * 80)
    print("GRID-POINT SPEI-12 MANN-KENDALL TREND ANALYSIS — IPCC AR6 LAND REGIONS")
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Excluded regions: {EXCLUDED_REGIONS}")
    print("=" * 80)

    # Step 1: Load precipitation + PET, compute water balance
    precip = load_cru_precipitation(CRU_PRE_FILE, START_YEAR, END_YEAR)
    pet = load_cru_pet(CRU_PET_FILE, START_YEAR, END_YEAR)
    water_balance = compute_water_balance(precip, pet)

    # Step 2: IPCC regions (40 regions after exclusion) + land mask
    ipcc_regions = get_ipcc_regions(excluded_regions=EXCLUDED_REGIONS)
    land_mask, mask_3D = get_land_mask(water_balance, ipcc_regions)

    # Step 3: Grid-point SPEI-12
    spei_gridded = compute_spei_gridded(water_balance, accum_months=ACCUM_MONTHS)

    # Step 4: Grid-point Mann-Kendall trend (monthly series)
    print("\n" + "=" * 80)
    print("MANN-KENDALL TREND TEST")
    print("=" * 80)
    trend_da, p_da, tau_da = compute_grid_trend_monthly(spei_gridded, land_mask, alpha=ALPHA)
    save_trend_to_csv(trend_da, p_da, tau_da)

    # Step 5: Regional trend percentages
    print("\n" + "=" * 80)
    print("REGIONAL TREND PERCENTAGES")
    print("=" * 80)
    trend_pct_df = compute_regional_trend_percentage(trend_da, ipcc_regions, mask_3D)

    # Step 6: Figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    plot_trend_map(trend_da, ipcc_regions)
    plot_total_trend_percentage_bar(trend_da)
    plot_regional_trend_percentage_bar_by_continent(trend_pct_df)
    plot_regional_trend_map(trend_pct_df, ipcc_regions, trend_da)
    plot_regional_trend_map_extremes(trend_pct_df, ipcc_regions)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("  csv/SPEI-12_gridpoint_trend.csv                              <- every grid cell")
    print("  csv/SPEI-12_regional_trend_percentages.csv                   <- % + counts per region")
    print("  figures/SPEI-12_gridpoint_trend_map.png                      <- grid-resolution map")
    print("  figures/SPEI-12_total_trend_percentage_bar.png               <- total % drying / no trend / wetting")
    print("  figures/SPEI-12_regional_trend_percentage_bar_by_continent.png <- 6 bar charts (area fraction 0-1)")
    print("  figures/SPEI-12_regional_trend_map.png                       <- grid points + boundary lines")
    print("  figures/SPEI-12_regional_trend_map_extremes.png              <- red/blue/black by grid-cell count")
    print("=" * 80)
