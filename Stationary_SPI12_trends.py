"""
Grid-Point SPI-12 Mann-Kendall Trend Analysis for IPCC AR6 Land Regions
============================================================================
- SPI-12 computed at EVERY grid cell (Thom's gamma estimator, vectorized)
- Excludes: EAN, WAN, GIC, SAH, ARP, ECA (analysis run on remaining regions)
- Grid-point Mann-Kendall trend test on the MONTHLY SPI-12 series
- Regional trend percentages (% wetting / % no-trend / % drying per region)
- Figures:
    1. Grid-point trend map (red=drying, blue=wetting, grey=no trend) + IPCC boundaries
    2. Regional trend percentage stacked bar chart
    3. Regional trend map (continuous net-trend choropleth)
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
from scipy.stats import gamma, norm, kendalltau
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
CRU_FILE = "/home/Data/CRU/ts4.10/cru_ts4.10.1901.2025.pre.dat.nc"
OUTPUT_DIR = "./SPI-12_trend_results"
ACCUM_MONTHS = 12
START_YEAR = 1901
END_YEAR = 2025

# Regions to EXCLUDE (analysis runs on the remaining 40 IPCC AR6 land regions)
EXCLUDED_REGIONS = ["EAN", "WAN", "GIC", "SAH", "ARP", "ECA"]

ALPHA = 0.05                # significance level for Mann-Kendall test
MIN_VALID_YEARS = 10        # minimum valid years to fit gamma at a grid cell/month
MIN_TREND_MONTHS = 120      # minimum valid monthly points to run MK test at a cell

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/csv", exist_ok=True)


# ============================================================================
# STEP 1: LOAD CRU PRECIPITATION DATA
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


def get_land_mask(pre, regions):
    """
    Returns:
      land_mask - boolean (lat, lon), True where grid cell is in ANY kept region
      mask_3D   - boolean (region, lat, lon), per-region masks
    """
    print("\nBuilding land mask from kept IPCC regions...")
    mask_3D = regions.mask_3D(pre.lon, pre.lat)
    land_mask = mask_3D.any(dim="region")
    n_land = int(land_mask.sum().values)
    n_total = land_mask.size
    print(f"  Land grid cells in kept regions: {n_land} / {n_total} ({n_land/n_total*100:.1f}%)")
    return land_mask, mask_3D


# ============================================================================
# STEP 3: GRID-POINT SPI-12 (VECTORIZED, THOM'S ESTIMATOR)
# ============================================================================
def thom_gamma_params(values_2d):
    """
    Thom (1958) closed-form gamma MLE approximation, vectorized across
    grid cells. Standard method used in operational SPI software.

    Parameters
    ----------
    values_2d : np.ndarray, shape (n_years, n_grid)
        Precipitation accumulation totals for ONE calendar month,
        across all years and grid cells.

    Returns
    -------
    shape, scale, zero_prob : np.ndarray, shape (n_grid,)
    """
    n_grid = values_2d.shape[1]
    shape_arr = np.full(n_grid, np.nan)
    scale_arr = np.full(n_grid, np.nan)
    zero_prob_arr = np.full(n_grid, np.nan)

    valid_mask = ~np.isnan(values_2d)
    n_valid = valid_mask.sum(axis=0)
    good_cells = n_valid >= MIN_VALID_YEARS
    if not np.any(good_cells):
        return shape_arr, scale_arr, zero_prob_arr

    vals = values_2d[:, good_cells]
    valid = valid_mask[:, good_cells]
    n_valid_good = valid.sum(axis=0).astype(float)

    zero_count = np.nansum((vals == 0) & valid, axis=0)
    zero_prob = zero_count / n_valid_good

    vals_nonzero = np.where((vals > 0) & valid, vals, np.nan)
    n_nonzero = np.sum(~np.isnan(vals_nonzero), axis=0).astype(float)
    enough_nonzero = n_nonzero >= 5

    mean_x = np.nanmean(vals_nonzero, axis=0)
    mean_lnx = np.nanmean(np.log(vals_nonzero, where=~np.isnan(vals_nonzero)), axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.log(mean_x) - mean_lnx
        shape = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A)
        scale = mean_x / shape

    valid_fit = enough_nonzero & (A > 0) & np.isfinite(shape) & np.isfinite(scale)

    shape_out = np.full(good_cells.sum(), np.nan)
    scale_out = np.full(good_cells.sum(), np.nan)
    shape_out[valid_fit] = shape[valid_fit]
    scale_out[valid_fit] = scale[valid_fit]

    shape_arr[good_cells] = shape_out
    scale_arr[good_cells] = scale_out
    zero_prob_arr[good_cells] = zero_prob

    return shape_arr, scale_arr, zero_prob_arr


def compute_spi_gridded(precip, accum_months=ACCUM_MONTHS):
    """Compute SPI at EVERY grid cell (vectorized across space, looped over 12 calendar months)."""
    print(f"\nComputing grid-point SPI-{accum_months}...")

    accum = precip.rolling(time=accum_months, min_periods=accum_months).sum(skipna=True)

    n_time, n_lat, n_lon = accum.shape
    n_grid = n_lat * n_lon

    accum_flat = accum.values.reshape(n_time, n_grid)
    spi_flat = np.full_like(accum_flat, np.nan)

    months = accum["time"].dt.month.values

    for m in range(1, 13):
        month_time_idx = np.where(months == m)[0]
        vals = accum_flat[month_time_idx, :]

        shape_arr, scale_arr, zero_prob_arr = thom_gamma_params(vals)

        with np.errstate(invalid="ignore"):
            cdf = zero_prob_arr[None, :] + (1 - zero_prob_arr[None, :]) * gamma.cdf(
                vals, shape_arr[None, :], loc=0, scale=scale_arr[None, :]
            )
            is_zero = (vals == 0)
            cdf = np.where(is_zero, zero_prob_arr[None, :], cdf)

        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spi_flat[month_time_idx, :] = norm.ppf(cdf)

        n_fit = np.sum(~np.isnan(shape_arr))
        print(f"  Month {m:2d}: fitted gamma at {n_fit}/{n_grid} grid cells")

    spi_array = spi_flat.reshape(n_time, n_lat, n_lon)

    spi_da = xr.DataArray(
        spi_array,
        coords={"time": accum.time, "lat": accum.lat, "lon": accum.lon},
        dims=["time", "lat", "lon"],
        name=f"SPI{accum_months}"
    )
    print(f"  Grid-point SPI-{accum_months} computation complete.")
    return spi_da


# ============================================================================
# STEP 4: GRID-POINT MANN-KENDALL TREND TEST (MONTHLY SERIES)
# ============================================================================
def compute_grid_trend_monthly(spi_gridded, land_mask, alpha=ALPHA):
    """
    Mann-Kendall trend test at EVERY grid cell, computed on the full
    MONTHLY SPI-12 series (not annual mean).

    Returns
    -------
    trend_da : xr.DataArray (lat, lon)
        1  = significant increasing (wetting)
       -1  = significant decreasing (drying)
        0  = no significant trend
       NaN = insufficient data / not in a kept region
    p_da, tau_da : xr.DataArray (lat, lon)
    """
    print("\nComputing grid-point Mann-Kendall trend on MONTHLY SPI-12 series...")

    data = spi_gridded.values  # (n_time, n_lat, n_lon)
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

    coords = {"lat": spi_gridded.lat, "lon": spi_gridded.lon}
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

    path = f"{output_dir}/csv/SPI-12_gridpoint_trend.csv"
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
        else:
            pct_inc = np.sum(cell_trend == 1) / total * 100
            pct_no = np.sum(cell_trend == 0) / total * 100
            pct_dec = np.sum(cell_trend == -1) / total * 100

        results.append({
            "region_abbrev": abbrev,
            "region_name": regions.names[r],
            "n_grid_cells": total,
            "pct_increasing": pct_inc,
            "pct_no_trend": pct_no,
            "pct_decreasing": pct_dec
        })
        print(f"  {abbrev:6s}: n={total:5d}  wetting={pct_inc:5.1f}%  "
              f"no-trend={pct_no:5.1f}%  drying={pct_dec:5.1f}%")

    df = pd.DataFrame(results)
    path = f"{OUTPUT_DIR}/csv/SPI-12_regional_trend_percentages.csv"
    df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")

    return df


# ============================================================================
# STEP 6: FIGURES
# ============================================================================
def plot_trend_map(trend_da, regions, output_dir=OUTPUT_DIR):
    """
    Global grid-resolution map of Mann-Kendall trend category:
        RED  = significant drying (-1)
        BLUE = significant wetting (+1)
        GREY = no significant trend (0)
    IPCC AR6 region boundaries overlaid.
    """
    print("\nPlotting grid-point trend map...")

    colors = ["#B2182B", "#BABABA", "#2166AC"]   # matches categories -1, 0, 1
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm_c = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(13, 6.5))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    ax.pcolormesh(
        trend_da.lon, trend_da.lat, trend_da.values,
        cmap=cmap, norm=norm_c,
        transform=ccrs.PlateCarree(), shading="auto"
    )

    for region_poly in regions.polygons:
        ax.add_geometries(
            [region_poly], crs=ccrs.PlateCarree(),
            facecolor="none", edgecolor="black", linewidth=0.7
        )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.4)

    legend_elements = [
        Patch(facecolor="#2166AC", edgecolor="black", label="Significant wetting trend"),
        Patch(facecolor="#BABABA", edgecolor="black", label="No significant trend"),
        Patch(facecolor="#B2182B", edgecolor="black", label="Significant drying trend"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, frameon=False)

    ax.set_title(f"Grid-Point SPI-12 Mann-Kendall Trend ({START_YEAR}–{END_YEAR})\n"
                 f"(Monthly series, α={ALPHA}; IPCC AR6 region boundaries overlaid)",
                 fontsize=13, fontweight="bold", pad=15)

    save_path = f"{output_dir}/figures/SPI-12_gridpoint_trend_map.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_regional_trend_percentage_bar(trend_pct_df, output_dir=OUTPUT_DIR):
    """Stacked bar: % of grid cells per region in each trend category."""
    print("\nCreating per-region trend percentage stacked bar chart...")

    df = trend_pct_df.dropna(subset=["pct_increasing"]).copy()
    df = df.sort_values("pct_decreasing", ascending=False).reset_index(drop=True)

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.35), 6))

    ax.bar(x, df["pct_increasing"], color="#2166AC", edgecolor="black",
           linewidth=0.3, label="Significant increasing (wetting)")
    ax.bar(x, df["pct_no_trend"], bottom=df["pct_increasing"],
           color="#BABABA", edgecolor="black", linewidth=0.3, label="No significant trend")
    ax.bar(x, df["pct_decreasing"], bottom=df["pct_increasing"] + df["pct_no_trend"],
           color="#B2182B", edgecolor="black", linewidth=0.3, label="Significant decreasing (drying)")

    ax.set_xticks(x)
    ax.set_xticklabels(df["region_abbrev"], rotation=90, fontsize=12)
   # ax.set_yticklabels( fontsize=16)
    ax.set_ylabel("Area Fraction", fontsize=18, fontweight="bold")
    ax.set_xlabel("IPCC AR6 Region", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title(f"Grid-Point SPI-12 Trend by IPCC Region (Monthly Series)\n"
                 f"(Mann-Kendall test, α={ALPHA}, {START_YEAR}–{END_YEAR})",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_path = f"{output_dir}/figures/SPI-12_regional_trend_percentage_bar.png"
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
    Labels show only the region abbreviation.
    """
    print("\nPlotting regional trend map (grid points + boundary lines colored by category)...")

    df = trend_pct_df.set_index("region_abbrev")

    boundary_colors = {
        "wet": "#2166AC",    # blue boundary = wetting dominant
        "none": "black",    # black boundary = no-trend dominant
        "dry": "#B2182B",    # red boundary = drying dominant
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
        # Display only the abbreviation without numerical percentage values
        label = abbrev

        ax.text(cx, cy, label, transform=ccrs.PlateCarree(),
                fontsize=6.5, fontweight="bold", ha="center", va="center",
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
              bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8.5, frameon=False)

    ax.set_title(f"Regional Moving-Window SPI-{ACCUM_MONTHS} Trend Summary by IPCC AR6 Region\n"
                 f"({WINDOW_YEARS}-yr trailing climatology; Mann-Kendall on monthly series, α={ALPHA})\n"
                 f"Dots = individual grid-cell trend; boundary color = dominant category by grid-cell count",
                 fontsize=12, fontweight="bold", pad=15)

    save_path = f"{output_dir}/figures/SPI-{ACCUM_MONTHS}_movingwindow_regional_trend_map.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()

def plot_regional_trend_map_dominant_category(trend_pct_df, regions, output_dir=OUTPUT_DIR):
    """
    IPCC region polygons colored by DOMINANT trend category (whichever of
    wetting/no-trend/drying covers the most grid cells within that region).
    Styled to match the duration/intensity map format: solid fills, bold
    white-boxed labels, clean discrete legend bar at the bottom.
    """
    print("\nPlotting regional trend map (dominant category style)...")

    df = trend_pct_df.set_index("region_abbrev")

    category_colors = {
        "wet": "#2166AC",    # blue = wetting
        "none": "#BABABA",   # grey = no trend
        "dry": "#B2182B",    # red = drying
    }

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    for i, region_poly in enumerate(regions.polygons):
        abbrev = regions.abbrevs[i]

        if abbrev not in df.index or pd.isna(df.loc[abbrev, "pct_increasing"]):
            color = "white"
            label = abbrev
        else:
            row = df.loc[abbrev]
            cats = {"wet": row["pct_increasing"],
                    "none": row["pct_no_trend"],
                    "dry": row["pct_decreasing"]}
            dominant = max(cats, key=cats.get)
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

    ax.set_title("Dominant SPI-12 Trend by IPCC Region\n"
                 f"(Mann-Kendall test on monthly series, α={ALPHA}; {START_YEAR}–{END_YEAR})",
                 fontsize=22, fontweight="bold", pad=20)

    # --- Discrete legend bar at the bottom, styled like a colorbar ---
    legend_elements = [
        Patch(facecolor=category_colors["wet"], edgecolor="black",
              label="Increasing (wetting) trend"),
        Patch(facecolor=category_colors["none"], edgecolor="black",
              label="No significant trend"),
        Patch(facecolor=category_colors["dry"], edgecolor="black",
              label="Decreasing (drying) trend"),
    ]
    legend = fig.legend(handles=legend_elements, loc="lower center",
                         bbox_to_anchor=(0.5, -0.02), ncol=3,
                         fontsize=14, frameon=False,
                         handlelength=2.5, handleheight=1.8,
                         columnspacing=2.0)

    plt.tight_layout()

    save_path = f"{output_dir}/figures/SPI-12_regional_trend_map_dominant.png"
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":

    print("=" * 80)
    print("GRID-POINT SPI-12 MANN-KENDALL TREND ANALYSIS — IPCC AR6 LAND REGIONS")
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Excluded regions: {EXCLUDED_REGIONS}")
    print("=" * 80)

    # Step 1: Load precipitation
    precip = load_cru_precipitation(CRU_FILE, START_YEAR, END_YEAR)

    # Step 2: IPCC regions (40 regions after exclusion) + land mask
    ipcc_regions = get_ipcc_regions(excluded_regions=EXCLUDED_REGIONS)
    land_mask, mask_3D = get_land_mask(precip, ipcc_regions)

    # Step 3: Grid-point SPI-12
    spi_gridded = compute_spi_gridded(precip, accum_months=ACCUM_MONTHS)

    # Step 4: Grid-point Mann-Kendall trend (monthly series)
    print("\n" + "=" * 80)
    print("MANN-KENDALL TREND TEST")
    print("=" * 80)
    trend_da, p_da, tau_da = compute_grid_trend_monthly(spi_gridded, land_mask, alpha=ALPHA)
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
    plot_regional_trend_percentage_bar(trend_pct_df)
    plot_regional_trend_map(trend_pct_df, ipcc_regions)
    plot_regional_trend_map_dominant_category(trend_pct_df, ipcc_regions)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("  csv/SPI-12_gridpoint_trend.csv                       <- every grid cell")
    print("  csv/SPI-12_regional_trend_percentages.csv            <- % per region")
    print("  figures/SPI-12_gridpoint_trend_map.png               <- grid-resolution map")
    print("  figures/SPI-12_regional_trend_percentage_bar.png     <- bar chart per region")
    print("  figures/SPI-12_regional_trend_map.png                <- net-trend choropleth")
    print("  figures/SPI-12_regional_trend_map_dominant.png       <- dominant-category choropleth")
    print("=" * 80)
