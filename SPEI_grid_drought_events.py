"""
SPEI_grid_drought_events.py
============================================================================
Grid-cell drought event extraction (run theory) from stationary and
non-stationary SPEI-12, saved as separate NetCDF files for Duration and
Intensity, plus publication-quality figures.

Event definition (applied independently at every grid cell):
  1. Onset: first month with SPEI <= -0.5.
  2. The event continues until SPEI >= +0.5 (recovery). The recovery month
     itself is NOT part of the event.
  3. Events shorter than 3 months are discarded.
  4. Intensity = minimum (most negative) SPEI during the event.
  5. Duration  = number of months in the event (onset month to last month
     before recovery). Set DURATION_MODE = "below_threshold" to instead count
     only months with SPEI <= -0.5 (as in the reference pipeline).
  6. A NaN month closes an open event. An event still open at the end of the
     record is kept if it satisfies the minimum duration.

Outputs (OUTPUT_DIR):
  nc/Drought_Duration_<tag>.nc, nc/Drought_Intensity_<tag>.nc  (per dataset)
  figures/Fig_Duration_<tag>.png/.pdf and Fig_Intensity_<tag>.png/.pdf (600 dpi)

Each NC file contains:
  event variable (event, lat, lon)   -> duration (months) or intensity (SPEI)
  onset_yyyymm (event, lat, lon)     -> onset date of each event
  n_events, mean_<var>, max_<var>    -> (lat, lon) summaries

Requirements: pip install xarray numpy netCDF4 matplotlib cartopy regionmask numba(optional)
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import matplotlib.colors as mcolors

try:
    from numba import njit
except ImportError:                       # slower, but works without numba
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f

# ============================================================================
# PARAMETERS -- EDIT THESE
# ============================================================================
IN_DIR = "/home/murtaza13/paper2/Grid_based_analysis/Trends/SPEI/data"
FILES = {
    "Stationary": "SPEI12_stationary.nc",
    "Non-stationary": "SPEI12_nonstationary_30yr.nc",
}
VAR_NAME = None            # None = auto-detect the SPEI variable
OUTPUT_DIR = "./SPEI12_grid_drought_events"

ONSET = -0.5
RECOVERY = 0.5
MIN_DURATION = 3
DURATION_MODE = "span"     # "span" or "below_threshold"
FILL_INT = -9999


# ============================================================================
# CORE: EVENT EXTRACTION FOR ONE TIME SERIES
# ============================================================================
@njit(cache=True)
def extract_events(x, onset, recov, min_dur, span_mode):
    n = x.shape[0]
    cap = n // 3 + 1
    start = np.empty(cap, np.int64)
    dur = np.empty(cap, np.int64)
    inten = np.empty(cap, np.float32)
    k = 0
    inev = False
    s = 0
    mn = 0.0
    nb = 0
    for i in range(n):
        v = x[i]
        if inev:
            if np.isnan(v) or v >= recov:
                e = i - 1
                d = (e - s + 1) if span_mode else nb
                if (e - s + 1) >= min_dur and d >= 1:
                    start[k] = s
                    dur[k] = d
                    inten[k] = mn
                    k += 1
                inev = False
            else:
                if v < mn:
                    mn = v
                if v <= onset:
                    nb += 1
        if (not inev) and (not np.isnan(v)) and v <= onset:
            inev = True
            s = i
            mn = v
            nb = 1
    if inev:                               # event still open at end of record
        e = n - 1
        d = (e - s + 1) if span_mode else nb
        if (e - s + 1) >= min_dur and d >= 1:
            start[k] = s
            dur[k] = d
            inten[k] = mn
            k += 1
    return k, start[:k], dur[:k], inten[:k]


# ============================================================================
# GRID PROCESSING
# ============================================================================
def load_spei(path):
    ds = xr.open_dataset(path)
    name = VAR_NAME or [v for v in ds.data_vars if ds[v].ndim == 3][0]
    da = ds[name]
    ren = {}
    for d in da.dims:
        if d.lower() in ("latitude", "y"):
            ren[d] = "lat"
        elif d.lower() in ("longitude", "x"):
            ren[d] = "lon"
    da = da.rename(ren).transpose("time", "lat", "lon")
    print(f"  variable '{name}', shape {da.shape}, "
          f"{str(da.time.values[0])[:7]} to {str(da.time.values[-1])[:7]}")
    return da


def grid_events(da):
    data = da.values.astype(np.float32)
    nt, ny, nx = data.shape
    yyyymm = (da.time.dt.year.values * 100 + da.time.dt.month.values).astype(np.int32)
    valid = np.isfinite(data).any(axis=0)
    cells = np.argwhere(valid)
    print(f"  valid grid cells: {len(cells)}")

    results, counts = {}, np.zeros((ny, nx), np.int16)
    for (j, i) in cells:
        k, st, du, it = extract_events(data[:, j, i], ONSET, RECOVERY,
                                       MIN_DURATION, DURATION_MODE == "span")
        counts[j, i] = k
        if k:
            results[(j, i)] = (st, du, it)

    nev = int(counts.max())
    print(f"  max events in one cell: {nev}")
    dur = np.full((nev, ny, nx), FILL_INT, np.int16)
    inten = np.full((nev, ny, nx), np.nan, np.float32)
    onset = np.full((nev, ny, nx), FILL_INT, np.int32)
    for (j, i), (st, du, it) in results.items():
        k = len(st)
        dur[:k, j, i] = du
        inten[:k, j, i] = it
        onset[:k, j, i] = yyyymm[st]
    return dur, inten, onset, counts, valid


def save_nc(tag, dur, inten, onset, counts, valid, da):
    os.makedirs(f"{OUTPUT_DIR}/nc", exist_ok=True)
    coords = {"event": np.arange(1, dur.shape[0] + 1), "lat": da.lat, "lon": da.lon}
    dims3, dims2 = ("event", "lat", "lon"), ("lat", "lon")
    nanmask = ~valid

    d_f = np.where(dur == FILL_INT, np.nan, dur).astype(np.float32)
    with np.errstate(all="ignore"):
        d_mean = np.nanmean(d_f, axis=0)
        d_max = np.nanmax(d_f, axis=0)
        i_mean = np.nanmean(inten, axis=0)
        i_min = np.nanmin(inten, axis=0)
    for a in (d_mean, d_max, i_mean, i_min):
        a[nanmask] = np.nan
    cnt = np.where(valid, counts, FILL_INT).astype(np.int16)

    def enc(**extra):
        return {"zlib": True, "complevel": 4, **extra}

    common = dict(
        onset_yyyymm=(dims3, onset, {"long_name": "Event onset (YYYYMM)"}),
        n_events=(dims2, cnt, {"long_name": "Number of drought events"}),
    )
    ds_d = xr.Dataset(
        dict(duration=(dims3, dur, {"long_name": "Drought duration", "units": "months"}),
             mean_duration=(dims2, d_mean.astype(np.float32), {"units": "months"}),
             max_duration=(dims2, d_max.astype(np.float32), {"units": "months"}),
             **common),
        coords=coords,
        attrs={"description": f"SPEI-12 drought events ({tag}); onset<={ONSET}, "
                              f"recovery>={RECOVERY}, min duration {MIN_DURATION} months, "
                              f"duration mode: {DURATION_MODE}"})
    ds_i = xr.Dataset(
        dict(intensity=(dims3, inten, {"long_name": "Drought intensity (minimum SPEI-12)",
                                       "units": "SPEI"}),
             mean_intensity=(dims2, i_mean.astype(np.float32), {"units": "SPEI"}),
             peak_intensity=(dims2, i_min.astype(np.float32), {"units": "SPEI"}),
             **common),
        coords=coords, attrs=ds_d.attrs)

    e_int = dict(_FillValue=FILL_INT, **{"zlib": True, "complevel": 4})
    ds_d.to_netcdf(f"{OUTPUT_DIR}/nc/Drought_Duration_{tag}.nc", encoding={
        "duration": enc(_FillValue=FILL_INT), "onset_yyyymm": enc(_FillValue=FILL_INT),
        "n_events": enc(_FillValue=FILL_INT)})
    ds_i.to_netcdf(f"{OUTPUT_DIR}/nc/Drought_Intensity_{tag}.nc", encoding={
        "intensity": enc(_FillValue=np.nan), "onset_yyyymm": enc(_FillValue=FILL_INT),
        "n_events": enc(_FillValue=FILL_INT)})
    print(f"  saved NC files for {tag}")
    return ds_d, ds_i


# ============================================================================
# FIGURES  (one clean map per metric per dataset)
# ============================================================================
EXCLUDED_REGIONS = ["EAN", "WAN", "GIC", "SAH", "ARP", "ECA"]   # same as reference
SHOW_GRID_LABELS = False        # True = add lat/lon tick labels (needs cartopy >= 0.20)
AXIS_FONT = 14                  # size of lat/lon labels if SHOW_GRID_LABELS = True

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def get_region_geoms():
    """IPCC AR6 land polygons: (kept, excluded). No abbreviations are drawn."""
    allr = regionmask.defined_regions.ar6.land
    kept, excl = [], []
    for ab, poly in zip(allr.abbrevs, allr.polygons):
        (excl if ab in EXCLUDED_REGIONS else kept).append(poly)
    return kept, excl


def make_map(da, title, cbar_label, levels, cmap_name, extend, fname, panel=None):
    kept, excl = get_region_geoms()
    n_bins = len(levels) - 1 + {"neither": 0, "min": 1, "max": 1, "both": 2}[extend]
    cmap = plt.get_cmap(cmap_name, n_bins)
    norm = mcolors.BoundaryNorm(levels, cmap.N, extend=extend)

    fig = plt.figure(figsize=(11, 6.6), facecolor="white")
    ax = fig.add_axes([0.03, 0.17, 0.94, 0.74], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor("white")

    im = ax.pcolormesh(da.lon, da.lat, np.ma.masked_invalid(da.values), cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto", rasterized=True, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6,
                   edgecolor="0.35", zorder=2)
    # excluded regions: light grey outline; analysed regions: clear black outline
    ax.add_geometries(excl, ccrs.PlateCarree(), facecolor="none",
                      edgecolor="0.65", linewidth=0.7, zorder=3)
    ax.add_geometries(kept, ccrs.PlateCarree(), facecolor="none",
                      edgecolor="black", linewidth=1.1, zorder=4)
    ax.spines["geo"].set_linewidth(1.2)

    if SHOW_GRID_LABELS:
        try:
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.6,
                              linestyle="--")
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = {"size": AXIS_FONT, "weight": "bold"}
            gl.ylabel_style = {"size": AXIS_FONT, "weight": "bold"}
        except Exception as e:
            print("  (grid labels not supported by this cartopy version:", e, ")")

    ax.set_title(title, fontsize=22, fontweight="bold", pad=12)
    if panel:
        fig.text(0.06, 0.93, panel, fontsize=24, fontweight="bold", ha="left", va="center")

    cax = fig.add_axes([0.22, 0.09, 0.56, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend, ticks=levels)
    cb.set_label(cbar_label, fontsize=16, fontweight="bold", labelpad=6)
    cb.ax.tick_params(labelsize=14, width=1.3, length=6)
    plt.setp(cb.ax.get_xticklabels(), fontweight="bold")
    cb.outline.set_linewidth(1.2)

    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUTPUT_DIR}/figures/{fname}.{ext}", dpi=600, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  saved figure: {fname}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    out = {}
    for tag, fn in FILES.items():
        print(f"\n=== {tag} ===")
        da = load_spei(os.path.join(IN_DIR, fn))
        dur, inten, onset, counts, valid = grid_events(da)
        tag_s = tag.replace("-", "").lower()
        out[tag] = save_nc(tag_s, dur, inten, onset, counts, valid, da)

    print("\nPlotting...")
    # Each grid cell can have several events; a map needs one value per cell.
    # "max"  -> longest event (duration) / strongest event (peak intensity)
    # "mean" -> average over all events in the cell
    PLOT_STAT = "max"
    if PLOT_STAT == "max":
        dur_var, int_var = "max_duration", "peak_intensity"
        dur_lab = "Drought Duration (months)"
        int_lab = "Drought intensity"
        dur_ttl, int_ttl = "Drought Duration", "Drought Intensity"
        dur_levels = np.array([10, 40, 70, 100, 130, 160, 190,])
        int_levels = np.round(np.arange(-4.0, -0.5, 0.5), 2)
    else:
        dur_var, int_var = "mean_duration", "mean_intensity"
        dur_lab = "Mean drought duration (months)"
        int_lab = "Mean drought intensity (minimum SPEI-12)"
        dur_ttl, int_ttl = "Drought Duration", "Drought Intensity"
        dur_levels = np.arange(3, 25, 3)
        int_levels = np.round(np.arange(-3.0, -0.5, 0.25), 2)

    for tag, (ds_d, ds_i) in out.items():
        tag_s = tag.replace("-", "").lower()
        make_map(ds_d[dur_var], f"{tag} SPEI-12: {dur_ttl}", dur_lab,
                 dur_levels, "YlOrRd", "max", f"Fig_Duration_{tag_s}")
        make_map(ds_i[int_var], f"{tag} SPEI-12: {int_ttl}", int_lab,
                 int_levels, "YlOrRd_r", "min", f"Fig_Intensity_{tag_s}")
    print("\nDone. Results in:", OUTPUT_DIR)
