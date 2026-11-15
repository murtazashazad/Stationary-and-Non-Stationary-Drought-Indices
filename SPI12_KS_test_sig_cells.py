"""
SPI12_KS_test_sig_cells.py
============================================================================
Grid-cell two-sample Kolmogorov-Smirnov test between the drought events of
STATIONARY and NON-STATIONARY SPI-12. DURATION and INTENSITY are analysed
completely separately (own significant cells, own files, own return periods).

  1. Drought events are extracted at every grid cell for both datasets with the
     same rules as SPI_grid_drought_events.py (onset <= -0.5, recovery >= +0.5,
     min 3 months).
  2. At each cell ks_2samp compares the per-event values of the two datasets,
     once for duration and once for intensity. p < ALPHA = significantly different.
  3. Outputs (OUTPUT_DIR), <var> = duration / intensity:
       nc/KS_test_results.nc                      KS statistic, p, flags (all cells)
       nc/Significant_cells_Duration.nc           events of duration-significant cells
       nc/Significant_cells_Intensity.nc          events of intensity-significant cells
       csv/KS_test_results_all_cells.csv
       csv/KS_significant_cells_<var>.csv
       csv/events_stationary_sig_cells_<var>.csv
       csv/events_nonstationary_sig_cells_<var>.csv   (input of the return-period code)
       figures/KS_map_<var>.png/.pdf              red = significant, blue = not significant
  4. SPI12_return_period_sig_cells.py is run at the end: the duration return
     periods use only duration-significant cells, the intensity return periods
     only intensity-significant cells.

Requires in the same folder: SPI_grid_drought_events.py,
                             SPI12_return_period_sig_cells.py
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import SPI_grid_drought_events as G      # event extraction, loaders, region polygons

# ============================================================================
# PARAMETERS -- EDIT THESE
# ============================================================================
OUTPUT_DIR = "./SPI-12_KS_significant_cells"   # keep equal to cfg.BASE in the return-period script
START_YEAR, END_YEAR = 1930, 2025               # common analysis period
ALPHA = 0.05
MIN_EVENTS = 3          # min events per dataset in a cell to run the KS test
APPLY_FDR = False       # True = Benjamini-Hochberg correction across cells (multiple testing)
RUN_RETURN_PERIOD = True

COL_SIG, COL_NS, COL_NT = "#d7191c", "#2c7bb6", "#d9d9d9"   # red, blue, grey


# ============================================================================
# EVENTS + KS TEST
# ============================================================================
def events_for(data, cells):
    span = G.DURATION_MODE == "span"
    out = {}
    for (j, i) in cells:
        _, st, du, it = G.extract_events(data[:, j, i], G.ONSET, G.RECOVERY,
                                         G.MIN_DURATION, span)
        out[(j, i)] = (st, du, it)
    return out


def fdr(p):
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok], method="bh")
    return out


def run_ks(ev_s, ev_n, cells, shape):
    ny, nx = shape
    A = {k: np.full(shape, np.nan, np.float32) for k in
         ("ks_dur", "p_dur", "ks_int", "p_int", "med_dur_s", "med_dur_n", "med_int_s", "med_int_n")}
    ns, nn = np.zeros(shape, np.int16), np.zeros(shape, np.int16)
    for (j, i) in cells:
        s, n = ev_s[(j, i)], ev_n[(j, i)]
        ns[j, i], nn[j, i] = len(s[1]), len(n[1])
        if len(s[1]) < MIN_EVENTS or len(n[1]) < MIN_EVENTS:
            continue
        r = stats.ks_2samp(s[1], n[1])
        A["ks_dur"][j, i], A["p_dur"][j, i] = r.statistic, r.pvalue
        r = stats.ks_2samp(s[2], n[2])
        A["ks_int"][j, i], A["p_int"][j, i] = r.statistic, r.pvalue
        A["med_dur_s"][j, i], A["med_dur_n"][j, i] = np.median(s[1]), np.median(n[1])
        A["med_int_s"][j, i], A["med_int_n"][j, i] = np.median(s[2]), np.median(n[2])
    return A, ns, nn


# ============================================================================
# SAVING
# ============================================================================
def events_table(ev, data, cells, lat, lon, times, nx):
    span = G.DURATION_MODE == "span"
    rows = []
    for (j, i) in cells:
        st, du, it = ev[(j, i)]
        for e, (s0, d0, i0) in enumerate(zip(st, du, it), 1):
            if span:
                e0 = s0 + d0 - 1
                pk = s0 + int(np.nanargmin(data[s0:e0 + 1, j, i]))
                end, peak = times[e0], times[pk]
            else:
                end = peak = pd.NaT
            rows.append(dict(cell_id=j * nx + i, lat=float(lat[j]), lon=float(lon[i]),
                             event_id=e, start=times[s0], end=end, peak_date=peak,
                             start_year=times[s0].year, Duration=int(d0),
                             Intensity=float(i0), I_abs=abs(float(i0))))
    return pd.DataFrame(rows)


def pack(ev, cells, idx, shape, integer):
    nmax = max([len(ev[c][idx]) for c in cells] or [1])
    arr = (np.full((nmax,) + shape, -9999, np.int16) if integer
           else np.full((nmax,) + shape, np.nan, np.float32))
    for (j, i) in cells:
        v = ev[(j, i)][idx]
        arr[:len(v), j, i] = v
    return arr


def save_sig_nc(name, unit, idx, integer, ev_s, ev_n, sigcells, ks, p, sigflag, lat, lon):
    shape = (len(lat), len(lon))
    a, b = pack(ev_s, sigcells, idx, shape, integer), pack(ev_n, sigcells, idx, shape, integer)
    ds = xr.Dataset({
        f"{name}_stationary": (("event_sta", "lat", "lon"), a, {"units": unit}),
        f"{name}_nonstationary": (("event_nsta", "lat", "lon"), b, {"units": unit}),
        "ks_statistic": (("lat", "lon"), np.where(sigflag, ks, np.nan).astype(np.float32)),
        "p_value": (("lat", "lon"), np.where(sigflag, p, np.nan).astype(np.float32)),
        "significant_cell": (("lat", "lon"), sigflag.astype(np.int8)),
    }, coords={"event_sta": np.arange(1, a.shape[0] + 1),
               "event_nsta": np.arange(1, b.shape[0] + 1), "lat": lat, "lon": lon},
        attrs={"description": f"{name} of drought events in cells where Stationary and "
                              f"Non-stationary SPI-12 differ (KS, alpha={ALPHA})"})
    enc = {"zlib": True, "complevel": 4}
    e = {f"{name}_stationary": dict(enc), f"{name}_nonstationary": dict(enc)}
    if integer:
        for k in e:
            e[k]["_FillValue"] = -9999
    ds.to_netcdf(f"{OUTPUT_DIR}/nc/Significant_cells_{name.capitalize()}.nc", encoding=e)


# ============================================================================
# MAP: red = significant, blue = not significant
# ============================================================================
def make_cat_map(cat, title, fname):
    kept, excl = G.get_region_geoms()
    cmap = mcolors.ListedColormap([COL_NT, COL_NS, COL_SIG])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

    fig = plt.figure(figsize=(11, 6.6), facecolor="white")
    ax = fig.add_axes([0.03, 0.14, 0.94, 0.77], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor("white")
    ax.pcolormesh(cat.lon, cat.lat, np.ma.masked_invalid(cat.values), cmap=cmap, norm=norm,
                  transform=ccrs.PlateCarree(), shading="auto", rasterized=True, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6, edgecolor="0.35", zorder=2)
    ax.add_geometries(excl, ccrs.PlateCarree(), facecolor="none", edgecolor="0.65",
                      linewidth=0.7, zorder=3)
    ax.add_geometries(kept, ccrs.PlateCarree(), facecolor="none", edgecolor="black",
                      linewidth=1.1, zorder=4)
    ax.spines["geo"].set_linewidth(1.2)
    ax.set_title(title, fontsize=21, fontweight="bold", pad=12)

    handles = [Patch(facecolor=COL_SIG, edgecolor="black", label=f"Significant (p < {ALPHA})"),
               Patch(facecolor=COL_NS, edgecolor="black", label=f"Not significant (p \u2265 {ALPHA})"),
               Patch(facecolor=COL_NT, edgecolor="black", label=f"< {MIN_EVENTS} events (not tested)")]
    lg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
                    prop={"weight": "bold", "size": 15}, bbox_to_anchor=(0.5, 0.02))
    lg.get_frame().set_edgecolor("black")
    lg.get_frame().set_linewidth(1.5)

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
    for d in ("nc", "csv", "figures"):
        os.makedirs(f"{OUTPUT_DIR}/{d}", exist_ok=True)

    print("Loading SPI-12 ...")
    sta = G.load_spi(os.path.join(G.IN_DIR, G.FILES["Stationary"]))
    nst = G.load_spi(os.path.join(G.IN_DIR, G.FILES["Non-stationary"]))
    sta, nst = xr.align(sta, nst, join="inner")
    sl = slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    sta, nst = sta.sel(time=sl), nst.sel(time=sl)
    if sta.size == 0:
        raise SystemExit("No overlapping time/lat/lon between the two files.")
    print(f"Common period: {str(sta.time.values[0])[:7]} to {str(sta.time.values[-1])[:7]}")

    lat, lon = sta.lat.values, sta.lon.values
    times = pd.DatetimeIndex(sta.time.values)
    xs, xn = sta.values.astype(np.float32), nst.values.astype(np.float32)
    ny, nx = len(lat), len(lon)
    valid = np.isfinite(xs).any(0) & np.isfinite(xn).any(0)
    cells = [tuple(c) for c in np.argwhere(valid)]
    print(f"Grid cells with data in both: {len(cells)}")

    print("Extracting events ...")
    ev_s, ev_n = events_for(xs, cells), events_for(xn, cells)

    print("KS tests ...")
    A, ns, nn = run_ks(ev_s, ev_n, cells, (ny, nx))
    p_dur, p_int = (fdr(A["p_dur"]), fdr(A["p_int"])) if APPLY_FDR else (A["p_dur"], A["p_int"])
    tested = np.isfinite(p_dur) & np.isfinite(p_int)
    sig_dur = tested & (p_dur < ALPHA)
    sig_int = tested & (p_int < ALPHA)
    nt = tested.sum()
    print(f"  tested cells: {nt}")
    print(f"  duration  significant: {sig_dur.sum()} ({100*sig_dur.sum()/nt:.1f}%)")
    print(f"  intensity significant: {sig_int.sum()} ({100*sig_int.sum()/nt:.1f}%)")

    # per-variable settings: (significant mask, KS stat, p, index in event tuple, unit, integer?)
    sets = {"duration": (sig_dur, A["ks_dur"], p_dur, 1, "months", True),
            "intensity": (sig_int, A["ks_int"], p_int, 2, "SPI", False)}

    # ---- CSV: all cells ----------------------------------------------------
    jj, ii = np.where(valid)
    tab = pd.DataFrame({
        "cell_id": jj * nx + ii, "lat": lat[jj], "lon": lon[ii],
        "n_events_sta": ns[jj, ii], "n_events_nsta": nn[jj, ii],
        "median_duration_sta": A["med_dur_s"][jj, ii], "median_duration_nsta": A["med_dur_n"][jj, ii],
        "median_intensity_sta": A["med_int_s"][jj, ii], "median_intensity_nsta": A["med_int_n"][jj, ii],
        "ks_duration": A["ks_dur"][jj, ii], "p_duration": p_dur[jj, ii],
        "ks_intensity": A["ks_int"][jj, ii], "p_intensity": p_int[jj, ii],
        "sig_duration": sig_dur[jj, ii], "sig_intensity": sig_int[jj, ii]})
    tab.to_csv(f"{OUTPUT_DIR}/csv/KS_test_results_all_cells.csv", index=False)

    co = {"lat": lat, "lon": lon}
    d2 = ("lat", "lon")
    pd_f = lambda a: np.where(valid, a, np.nan).astype(np.float32)
    xr.Dataset({
        "ks_duration": (d2, pd_f(A["ks_dur"])), "p_duration": (d2, pd_f(p_dur)),
        "ks_intensity": (d2, pd_f(A["ks_int"])), "p_intensity": (d2, pd_f(p_int)),
        "sig_duration": (d2, sig_dur.astype(np.int8)), "sig_intensity": (d2, sig_int.astype(np.int8)),
        "n_events_sta": (d2, ns), "n_events_nsta": (d2, nn)}, coords=co,
        attrs={"description": f"Two-sample KS test, Stationary vs Non-stationary SPI-12 events; "
                              f"alpha={ALPHA}, FDR={APPLY_FDR}"}
    ).to_netcdf(f"{OUTPUT_DIR}/nc/KS_test_results.nc",
                encoding={"p_duration": {"zlib": True}, "p_intensity": {"zlib": True}})

    # ---- per-variable significant-cell files --------------------------------
    for v, (sg, ks, p, idx, unit, integer) in sets.items():
        cl = [tuple(c) for c in np.argwhere(sg)]
        tab[tab[f"sig_{v}"]].to_csv(f"{OUTPUT_DIR}/csv/KS_significant_cells_{v}.csv", index=False)
        events_table(ev_s, xs, cl, lat, lon, times, nx).to_csv(
            f"{OUTPUT_DIR}/csv/events_stationary_sig_cells_{v}.csv", index=False)
        events_table(ev_n, xn, cl, lat, lon, times, nx).to_csv(
            f"{OUTPUT_DIR}/csv/events_nonstationary_sig_cells_{v}.csv", index=False)
        save_sig_nc(v, unit, idx, integer, ev_s, ev_n, cl, ks, p, sg, lat, lon)
        print(f"  saved CSV/NC for {v}: {len(cl)} significant cells")

    # ---- maps ---------------------------------------------------------------
    print("Plotting ...")
    def category(sigmask):
        c = np.full((ny, nx), np.nan, np.float32)
        c[valid] = 0
        c[tested] = 1
        c[sigmask] = 2
        return xr.DataArray(c, coords=co, dims=d2)
    for v, (sg, *_rest) in sets.items():
        make_cat_map(category(sg),
                     f"KS test: Stationary vs Non-stationary\nDrought {v.capitalize()}",
                     f"KS_map_{v}")

    # ---- return periods, each variable on its own significant cells ----------
    if RUN_RETURN_PERIOD:
        print("\nReturn-period analysis (separately for duration and intensity) ...")
        import SPI12_return_period_sig_cells as RP
        RP.main()
    print("\nDone. Results in:", OUTPUT_DIR)
