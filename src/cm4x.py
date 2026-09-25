"""Loading the coarsened CM4X output and defining the Antarctic continental shelf (ACS) region."""
import glob

import numpy as np
import xarray as xr
import CM4Xutils
from regionate import MaskRegions, GriddedRegion

from paths import datadir


def cm4x_budget_files(first_year=1850):
    """List the 5-year CM4X budget files, which hold monthly output on a 1.5 degree grid in sigma2
    coordinates. Files before `first_year` are skipped, so the default starts in 1850 and leaves out
    the piControl spin-up.

    Parameters
    ----------
    first_year : int, default 1850
        First year to include; 1850 skips the piControl spin-up.

    Returns
    -------
    list of str
        Paths to the budget zarr stores, sorted in time.
    """
    files = sorted(glob.glob(datadir("budget_sigma2_v1.2.0/CM4Xp125*")))
    return [f for f in files if int(f.split("_")[-1][:4]) >= first_year]


def open_cm4x_budget(files):
    """Open one or more consecutive 5-year budget files as a single dataset. Monthly fields and
    boundary snapshots (the *_bounds variables) are joined along their own time dimensions, and the
    snapshot shared by neighbouring files is kept only once.

    Parameters
    ----------
    files : str or list of str
        One budget file or several consecutive ones, e.g. from `cm4x_budget_files`.

    Returns
    -------
    xarray.Dataset
        Lazy dataset with one more `time_bounds` than `time`.
    """
    files = [files] if isinstance(files, str) else list(files)
    dss = [
        xr.open_mfdataset(f, data_vars="minimal", coords="minimal", compat="override", engine="zarr")
        .drop_vars(["time_since_init", "time_bounds_since_init"], errors="ignore")
        for f in files
    ]
    concat = dict(data_vars="minimal", coords="minimal", compat="override")
    on_time = xr.concat([ds.drop_dims("time_bounds") for ds in dss], dim="time", **concat)
    bounds_vars = [v for v in dss[0].data_vars if "time_bounds" in dss[0][v].dims]
    on_bounds = xr.concat([ds[bounds_vars] for ds in dss], dim="time_bounds", **concat)
    return xr.merge([on_time, on_bounds.drop_duplicates("time_bounds")], compat="override")


def cm4x_acs_region(ds, grid):
    """Build the Antarctic Continental Shelf (ACS) region used for all budgets: ocean south of 50S
    and shallower than 1001 m. Only the largest connected area is kept, so the region is the
    continental shelf itself rather than isolated shallow banks.

    Parameters
    ----------
    ds : xarray.Dataset
        Budget dataset with `geolat` and `deptho`.
    grid : xgcm.Grid
        Grid of `ds`, e.g. from `CM4Xutils.ds_to_grid`.

    Returns
    -------
    regionate.GriddedRegion
        The ACS region; its `.mask` is the cell mask and its contour is the shelf boundary.
    """
    mask = (ds["geolat"] <= -50.0) * (ds["deptho"].fillna(0.0) <= 1001.0)
    antarctic = MaskRegions(mask, grid).region_dict[0]  # the largest contour is Antarctica
    return GriddedRegion(
        "antarctic", antarctic.lons_c, antarctic.lats_c, grid,
        ij=(antarctic.i_c, antarctic.j_c),
    )


def cm4x_acs_mask():
    """Return the ACS region as a mask on the 1.5 degree budget grid. The figure notebooks use it to
    outline the shelf on maps, so it only reads the grid from a single budget file.

    Returns
    -------
    xarray.DataArray
        Boolean mask of the ACS region on the 1.5 degree budget grid.
    """
    ds = xr.open_mfdataset(cm4x_budget_files(first_year=0)[1], engine="zarr")
    grid = CM4Xutils.ds_to_grid(ds)
    return cm4x_acs_region(ds, grid).mask


def _preprocess_tracers(ds):
    """Trim one tracer store before the stores are joined in `open_cm4x_tracers`. It drops the
    piControl calendar coordinate `year_ctrl` and keeps only the Southern Ocean rows and the years
    up to 2099.

    Parameters
    ----------
    ds : xarray.Dataset
        One yearly tracer store as opened by `xr.open_mfdataset`.

    Returns
    -------
    xarray.Dataset
        The first 150 rows of `yh` (the Southern Ocean), for years up to 2099.
    """
    n_southern_rows = 150  # rows of yh containing the Southern Ocean
    if "year_ctrl" in ds.coords:
        ds = ds.drop_vars(["year_ctrl"])
    return ds.isel(yh=slice(0, n_southern_rows)).sel(year=slice(None, 2099))


def open_cm4x_tracers(parallel=True):
    """Open the yearly CM4X tracer output (0.75 degree, on depth levels) for both experiments. The
    piControl run and the joined Historical and SSP5-8.5 runs are stacked along a new `expt`
    dimension covering 1850-2099.

    Parameters
    ----------
    parallel : bool, default True
        Open the stores on the dask workers. Use False if the workers can't import this module
        (e.g. a SLURM cluster whose workers don't have src/ on their path).

    Returns
    -------
    xarray.Dataset
        Lazy tracers for 1850-2099 with dimension `expt` ("control" = piControl,
        "forced" = Historical/SSP5-8.5) and depth interfaces `z_i`.
    """
    tracer_files = {  # yearly tracer products for each experiment
        "control": ["CM4Xp125_piControl_transient_tracers_z.zarr"],
        "forced": ["CM4Xp125_historical_transient_tracers_z.zarr", "CM4Xp125_ssp585_transient_tracers_z.zarr"],
    }
    # depth of the z-level interfaces in the tracer product (in m)
    z_i = np.array([
        0.0, 5.0, 15.0, 25.0, 40.0, 62.5, 87.5, 112.5, 137.5, 175.0, 225.0, 275.0,
        350.0, 450.0, 550.0, 650.0, 750.0, 850.0, 950.0, 1050.0, 1150.0, 1250.0, 1350.0, 1450.0,
        1625.0, 1875.0, 2250.0, 2750.0, 3250.0, 3750.0, 4250.0, 4750.0, 5250.0, 5750.0, 6250.0, 6750.0,
    ])

    dss = []
    for expt, files in tracer_files.items():
        paths = [datadir("transient_tracers_z/" + f) for f in files]
        ds = xr.open_mfdataset(
            paths, combine="nested", concat_dim=["year"], parallel=parallel,
            preprocess=_preprocess_tracers, engine="zarr",
        )
        dss.append(ds.expand_dims(expt=[expt]))
    return xr.concat(dss, dim="expt").assign_coords(z_i=z_i)
