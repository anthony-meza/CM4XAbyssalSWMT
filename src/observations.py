"""Loading observational datasets: the GIOMAS sea ice reanalysis (Zhang & Rothrock, 2003)."""
import numpy as np
import xarray as xr

from paths import obsdir


def _preprocess_giomas(ds):
    """Tidy one yearly GIOMAS file before the files are joined in `open_giomas`. It builds a monthly
    `time` axis, adds the area of each grid cell, and renames the grid coordinates to `lon` and
    `lat`.

    Parameters
    ----------
    ds : xarray.Dataset
        One yearly GIOMAS file, with grid spacings `dxt` and `dyt` in km.

    Returns
    -------
    xarray.Dataset
        The same fields on a monthly `time` axis, plus the cell area `darea` (in m^2).
    """
    year = int(ds["year"])
    time = np.array([np.datetime64(f"{year}-{month:02d}-01") for month in ds["month"].values])
    ds = ds.assign_coords(
        time=("n", time),
        lon_scaler=ds["lon_scaler"],
        lat_scaler=ds["lat_scaler"],
        z=ds["z"],
        dz=ds["dz"],
        dxt=ds["dxt"],
        dyt=ds["dyt"],
        darea=ds["dxt"] * 1e3 * ds["dyt"] * 1e3,  # km -> m
    )
    ds = ds.drop_vars(["kmt", "year", "month"]).swap_dims({"n": "time"})
    return ds.rename({"lon_scaler": "lon", "lat_scaler": "lat"})


def open_giomas():
    """Open the GIOMAS sea ice reanalysis (Zhang & Rothrock, 2003) from `data/GIOMASS/`. Fill values
    in the sea ice concentration and thickness are replaced with NaN.

    Returns
    -------
    xarray.Dataset
        Monthly sea ice concentration (`area`) and thickness (`heff`, in m) with fill values
        removed, and the cell area (`darea`, in m^2).
    """
    ds = xr.open_mfdataset(obsdir("GIOMASS/*"), preprocess=_preprocess_giomas, combine="by_coords")
    ds["area"] = ds["area"].where((ds["area"] < 9999.9) * (ds["area"] > 0.0))
    ds["heff"] = ds["heff"].where((ds["heff"] < 9999.9) * (ds["heff"] > 0.0))
    return ds
