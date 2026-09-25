"""Approximate depth of density layers in the sigma2-coordinate CM4X output."""
import xarray as xr


def approximate_z_on_boundaries_top_down(ds, dim="sigma2"):
    """Estimate the depth of each density-layer interface in the sigma2-coordinate output. Starting
    from the sea surface, it adds up the layer thicknesses downward.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with `thkcello`, `wet`, and either `zos` or `deptho`, on `{dim}_l` / `{dim}_i`.
    dim : str, default "sigma2"
        Name of the vertical coordinate.

    Returns
    -------
    xarray.DataArray
        Depth of each layer interface (in m, negative downward); NaN over land.
    """
    ldim = f"{dim}_l"
    idim = f"{dim}_i"

    thickness = ds["thkcello"].fillna(0.0)

    if "zos" in ds:
        zos = ds["zos"]
    else:
        zos = thickness.sum(ldim) - ds["deptho"]

    z_top = xr.zeros_like(thickness.isel({ldim: 0}, drop=True)) + zos
    z_top = z_top.expand_dims({ldim: [-1]})

    z_i = xr.concat([z_top, -thickness], dim=ldim).cumsum(ldim)
    z_i = z_i.rename({ldim: idim})
    z_i = z_i.assign_coords({idim: ds[idim]})

    return z_i.where(ds["wet"] > 0)  # remove land points


def approximate_z_top_down(ds, dim="sigma2"):
    """Estimate the depth of each density layer in the sigma2-coordinate output. Each layer is
    placed halfway between its upper and lower interfaces from
    `approximate_z_on_boundaries_top_down`.

    Parameters
    ----------
    ds : xarray.Dataset
        As for `approximate_z_on_boundaries_top_down`.
    dim : str, default "sigma2"
        Name of the vertical coordinate.

    Returns
    -------
    xarray.DataArray
        Depth of each layer (in m, negative downward); NaN where the layer is empty or over land.
    """
    ldim = f"{dim}_l"
    idim = f"{dim}_i"

    z_i = approximate_z_on_boundaries_top_down(ds, dim=dim)

    z_upper = z_i.isel({idim: slice(0, -1)})
    z_lower = z_i.isel({idim: slice(1, None)})
    z_lower = z_lower.assign_coords({idim: z_upper[idim]})

    z_l = 0.5 * (z_upper + z_lower)
    z_l = z_l.rename({idim: ldim})
    z_l = z_l.assign_coords({ldim: ds[ldim]})

    return z_l.where((ds["thkcello"] > 0) & (ds["wet"] > 0))
