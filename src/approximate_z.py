import xarray as xr

def approximate_z_on_boundaries_top_down(ds, dim="sigma2"):
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

    return z_i.where(ds["wet"] > 0) #this removes any non-ocean points from the field


def approximate_z_top_down(ds, dim="sigma2"):
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