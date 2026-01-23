import numpy as np
import CM4Xutils #needed to run: pip install nc-time-axis
import xarray as xr
from approximate_z import * 
from CM4XUtilsFunctions import * 
import warnings

from vertical_grid_utils import * 

from scipy.optimize import fsolve

def remap_vcoord_to_depth(ds, z_i = None, z_l = None, print_warning = False, vcoord = "sigma2", new_vcoord = "z"): 
    if z_i is None: 
        min_dz = 5.0 
        max_dz = 1000.0
        n_lev = 36
        z0 = 0
        zf = 6.750e+03
        H = zf - z0
        if print_warning:
            print("no z_i provided")
            print("using a default setup")
            print(f"N = {N}; H = {H} meters; eta = {eta} meters")
            
        dz, z_i, z_l = generate_tanh_vertical_grid(z0, min_dz, max_dz, n_lev, H)
        # z_i = make_tanh_grid(N = N, H = H, eta = eta)
        #                                      #np.arange(-6500, 251, 250)
        # z_l = (z_i[1:] + z_i[0:-1]) / 2
    ds = ds.assign_coords({f"{new_vcoord}_l": -z_l[::-1], f"{new_vcoord}_i":-z_i[::-1]})
    # print(ds.z_l)
    ds[f"{new_vcoord}"] = approximate_z_on_boundaries_bottom_up(ds, dim = f"{vcoord}")
    # print(ds["z"].max().compute())
    
  
    # Interpolate to fill NaNs and ensure monotonicity
    ds = ds.chunk({f"{vcoord}_l":-1, f"{vcoord}_i":-1, "year":1})
    ds[f"{new_vcoord}"] = ds[f"{new_vcoord}"].interpolate_na(dim=f"{vcoord}_i", method='linear')
    z_max_grid = float(ds[f"{new_vcoord}_i"].min())  # Most negative value (deepest)
    ds[f"{new_vcoord}"] = ds[f"{new_vcoord}"].clip(min=z_max_grid)  # Everything deeper than -6500m becomes -6500m (done to deal with trench artifacts)
      
    grid = CM4Xutils.ds_to_grid(ds)
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        ds_remap = remap_vertical_coord_custom(f"{new_vcoord}", ds, grid, ds[f"{new_vcoord}"])
        return ds_remap





def remap_sigma_to_depth(ds, z_i = None, z_l = None, print_warning = False): 
    if z_i is None: 
        min_dz = 5.0 
        max_dz = 1000.0
        n_lev = 45
        z0 = -5
        zf = 5250.0
        H = zf - z0
        if print_warning:
            print("no z_i provided")
            print("using a default setup")
            print(f"N = {N}; H = {H} meters; eta = {eta} meters")
            
        dz, z_i, z_l = generate_tanh_vertical_grid(z0, min_dz, max_dz, n_lev, H)
        # z_i = make_tanh_grid(N = N, H = H, eta = eta)
        #                                      #np.arange(-6500, 251, 250)
        # z_l = (z_i[1:] + z_i[0:-1]) / 2
    ds = ds.assign_coords({"z_l": -z_l[::-1], "z_i":-z_i[::-1]})
    # print(ds.z_l)
    ds["z"] = approximate_z_on_boundaries_bottom_up(ds, dim = "sigma2")
    # print(ds["z"].max().compute())
    
  
    # Interpolate to fill NaNs and ensure monotonicity
    ds = ds.chunk({"sigma2_l":-1, "sigma2_i":-1, "time":1})
    ds["z"] = ds["z"].interpolate_na(dim="sigma2_i", method='linear')
    z_max_grid = float(ds['z_i'].min())  # Most negative value (deepest)
    ds["z"] = ds["z"].clip(min=z_max_grid)  # Everything deeper than -6500m becomes -6500m (done to deal with trench artifacts)
      
    grid = CM4Xutils.ds_to_grid(ds)
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        ds_remap = remap_vertical_coord_custom("z", ds, grid, ds["z"])
        return ds_remap



