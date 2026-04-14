import xarray as xr 
import numpy as np 
import pandas as pd
import xgcm 
import warnings
from tqdm import tqdm
import CM4Xutils #needed to run pip install nc-time-axis
from approximate_z import * 
def check_depth_sign(z): 

    zmin = z.min(skipna=True)
    zmax = z.max(skipna=True)
    
    if abs(zmin) > abs(zmax):
        return -z
    elif abs(zmax) > abs(zmin):
        return z
    else:
        raise ValueError("Cannot infer sign convention from equal-magnitude min/max.")



def get_psi_mean_basic(ds):
    """
    Calculates time-mean isopycnal overturning streamfunction using a naive averaging method.
    This does not take the changing latitudes of the tripolar grid into account and thus
    is really only valid south of 75°N.
    """

    # time mean and grid
    ds_mean = ds.mean(dim="time", skipna=True)
    grid = CM4Xutils.ds_to_grid(ds_mean, Zprefix="sigma2")

    # time-mean transport
    vmo = ds_mean["vmo"].fillna(0.0)

    # infer interface depth from layer thickness
    thk = ds_mean["thkcello"].fillna(0.0)
    z_bot = thk.cumsum(dim="sigma2_l")
    z_top = xr.zeros_like(z_bot.isel(sigma2_l=[0]))

    z_bot = z_bot.rename({"sigma2_l": "sigma2_i"})
    z_top = z_top.rename({"sigma2_l": "sigma2_i"})

    z_top["sigma2_i"] = [ds_mean["sigma2_i"][0].values]
    z_bot["sigma2_i"] = ds_mean["sigma2_i"][1:].values

    z_i = xr.concat([z_top, z_bot], dim="sigma2_i")

    # interpolate interface depth to v-grid
    z_i_yq = grid.interp(z_i, axis="Y", boundary="fill")

    # zonal mean depth on the v-grid
    dxCv = ds_mean["dxCv"].fillna(0.0)
    wet_v = ds_mean["wet_v"].fillna(0.0)
    valid = (wet_v == 1) & z_i_yq.notnull()

    z_yq_mean = (
        (z_i_yq.where(valid, 0.0) * dxCv.where(valid, 0.0)).sum(dim="xh")
        / dxCv.where(valid, 0.0).sum(dim="xh").where(lambda x: x > 0)
    )

    # overturning streamfunction
    vmo_xsum = vmo.sum(dim="xh")
    psi_l = (vmo_xsum.cumsum(dim="sigma2_l") - vmo_xsum.sum(dim="sigma2_l")) / (1035 * 1e6)

    psi_top = xr.zeros_like(psi_l.isel(sigma2_l=[0]))
    psi_l = psi_l.rename({"sigma2_l": "sigma2_i"})
    psi_top = psi_top.rename({"sigma2_l": "sigma2_i"})

    psi_top["sigma2_i"] = [ds_mean["sigma2_i"][0].values]
    psi_l["sigma2_i"] = ds_mean["sigma2_i"][1:].values

    psi = xr.concat([psi_top, psi_l], dim="sigma2_i")
    psi_ds = psi.rename("psi").to_dataset()

    # assemble output
    psi_ds["depth"] = z_yq_mean
    psi_ds.coords["lat"] = ds_mean["geolat_v"].mean(dim="xh")

    psi_ds["depth"].attrs = {
        "long_name": "Depth",
        "standard_name": "depth",
        "units": "m",
        "positive": "down",
        "axis": "Z",
    }

    psi_ds["psi"].attrs = {
        "long_name": "Isopycnal overturning streamfunction",
        "units": "Sv (10^6 m^3/s)",
        "long_units": "Sverdrups",
        "description": "Meridional overturning streamfunction",
    }

    return psi_ds
    
def get_psi_mean_simple(ds):
    """
    Calculates time-mean isopycnal overturning streamfunction using a naive averaging method.
    This does not take the changing latitudes of the tripolar grid into account and thus 
    is really only valid south of 75°N. 
    """
    # ====================================================
    # 0. take time average (to remove isopycnal heave) and setup grid
    # ====================================================
    
    ds_mean = ds.fillna(0.0).mean(dim="time")
    grid = CM4Xutils.ds_to_grid(ds_mean, Zprefix="sigma2")

    # ====================================================
    # 1. Calculate time-averaged transport and depth 
    # ====================================================
    # For the simple method, we just use the raw vmo directly
    vmo = ds_mean['vmo'].fillna(0.0)
    
    z_3d = approximate_z_on_boundaries_top_down(ds_mean, dim="sigma2") #z is NaN at land cells
    z_3d = check_depth_sign(z_3d)

    # Interpolate depth to cell faces (yq) using the grid
    z_3d_yq = grid.interp(z_3d, axis="Y", boundary="fill")
    
    # Load native v-grid metrics
    wet_v = ds_mean["wet_v"].fillna(0.0)
    dxCv = ds_mean["dxCv"].fillna(0.0)

    # Valid mask where there is ocean and a valid depth
    valid_mask = (wet_v == 1) & z_3d_yq.notnull()
    
    # Extract just the valid ocean face depths and widths
    z_bnd = z_3d_yq.where(valid_mask).fillna(0.0)
    dx_bnd = dxCv.where(valid_mask).fillna(0.0)
    
    # Length-weighted zonal mean on the yq grid
    z_weighted_sum = (z_bnd * dx_bnd).sum(dim='xh')
    total_width = dx_bnd.sum(dim='xh')
    z_yq_mean = z_weighted_sum / total_width.where(total_width > 0)
    
    # ========================================================
    # 3. Construct streamfunction and interpolate depth grid 
    # ========================================================
    vmo_xsum = vmo.sum(dim='xh')
    
    # Calculate streamfunction in layers. Make sure that it vertically integrates to zero. Convert from kg/s to Sv
    psi_l = (vmo_xsum.cumsum(dim='sigma2_l') - vmo_xsum.sum(dim='sigma2_l')) / (1035 * 1e6)
    
    # Create a 0 array for the ocean surface (Size 1)
    psi_top = xr.zeros_like(psi_l.isel(sigma2_l=[0]))
    
    # Rename dimensions to interfaces (sigma2_i)
    psi_l = psi_l.rename({'sigma2_l': 'sigma2_i'})
    psi_top = psi_top.rename({'sigma2_l': 'sigma2_i'})
    
    # Apply the correct interface coordinates 
    psi_top['sigma2_i'] = [ds_mean['sigma2_i'][0].values]
    psi_l['sigma2_i'] = ds_mean['sigma2_i'][1:].values
    
    # Concatenate to put streamfunction on the layer interface grid. 
    psi = xr.concat([psi_top, psi_l], dim='sigma2_i')
    psi_ds = psi.rename("psi").to_dataset()

    # ========================================================
    # 4. Assemble isopycnal overturning dataset
    # ========================================================
    psi_ds['depth'] = z_yq_mean
    psi_ds.coords['lat'] = ds_mean["geolat_v"].mean(dim='xh') 

    psi_ds["depth"].attrs = {
        "long_name": "Depth",
        "standard_name": "depth",
        "units": "m",
        "positive": "down",
        "axis": "Z",
    }
    
    psi_ds["psi"].attrs = {
        "long_name": "Isopycnal overturning streamfunction",
        "units": "Sv (1e-6 m^3/s)",
        "long_units": "Sverdrups",
        "description": "Meridional overturning streamfunction",
    }

    return psi_ds

def get_psi_mean_convergence(ds, lats, grid=None):
    """
    Calculates time-mean meridional isopycnal overturning streamfunction using
    meridional convergence, which accounts for the tripolar grid .
    """
    # ====================================================
    # 0. take time average (to remove isopycnal heave) and setup grid
    # ====================================================
    ds_mean = ds.fillna(0.0).mean(dim="time")
    if grid is None:
        grid = CM4Xutils.ds_to_grid(ds_mean, Zprefix="sigma2")
        
    # ====================================================
    # 1. calculate transport across a latitudinal boundary by taking the convergence)
    # ====================================================
    div_v = grid.diff(ds_mean["vmo"].fillna(0.0), axis="Y", boundary="fill").fillna(0.0)
    
    # ====================================================
    # 2. Calculate z at lateral cell center and layer interfaces. Check the sign convention. Interpolate from yh to yq
    # ====================================================
    z_3d = approximate_z_on_boundaries_top_down(ds_mean, dim="sigma2") #z is NaN at land cells
    z_3d = check_depth_sign(z_3d) #could this happen later? 

    # Interpolate depth to cell faces (yq)
    z_3d_yq = grid.interp(z_3d, axis="Y", boundary="fill")
    
    # Load native v-grid metrics
    wet_v = ds_mean["wet_v"].fillna(0.0)
    dxCv = ds_mean["dxCv"].fillna(0.0)

    # ====================================================
    # 3. Calculate net transport across each latitude. Also, calculate average depth at that latitude. 
    # ====================================================
    psi_list = []
    depth_list = []
    
    for l in lats:
        # a. Calculate the convergence across a latitude mask 
        #(-div_v, since grid.diff calculate north_point - south_point but we need  south_point - north_point
        mask_south_yh = ds_mean["geolat"] <= l
        term_div = (-div_v * mask_south_yh).sum(dim=["xh", "yh"])
        psi_list.append(term_div)
        
        # b. Find the yq points that outline the mask
        mask_float = mask_south_yh.astype(float)
        boundary_mask_yq = grid.diff(mask_float, axis="Y", boundary="fill") < 0 #grid.diff calculate north_point - south_point
        
        # c. pull out the valid ocean points
        valid_bnd = boundary_mask_yq & (wet_v == 1)
        
        # d. Extract depth and width exactly along the wet boundary
        z_bnd = z_3d_yq.where(valid_bnd)
        dx_bnd = dxCv.where(valid_bnd)
        
        # e. Length-weighted boundary average
        z_weighted = (z_bnd * dx_bnd).sum(dim=["xh", "yq"], skipna=True)
        length_tot = dx_bnd.sum(dim=["xh", "yq"], skipna=True)
        
        term_depth = z_weighted / length_tot.where(length_tot > 0)
        depth_list.append(term_depth)
        
    # ====================================================
    # 4. Integrate transport and construct the psi dataset. 
    # ====================================================
    lat_dim = xr.DataArray(lats, dims='lat', name='lat')
    psi_lat_sum = xr.concat(psi_list, dim=lat_dim)
    depth_lat = xr.concat(depth_list, dim=lat_dim)
    
    # Calculate streamfunction in layers. Make sure that it vertically integrates to zero. Convert from kg/s to Sv
    psi_l = -((psi_lat_sum.cumsum("sigma2_l") - psi_lat_sum.sum("sigma2_l")) / (1035 * 1e6))
    
    # Align to interfaces
    psi_top = xr.zeros_like(psi_l.isel(sigma2_l=[0]))
    psi_l = psi_l.rename({'sigma2_l': 'sigma2_i'})
    psi_top = psi_top.rename({'sigma2_l': 'sigma2_i'})
    
    psi_top['sigma2_i'] = [ds_mean['sigma2_i'][0].values]
    psi_l['sigma2_i'] = ds_mean['sigma2_i'][1:].values
    psi = xr.concat([psi_top, psi_l], dim='sigma2_i')

    psi_ds = psi.rename("psi").to_dataset()
    psi_ds['depth'] = depth_lat

    psi_ds["depth"].attrs = {
        "long_name": "Depth",
        "standard_name": "depth",
        "units": "m",
        "positive": "down",
        "axis": "Z",
    }
    
    psi_ds["psi"].attrs = {
        "long_name": "Isopycnal overturning streamfunction",
        "units": "Sv (1e6 m^3/s)",
        "long_units": "Sverdrups",
        "description": "Meridional overturning streamfunction",
    }


    return psi_ds

def get_psi_convergence(ds, lats, grid=None):
    """
    Calculates time-mean meridional isopycnal overturning streamfunction using
    meridional convergence, which accounts for the tripolar grid .
    """
    # ====================================================
    # 0. take time average (to remove isopycnal heave) and setup grid
    # ====================================================
    ds_mean = ds.fillna(0.0)
    if grid is None:
        grid = CM4Xutils.ds_to_grid(ds_mean, Zprefix="sigma2")
        
    # ====================================================
    # 1. calculate transport across a latitudinal boundary by taking the convergence)
    # ====================================================
    div_v = grid.diff(ds_mean["vmo"].fillna(0.0), axis="Y", boundary="fill").fillna(0.0)
    
    # ====================================================
    # 2. Calculate z at lateral cell center and layer interfaces. Check the sign convention. Interpolate from yh to yq
    # ====================================================
    z_3d = approximate_z_on_boundaries_top_down(ds_mean, dim="sigma2") #z is NaN at land cells
    z_3d = check_depth_sign(z_3d) #could this happen later? 

    # Interpolate depth to cell faces (yq)
    z_3d_yq = grid.interp(z_3d, axis="Y", boundary="fill")
    
    # Load native v-grid metrics
    wet_v = ds_mean["wet_v"].fillna(0.0)
    dxCv = ds_mean["dxCv"].fillna(0.0)

    # ====================================================
    # 3. Calculate net transport across each latitude. Also, calculate average depth at that latitude. 
    # ====================================================
    psi_list = []
    depth_list = []
    
    for l in lats:
        # a. Calculate the convergence across a latitude mask 
        #(-div_v, since grid.diff calculate north_point - south_point but we need  south_point - north_point
        mask_south_yh = ds_mean["geolat"] <= l
        term_div = (-div_v * mask_south_yh).sum(dim=["xh", "yh"])
        psi_list.append(term_div)
        
        # b. Find the yq points that outline the mask
        mask_float = mask_south_yh.astype(float)
        boundary_mask_yq = grid.diff(mask_float, axis="Y", boundary="fill") < 0 #grid.diff calculate north_point - south_point
        
        # c. pull out the valid ocean points
        valid_bnd = boundary_mask_yq & (wet_v == 1)
        
        # d. Extract depth and width exactly along the wet boundary
        z_bnd = z_3d_yq.where(valid_bnd)
        dx_bnd = dxCv.where(valid_bnd)
        
        # e. Length-weighted boundary average
        z_weighted = (z_bnd * dx_bnd).sum(dim=["xh", "yq"], skipna=True)
        length_tot = dx_bnd.sum(dim=["xh", "yq"], skipna=True)
        
        term_depth = z_weighted / length_tot.where(length_tot > 0)
        depth_list.append(term_depth)
        
    # ====================================================
    # 4. Integrate transport and construct the psi dataset. 
    # ====================================================
    lat_dim = xr.DataArray(lats, dims='lat', name='lat')
    psi_lat_sum = xr.concat(psi_list, dim=lat_dim)
    depth_lat = xr.concat(depth_list, dim=lat_dim)
    
    # Calculate streamfunction in layers. Make sure that it vertically integrates to zero. Convert from kg/s to Sv
    psi_l = -((psi_lat_sum.cumsum("sigma2_l") - psi_lat_sum.sum("sigma2_l")) / (1035 * 1e6))
    
    # Align to interfaces
    psi_top = xr.zeros_like(psi_l.isel(sigma2_l=[0]))
    psi_l = psi_l.rename({'sigma2_l': 'sigma2_i'})
    psi_top = psi_top.rename({'sigma2_l': 'sigma2_i'})
    
    psi_top['sigma2_i'] = [ds_mean['sigma2_i'][0].values]
    psi_l['sigma2_i'] = ds_mean['sigma2_i'][1:].values
    psi = xr.concat([psi_top, psi_l], dim='sigma2_i')

    psi_ds = psi.rename("psi").to_dataset()
    psi_ds['depth'] = depth_lat

    psi_ds["depth"].attrs = {
        "long_name": "Depth",
        "standard_name": "depth",
        "units": "m",
        "positive": "down",
        "axis": "Z",
    }
    
    psi_ds["psi"].attrs = {
        "long_name": "Isopycnal overturning streamfunction",
        "units": "Sv (1e6 m^3/s)",
        "long_units": "Sverdrups",
        "description": "Meridional overturning streamfunction",
    }


    return psi_ds
