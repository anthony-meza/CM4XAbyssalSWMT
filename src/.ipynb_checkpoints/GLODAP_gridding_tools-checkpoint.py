import pyinterp
import pyinterp.backends.xarray
from scipy import interpolate
import xarray as xr
import dask
import numpy as np
import matplotlib.pyplot as plt
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
from scipy.interpolate import CubicSpline
import copy
import pandas as pd

def extract_GLODAP_vars(ds, varn): 
    lat, lon = ds["G2latitude"].to_xarray(), ds["G2longitude"].to_xarray()
    years = ds["G2year"].to_xarray()
    depths = ds["G2depth"].to_xarray()
    
    cast_num = ds["G2cast"].to_xarray().astype("int").astype("str")
    cruise_num = ds["G2cruise"].to_xarray().astype("int").astype("str")
    station_num = ds["G2station"].to_xarray().astype("int").astype("str")

    cast_codes = ds["G2expocode"].to_xarray().astype("str")
    
    tmp = np.char.add(cast_codes.values, "_")
    tmp = np.char.add(tmp, cruise_num)
    tmp = np.char.add(tmp, "_")
    tmp = np.char.add(tmp, station_num)
    tmp = np.char.add(tmp, "_")
    tmp = np.char.add(tmp, cast_num)
    
    all_casts = copy.deepcopy(cast_codes)
    all_casts.values = tmp

    return lat, lon, years, depths, all_casts

def GLODAP_2_regular_grid(ds, regular_z, regular_lat, regular_lon, varns, year = 1990.):

    nvars = len(varns)
    nreglon = len(regular_lon)
    nreglat = len(regular_lat)
    nregz = len(regular_z)
    
    vars_dicts = dict()
    casts_dict = dict()
    
    for (v, varn) in enumerate(varns): 
        vars_gridded_nearest = np.nan * np.zeros((nreglon, nreglat, nregz))
        lat, lon, years, depths, all_casts = extract_GLODAP_vars(ds, varn)
        var_df = ds[varn].to_xarray()
        try: 
            var_df_flag = ds[varn + "f"].to_xarray()
            which_year_and_notnan = (years == year) * (~np.isnan(var_df)) * ( var_df_flag == 2)
        except: 
            which_year_and_notnan = (years == year) * (~np.isnan(var_df))

        depths_ = depths[which_year_and_notnan]
        lat_  = lat[which_year_and_notnan]
        lon_ = lon[which_year_and_notnan]
        var_ = var_df[which_year_and_notnan]
        all_casts_ = all_casts[which_year_and_notnan]
        all_casts_ = np.array(all_casts_.values.tolist())
        unique_casts_ = np.unique(all_casts_)
        
        interp_casts = np.zeros((len(regular_z), len(unique_casts_)))
        LATS = np.zeros(len(unique_casts_))
        LONS = np.zeros(len(unique_casts_))
        cast_list = []
        for j in range(len(unique_casts_)):
            which_cast = (all_casts_ == unique_casts_[j])
            cast_list += [unique_casts_[j]]
            cast_var = var_[which_cast]; cast_depths = depths_[which_cast]
            
            unique_depths = np.unique(cast_depths)
            LATS[j] = lat_[which_cast][0]
            LONS[j] = lon_[which_cast][0]
            time_averaged_cast = np.zeros(len(unique_depths))
            if len(unique_depths) < len(cast_depths):
                for (i, d) in enumerate(unique_depths):
                    time_averaged_cast[i] = np.nanmean(cast_var[cast_depths == d]) 
            else:
                time_averaged_cast[:] = 1 * cast_var[:]  

            f = interpolate.interp1d(unique_depths, time_averaged_cast, fill_value = np.nan, 
                                     bounds_error = False, kind = "linear")
            interp_casts[:, j] = f(regular_z)
            
        interp_casts_ds = xr.Dataset(
                        coords={
                        "points":(["points"], np.arange(len(unique_casts_))),
                        "z":(["z"], regular_z)
                        })
        interp_casts_ds[varn] = (["z", "points"], interp_casts)
        interp_casts_ds["lat"] = (["points"], LATS)
        interp_casts_ds["lon"] = (["points"], LONS)
        interp_casts_ds["cast_id"] = (["points"], cast_list)
        interp_casts_ds["time"] = year

        for k in range(len(regular_z)):
            # binning = pyinterp.Binning2D(
            #     pyinterp.Axis(regular_lon, is_circle=True),
            #     pyinterp.Axis(regular_lat))
            # binning.clear()
            binning = pyinterp.Histogram2D(
                pyinterp.Axis(regular_lon, is_circle=True),
                pyinterp.Axis(regular_lat))
            binning.clear()
            
            binning.push(LONS, LATS, 1 * interp_casts[k, :][:])        
            vars_gridded_nearest[:, :, k] = 1 * binning.variable('mean')            
            
        regular_ds = xr.Dataset(
                        coords={
                        "lon":(["lon"], regular_lon),
                        "lat":(["lat"], regular_lat),
                        "z":(["z"], regular_z), 
                        })
        regular_ds[varn] = (["lon", "lat", "z"], vars_gridded_nearest)
        regular_ds["time"] = year

        vars_dicts[varn] = regular_ds
        casts_dict[varn] = interp_casts_ds

    
    
    return (vars_dicts, casts_dict)


def GLODAP_2_regular_grid_custom_vertical(ds, regular_vertical, regular_lat, regular_lon, varns, 
                                          vertical_coord='G2depth', year=1990., binning_method='mean'):
    """
    Interpolate GLODAP data onto a regular grid with customizable vertical coordinate.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        GLODAP dataset
    regular_vertical : array-like
        Regular vertical grid to interpolate onto
    regular_lat : array-like
        Regular latitude grid
    regular_lon : array-like
        Regular longitude grid
    varns : list of str
        Variable names to process
    vertical_coord : str
        Name of the vertical coordinate variable in the dataset (e.g., 'G2depth', 'G2pressure')
        Default is 'G2depth'
    year : float
        Year to filter data
    binning_method : str
        Method for binning data in each grid cell. Options: 'mean', 'median'
        Default is 'mean'
        
    Returns:
    --------
    tuple : (vars_dicts, casts_dict)
        vars_dicts: dict of gridded datasets for each variable
        casts_dict: dict of cast profiles for each variable
    """
    
    # Check if vertical coordinate exists in dataset
    if vertical_coord not in ds:
        raise ValueError(f"Vertical coordinate '{vertical_coord}' not found in dataset. "
                        f"Available variables: {list(ds.keys())}")
    
    # Check if binning method is valid
    valid_methods = ['mean', 'median']
    if binning_method not in valid_methods:
        raise ValueError(f"Binning method '{binning_method}' not recognized. "
                        f"Valid options: {valid_methods}")
    
    nvars = len(varns)
    nreglon = len(regular_lon)
    nreglat = len(regular_lat)
    nregvert = len(regular_vertical)
    
    vars_dicts = dict()
    casts_dict = dict()
    
    for (v, varn) in enumerate(varns): 
        vars_gridded_nearest = np.nan * np.zeros((nreglon, nreglat, nregvert))
        lat, lon, years, depths, all_casts = extract_GLODAP_vars(ds, varn)
        
        # Get the vertical coordinate data
        vertical_data = ds[vertical_coord].to_xarray()
        
        var_df = ds[varn].to_xarray()
        try: 
            var_df_flag = ds[varn + "f"].to_xarray()
            which_year_and_notnan = (years == year) * (~np.isnan(var_df)) * (var_df_flag == 2)
        except: 
            which_year_and_notnan = (years == year) * (~np.isnan(var_df))

        vertical_ = vertical_data[which_year_and_notnan]
        lat_ = lat[which_year_and_notnan]
        lon_ = lon[which_year_and_notnan]
        var_ = var_df[which_year_and_notnan]
        all_casts_ = all_casts[which_year_and_notnan]
        all_casts_ = np.array(all_casts_.values.tolist())
        unique_casts_ = np.unique(all_casts_)
        
        interp_casts = np.zeros((len(regular_vertical), len(unique_casts_)))
        LATS = np.zeros(len(unique_casts_))
        LONS = np.zeros(len(unique_casts_))
        cast_list = []
        
        for j in range(len(unique_casts_)):
            which_cast = (all_casts_ == unique_casts_[j])
            cast_list += [unique_casts_[j]]
            cast_var = var_[which_cast]
            cast_vertical = vertical_[which_cast]
            
            unique_vertical = np.unique(cast_vertical)
            LATS[j] = lat_[which_cast][0]
            LONS[j] = lon_[which_cast][0]
            time_averaged_cast = np.zeros(len(unique_vertical))
            
            if len(unique_vertical) < len(cast_vertical):
                # print(unique_vertical)
                # print(cast_vertical)

                for (i, v_val) in enumerate(unique_vertical):
                    time_averaged_cast[i] = np.nanmean(cast_var[cast_vertical == v_val]) 
            else:
                time_averaged_cast[:] = 1 * cast_var[:]  

            f = interpolate.interp1d(unique_vertical, time_averaged_cast, fill_value=np.nan, 
                                     bounds_error=False, kind="linear")
            interp_casts[:, j] = f(regular_vertical)
            
        interp_casts_ds = xr.Dataset(
                        coords={
                        "points": (["points"], np.arange(len(unique_casts_))),
                        vertical_coord: ([vertical_coord], regular_vertical)
                        })
        interp_casts_ds[varn] = ([vertical_coord, "points"], interp_casts)
        interp_casts_ds["lat"] = (["points"], LATS)
        interp_casts_ds["lon"] = (["points"], LONS)
        interp_casts_ds["cast_id"] = (["points"], cast_list)
        interp_casts_ds["time"] = year

        for k in range(len(regular_vertical)):
            binning = pyinterp.Histogram2D(
                pyinterp.Axis(regular_lon, is_circle=True),
                pyinterp.Axis(regular_lat))
            binning.clear()
            
            binning.push(LONS, LATS, 1 * interp_casts[k, :][:])        
            vars_gridded_nearest[:, :, k] = 1 * binning.variable(binning_method)            
            
        regular_ds = xr.Dataset(
                        coords={
                        "lon": (["lon"], regular_lon),
                        "lat": (["lat"], regular_lat),
                        vertical_coord: ([vertical_coord], regular_vertical), 
                        })
        regular_ds[varn] = (["lon", "lat", vertical_coord], vars_gridded_nearest)
        regular_ds["time"] = year

        vars_dicts[varn] = regular_ds
        casts_dict[varn] = interp_casts_ds
    
    return (vars_dicts, casts_dict)