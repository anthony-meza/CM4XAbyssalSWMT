import xarray as xr 
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import numpy as np
import geopy
from geopy import distance
import xesmf as xe
# import sectionate
import re
import gsw


rootdir = "/user/anthony.meza/CM4XAbyssalSWMT/notebooks"
plotsdir = lambda x="": rootdir + "/figures/" + x
datadir = lambda x="" : "/proj/ecco/CM4X/" + x
outputdir = lambda x="" : "/proj/ecco/CM4X/postprocessing/" + x

from CM4XUtilsFunctions import *
from approximate_z import * 
from select_max_loc import * 
from spatial_decomposition import * 
from SWMT_decomposition import * 
from ant_plots import * 

def get_sigma2_at_surface(ds, keep_vars = False): 
    
    #this is how it is calculated in CM4X, no ct or sa 
    sigma2_surf = xr.apply_ufunc(
        gsw.sigma2,
        ds.sos,
        ds.tos,
        dask="parallelized"
    )

    ds["rho2"] = sigma2_surf

    
    return ds
    