import xarray as xr 
import numpy as np 
import pandas as pd

def approximate_z_on_boundaries_top_down(ds, dim="sigma2", zos=None):
    thicknesses = ds.thkcello.fillna(0.0)

    if zos is None:
        if "zos" in ds:
            zos = ds["zos"]
        else:
            zos = thicknesses.sum(f"{dim}_l") - ds.deptho

    eta = (0.0 * thicknesses.isel({f"{dim}_l": 0})) + zos
    eta.coords[f"{dim}_l"] = -100

    cell_boundaries = xr.concat([eta, -thicknesses], dim=f"{dim}_l")
    cell_boundaries = cell_boundaries.cumsum(dim=f"{dim}_l")
    cell_boundaries = cell_boundaries.rename({f"{dim}_l": f"{dim}_i"})
    cell_boundaries.coords[f"{dim}_i"] = ds.coords[f"{dim}_i"]

    return cell_boundaries.where(ds.wet > 0)

def approximate_z_top_down(ds, dim="sigma2", zos=None):
    cell_boundary_height = approximate_z_on_boundaries_top_down(ds, dim=dim, zos=zos)

    h_n = cell_boundary_height.isel({f"{dim}_i": slice(0, -1)})
    h_np1 = cell_boundary_height.isel({f"{dim}_i": slice(1, None)})
    h_np1.coords[f"{dim}_i"] = h_n.coords[f"{dim}_i"]

    cell_center_height = 0.5 * (h_np1 + h_n)
    cell_center_height = cell_center_height.rename({f"{dim}_i": f"{dim}_l"})
    cell_center_height.coords[f"{dim}_l"] = ds.coords[f"{dim}_l"]

    return cell_center_height.where(ds.wet > 0)
    
def approximate_z_on_boundaries_bottom_up(ds, dim = "sigma2"):
    H = ds.deptho
    
    thicknesses = ds.thkcello.fillna(0.0)
    
    # Flip the thickness array to go from densest to least dense waters
    flipped_thicknesses = thicknesses.isel({f"{dim}_l": slice(None, None, -1)})
    #set H to be maximum depth within a given column
    h_bottom = (0.0 * flipped_thicknesses.isel({f"{dim}_l":0})) + H
    h_bottom.coords[f"{dim}_l"] = 100
    #calculate the vertical position of cell boundaries using 
    #cumsum
    cell_boundaries = xr.concat([h_bottom, (-flipped_thicknesses)], dim = f"{dim}_l")
    cell_boundaries = cell_boundaries.cumsum(dim = f"{dim}_l")
    cell_boundaries = cell_boundaries.rename({f"{dim}_l":f"{dim}_i"})
    #force grid to be ordered from least to most dense 
    cell_boundaries = cell_boundaries.isel({f"{dim}_i": slice(None, None, -1)})
    #make negative values mean "below" surface
    cell_boundaries *= -1
    cell_boundaries.coords[f"{dim}_i"] = ds.coords[f"{dim}_i"]
    
    # NEW: Set repeated values to NaN (these correspond to missing density classes)
    # Where thickness was zero, cell_boundaries will have repeated values
    # Mark these as NaN since they don't represent real interfaces
    diff = cell_boundaries.diff(f"{dim}_i")
    # Depths should get more negative (diff < 0) as density increases
    # If diff == 0, it's a repeat -> set to NaN
    repeats = (diff == 0)
    repeats = np.abs(diff) < 1e-10  # Use tolerance instead of == 0
    # Propagate NaN to the repeated interfaces (shifted by 1)
    cell_boundaries = cell_boundaries.where(~repeats.shift({f"{dim}_i": 1}, fill_value=False))
    
    return cell_boundaries
    
def approximate_z_bottom_up(ds, dim="sigma2"):
    H = ds.deptho

    thicknesses = ds.thkcello.fillna(0.0)
    zos = thicknesses.sum(f"{dim}_l") - H
    
    return approximate_z_top_down(ds, dim = dim, zos = zos)

    