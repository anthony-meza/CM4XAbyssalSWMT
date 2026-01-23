from scipy.optimize import fsolve
import numpy as np
def define_tanhdz(DELTA, min_dz, max_dz, n_lev, H):
    """
    Compute layer thicknesses using hyperbolic tangent distribution.
    
    PARAMETERS:
    -----------
        DELTA : float
            Transition steepness parameter (larger = more abrupt transition)
        min_dz : float
            Minimum layer thickness (surface resolution)
        max_dz : float
            Maximum layer thickness (deep resolution)
        n_lev : int
            Number of vertical levels
        H : float
            Total domain depth
    
    RETURNS:
    --------
        dz : ndarray
            Array of layer thicknesses as float32
    """
    k0 = 0
    k = np.arange(k0, n_lev)
    dz = ((max_dz - min_dz) * np.tanh(np.pi * (k - k0) / DELTA)) + min_dz
    return np.float32(dz)
    
def generate_tanhdz(min_dz, max_dz, n_lev, H):
    """
    Optimize DELTA parameter so layer thicknesses sum exactly to total depth H.
    
    PARAMETERS:
    -----------
        min_dz : float
            Minimum layer thickness (surface resolution)
        max_dz : float
            Maximum layer thickness (deep resolution)
        n_lev : int
            Number of vertical levels
        H : float
            Total domain depth
    
    RETURNS:
    --------
        dz_opt : ndarray
            Optimized array of layer thicknesses
    
    RAISES:
    -------
        ValueError
            If solver fails to converge to a solution
    """
    tanhdz_problem = lambda DELTA: np.sum(define_tanhdz(DELTA, min_dz, max_dz, n_lev, H)) - H
    DELTA_OPT = fsolve(tanhdz_problem, [1e4], xtol = 1e-12)
    dz_opt = define_tanhdz(DELTA_OPT, min_dz, max_dz, n_lev, H)
    if np.isclose(tanhdz_problem(DELTA_OPT), 0.0):
        return dz_opt
    else: 
        raise ValueError("Sorry, solver could not converge")

def zf_from_dz(dz, z0):
    """
    Compute interface depths from layer thicknesses via cumulative sum.
    
    PARAMETERS:
    -----------
        dz : ndarray
            Array of layer thicknesses
        z0 : float
            Starting depth (typically 0 for surface)
    
    RETURNS:
    --------
        zf : ndarray
            Array of interface depths (n_lev + 1 values)
    """
    return np.cumsum(np.concatenate([np.float32([z0]), dz]))

def z_from_zf(zf):
    """
    Compute layer center depths from interface depths.
    
    PARAMETERS:
    -----------
        zf : ndarray
            Array of interface depths (n_lev + 1 values)
    
    RETURNS:
    --------
        z : ndarray
            Array of layer center depths (n_lev values, midpoints between interfaces)
    """
    return (zf[1:] + zf[:-1]) / np.array(np.float32([2.0]))

def generate_tanh_vertical_grid(z0, min_dz, max_dz, n_lev, H):
    """
    Generate complete vertical grid with tanh-based smooth spacing transition.
    
    PARAMETERS:
    -----------
        min_dz : float
            Minimum layer thickness (surface resolution)
        max_dz : float
            Maximum layer thickness (deep resolution)
        n_lev : int
            Number of vertical levels
        H : float
            Total domain depth
    
    RETURNS:
    --------
        dz : ndarray
            Layer thicknesses (n_lev values)
        zf : ndarray
            Interface depths (n_lev + 1 values)
        z : ndarray
            Layer center depths (n_lev values)
    """
    # print("using tanh")
    dz = generate_tanhdz(min_dz, max_dz, n_lev, H)
    zf = zf_from_dz(dz, z0)
    z = z_from_zf(zf)
    return dz, zf, z