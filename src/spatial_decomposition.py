import xarray as xr
from typing import Sequence, Hashable
import numpy as np
def decompose_SWMT_spatial(
    f: xr.DataArray,
    g: xr.DataArray,
    dims: Sequence[Hashable] = ("lat", "lon"),
) -> xr.Dataset:
    """
    Perform a six‑term discrete decomposition of the change
        Σ_{i∈D_g} g[i] * w_g[i]
      − Σ_{i∈D_f} f[i] * w_f[i]

    If D_f or D_g is None, it is assumed to be True everywhere.
    If w_f or w_g is None, it is assumed to be 1 everywhere.

    Parameters
    ----------
    f : xr.DataArray
        Original field defined on at least the coords in `dims`.
    g : xr.DataArray
        Updated field on the same coords as `f`.
    D_f : xr.DataArray (bool), optional
        Mask of the “old” domain.  Defaults to all True.
    D_g : xr.DataArray (bool), optional
        Mask of the “new” domain.  Defaults to all True.
    w_f : xr.DataArray, optional
        Original weights.  Defaults to 1 everywhere.
    w_g : xr.DataArray, optional
        Updated weights.  Defaults to 1 everywhere.
    dims : sequence of hashable, default=("lat","lon")
        The dimensions over which to integrate / sum.

    Returns
    -------
    xr.Dataset
        DataVariables:
          domain_shift,
          integrand_change,
          weight_change,
          overlap_interaction,
          new_domain_interaction,
          total_decomposed,
          total_direct,
          D_f_minus_g,
          D_g_minus_f,
          D_intersect
    """
    # --- set defaults ---
    w_f = xr.ones_like(f)
    w_g = xr.ones_like(g)

    # full‐domain mask over provided dims
    coords = {d: f.coords[d] for d in dims}
    
    D_f = ~np.isnan(f)
    D_g = ~np.isnan(g)

    # --- perturbations ---
    df = g - f
    dw = w_g - w_f

    # --- sub‑domains ---
    D_f_minus_g = D_f & ~D_g
    D_g_minus_f = D_g & ~D_f
    D_intersect     = D_f & D_g

    # helper to sum only where mask is True
    def integrate(field: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
        return field.where(mask).sum(dim=dims)
    # 1) Domain shift, same weights and integrand
    domain_shift_loss = integrate(-f * w_f, D_f_minus_g)

    domain_shift_gain = integrate(g * w_g, D_g_minus_f)
    # 1) is equivalent to:   
    # domain_shift = (
    #     integrate(f * w_f, D_g)
    #   - integrate(f * w_f, D_f)
    # )
    # 2) Flux change on overlap
    integrand_change = integrate(df * w_f, D_intersect)

    # totals
    reconstructed_total = (
        integrand_change
      + domain_shift_gain
      + domain_shift_loss
    )

    integrate_g_wg = integrate(g * w_g, D_g)
    integrate_f_wf = integrate(f * w_f, D_f)

    true_total = (
        integrate_g_wg
      - integrate_f_wf
    )
    
    return xr.Dataset({
        "domain_shift_gain":           domain_shift_gain,
        "domain_shift_loss":domain_shift_loss, 
        "integrand_change":       integrand_change,
        "reconstructed_total":       reconstructed_total,
        "true_total":           true_total,
        "D_f_minus_g":        D_f_minus_g,
        "D_g_minus_f":        D_g_minus_f,
        "D_intersect":            D_intersect,
        "integrate_g_wg": integrate_g_wg, 
        "integrate_f_wf": integrate_f_wf, 
        "D_f":D_f, 
        "D_g":D_g, 
    })
