"""Water mass transformation calculations shared by the notebooks."""
import numpy as np
import xarray as xr


def get_SWMT_heat(ds):
    """Compute the surface water mass transformation (SWMT) caused by surface heat fluxes. It also
    splits this transformation into the contribution of each heat flux, such as sensible or latent
    heat.

    Parameters
    ----------
    ds : xarray.Dataset
        Water mass budget from xwmb, including the heat flux terms and the individual surface heat
        fluxes (in kg/s).

    Returns
    -------
    xarray.Dataset
        The heat-driven SWMT (SWMT_heat) and its contribution from each heat flux: latent, longwave,
        shortwave, sensible, frazil ice, and mass transfer. Also the sum of those contributions
        (SWMT_heat_approx) and what they miss (SWMT_heat_residual). All in kg/s.
    """
    ds_heat = (ds["boundary_fluxes_heat"] - ds["bottom_flux_heat"]).rename("SWMT_heat").to_dataset()
    ds_heat["SWMT_heat_latent"] = ds["surface_exchange_flux_nonadvective_latent_heat"]
    ds_heat["SWMT_heat_longwave"] = ds["surface_exchange_flux_nonadvective_longwave_heat"]
    ds_heat["SWMT_heat_shortwave"] = ds["surface_exchange_flux_nonadvective_shortwave_heat"]
    ds_heat["SWMT_heat_sensible"] = ds["surface_exchange_flux_nonadvective_sensible_heat"]
    ds_heat["SWMT_heat_frazil"] = ds["frazil_ice_heat"]
    ds_heat["SWMT_heat_mass_transfer"] = -ds["surface_exchange_flux_advective_mass_transfer_heat"]
    ds_heat["SWMT_heat_approx"] = (
        ds_heat["SWMT_heat_latent"] + ds_heat["SWMT_heat_longwave"] + ds_heat["SWMT_heat_shortwave"]
        + ds_heat["SWMT_heat_sensible"] + ds_heat["SWMT_heat_mass_transfer"] + ds_heat["SWMT_heat_frazil"]
    )
    ds_heat["SWMT_heat_residual"] = ds_heat["SWMT_heat"] - ds_heat["SWMT_heat_approx"]
    return ds_heat


def get_SWMT_salt(ds, combine_precipitation=True, combine_P_minus_E=True):
    """Compute the SWMT caused by surface salt and freshwater fluxes. It also splits this
    transformation into the contribution of each flux, such as sea ice formation or precipitation
    minus evaporation.

    Parameters
    ----------
    ds : xarray.Dataset
        Water mass budget from xwmb, including the salt flux terms and the individual surface
        freshwater and salt fluxes (in kg/s).
    combine_precipitation : bool, default True
        Combine rain/ice and snow into one precipitation term.
    combine_P_minus_E : bool, default True
        Combine precipitation and evaporation into precipitation minus evaporation.

    Returns
    -------
    xarray.Dataset
        The salt-driven SWMT (SWMT_salt) and its contribution from each flux: rivers, icebergs, sea
        ice, basal salt, and precipitation and evaporation. Also the sum of those contributions
        (SWMT_salt_approx) and what they miss (SWMT_salt_residual). All in kg/s.
    """
    ds_salt = (ds["surface_ocean_flux_advective_negative_rhs_salt"] + ds["surface_exchange_flux_salt"]).rename("SWMT_salt").to_dataset()
    ds_salt["SWMT_salt_rivers"] = -ds["surface_exchange_flux_advective_rivers_salt"]
    ds_salt["SWMT_salt_icebergs"] = -ds["surface_exchange_flux_advective_icebergs_salt"]
    ds_salt["SWMT_salt_sea_ice"] = -ds["surface_exchange_flux_advective_sea_ice_salt"]
    ds_salt["SWMT_salt_basal_salt"] = ds["surface_exchange_flux_nonadvective_basal_salt"]
    ds_salt["SWMT_salt_evaporation"] = -ds["surface_exchange_flux_advective_evaporation_salt"]

    if combine_precipitation:
        ds_salt["SWMT_salt_precipitation"] = -(
            ds["surface_exchange_flux_advective_rain_and_ice_salt"] + ds["surface_exchange_flux_advective_snow_salt"]
        )
        precipitation = ds_salt["SWMT_salt_precipitation"]
    else:
        ds_salt["SWMT_salt_rain_and_ice"] = -ds["surface_exchange_flux_advective_rain_and_ice_salt"]
        ds_salt["SWMT_salt_snow"] = -ds["surface_exchange_flux_advective_snow_salt"]
        precipitation = ds_salt["SWMT_salt_rain_and_ice"] + ds_salt["SWMT_salt_snow"]

    ds_salt["SWMT_salt_approx"] = (
        ds_salt["SWMT_salt_evaporation"] + precipitation + ds_salt["SWMT_salt_rivers"]
        + ds_salt["SWMT_salt_sea_ice"] + ds_salt["SWMT_salt_icebergs"] + ds_salt["SWMT_salt_basal_salt"]
    )

    if combine_P_minus_E:
        ds_salt["SWMT_salt_precip_minus_evaporation"] = precipitation + ds_salt["SWMT_salt_evaporation"]
        ds_salt = ds_salt.drop_vars(
            [v for v in ["SWMT_salt_precipitation", "SWMT_salt_evaporation"] if v in ds_salt]
        )

    ds_salt["SWMT_salt_residual"] = ds_salt["SWMT_salt"] - ds_salt["SWMT_salt_approx"]
    return ds_salt


def get_SWMT(ds):
    """Compute the total SWMT from the water mass budget. The heat and salt contributions from
    `get_SWMT_heat` and `get_SWMT_salt` are returned alongside it.

    Parameters
    ----------
    ds : xarray.Dataset
        Water mass budget from xwmb with the terms needed by `get_SWMT_heat` and
        `get_SWMT_salt` (in kg/s).

    Returns
    -------
    xarray.Dataset
        The total SWMT, together with its heat and salt contributions from `get_SWMT_heat` and
        `get_SWMT_salt`. All in kg/s.
    """
    exact_SWMT = ds["boundary_fluxes"] - ds["bottom_flux_heat"]
    return xr.merge([get_SWMT_salt(ds), get_SWMT_heat(ds), exact_SWMT.rename("SWMT")])


def decompose_SWMT_spatial(f, g, dims=("lat", "lon")):
    """Split the change in integrated SWMT between two periods into the three terms of the spatial
    decomposition in Methods. The terms are the change over outcrops present in both periods
    (integrand_change), the transformation gained from outcrops that only appear later
    (domain_shift_gain), and the transformation lost from outcrops that disappear
    (domain_shift_loss); cells where a field is NaN are not outcrops.

    Parameters
    ----------
    f, g : xarray.DataArray
        Time-mean SWMT per cell in the reference and the later period.
    dims : tuple of str, default ("lat", "lon")
        Horizontal dimensions to integrate over.

    Returns
    -------
    xarray.Dataset
        The three terms (integrand_change, domain_shift_gain, domain_shift_loss) and their sum
        (reconstructed_total), which matches the directly computed change (true_total). Also the
        SWMT integrated over each period (integrate_f_wf, integrate_g_wg) and the outcrop masks
        used for each term (D_f, D_g, D_intersect, D_f_minus_g, D_g_minus_f).
    """
    D_f, D_g = f.notnull(), g.notnull()
    D_intersect = D_f & D_g
    D_f_minus_g = D_f & ~D_g
    D_g_minus_f = D_g & ~D_f

    def integrate(field, mask):
        """Sum `field` over the cells where `mask` is True, along `dims`."""
        return field.where(mask).sum(dim=dims)

    integrand_change = integrate(g - f, D_intersect)
    domain_shift_gain = integrate(g, D_g_minus_f)
    domain_shift_loss = integrate(-f, D_f_minus_g)
    integrate_f_wf = integrate(f, D_f)
    integrate_g_wg = integrate(g, D_g)

    return xr.Dataset({
        "domain_shift_gain": domain_shift_gain,
        "domain_shift_loss": domain_shift_loss,
        "integrand_change": integrand_change,
        "reconstructed_total": integrand_change + domain_shift_gain + domain_shift_loss,
        "true_total": integrate_g_wg - integrate_f_wf,
        "D_f_minus_g": D_f_minus_g,
        "D_g_minus_f": D_g_minus_f,
        "D_intersect": D_intersect,
        "integrate_g_wg": integrate_g_wg,
        "integrate_f_wf": integrate_f_wf,
        "D_f": D_f,
        "D_g": D_g,
    })


def select_from_location(ds, ds_loc, dim="sigma2_l_target", method=None):
    """Pick out a field at a density that changes from year to year. This follows a quantity at the
    DSW formation or export density, which is found separately for each experiment and year.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Field with dimensions exp ("control", "forced"), year, and `dim`.
    ds_loc : xarray.DataArray
        Density to select for each exp and year; NaN where undefined.
    dim : str, default "sigma2_l_target"
        Density dimension of `ds`.
    method : str, optional
        Passed to `.sel`, e.g. "nearest" when the densities don't match exactly.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        `ds` at the selected density, with dimensions exp and year; NaN where `ds_loc` is NaN.
    """
    results = []
    for exp in ["control", "forced"]:
        exp_data = []
        for y in ds.year:
            ds_sub = ds.sel(exp=exp, year=y)
            loc = ds_loc.sel(exp=exp, year=y).values
            if np.isnan(loc):
                exp_data.append(ds_sub.isel({dim: 0}).where(False))
            else:
                exp_data.append(ds_sub.sel({dim: loc}, method=method))
        results.append(xr.concat(exp_data, dim="year"))
    return xr.concat(results, dim="exp")
