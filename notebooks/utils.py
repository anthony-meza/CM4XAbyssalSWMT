"""Shared notebook helpers for Bayesian TTD examples."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import arviz as az
import numpy as np
import xarray as xr

__all__ = [
    "credible_interval",
    "flatten_numpyro_output",
    "mse_score",
    "numpyro_likelihood_score",
    "relabel_source_vars",
]

def _get_arviz_group_as_dataset(idata, group):
    """Return an ArviZ group as an xarray.Dataset for old and new ArviZ APIs."""
    obj = getattr(idata, group, None)

    if obj is None:
        obj = idata[group]

    if isinstance(obj, xr.Dataset):
        return obj

    if hasattr(obj, "ds"):
        return obj.ds

    if hasattr(obj, "to_dataset"):
        return obj.to_dataset()

    raise TypeError(
        f"Could not convert group {group!r} of type {type(obj)} "
        "to xarray.Dataset."
    )


def flatten_numpyro_output(
    group="posterior",
    sample=None,
    source=None,
    source_names=("gamma", "delta_gamma_ratio", "mass_fraction"),
    **from_numpyro_kwargs,
):
    """Flatten NumPyro/ArviZ chain and draw dimensions into one sample axis."""

    if (
        "posterior" not in from_numpyro_kwargs
        and "sample_dims" not in from_numpyro_kwargs
    ):
        from_numpyro_kwargs["sample_dims"] = ["sample"]

    idata = az.from_numpyro(**from_numpyro_kwargs)
    output = _get_arviz_group_as_dataset(idata, group)

    if {"chain", "draw"}.issubset(output.dims):
        output = output.stack(sample=("chain", "draw")).reset_index(
            "sample",
            drop=True,
        )
    elif "sample" in output.dims:
        pass
    elif "draw" in output.dims:
        output = output.rename({"draw": "sample"})
    else:
        raise ValueError(
            f"Cannot infer sample dimension from dims: {dict(output.sizes)}"
        )

    if sample is not None:
        output = output.assign_coords(sample=sample)

    if source is not None:
        output = relabel_source_vars(output, source, names=source_names)

    return output
    
def relabel_source_vars(ds, source, names=("gamma", "delta_gamma_ratio", "mass_fraction")):
    """Rename NumPyro vector dimensions to ``source`` and attach source labels."""
    ds = ds.copy()
    for name in names:
        if name not in ds:
            continue
        da = ds[name]
        non_sample_dims = [dim for dim in da.dims if dim != "sample"]
        if non_sample_dims and non_sample_dims[0] != "source":
            da = da.rename({non_sample_dims[0]: "source"})
        if "source" in da.dims:
            da = da.assign_coords(source=source)
        ds[name] = da
    return ds


def credible_interval(ds, dim="sample"):
    """Return median, 2.5%, and 97.5% quantiles along a sample dimension."""
    return (
        ds.quantile(0.5, dim=dim),
        ds.quantile(0.025, dim=dim),
        ds.quantile(0.975, dim=dim),
    )


def mse_score(predicted, observed, error, dim=None):
    """Return sample-wise MSE and RMSE between predicted and observed values."""
    residual = (predicted - observed) / error
    if dim is None:
        dim = [name for name in residual.dims if name != "sample"]
    mse = (residual**2).mean(dim=dim)
    return xr.Dataset({"mse": mse, "rmse": np.sqrt(mse)})


def numpyro_likelihood_score(
    model,
    posterior_samples,
    *model_args,
    obs_site="obs",
    sample_coord=None,
    **model_kwargs,
):
    """Evaluate posterior log-likelihood scores from NumPyro posterior samples."""
    from numpyro.infer.util import log_likelihood

    ll_pointwise = np.asarray(
        log_likelihood(model, posterior_samples, *model_args, **model_kwargs)[obs_site]
    )
    n_observation = int(np.prod(ll_pointwise.shape[1:]))
    ll = ll_pointwise.reshape(ll_pointwise.shape[0], -1).sum(axis=1)
    sample_coord = np.arange(ll.size) if sample_coord is None else sample_coord
    log_likelihood_da = xr.DataArray(
        ll,
        dims="sample",
        coords={"sample": sample_coord},
        name="log_likelihood",
    )
    return xr.Dataset(
        {
            "log_likelihood": log_likelihood_da,
            "mean_log_likelihood": log_likelihood_da / n_observation,
            "deviance": -2.0 * log_likelihood_da,
        }
    )
