"""Reusable plotting helpers for inverse-Gaussian TTD examples."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import seaborn as sns
from geom_median.numpy import compute_geometric_median

from utils import credible_interval


advective_timescale = lambda gamma, delta: gamma
delta_gamma_ratio = lambda gamma, delta: delta / gamma
diffusive_timescale = lambda gamma, delta: delta**2 / (2.0 * gamma)
peclet_number = lambda gamma, delta: delta**2 / (2.0 * gamma**2)


def plot_distribution(
    ax,
    x,
    ds,
    color,
    label=None,
    legend=None,
    dim="sample",
    ls="-",
    lw=2,
    alpha=0.35,
):
    """Plot a median line and credible interval band from sampled values."""
    median, lower, upper = credible_interval(ds, dim=dim)
    line, = ax.plot(x, median, color=color, ls=ls, lw=lw)
    band = ax.fill_between(x, lower, upper, color=color, alpha=alpha)
    handle = (line, band)
    if legend is not None and label is not None:
        legend[label] = handle
    return handle


def plot_hist_distribution(
    ax,
    prior_values,
    posterior_values,
    prior_color="tab:orange",
    posterior_color="tab:blue",
    hist_kws=None,
):
    """Overlay prior and posterior one-dimensional distributions."""
    hist_kws = {} if hist_kws is None else dict(hist_kws)
    sns.histplot(
        prior_values,
        edgecolor=prior_color,
        color=prior_color,
        label="Prior",
        ax=ax,
        **hist_kws,
    )
    sns.histplot(
        posterior_values,
        edgecolor=posterior_color,
        color=posterior_color,
        label="Posterior",
        ax=ax,
        **hist_kws,
    )


def _source_values(samples, source_dim="source"):
    if source_dim in samples.dims:
        return list(samples[source_dim].values)
    return [None]


def _sel_source(da, source, source_dim="source"):
    return da.sel({source_dim: source}) if source is not None and source_dim in da.dims else da


def plot_parameter_marginals(
    prior,
    posterior,
    parameters,
    *,
    source_dim="source",
    prior_color="tab:orange",
    posterior_color="tab:blue",
    hist_kws=None,
    figsize_per_source=5.2,
):
    """Plot prior/posterior marginal distributions for any parameters and sources."""
    sources = _source_values(posterior[parameters[0]], source_dim=source_dim)
    n_sources = len(sources)
    n_parameters = len(parameters)
    hist_kws = {} if hist_kws is None else dict(hist_kws)
    titles = {
        "gamma": r"$\mathbf{\Gamma}$",
        "delta": r"$\mathbf{\Delta}$",
        "mass_fraction": "Mass Fraction",
    }
    units = {
        "gamma": "years",
        "delta": "years",
        "mass_fraction": "unitless",
    }

    fig, axes = plt.subplots(
        n_parameters,
        n_sources,
        figsize=(figsize_per_source * n_sources, 3.0 * n_parameters),
        squeeze=False,
        sharey="row",
    )
    for col, source in enumerate(sources):
        source_label = source if source is not None else "Source"
        for row, parameter in enumerate(parameters):
            ax = axes[row, col]
            prior_values = _sel_source(prior[parameter], source, source_dim=source_dim)
            posterior_values = _sel_source(posterior[parameter], source, source_dim=source_dim)
            prior_handle = plot_hist_with_median(
                ax, prior_values, prior_color, "Prior", hist_kws=hist_kws
            )
            posterior_handle = plot_hist_with_median(
                ax, posterior_values, posterior_color, "Posterior", hist_kws=hist_kws
            )
            ax.set_title(
                f"{source_label} {titles.get(parameter, parameter.replace('_', ' ').title())}",
                fontweight="bold",
            )
            ax.set_xlabel(units.get(parameter, ""))
            ax.grid(alpha=0.4)
            ax.legend(
                [prior_handle, posterior_handle],
                [
                    f"Prior median = {float(prior_values.median()):.2f}",
                    f"Posterior median = {float(posterior_values.median()):.2f}",
                ],
                handler_map={tuple: HandlerTuple(ndivide=None)},
            )
    fig.tight_layout()
    return fig, axes


def plot_variable_marginals(
    prior,
    posterior,
    variables,
    titles,
    *,
    source=None,
    source_dim="source",
    ncol=4,
    prior_color="tab:orange",
    posterior_color="tab:blue",
    hist_kws=None,
    figsize=(15, 4),
):
    """Plot prior/posterior marginals for named variables in an nrow x ncol grid."""
    hist_kws = {} if hist_kws is None else dict(hist_kws)
    ncol = min(ncol, len(variables))
    nrow = int(np.ceil(len(variables) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0], figsize[1] * nrow), squeeze=False)

    for ax, name, title in zip(axes.ravel(), variables, titles, strict=False):
        prior_values = _sel_source(prior[name], source, source_dim=source_dim)
        posterior_values = _sel_source(posterior[name], source, source_dim=source_dim)
        prior_handle = plot_hist_with_median(
            ax, prior_values, prior_color, "Prior", hist_kws=hist_kws
        )
        posterior_handle = plot_hist_with_median(
            ax, posterior_values, posterior_color, "Posterior", hist_kws=hist_kws
        )
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.4)
        ax.legend(
            [prior_handle, posterior_handle],
            [
                f"Prior median = {float(prior_values.median()):.2f}",
                f"Posterior median = {float(posterior_values.median()):.2f}",
            ],
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )

    for ax in axes.ravel()[len(variables):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_hist_with_median(ax, values, color, label, hist_kws=None):
    """Plot a histogram/KDE and median line as one legend handle."""
    hist_kws = {} if hist_kws is None else dict(hist_kws)
    hist_kws.setdefault("line_kws", {"color": color, "lw": 1.5})
    sns.histplot(
        values,
        edgecolor=color,
        color=color,
        label=label,
        ax=ax,
        **hist_kws,
    )
    median_line = ax.axvline(float(values.median()), c=color, linestyle="-", lw=1.5)
    patch = ax.patches[-1] if ax.patches else plt.Line2D([], [], color=color, lw=6)
    return patch, median_line


def plot_gamma_delta_joint(
    posterior, *, fig=None, gs=None, axes=None, source_label=None,
    posterior_color="tab:blue", bins=40, cmap=None,
    figsize_per_source=(7.2, 5.6), suptitle=True,
):
    cmap = _default_joint_cmap(posterior_color) if cmap is None else cmap
    fig, axes = _get_gamma_delta_fig_axes(fig, gs, axes, figsize_per_source)

    _plot_one_gamma_delta_joint(
        axes,
        gamma=posterior["gamma"],
        delta=posterior["delta"],
        source_label="Source" if source_label is None else source_label,
        posterior_color=posterior_color,
        bins=bins,
        cmap=cmap,
    )

    if suptitle:
        fig.suptitle(
            r"Joint posterior distributions of $\mathbf{\Gamma}$ and $\mathbf{\Delta}$",
            fontsize=15,
            fontweight="bold",
        )

    return fig, axes


def plot_mixed_gamma_delta_joint(
    posterior, *, fig=None, gs=None, axes=None, source_label=None,
    posterior_color="tab:blue", bins=40, cmap=None,
    figsize_per_source=(7.2, 5.6), suptitle=True,
):
    cmap = _default_joint_cmap(posterior_color) if cmap is None else cmap
    fig, axes = _get_gamma_delta_fig_axes(fig, gs, axes, figsize_per_source)

    _plot_one_gamma_delta_joint(
        axes,
        gamma=posterior["gamma_mix"],
        delta=posterior["delta_mix"],
        source_label=source_label,
        posterior_color=posterior_color,
        bins=bins,
        cmap=cmap,
    )

    if suptitle:
        fig.suptitle(
            r"Joint posterior distributions of $\mathbf{\Gamma}$ and $\mathbf{\Delta}$",
            fontsize=15,
            fontweight="bold",
        )

    return fig, axes


def _get_gamma_delta_fig_axes(fig=None, gs=None, axes=None, figsize=(7.2, 5.6)):
    if axes is not None:
        return axes[1].figure, axes

    fig = plt.figure(figsize=figsize, constrained_layout=True) if fig is None else fig
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=(4.0, 0.18, 1.15),
        height_ratios=(1.15, 4.0),
        wspace=0.06, hspace=0.06,
    ) if gs is None else gs

    ax_g = fig.add_subplot(gs[0, 0])
    ax_j = fig.add_subplot(gs[1, 0], sharex=ax_g)
    cax = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2], sharey=ax_j)

    return fig, (ax_g, ax_j, ax_d, cax)


def _plot_one_gamma_delta_joint(
    axes, *, gamma, delta, source_label, posterior_color, bins, cmap,
):
    ax_g, ax_j, ax_d, cax = axes

    gamma_geo_med, delta_geo_med = compute_geometric_median(
        np.column_stack([gamma.to_numpy(), delta.to_numpy()])
    ).median

    sns.histplot(x=gamma, y=delta, bins=bins, cmap=cmap, cbar=True, cbar_ax=cax, ax=ax_j, stat="probability")
    sns.histplot(x=gamma, bins=35, stat="probability", alpha=0.35, kde=True, color=posterior_color, ax=ax_g)
    sns.histplot(y=delta, bins=35, stat="probability", alpha=0.35, kde=True, color=posterior_color, ax=ax_d)

    ax_j.scatter(
        gamma_geo_med, delta_geo_med,
        color=posterior_color, marker="o", s=120, lw=1.0,
        edgecolor="k", zorder=100,
        label="Geometric posterior median\n"
        + rf"$\Gamma, \Delta$ = {gamma_geo_med:.1f}, {delta_geo_med:.1f}",
    )

    prefix = "" if source_label is None else f"{source_label} "
    ax_g.set(title=rf"{prefix}$\mathbf{{\Gamma}}$", ylabel="Density")
    ax_d.set(title=rf"{prefix}$\mathbf{{\Delta}}$", xlabel="Density")
    ax_j.set(xlabel=r"$\mathbf{\Gamma}$ [years]", ylabel=r"$\mathbf{\Delta}$ [years]")
    cax.set_ylabel("Count")

    for ax in (ax_g, ax_d):
        ax.title.set(fontsize=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    ax_j.xaxis.label.set_fontweight("bold")
    ax_j.yaxis.label.set_fontweight("bold")
    ax_g.tick_params(labelbottom=False)
    ax_d.tick_params(labelleft=False)
    ax_j.legend(loc="lower right", frameon=True)

    return axes


def _default_joint_cmap(color):
    return mcolors.LinearSegmentedColormap.from_list(
        f"joint_{mcolors.to_hex(color).replace('#', '')}",
        ["white", mcolors.to_rgba(color)],
    )