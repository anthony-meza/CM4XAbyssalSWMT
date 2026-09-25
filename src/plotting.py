"""Figure helpers shared by the fig* notebooks."""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.path import Path
from matplotlib.ticker import FixedLocator, FormatStrFormatter

to_tg = 1e-9  # kg/s -> Tg/s (1 Tg = 1e9 kg)

expt_colors = {"control": "#2f74b3", "forced": "#cc415a"}  # line colors for each experiment
expt_labels = {"control": "piControl", "forced": "Historical/SSP5-8.5"}  # legend labels for each experiment


def set_style(dpi=150):
    """Apply the plotting style shared by all figures. It sets the seaborn "ticks" theme and the on-
    screen figure resolution.

    Parameters
    ----------
    dpi : int, default 150
        Figure resolution on screen (saved figures set their own dpi).
    """
    sns.set_theme(context="notebook", style="ticks")
    mpl.rcParams["figure.dpi"] = dpi


def make_center_extended_cmap(levels, cmap_name="RdBu_r", center_value=0.0, zero_color=None):
    """Build a discrete colormap and norm for fields with both signs, such as anomalies. The two
    bins on either side of `center_value` share one neutral color, so values close to zero don't
    stand out.

    Parameters
    ----------
    levels : array-like
        Bin edges; must contain `center_value`, but not as the first or last edge.
    cmap_name : str, default "RdBu_r"
        Matplotlib colormap to sample.
    center_value : float, default 0.0
        Value whose neighbouring bins are drawn in `zero_color`.
    zero_color : color, optional
        Color of the two center bins; defaults to the middle of the colormap.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
    norm : matplotlib.colors.BoundaryNorm
    """
    levels = np.asarray(levels)
    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(0, 1, len(levels) - 1))
    zero_color = base(0.5) if zero_color is None else mcolors.to_rgba(zero_color)

    center_matches = np.where(np.isclose(levels, center_value))[0]
    if len(center_matches) == 0:
        raise ValueError(f"center_value={center_value} must appear in levels.")
    center_idx = center_matches[0]
    if center_idx == 0 or center_idx == len(levels) - 1:
        raise ValueError("center_value cannot be the first or last level.")

    colors[center_idx - 1:center_idx + 1] = zero_color
    cmap = mcolors.ListedColormap(colors)
    return cmap, mcolors.BoundaryNorm(levels, cmap.N)


land_feature = cfeature.NaturalEarthFeature("physical", "land", "110m", facecolor="lightgrey", edgecolor="lightgrey")


def plot_antarctic(
    ds,
    ax,
    *,
    cmap=None,
    norm=None,
    mask_threshold=1e-8,
    draw_gridlines=True,
    gridline_lons=range(-180, 180, 60),
    gridline_lats=range(-90, -60, 10),
    draw_labels=True,
    plot_kwargs=None,
    label_fontsize=10,
):
    """Draw a field on a circular south polar map that extends from the pole to 60S. It adds land
    and, optionally, gridlines with longitude and latitude labels, in the style of the map figures.

    Parameters
    ----------
    ds : xarray.DataArray
        Field with `geolon` and `geolat` coordinates.
    ax : cartopy GeoAxes
        Axes with a south polar projection.
    cmap, norm : optional
        Colormap and norm for pcolormesh.
    mask_threshold : float or None, default 1e-8
        Values with magnitude at or below this are left blank; None plots everything.
    draw_gridlines : bool, default True
        Draw longitude and latitude lines.
    gridline_lons, gridline_lats : iterable of float
        Longitudes and latitudes of the gridlines (in degrees).
    draw_labels : bool, default True
        Label the gridlines.
    plot_kwargs : dict, optional
        Extra arguments for pcolormesh.
    label_fontsize : float, default 10
        Font size of the gridline labels.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The plotted mesh, e.g. for a colorbar.
    """
    plate_carree = ccrs.PlateCarree()  # coordinate system of geolon/geolat
    theta = np.linspace(0, 2 * np.pi, 200)
    antarctic_circle = Path(np.c_[np.sin(theta), np.cos(theta)] * 0.5 + 0.5)  # circular map boundary
    max_lat = -60

    plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
    plot_kwargs.setdefault("cmap", cmap)
    plot_kwargs.setdefault("norm", norm)
    plot_kwargs.setdefault("shading", "auto")
    plot_kwargs.setdefault("rasterized", True)
    plot_kwargs["transform"] = plate_carree

    field = ds if mask_threshold is None else ds.where(np.abs(ds) > mask_threshold)

    ax.set_extent([-180, 180, -90, max_lat], plate_carree)
    ax.set_boundary(antarctic_circle, transform=ax.transAxes)

    mesh = ax.pcolormesh(
        field.geolon,
        field.geolat,
        field,
        **plot_kwargs,
    )

    ax.add_feature(land_feature, zorder=1)

    if draw_gridlines:
        gl = ax.gridlines(
            draw_labels=False,
            color="gray",
            alpha=0.15,
            linewidth=1,
            linestyle="-",
            zorder=10,
        )
        gl.xlocator = mticker.FixedLocator(gridline_lons)
        gl.ylocator = mticker.FixedLocator(gridline_lats)

        if draw_labels:
            for lon in gridline_lons:
                if lon == -180:
                    continue

                lon_wrapped = ((lon + 180) % 360) - 180
                rot = -lon
                rot = rot - 180 if rot > 90 else rot + 180 if rot < -90 else rot

                label = (
                    "0°" if lon_wrapped == 0 else
                    "180°" if abs(lon_wrapped) == 180 else
                    f"{abs(lon_wrapped):g}°{'E' if lon_wrapped > 0 else 'W'}"
                )

                ax.text(
                    lon,
                    max_lat + 1.5,
                    label,
                    rotation=rot,
                    color="gray",
                    fontsize=label_fontsize,
                    ha="center",
                    va="center",
                    transform=plate_carree,
                )

            for lat in gridline_lats:
                if -90 < lat < max_lat:
                    ax.text(
                        105,
                        lat + 1.5,
                        f"{abs(lat):g}°S",
                        rotation=75,
                        color="gray",
                        fontsize=label_fontsize,
                        ha="center",
                        va="center",
                        transform=plate_carree,
                    )

    return mesh


def set_uniform_tickscale_y(ax, yticks, *, fmt="%.2f"):
    """Place unevenly spaced y ticks at even intervals along an axis. Values between ticks are
    positioned by linear interpolation, which stretches part of the range, e.g. near zero in Figure
    12(c).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    yticks : array-like
        Strictly increasing tick values; the axis spans the first to the last.
    fmt : str, default "%.2f"
        Tick label format.

    Returns
    -------
    matplotlib.axes.Axes
    """
    yticks = np.asarray(yticks, dtype=float)
    ypos = np.arange(yticks.size, dtype=float)
    if np.any(np.diff(yticks) <= 0):
        raise ValueError("yticks must be strictly increasing.")

    ax.set_yscale("function", functions=(
        lambda y: np.interp(y, yticks, ypos),
        lambda yp: np.interp(yp, ypos, yticks),
    ))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    ax.set_ylim(yticks[0], yticks[-1])
    return ax


def plot_row(
    fig, axs, var, levels, cmap, year, units, cbar_label, section_titles, timeseries_titles, s_tie,
    z=slice(None), scale=1.0, contour_fmt="%g", surf_loc=(380, -430), bot_loc=(680, -3000),
    plot_contours=True, clabel_colors="k", clabel_fontsize=10,
    hide_section_xlabel=False, cbar_width=0.014, cbar_pad=0.06,
):
    """Plot one row of Figure 11 for a single variable, such as sigma2 or SF6. The row shows the
    transect section in both experiments for one year, plus time series at a point on the
    continental slope and a point in the deep ocean.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    axs : dict of Axes
        Axes named "section_control", "section_forced", "surf" (slope time series), and "bot" (deep time series).
    var : xarray.DataArray
        Transect field with dimensions expt, year, depth, and transect_length.
    levels : array-like
        Contour levels for the sections.
    cmap : matplotlib.colors.Colormap
    year : int
        Year shown in the sections.
    units : str
        Units for the time-series y labels.
    cbar_label : str
        Colorbar label.
    section_titles, timeseries_titles : list of str
        Titles for the two sections and the two time series.
    s_tie : array-like
        Distance of the tie points along the transect (in km), marked under each section.
    z : slice, optional
        Depth range to plot (in m).
    scale : float, default 1.0
        Factor applied to `var` (unit conversion).
    contour_fmt : str, default "%g"
        Format of contour and colorbar labels.
    surf_loc, bot_loc : tuple
        Location of the slope and deep time-series points, as (distance in km, depth in m).
    plot_contours : bool, default True
        Draw labelled contour lines on the sections.
    clabel_colors, clabel_fontsize : optional
        Color and size of the contour labels.
    hide_section_xlabel : bool, default False
        Hide the x axis label of the lower section.
    cbar_width, cbar_pad : float
        Colorbar width and gap to the sections, in figure coordinates.
    """
    section_specs = [
        (axs["section_control"], "control", section_titles[0]),
        (axs["section_forced"], "forced", section_titles[1]),
    ]
    site_specs = {
        "surf": {"axis": axs["surf"], "loc": surf_loc, "label": "slope", "title": timeseries_titles[0],
                 "marker": "*", "ms": 20, "text_dx": -180, "text_dz": 600},
        "bot": {"axis": axs["bot"], "loc": bot_loc, "label": "deep", "title": timeseries_titles[1],
                "marker": "o", "ms": 16, "text_dx": -260, "text_dz": 100},
    }
    expt_colors = {"control": "grey", "forced": "k"}
    expt_labels = {"control": "piControl", "forced": "Historical/SSP5-8.5"}

    var = var.sel(depth=z) * scale
    v = var.sel(year=year)

    cmap = cmap.copy()
    n_regions = len(levels) - 1 + 2
    if cmap.N < n_regions:
        cmap = cmap.resampled(n_regions)
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend="both")

    for ax, expt, title in section_specs:
        da = v.sel(expt=expt).transpose("depth", "transect_length")
        ax.contourf(da.transect_length, np.abs(da.depth), da, cmap=cmap, norm=norm, levels=levels, extend="both")

        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.scatter(s_tie, 0.02 + np.zeros_like(s_tie), edgecolor="k", c="limegreen", marker="^",
                   transform=trans, clip_on=False, s=100, zorder=20)

        if plot_contours:
            cs = ax.contour(da.transect_length, np.abs(da.depth), da, levels=levels,
                            colors=clabel_colors, linewidths=0.8)
            ax.clabel(cs, cs.levels, inline=True, fontsize=clabel_fontsize, fmt=contour_fmt)

        for site in site_specs.values():
            x, z0 = site["loc"]
            ax.plot(x, np.abs(z0), marker=site["marker"], ms=site["ms"], mfc="yellow", mec="k", mew=0.8,
                    zorder=10, alpha=0.8)
            ax.text(x + site["text_dx"], np.abs(z0 - site["text_dz"]), site["label"], ha="left", va="center",
                    color="yellow", fontweight="bold", zorder=11,
                    path_effects=[pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()])

        ax.invert_yaxis()
        ax.set_facecolor("grey")
        ax.set_ylabel("Depth [m]")
        ax.set_title(title, loc="left", fontweight="bold")

    axs["section_control"].set_xlabel("")
    axs["section_control"].tick_params(labelbottom=False)
    if hide_section_xlabel:
        axs["section_forced"].set_xlabel("")
        axs["section_forced"].tick_params(labelbottom=False)
    else:
        axs["section_forced"].set_xlabel("Offshore Distance [km]")

    # stack the two sections with a small gap, and put the colorbar to their left
    section_gap = 0.05
    pos0 = axs["section_control"].get_position()
    pos1 = axs["section_forced"].get_position()
    bottom, top = pos1.y0, pos0.y1
    height = (top - bottom - section_gap) / 2
    axs["section_forced"].set_position([pos1.x0, bottom, pos1.width, height])
    axs["section_control"].set_position([pos0.x0, bottom + height + section_gap, pos0.width, height])

    cbar_height = (top - bottom) * 0.7
    cbar_bottom = bottom + 0.5 * ((top - bottom) - cbar_height)
    cax = fig.add_axes([pos0.x0 - cbar_pad - cbar_width, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical",
                        ticks=levels, spacing="uniform", extend="both")
    cbar.set_ticklabels([contour_fmt % val for val in levels])
    cbar.set_label(cbar_label)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    for site in site_specs.values():
        x, z0 = site["loc"]
        ts = var.interp(transect_length=x, depth=z0, method="nearest", kwargs={"fill_value": np.nan})
        ax = site["axis"]
        for expt, color in expt_colors.items():
            ax.plot(ts.year, ts.sel(expt=expt), c=color, label=expt_labels[expt])
        ax.set_title(site["title"], loc="left", fontweight="bold")
        ax.set_ylabel(f"[{units}]")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    axs["surf"].tick_params(labelbottom=False)
