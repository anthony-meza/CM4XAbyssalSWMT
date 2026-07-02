import numpy as np
from matplotlib.path import Path
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PROJ = ccrs.PlateCarree()

theta = np.linspace(0, 2 * np.pi, 200)
ANTARCTIC_CIRCLE = Path(np.c_[np.sin(theta), np.cos(theta)] * 0.5 + 0.5)

LAND = cfeature.NaturalEarthFeature(
    "physical",
    "land",
    "110m",
    facecolor="lightgrey",
    edgecolor = "lightgrey", 
)


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
    label_fontsize = 10, 
):
    max_lat = -60

    plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
    plot_kwargs.setdefault("cmap", cmap)
    plot_kwargs.setdefault("norm", norm)
    plot_kwargs.setdefault("shading", "auto")
    plot_kwargs.setdefault("rasterized", True)
    plot_kwargs["transform"] = PROJ

    field = ds if mask_threshold is None else ds.where(np.abs(ds) > mask_threshold)

    ax.set_extent([-180, 180, -90, max_lat], PROJ)
    ax.set_boundary(ANTARCTIC_CIRCLE, transform=ax.transAxes)

    mesh = ax.pcolormesh(
        field.geolon,
        field.geolat,
        field,
        **plot_kwargs,
    )

    # ax.coastlines(linewidth=0.5, color = "lightgrey")
    ax.add_feature(LAND, zorder=1)

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

                lw = ((lon + 180) % 360) - 180
                rot = -lon
                rot = rot - 180 if rot > 90 else rot + 180 if rot < -90 else rot

                label = (
                    "0°" if lw == 0 else
                    "180°" if abs(lw) == 180 else
                    f"{abs(lw):g}°{'E' if lw > 0 else 'W'}"
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
                    transform=PROJ,
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
                        transform=PROJ,
                    )

    return mesh