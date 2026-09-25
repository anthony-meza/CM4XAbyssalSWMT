# Dense Shelf Water Surface Water Mass Transformation in CM4X

This repository contains the code for "Sea ice-driven decline of abyssal ocean ventilation in an eddy-permitting coupled climate model" by Anthony Meza, Henri Drake, and Geoffrey Gebbie (in preparation for *JGR: Oceans*). It computes water mass transformation (WMT) budgets of Dense Shelf Water (DSW) on the Antarctic Continental Shelf (ACS) in the GFDL CM4X climate model and reconstructs the analysis and figures in the paper.

## Walkthrough

### Data processing

- `01_prepare_dsw_wmt_budget.ipynb` computes the WMT budget of the ACS (south of 50°S, shallower than 1000 m) from the monthly CM4X budget output, 1850–2099.
- `02_prepare_spatial_swmt.ipynb` computes yearly maps of surface water mass transformation (SWMT) on the ACS.
- `03_prepare_dsw_densities_and_periods.ipynb` finds the DSW formation and export densities each year and averages the SWMT maps over 1900–2000 and 2090–2100 (needs 01 and 02).
- `04_prepare_surface_properties.ipynb` extracts monthly surface fluxes, surface properties, and sea ice.
- `05_prepare_bottom_tracers.ipynb` averages bottom CFC-11, SF6, and σ2 over 1900–1920, 1960–1980, 1980–2000, 2000–2020, and 2020–2040.
- `06_prepare_weddell_transect.ipynb` samples tracers along a Weddell Sea transect.

### Data visualization

Each figure notebook reads the products above and saves its figures to `figures/`:
- `fig01-02_shelf_swmt_and_export.ipynb` plots SWMT and cross-shelf export on the ACS (Figures 1–2).
- `fig03-05_dsw_budgets.ipynb` plots the DSW budgets and the spatial breakdown of the SWMT change (Figures 3–5).
- `fig06_shelf_fluxes_and_sea_ice.ipynb` plots the ACS surface fluxes and sea ice, compared with the NOAA/NSIDC and GIOMAS estimates (Figure 6).
- `fig07-09_bottom_tracers.ipynb` maps the bottom tracers (Figures 7–9).
- `fig10-12_weddell.ipynb` covers the Weddell Sea transect (Figures 10–12).
- `figS01-S02_surface_fluxes.ipynb` makes the surface flux maps in the Supporting Information (Figures S1–S2).

The coarsened CM4X output used in this analysis can be regenerated from the native model output with [CM4Xutils](https://github.com/hdrake/CM4Xutils).

## Environment

Create and activate the conda environment before running the notebooks:

```bash
conda env create -f environment.yml
conda activate CM4X_DSW_SWMT
```

Available here: DOI (TODO)
