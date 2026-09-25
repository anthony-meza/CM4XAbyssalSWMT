"""Where the data, derived products, and figures live."""
import os

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plotsdir = lambda x="": rootdir + "/figures/" + x  # figures in the manuscript
obsdir = lambda x="": rootdir + "/data/" + x  # observational data (not tracked)
datadir = lambda x="": "/proj/ecco/CM4X/" + x  # coarsened CM4X output
outputdir = lambda x="": "/proj/ecco/CM4X/postprocessing/" + x  # products written by the prepare_* notebooks
