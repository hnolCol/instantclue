# Instant Clue Plots

Each chart is a subclass of ICChart. ICPlotter (module: plotManager.py) inititates individual charts and holds them in the attribute graph. Therefore ICPlotter.graph.fn(..) can be used from outside to target the current graph/chart. 

All individual function of the different charts (ICHistogram, ICBoxplot, ...) should be present in ICChart. 