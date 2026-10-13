from .base import PlotterBrainBase
from .categorical import CategoricalPlotMixin
from .matrix import MatrixPlotMixin
from .scatter import ScatterPlotMixin


class PlotterBrain(PlotterBrainBase, CategoricalPlotMixin, MatrixPlotMixin, ScatterPlotMixin):
    """Composed plotter brain: dispatch/state + categorical, matrix and scatter plot families."""
