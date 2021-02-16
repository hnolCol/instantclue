from PyQt5.QtCore import QObject, pyqtSignal
from .ICPlots.ICBoxplot import ICBoxplot
from .ICPlots.ICBarplot import ICBarplot
from .ICPlots.ICPointplot import ICPointplot
from .ICPlots.ICScatter import ICScatterPlot
from .ICPlots.ICPCAPlot import ICPCAPlot
from .ICPlots.ICHClustermap import ICClustermap
from .ICPlots.ICSwarmplot import ICSwarmplot
from .ICPlots.ICViolinplot import ICViolinplot
from .ICPlots.ICLineplot import ICLineplot
from .ICPlots.ICHistogram import ICHistogram
from .ICPlots.ICCountplot import ICCountplot
from .ICPlots.ICXYPlot import ICXYPlot
from .ICPlots.ICForestplot import ICForestplot
from .ICPlots.ICWordCloud import ICWordCloud

plotTypeGraph = {
                "scatter"       :       ICScatterPlot,
                "hclust"        :       ICClustermap,
                "boxplot"       :       ICBoxplot,
                "swarmplot"     :       ICSwarmplot,
                "barplot"       :       ICBarplot,
                "pointplot"     :       ICPointplot,
                "corrmatrix"    :       ICClustermap,
                "violinplot"    :       ICViolinplot,
                "lineplot"      :       ICLineplot,
                "histogram"     :       ICHistogram,
                "countplot"     :       ICCountplot,
                "x-ys-plot"     :       ICXYPlot,
                "dim-red-plot"  :       ICBarplot,      
                "forestplot"    :       ICForestplot,
                "wordcloud"     :       ICWordCloud
                }

additionToGraph = {
                "addSwarm"      :       "addSwarmData"
}
class ICPlotter(object):
	
    def __init__(self,mainController, figure):
        self.mC = mainController
        self.f = figure
        self.f.canvas.mpl_connect("draw_event",self.updateBackground)

    def setGraph(self,plotType):

        if plotType in plotTypeGraph:
            self.graph = plotTypeGraph[plotType](self.mC,self,plotType)
        elif plotType in additionToGraph and hasattr(self.graph,additionToGraph[plotType]):
            getattr(self.graph,additionToGraph[plotType])()

    def clearFigure(self):
        ""
        if hasattr(self,"graph"):
            self.graph.disconnectBindings()
            del self.graph
        self.f.clf()

    def redraw(self, forceRedraw = False):
        ""
        if forceRedraw or self.f.get_axes(): #redraw only if axes exist
            
            if hasattr(self,"graph"):
                self.graph.setHoverObjectsInvisible()
            
            self.f.canvas.draw()
          
    def updateBackground(self,event):
        ""
        if hasattr(self,"graph"):
            self.graph.updateBackgrounds()

    




	