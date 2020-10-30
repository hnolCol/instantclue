
from matplotlib import gridspec

class CategoricalCountPlot(object):

    def __init__(self,plotter):
        self.plotter = plotter

        self.defineVariables()
        self.createAxis() 


    def defineVariables(self):

        self.axisDict = dict() 
    

    def setData(self,data):
        ""
        self.data = data

        self.replot()


    def createAxis(self):
        ""
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
        self.axisDict[0] = self.plotter.figure.add_subplot(gs[0])
        self.axisDict[1] = self.plotter.figure.add_subplot(gs[1])


    def replot(self, specificAxis = None):
        ""

        ax = specificAxis if specificAxis is not None else self.ax

