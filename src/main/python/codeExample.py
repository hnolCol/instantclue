from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from backend.data.data import DataCollection 
from backend.plotting.plotterCalculations import PlotterBrain 
from backend.config.config import Config
from ui.plotter.plotManager import ICPlotter 

from ui.plotter.ICPlots.ICScatter import ICScatterPlot
from matplotlib.pyplot import figure

import os 
import sys 
#link to tab delimted file
tabDelFile = "./examples/Tutorial_Data_02.txt" 

#create fake app to get a config which is required for some plot types
class FakeApp(object):
    ""
    def __init__(self,*args,**kwargs):
        self.mainPath = os.path.dirname(sys.argv[0])
        self.config = Config(self)
fakeApp = FakeApp()

#add the data
sourceData = DataCollection(parent=fakeApp)
sourceData.addDataFrameFromTxtFile(tabDelFile,"TutorialData2")
dataID = list(sourceData.dfs.keys())[0]
print(dataID)
#print columns by data type
print(sourceData.dfsDataTypesAndColumnNames[dataID])

#define plot type and get plotData 
plotType = "scatter"
plotter = PlotterBrain(sourceData)
plotData = plotter.getPlotProps(dataID,numericColumns=["CTRL_1","CTRL_2"], categoricalColumns=[],plotType=plotType)

#plot the actual graph
print(plotData)


#show the figure
class MainWindow(QMainWindow):

    def __init__(self, plotData, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.figure = figure()
        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        ICP = ICPlotter(fakeApp,self.figure)
        initGraph = ICP.setGraph(plotType)
        if initGraph:
            ICP.graph.setData(plotData["data"])
            self.__layout() 
            ICP.redraw()
    
    def __layout(self):
        "Add items"
        ww = QWidget()
        ww.setLayout(QVBoxLayout())
        ww.layout().addWidget(self.canvas)
        ww.layout().addWidget(self.toolbar)
        self.setCentralWidget(ww)
        
    

app = QApplication(sys.argv)
w = MainWindow(plotData)
w.show()
app.exec()