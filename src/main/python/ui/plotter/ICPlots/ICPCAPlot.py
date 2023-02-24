#
from .ICChart import ICChart
from .charts.scatter_plotter import scatterPlot

class ICPCAPlot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICPCAPlot,self).__init__(*args,**kwargs)
        self.scatterPlots = dict()
        self.staticScatterPlots = dict()
        
    def initPCAScatterPlots(self):
        ""
        
        #init scatters
        for n,ax in self.axisDict.items():
            if n < 2:
                #plot projected data
                columnPair = self.data["columnPairs"][n]
                self.scatterPlots[columnPair] = scatterPlot(
                                    self,
                                    data = self.data["plotData"]["projection"],
                                    plotter = self.p,
                                    ax = ax,
                                    numericColumns = list(columnPair),
                                    dataID = self.data["dataID"],
                                    scatterKwargs = self.getScatterKwargs(),
                                    hoverKwargs = self.getHoverKwargs()
                                    )
            else:
                #plot eigenvalues
                columnPair = self.data["columnPairs"][n-2]
                self.staticScatterPlots[columnPair] = scatterPlot(
                                    self,
                                    data = self.data["plotData"]["eigV"],
                                    plotter = self.p,
                                    ax = ax,
                                    numericColumns = list(columnPair),
                                    dataID = self.data["dataID"],
                                    scatterKwargs = self.getScatterKwargs(),
                                    hoverKwargs = self.getHoverKwargs(),
                                    interactive = False
                                    )

    def onDataLoad(self, data):
        ""
        self.data = data
        self.initAxes(data["axisPositions"])
        self.initPCAScatterPlots()
        self.updateFigure.emit()

    def setHoverData(self,dataIndex, sender = None):
        "Sets hover data in scatter plots"
        for scatterPlot in self.scatterPlots.values():
            if sender is None:
                scatterPlot.setHoverData(dataIndex)
            elif sender != scatterPlot:
                scatterPlot.setHoverData(dataIndex)

    def setMask(self,dataIndex):
        "Sets a mask on the data to hide some"
        self.requestUpdateData()

        #for scatterPlot in self.scatterPlots.values():
         #   scatterPlot.setMask(dataIndex)

    def resetMask(self):
        "Resets mask"
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.resetMask()

    def updateBackgrounds(self, redraw = False):
        ""
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.updateBackground(redraw=redraw)

    def requestUpdateData(self):
        ""
        funcProps = {"key":"dimReduction::PCAForPlot",
                    "kwargs":{
                        "dataID":self.data["dataID"],
                        "liveGraph":True,
                        "columnNames":self.data["numericColumns"]}}
        self.mC.sendRequestToThread(funcProps)
    
    def updateData(self, data):
        ""
        try:
            for scatterPlot in self.scatterPlots.values():
                scatterPlot.setData(data["projection"])

            for scatterPlot in self.staticScatterPlots.values():
                scatterPlot.setData(data["eigV"])
        except Exception as e:
            print(e)

        self.updateFigure.emit()
