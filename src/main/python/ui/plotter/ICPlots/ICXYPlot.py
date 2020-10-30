

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
import numpy as np

class ICXYPlot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICXYPlot,self).__init__(*args,**kwargs)

        self.xyplotItems = dict() 

    def addHoverLine(self):
        ""
        self.hoverLines = {}
        for ax in self.axisDict.values():
            hoverLine = ax.plot([],[],
                            linewidth=self.getParam("linewidth.median"), 
                            marker= self.getParam("marker.median"), 
                            color = self.getParam("scatter.hover.color"),
                            markeredgecolor = "black", 
                            linestyle = "-",
                            markeredgewidth = self.getParam("markeredgewidth.median"))
            self.hoverLines[ax] = hoverLine[0]

        self.setHoverLinesInivisble()

    def setHoverLinesInivisble(self):
        ""
        for l in self.hoverLines.values():
            l.set_visible(False)

   

    def initXYPlot(self, onlyForID = None, targetAx = None):
        ""
        for n,ax in self.axisDict.items():
            for l in self.data["lines"][n]:
                ax.add_line(l)

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
           
            self.initAxes(data["axisPositions"])
            self.setAxisLabels(self.axisDict,self.data["axisLabels"])
            self.initXYPlot()
            for n,ax in self.axisDict.items():
                if n in self.data["axisLimits"]:
                    self.setAxisLimits(ax,
                            self.data["axisLimits"][n]["xLimit"],
                            self.data["axisLimits"][n]["yLimit"])

            if self.interactive:
               self.addHoverLine()
               self.addHoverBinding() 

            #self.addTitles()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            # self.setXTicksForAxes(self.axisDict,
            #             data["tickPositions"],
            #             data["tickLabels"],
            #             rotation=90)
            # qsData = self.getQuickSelectData()
            # if qsData is not None:
            #     self.mC.quickSelectTrigger.emit()
            # else:
            #     self.updateFigure.emit() 
            self.updateFigure.emit()
           
        except Exception as e:
            print(e)
        

    def setHoverData(self,dataIndex, showText = False):
        ""
       # print(dataIndex)
       # if dataIndex in self.data["plotData"].index:


    def setHoverObjectsInvisible(self):
        ""
        if hasattr(self,"hoverLines"):
            for l in self.hoverLines.values():
                l.set_visible(False)

    def getInternalIDByColor(self, color):
        ""
        colorGroupData = self.data["dataColorGroups"]
        boolIdx = colorGroupData["color"].values ==  color
        if np.any(boolIdx):
            return colorGroupData.loc[boolIdx,"internalID"].values[0]

    def updateGroupColors(self,colorGroup,changedCategory=None):
        "changed category is encoded in a internalID"
        if "linesByInternalID" in self.data and changedCategory in self.data["linesByInternalID"]:
            l = self.data["linesByInternalID"][changedCategory]
            changedColor = colorGroup.loc[colorGroup["internalID"] == changedCategory]["color"].values[0]
            l.set_color(changedColor)
            
        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)
        self.updateFigure.emit()

    def updateBackgrounds(self):
        "Update Background for blitting"
        self.axBackground = dict()
        for ax in self.axisDict.values():
            self.axBackground[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)	
    
    def updateQuickSelectItems(self,propsData=None):
        "Saves lines by idx id"

    
    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        
        
         