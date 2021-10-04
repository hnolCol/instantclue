

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

class ICProteinPeptidePlot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICProteinPeptidePlot,self).__init__(*args,**kwargs)

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

    def addHoverArea(self):
        ""
        self.hoverAreas = {}
        for ax in self.axisDict.values():
            hoverArea  = Polygon([[0,0],[1,1]],visible=False,
                            alpha=self.getParam("alpha.IQR"),
                            facecolor=self.getParam("scatter.hover.color"),
                            edgecolor="black",
                            linewidth=0.1,
                            fill=True, 
                            closed=True)
            
            ax.add_patch(hoverArea)
            self.hoverAreas[ax] = hoverArea
        
        
    def setHoverLinesInivisble(self):
        ""
        for l in self.hoverLines.values():
            l.set_visible(False)

    

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
           
            self.initAxes(data["axisPositions"])
            self.initLineplot()
            for n,ax in self.axisDict.items():
                if n in self.data["axisLimits"]:
                    self.setAxisLimits(ax,
                            self.data["axisLimits"][n]["xLimit"],
                            self.data["axisLimits"][n]["yLimit"])

            if self.interactive:
               self.addHoverLine()
               self.addHoverArea()
               self.addHoverBinding() 

            self.addTitles()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.setXTicksForAxes(self.axisDict,
                        data["tickPositions"],
                        data["tickLabels"],
                        rotation=90)
            self.checkForQuickSelectDataAndUpdateFigure()
           
           
        except Exception as e:
            print(e)
        

    def setHoverData(self,dataIndex, showText = False):
        ""
        
       

    def setHoverObjectsInvisible(self):
        ""
        if hasattr(self,"hoverLines"):
            for l in self.hoverLines.values():
                l.set_visible(False)
        if hasattr(self,"hoverAreas"):
            for area in self.hoverAreas.values():
                area.set_visible(False)

    def getInternalIDByColor(self, color):
        ""
       

    def updateGroupColors(self,colorGroup,changedCategory=None):
        "changed category is encoded in a internalID"
 

    def updateBackgrounds(self):
        "Update Background for blitting"
        self.axBackground = dict()
        for ax in self.axisDict.values():
            self.axBackground[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)	
    
    def updateQuickSelectItems(self,propsData=None):
        "Saves lines by idx id"

    
    def updateQuickSelectData(self,quickSelectGroup,changedCategory=None):
        ""
       
    
    def mirrorQuickSelectArtists(self,axisID,targetAx):
        ""
        if axisID in self.axisDict:
            sourceAx = self.axisDict[axisID]
            if sourceAx in self.quickSelectLineKwargs:
                for lineKwargs in self.quickSelectLineKwargs[sourceAx].values():
                    targetAx.add_line(Line2D(**lineKwargs))
                for poylgonKwargs in self.quickSelectPolygonKwargs[sourceAx].values():
                    targetAx.add_patch(Polygon(**poylgonKwargs))
