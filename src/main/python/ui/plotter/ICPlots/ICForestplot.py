

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

import numpy as np

class ICForestplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICForestplot,self).__init__(*args,**kwargs)


    def initGraphElements(self, onlyForID = None, targetAx = None):
        ""
        if "plotData" in self.data:
            #print(self.data["plotData"])
            self.circlesByInternalID = OrderedDict() 

            for axisID, categoricalGraphElements in self.data["plotData"].items():
                if onlyForID is not None and targetAx is not None:
                    if onlyForID != axisID:
                        continue
                    else:
                        ax = targetAx
                else:
                    ax = self.axisDict[axisID]
                
                
                if self.getParam("forest.plot.line.ratio.one"):
                    ax.axvline(1, linewidth = 0.5, color = "darkgrey")

                for variableName, statData in categoricalGraphElements.items():
                    ratioName = self.getParam("forest.plot.cont.table.ratio") if not self.getParam("forest.plot.calculated.data") else "ratio"
                    minCI, maxCI = statData["{}CI".format(ratioName)]
                    pos = statData["yPosition"]

                    ax.plot([minCI,maxCI], [pos,pos], 
                                color = self.getParam("forest.plot.line.color"), 
                                linewidth = self.getParam("forest.plot.line.width"))
        
                    circleLine  = ax.plot([statData[ratioName]],[pos], 
                                    marker=self.getParam("forest.plot.marker"), 
                                    markersize=self.getParam("forest.plot.markersize"), 
                                    markerfacecolor = self.data["faceColors"][axisID][variableName],
                                    markeredgewidth = 0.5,
                                    markeredgecolor = "darkgrey"
                                    )
                    self.circlesByInternalID[statData["internalID"]] = circleLine[0] #ax.plot returns a list 
                    ax.scatter([minCI, maxCI],[pos,pos], 
                                c = [self.getParam("forest.plot.lower.bound.color"),
                                    self.getParam("forest.plot.upper.bound.color")],
                                s = 70,
                                linewidths = 0.5,
                                edgecolors = "darkgrey",
                                marker = self.getParam("forest.plot.bound.marker")
                                    )
            
            
                    

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            self.initAxes(data["axisPositions"])
            
            self.setYTicksForAxes(self.axisDict,data["tickPositions"],data["tickLabels"])
            self.initGraphElements()
            if "axisLimits" in self.data:
                for n,ax in self.axisDict.items():
                    if n in data["axisLimits"]:
                        self.setAxisLimits(ax,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
        
            self.setAxisLabels(self.axisDict,data["axisLabels"])

            self.setDataInColorTable(self.data["dataColorGroups"], title = "Variables")

            
            self.updateFigure.emit() 
           
           
        except Exception as e:
            print(e)
        

    def updateBackgrounds(self):
        "Update Background for blitting"
        self.axBackground = dict()
        for ax in self.axisDict.values():
            self.axBackground[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)	

    def drawBackgrounds(self):
        ""
        for ax, background in self.axBackground.items():
            self.p.f.canvas.restore_region(background)
            self.p.f.canvas.blit(ax.bbox)

    def updateQuickSelectItems(self):
        "Saves lines by idx id"


    def updateGroupColors(self,colorGroup,changedCategory=None):
        ""
        for color, _ , intID in colorGroup.values:
            if intID in self.circlesByInternalID:
                if self.circlesByInternalID[intID].get_markerfacecolor() != color:
                    self.circlesByInternalID[intID].set_markerfacecolor(color)

        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)
        self.updateFigure.emit()

    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        data = self.data
        self.setAxisLabels({axisID:targetAx},data["axisLabels"],onlyForID=axisID)
        self.initGraphElements(onlyForID=axisID,targetAx=targetAx)
        self.setXTicksForAxes({axisID:targetAx},data["tickPositions"],data["tickLabels"], onlyForID = axisID, rotation=90)          

   # self.circlesByInternalID[statData["internalID"]]

        # colorData = self.getQuickSelectData()
        # dataIndex = self.getDataIndexOfQuickSelectSelection()
        # if not hasattr(self,"quickSelectLines"):
        #     self.quickSelectLines = dict() 
        # #dataIndexInClust = [idx for idx in dataIndex if idx in self.data["plotData"].index]
        # for n,ax in self.axisDict.items():
        #     idxInAxSet = self.data["hoverData"][n].index.intersection(dataIndex)
        #     if idxInAxSet.size > 0:
        #         for idx in idxInAxSet.values:

        #             y = self.data["hoverData"][n].loc[idx,self.data["numericColumns"]].values.flatten()
        #             x = np.arange(y.size)
        #             c = colorData.loc[idx,"color"]
        #             lines = ax.plot(
        #                             x,
        #                             y, 
        #                             marker = self.getParam("marker.quickSelect"), 
        #                             markerfacecolor = c, 
        #                             color = c, 
        #                             linewidth = self.getParam("linewidth.quickSelect"), 
        #                             markeredgecolor = "black", 
        #                             markeredgewidth = self.getParam("markeredgewidth.quickSelect")
        #                             )
        #             self.quickSelectLines[idx] = lines[0]
                    
