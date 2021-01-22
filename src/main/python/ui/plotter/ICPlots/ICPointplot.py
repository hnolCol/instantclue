

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
import numpy as np

class ICPointplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICPointplot,self).__init__(*args,**kwargs)

        self.pointplotItems = dict() 

    def initPointplots(self, onlyForID = None, targetAx = None):
        ""
        try:
            for n, lineData in self.data["plotData"].items():
                if onlyForID is not None and targetAx is not None:
                    errorData = self.data["errorData"][onlyForID]
                    for eKwargs, lKwargs in zip(errorData,lineData):
                        line, _, _ = targetAx.errorbar(**eKwargs,**lKwargs)
                        targetAx.add_artist(line)
                elif onlyForID is None and n in self.axisDict:
                    errorData = self.data["errorData"][n]
                    for eKwargs, lKwargs in zip(errorData,lineData):
                        line, _, _ = self.axisDict[n].errorbar(**eKwargs,**lKwargs)
                        internalID = self.getInternalIDByColor(lKwargs["markerfacecolor"])
                        if internalID not in self.pointplotItems:
                            self.pointplotItems[internalID] = []
                        self.pointplotItems[internalID].append(line)
                        self.axisDict[n].add_artist(line)
        except Exception as e:
            print(e)

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            self.initAxes(data["axisPositions"])
            self.initPointplots()
            for n,ax in self.axisDict.items():
                if n in self.data["axisLimits"]:
                    self.setAxisLimits(ax,
                            self.data["axisLimits"][n]["xLimit"],
                            self.data["axisLimits"][n]["yLimit"])
            self.setXTicksForAxes(self.axisDict,
                        data["tickPositions"],
                        data["tickLabels"],
                        rotation=90)
            self.setAxisLabels(self.axisDict,self.data["axisLabels"])
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            #hoverGroupItems = self.reorderBoxplotItemsForHover()
            #self.setHoverItemGroups(hoverGroupItems)
            self.updateFigure.emit()
        except Exception as e:
            print(e)
    
    def getInternalIDByColor(self, color):
        ""
        colorGroupData = self.data["dataColorGroups"]
        boolIdx = colorGroupData["color"].values ==  color
        if np.any(boolIdx):
            return colorGroupData.loc[boolIdx,"internalID"].values[0]

    def updateGroupColors(self,colorGroup,changedCategory=None):
        ""
        if changedCategory is not None:
            if changedCategory in self.pointplotItems:
                lines = self.pointplotItems[changedCategory]
                for l in lines:
                    boolIdx = colorGroup["internalID"].values ==  changedCategory
                    newColor = colorGroup.loc[boolIdx,"color"].values[0]
                    l.set_markerfacecolor(newColor)
        else:
            for color, group, internalID in colorGroup.values:
                if internalID in self.pointplotItems:
                    lines = self.pointplotItems[internalID]
                    for l in lines:
                       # boolIdx = colorGroup["internalID"].values ==  changedCategory
                        #newColor = colorGroup.loc[boolIdx,"color"].values[0]
                        l.set_markerfacecolor(color)
        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)
        self.updateFigure.emit()


    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        data = self.data
        self.setAxisLabels({axisID:targetAx},data["axisLabels"],onlyForID=axisID)
        self.setXTicksForAxes({axisID:targetAx},
                        data["tickPositions"],
                        data["tickLabels"],
                        onlyForID= axisID,
                        rotation=90)
        self.initPointplots(axisID,targetAx)
       
                
