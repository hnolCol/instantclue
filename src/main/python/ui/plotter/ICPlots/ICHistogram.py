

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from typing import Iterable

class ICHistogram(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICHistogram,self).__init__(*args,**kwargs)
        self.histogramPatches = dict() 
        self.histogramKey = dict()

    def addGraphSpecActions(self,menus : dict) -> None:
        ""
        if "main" in menus and hasattr(menus["main"],"addAction"):
            menus["main"].addAction("Histogram Style",lambda : self.mC.openSettings(specificSettingsTab ="Histogram (Density)"))
       
    def addPatches(self, onlyForID = None, targetAx = None):
        ""
        for n,patches in self.data["patches"].items():
            
            if n in self.axisDict and onlyForID is None:
                
                for pProps in patches:
                    internalID = pProps["internalID"]
                    if not internalID in self.histogramPatches:
                        self.histogramPatches[internalID] = []
                        self.histogramKey[internalID] = []
                    p = self.getPatch(pProps)
                    self.histogramPatches[internalID].append(p)
                    self.histogramKey[internalID] = n
                    self.axisDict[n].add_patch(p)
            else:
                if n != onlyForID:
                    continue
                else:
                    for pProps in patches:
                        p = self.getPatch(pProps)
                        targetAx.add_patch(p)

    def getPatch(self,pProps):
        ""
        if pProps["type"] == "Rectangle":
            p = Rectangle(**pProps["p"])
        elif pProps["type"] == "Polygon":
            p = Polygon(**pProps["p"])
        return p 

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            self.initAxes(data["axisPositions"])
            #set axis Limits
            for n,ax in self.axisDict.items():
                if n in self.data["axisLimits"]:
                    self.setAxisLimits(ax,
                            self.data["axisLimits"][n]["xLimit"],
                            self.data["axisLimits"][n]["yLimit"])
            self.setAxisLabels(self.axisDict,self.data["axisLabels"])
            self.addPatches()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            if self.interactive:
                for ax in self.axisDict.values():
                    self.addHoverScatter(ax)
                    
            self.checkForQuickSelectDataAndUpdateFigure()
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
        for color, group, internalID in colorGroup.values:
            if internalID in self.histogramPatches:
                patches = self.histogramPatches[internalID]

                for p in patches:
                    if isinstance(p,Polygon):
                        p.set_edgecolor(color)
                        
                    else:
                        p.set_facecolor(color)
                self.updateColorInProps(internalID,color)

        self.updateFigure.emit()


    def updateColorInProps(self,internalID,color):
        "Update the props to allow for export to main figure."
        for n,patches in self.data["patches"].items():
            
            for pProps in patches:
                if pProps["internalID"] == internalID:
                    if pProps["type"] == "Rectangle":
                        pProps["p"]["facecolor"] = color
                    else:
                        pProps["p"]["edgecolor"] = color
    


    def updateBackgrounds(self):
        ""
        if not hasattr(self,"background"):
            self.background = dict()
        for ax in self.axisDict.values():
            self.background[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)

    def setHoverObjectsInvisible(self):
        ""
        if hasattr(self,"hoverScatter"):
            for hoverScatter in self.hoverScatter.values():
                hoverScatter.set_visible(False)

    def setHoverData(self,dataIndex : Iterable):
        ""
        dataIndex = np.array(dataIndex)
        #print(dataIndex)
        for n, ax in self.axisDict.items():
            if ax in self.background and n in self.data["hoverData"]:
                self.p.f.canvas.restore_region(self.background[ax])
                xValues = self.data["hoverData"][n].loc[self.data["hoverData"][n].index.intersection(dataIndex)]
                if xValues.empty:
                    #if empty just draw backgrond
                    self.p.f.canvas.blit(ax.bbox)
                else:
                    yLim = self.getYlim(ax)
                    
                    dist = np.sqrt(yLim[0]**2 + yLim[1]**2)
                   
                    #create numpy array with scatter offsets
                    coords = np.zeros(shape=(xValues.size,2))
                    coords[:,0] = xValues.values
                    coords[:,1] = [yLim[0] + dist * 0.025] * xValues.size #2.5% of ylim
                    
                    self.setHoverScatterData(coords, ax)
    
    def updateQuickSelectItems(self, propsData = None):
        ""
        colorData = self.getQuickSelectData()
        dataIndex = self.getDataIndexOfQuickSelectSelection()
        if not hasattr(self,"quickSelectScatter"):
            self.quickSelectScatter = dict()
        for ax in self.axisDict.values():
            if not ax in self.quickSelectScatter:
                self.quickSelectScatter[ax] = ax.scatter(x=[],y=[],**self.getScatterKwargs(), zorder=1e6)
        for n, ax in self.axisDict.items():
            if n in self.data["hoverData"]:
                xValues = self.data["hoverData"][n].loc[self.data["hoverData"][n].index.intersection(dataIndex)]
                yLim = self.getYlim(ax)        
                dist = np.sqrt(yLim[0]**2 + yLim[1]**2)
                #create numpy array with scatter offsets
                coords = np.zeros(shape=(xValues.size,2))
                coords[:,0] = xValues.values
                coords[:,1] = [yLim[0] + dist * 0.025] * xValues.size #2.5% of ylim
           
                self.quickSelectScatter[ax].set_offsets(coords)
                self.quickSelectScatter[ax].set_visible(True)
                self.quickSelectScatter[ax].set_facecolor(colorData.loc[xValues.index,"color"])

    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        data = self.data 
        self.setAxisLabels({axisID:targetAx},data["axisLabels"],onlyForID=axisID)
        for n,_ in self.axisDict.items():
            if axisID == n and axisID in data["axisLimits"]:
                self.setAxisLimits(targetAx,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
        self.addPatches(axisID,targetAx)
            