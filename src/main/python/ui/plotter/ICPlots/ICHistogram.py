

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
import numpy as np

class ICHistogram(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICHistogram,self).__init__(*args,**kwargs)
        self.histogramPatches = dict() 

       
    def addPatches(self, onlyForID = None, targetAx = None):
        ""
        for n,patches in self.data["patches"].items():
            
            if n in self.axisDict and onlyForID is None:
                
                for pProps in patches:
                    internalID = pProps["internalID"]
                    if not internalID in self.histogramPatches:
                        self.histogramPatches[internalID] = []
                    p = self.getPatch(pProps)
                    self.histogramPatches[internalID].append(p)
                    self.axisDict[n].add_patch(p)
            else:
                if n != onlyForID:
                    continue
                else:
                    print("here", n)
                    for pProps in patches:
                        print(pProps)
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
        
        for color, group, internalID in colorGroup.values:
            if internalID in self.histogramPatches:
                patches = self.histogramPatches[internalID]
                for p in patches:
                    if isinstance(p,Polygon):
                        p.set_edgecolor(color)
                    else:
                        p.set_facecolor(color)

        self.updateFigure.emit()

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

    def setHoverData(self,dataIndex):
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
    
    def updateQuickSelectItems(self):
        ""
       # print("updaing")
        colorData = self.getQuickSelectData()
        dataIndex = self.getDataIndexOfQuickSelectSelection()
        #dataIndexInClust = [idx for idx in dataIndex if idx in self.data["plotData"].index]
        #idxPosition = [self.data["plotData"].index.get_loc(idx) + 0.5 for idx in dataIndexInClust]
      #  if len(idxPosition) == 0:
       #     return
        if not hasattr(self,"quickSelectScatter"):
            self.quickSelectScatter = dict()
        for ax in self.axisDict.values():
            if not ax in self.quickSelectScatter:
                self.quickSelectScatter[ax] = ax.scatter(x=[],y=[],**self.getScatterKwargs())
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
            