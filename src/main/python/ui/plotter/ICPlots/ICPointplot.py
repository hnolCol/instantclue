

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
                        line, errorCaps, errorBarRangeLine = self.axisDict[n].errorbar(**eKwargs,**lKwargs)
                        internalID = self.getInternalIDByColor(lKwargs["markerfacecolor"])
                        if internalID not in self.pointplotItems:
                            self.pointplotItems[internalID] = []
                        self.pointplotItems[internalID].extend([line,errorBarRangeLine,errorCaps])
                        
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
            if self.interactive:
                self.addQuickSelectHoverScatter()
                for ax in self.axisDict.values():
                    self.addHoverScatter(ax) 
                self.addHoverBinding()
            self.checkForQuickSelectDataAndUpdateFigure()
        except Exception as e:
            print("==")
            print(self.data)
            print(str(e))
    
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
                    if isinstance(l,tuple): #errorbar collection
                            for el in l:
                                el.set_color(newColor)
                    else:
                        l.set_markerfacecolor(newColor)
                        if self.getParam("pointplot.line.marker.same.color"):
                                l.set_color(newColor)
                    
        else:
            for color, group, internalID in colorGroup.values:
                if internalID in self.pointplotItems:
                    lines = self.pointplotItems[internalID]
                    for l in lines:
                       # boolIdx = colorGroup["internalID"].values ==  changedCategory
                        #newColor = colorGroup.loc[boolIdx,"color"].values[0]
                        if isinstance(l,tuple): #errorbar collection
                            for el in l:
                                el.set_color(color)
                        else:
                            l.set_markerfacecolor(color)
                            if self.getParam("pointplot.line.marker.same.color"):
                                l.set_color(color)
        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)
        self.updateFigure.emit()

    def updateBackgrounds(self):
        ""
        if not hasattr(self,"backgrounds"):
            self.backgrounds = {}
        self.backgrounds.clear() 
        if hasattr(self.p.f.canvas,"copy_from_bbox"):
            for ax in self.axisDict.values():
                self.backgrounds[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)

    def setHoverData(self,dataIndex):
        ""
        coords = np.array([])
        if hasattr(self,"backgrounds"):
            for n, ax in self.axisDict.items():
                if n in self.data["hoverData"] and ax in self.backgrounds:
                    data = self.data["hoverData"][n]
                    idx = data.index.intersection(dataIndex)
                    data = data.loc[idx]
                    if not data.empty:
                            errorData = self.data["errorData"][n]
                            xValues = np.array([errorData[m]["x"][0] for m in range(len(errorData))] * data.index.size).flatten()
                            yValues = data.values.flatten() 
                            coords = np.array([(xValues[n],yValues[n]) for n in range(len(xValues))])

                    self.p.f.canvas.restore_region(self.backgrounds[ax])
                    if coords.size > 0:
                        
                        self.setHoverScatterData(coords,ax)
                    else:
                        self.p.f.canvas.blit(ax.bbox)

    def updateQuickSelectItems(self,propsData=None):
               
        if self.isQuickSelectModeUnique() and hasattr(self,"quickSelectCategoryIndexMatch"):
            dataIndex = np.concatenate([idx for idx in self.quickSelectCategoryIndexMatch.values()])
        else:
            dataIndex = self.getDataIndexOfQuickSelectSelection()

        if not hasattr(self,"backgrounds"):
            self.updateBackgrounds()
        
        if hasattr(self,"quickSelectScatter"):
            try:
                for n,ax in self.axisDict.items():
                    if n in self.data["hoverData"] and ax in self.backgrounds and ax in self.quickSelectScatter:                        
                        data = self.data["hoverData"][n]
                        idx = data.index.intersection(dataIndex)
                        data = data.loc[idx]
                        if not data.empty:
                            errorData = self.data["errorData"][n]
                            xValues = np.array([errorData[m]["x"][0] for m in range(len(errorData))] * data.index.size).flatten()
                            yValues = data.values.flatten() 
                            coords = np.array([(xValues[n],yValues[n]) for n in range(len(xValues))]) #create numpy array for coords (x,y)
                            scatterColors = np.repeat([propsData.loc[idx,"color"] for idx in dataIndex],data.shape[1])
                            scatterSizes = np.repeat([propsData.loc[idx,"size"] for idx in dataIndex], data.shape[1])
                            self.quickSelectScatterDataIdx[ax] = np.repeat(idx.values,data.shape[1])
                            self.updateQuickSelectScatter(ax,coords,scatterColors,scatterSizes)

            except Exception as e:
                print(e)


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
       
                
