
from .ICChart import ICChart
from collections import OrderedDict
import numpy as np
import pandas as pd
class ICBarplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICBarplot,self).__init__(*args,**kwargs)

        self.barplotItems = dict() 

    
    def initBarplots(self, onlyForID = None, targetAx = None):
        ""
        try:
            for n, barplotProps in self.data["plotData"].items():
                if onlyForID is not None and n != onlyForID:
                    continue
                if targetAx is None and n in self.axisDict:
                    self.barplotItems[n] = self.axisDict[n].bar(**barplotProps)
                elif targetAx is not None:
                    barplotProps["color"] = [a.get_facecolor() for a in self.barplotItems[n].patches]
                    targetAx.bar(**barplotProps)
        except Exception as e:
            print(e)

    def onDataLoad(self, data):
        ""
        self.data = data
        self.initAxes(data["axisPositions"])
        self.setXTicksForAxes(self.axisDict,data["tickPositions"],data["tickLabels"],rotation=90)
        self.setAxisLabels(self.axisDict,data["axisLabels"])
        for n,ax in self.axisDict.items():
            if n in data["axisLimits"]:
                self.setAxisLimits(ax,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
        self.initBarplots()
        self.savePatches()
        if self.interactive:
                for ax in self.axisDict.values():
                    self.addHoverScatter(ax) 
                self.addQuickSelectHoverScatter()
        self.addTitles()
        self.addVerticalLines()
        self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
       
        self.checkForQuickSelectDataAndUpdateFigure()

    def savePatches(self):
        ""
        colorGroupData = self.data["dataColorGroups"]
        self.colorGroupArtists = OrderedDict([(intID,[]) for intID in colorGroupData["internalID"].values])
        self.groupColor = dict() 
        
        for n, barprops in self.barplotItems.items():
            # print(barprops["patches"])
            for artist, color in zip(barprops.patches,self.data["facecolors"][n]):
                idx = colorGroupData.index[colorGroupData["color"] == color]
                intID = colorGroupData.loc[idx,"internalID"].iloc[0]
                self.colorGroupArtists[intID].append(artist)
                if intID not in self.groupColor:
                    self.groupColor[intID] = color

       
    def updateGroupColors(self,colorGroup,changedCategory=None):
        ""
        for color, _ , intID in colorGroup.values:
            if intID in self.colorGroupArtists:
                if self.groupColor[intID] != color:
                    artists = self.colorGroupArtists[intID]
                    for artist in artists:
                        artist.set_facecolor(color)
                    self.groupColor[intID] = color
        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)
        self.updateFigure.emit()
        
    def updateBackgrounds(self):
        ""
        if not hasattr(self,"backgrounds"):
            self.backgrounds = {}
        self.backgrounds.clear() 
        for ax in self.axisDict.values():
            self.backgrounds[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)


    def setHoverData(self,dataIndex):
        ""
        if hasattr(self,"backgrounds"):
            for n, ax in self.axisDict.items():
                if n in self.data["hoverData"] and ax in self.backgrounds:
                    #print(self.data["plotData"][n])
                    data = self.data["hoverData"][n]["x"]
                    coords = np.array([(self.data["plotData"][n]["x"][m], X.loc[dataIdx]) for dataIdx in dataIndex for m,X in enumerate(data) if dataIdx in X.index.values ])
                    if coords.size > 0:
                        self.p.f.canvas.restore_region(self.backgrounds[ax])
                        self.setHoverScatterData(coords,ax)


    def updateQuickSelectItems(self,propsData=None):
        ""
        if self.isQuickSelectModeUnique() and hasattr(self,"quickSelectCategoryIndexMatch"):
            dataIndex = np.concatenate([idx for idx in self.quickSelectCategoryIndexMatch.values()])
            intIDMatch = np.concatenate([np.full(idx.size,intID) for intID,idx in self.quickSelectCategoryIndexMatch.items()]).flatten()
            
        else:
            dataIndex = self.getDataIndexOfQuickSelectSelection()
            intIDMatch = np.array(list(self.quickSelectCategoryIndexMatch.keys()))

        if not hasattr(self,"backgrounds"):
            self.updateBackgrounds()
        if hasattr(self,"quickSelectScatter"):
            try:
                for n,ax in self.axisDict.items():
                    if n in self.data["hoverData"] and ax in self.backgrounds and ax in self.quickSelectScatter:                        
                        data = self.data["hoverData"][n]["x"]
                    
                        coords = [(self.data["plotData"][n]["x"][m], X.loc[dataIdx], intIDMatch[mIdx], dataIdx) for mIdx,dataIdx in enumerate(dataIndex) for m,X in enumerate(data) if dataIdx in X.index.values ]
                        #idxPlotted = [idx for idx in dataIndex if idx in coords[:,2]] 
                        coords = pd.DataFrame(coords, columns = ["x","y","intID","idx"])
                        
                        sortedDataIndex = coords["idx"].values
                        scatterColors = [propsData.loc[idx,"color"] for idx in sortedDataIndex]
                        scatterSizes = [propsData.loc[idx,"size"] for idx in sortedDataIndex]
                        self.quickSelectScatterDataIdx[ax] = {"idx":sortedDataIndex,"coords":coords}

                        self.updateQuickSelectScatter(ax,coords,scatterColors,scatterSizes)

            except Exception as e:
                
                print(e) 

    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
           
        data = self.data
        self.setXTicksForAxes({axisID:targetAx},data["tickPositions"],data["tickLabels"], onlyForID = axisID, rotation=90)
        self.setAxisLabels({axisID:targetAx},data["axisLabels"],onlyForID=axisID)
        self.initBarplots(targetAx=targetAx)
        self.addVerticalLines(axisID,targetAx)
        self.addSwarm("", [], [], onlyForID=axisID,targetAx=targetAx)


   