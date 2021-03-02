 
from .ICChart import ICChart
from collections import OrderedDict
import numpy as np 
import pandas as pd


class ICClusterplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICClusterplot,self).__init__(*args,**kwargs)
        self.boxplotItems = {} 
        
    def initClusters(self,onlyForID = None, targetAx = None):
        ""
        for n, boxplotProps in self.data["plotData"].items():
            self.boxplotItems[n] = self.axisDict[n].boxplot(**boxplotProps)
           
    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            self.initAxes(data["axisPositions"])
            
            if "tickPositions" in data and "tickLabels" in data:
                self.setXTicksForAxes(self.axisDict,data["tickPositions"],data["tickLabels"],rotation=90)
            if "axisLimits" in data:
                for n,ax in self.axisDict.items():
                    if n in data["axisLimits"]:
                        self.setAxisLimits(ax,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
            if "axisLabels" in data:
                self.setAxisLabels(self.axisDict,data["axisLabels"])
         
            self.addTitles()
            self.initClusters()
          
            if self.interactive:
                for ax in self.axisDict.values():
                    self.addHoverScatter(ax) 
                #adda qucik select hover
                self.addQuickSelectHoverScatter()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.checkForQuickSelectDataAndUpdateFigure()
        except Exception as e:
            print(e)
        

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

 
    def updateQuickSelectItems(self,propsData=None):
               
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
                    if n in self.data["plotData"] and ax in self.backgrounds and ax in self.quickSelectScatter:                        
                        data = self.data["plotData"][n]["x"]
                        coords = [(self.data["plotData"][n]["positions"][m], X.loc[dataIdx], intIDMatch[mIdx],dataIdx) for mIdx,dataIdx in enumerate(dataIndex) for m,X in enumerate(data) if dataIdx in X.index.values ]
                        coords = pd.DataFrame(coords, columns = ["x","y","intID","idx"])
                        
                        sortedDataIndex = coords["idx"].values
                        scatterColors = [propsData.loc[idx,"color"] for idx in sortedDataIndex]
                        scatterSizes = [propsData.loc[idx,"size"] for idx in sortedDataIndex]
                        self.quickSelectScatterDataIdx[ax] = {"idx":sortedDataIndex,"coords":coords}
                        self.updateQuickSelectScatter(ax,coords,scatterColors,scatterSizes)

            except Exception as e:
                print(e)

    
    def updateQuickSelectData(self,quickSelectGroup,changedCategory=None):
        ""
        for ax in self.axisDict.values():
            if self.isQuickSelectModeUnique():

                scatterSizes, scatterColors, _ = self.getQuickSelectScatterProps(ax,quickSelectGroup)
                

            elif ax in self.quickSelectScatterDataIdx: #mode == "raw"

                dataIdx = self.quickSelectScatterDataIdx[ax]["idx"]
                scatterSizes = [quickSelectGroup["size"].loc[idx] for idx in dataIdx]	
                scatterColors = [quickSelectGroup["color"].loc[idx] for idx in dataIdx]

            else:
                
                continue

            self.updateQuickSelectScatter(ax, scatterColors = scatterColors, scatterSizes = scatterSizes)
	

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
                if n in self.data["plotData"] and ax in self.backgrounds:
                    data = self.data["plotData"][n]["x"]
                    coords = np.array([(self.data["plotData"][n]["positions"][m], X.loc[dataIdx]) for dataIdx in dataIndex for m,X in enumerate(data) if dataIdx in X.index.values ])
                    self.p.f.canvas.restore_region(self.backgrounds[ax])
                    if coords.size > 0:
                        
                        self.setHoverScatterData(coords,ax)
                    else:
                        self.p.f.canvas.blit(ax.bbox)
            
    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        


