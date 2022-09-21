 
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from .ICChart import ICChart
from collections import OrderedDict
import numpy as np 
import pandas as pd

class ICBoxenplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICBoxenplot,self).__init__(*args,**kwargs)
        self.colorGroupArtists = OrderedDict()
        self.groupColor = dict()
        self.boxplotItems = dict() 
    
    def addGraphSpecActions(self,menus):
        ""
        # menus["main"].addAction("Show summary data", self.displaySummaryData)

    def displaySummaryData(self,*args,**kwargs):
        ""
        # if "groupedPlotData" in self.data:
        #     self.mC.mainFrames["data"].openDataFrameinDialog(self.data["groupedPlotData"], 
        #                             ignoreChanges=True, 
        #                             headerLabel="Boxplot data.", 
        #                             tableKwargs={"forwardSelectionToGraph":False})
        
   
    def initBoxenplots(self,onlyForID=None,targetAx=None):
        ""
        self.boxenCollection = OrderedDict()
        for n,rectProps in self.data["plotData"].items():
            if n in self.axisDict and onlyForID is None:
                ax = self.axisDict[n]
                
                for r in rectProps:
                    if "rectProps" in r and len(r["rectProps"]) > 0 and r["cmap"] is not None:
                        boxes = [Rectangle(**rProps) for rProps in r["rectProps"]]
                        collection = PatchCollection(
                            boxes, 
                            cmap=r["cmap"], 
                            edgecolor="black", 
                            linewidth=self.getParam("boxen.boxprops.linewidth")
                            )
                        
                            # Set the color gradation, first box will have color=hex_color
                        collection.set_array(np.array(np.linspace(1, 0, len(boxes))))
                        
                        ax.add_collection(collection)
                    if "medianLine" in r and len(r["medianLine"]) > 0:
                        l = Line2D(**r["medianLine"])
                        ax.add_line(l)
                    
                    if "internalID" in r and "faceColor" in r:
                        intID = r["internalID"]
                        if intID not in self.colorGroupArtists:
                            self.colorGroupArtists[intID] = []
                        self.colorGroupArtists[intID].append(collection)
                        if intID not in self.groupColor:
                            self.groupColor[intID] = r["faceColor"]
                    
                    if "outlierProps" in r and len(r["outlierProps"]) > 0:
                        l = Line2D(**r["outlierProps"])
                        ax.add_line(l)

            elif targetAx is not None and n == onlyForID:
                for r in rectProps:
                    ax = targetAx
                    boxes = [Rectangle(**rProps) for rProps in r["rectProps"]]
                    intID = r["internalID"]
                    hexColor = self.groupColor[intID]
                    cmap = self._getCampForBoxes(hexColor)
                    collection = PatchCollection(
                                boxes, 
                                cmap=cmap, 
                                edgecolor="black", 
                                linewidth=self.getParam("boxen.boxprops.linewidth")
                                )
                            
                                # Set the color gradation, first box will have color=hex_color
                    collection.set_array(np.array(np.linspace(1, 0, len(boxes))))
                    ax.add_collection(collection)
                    l = Line2D(**r["medianLine"])
                    ax.add_line(l)
                        
    def _getCampForBoxes(self, hexColor):
        ""
        rgb = [hexColor, (1, 1, 1)]
        cmap = LinearSegmentedColormap.from_list('new_map', rgb)
        # Make sure that the last boxes contain hue and are not pure white
        rgb = [hexColor, cmap(.85)]
        cmap = LinearSegmentedColormap.from_list('new_map', rgb)

        return cmap

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            self.initAxes(data["axisPositions"])
            
            self.initBoxenplots()
            
            self.setXTicksForAxes(self.axisDict,data["tickPositions"],data["tickLabels"],rotation=90)
     
            for n,ax in self.axisDict.items():
                if n in data["axisLimits"]:
                    self.setAxisLimits(ax,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
     
            self.setAxisLabels(self.axisDict,data["axisLabels"])
         
            self.addTitles()
            #self.setHoverItemGroups(hoverGroupItems)

            self.addVerticalLines()
                
            if self.interactive:
                for ax in self.axisDict.values():
                    self.addHoverScatter(ax) 
                #adda qucik select hover
                self.addQuickSelectHoverScatter()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])

            self.checkForQuickSelectDataAndUpdateFigure()
        except Exception as e:
        
            print(e)
    

    def highlightGroupByColor(self,colorGroup,highlightCategory):
        """
        highlightCategory = None -> reset
        """
       
        nanColor = self.getParam("nanColor")
        for color, _ , intID in colorGroup.values:

            if intID in self.colorGroupArtists:
                    
                if intID != highlightCategory and highlightCategory is not None:
                    cmap = self._getCampForBoxes(nanColor)
                    
                else:
                    cmap = self._getCampForBoxes(color)
                collections = self.colorGroupArtists[intID]
                for collection in collections:
                    collection.set_cmap(cmap)
                
                self.groupColor[intID] = color
       
        self.updateFigure.emit()


    def updateGroupColors(self,colorGroup,changedCategory=None):
        ""
        
        for color, _ , intID in colorGroup.values:
            if intID in self.colorGroupArtists:
                if self.groupColor[intID] != color:
                    collections = self.colorGroupArtists[intID]
                    cmap = self._getCampForBoxes(color)
                    
                    for collection in collections:
                        collection.set_cmap(cmap)
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
        
        #print(dataIndex)

        if hasattr(self,"quickSelectScatter"):
            try:
                for n,ax in self.axisDict.items():
                    if n in self.data["hoverData"] and ax in self.backgrounds and ax in self.quickSelectScatter: 
                        data = self.data["hoverData"][n]["x"]
                    
                        coords = [(self.data["positions"][n][m], X.loc[dataIdx], intIDMatch[mIdx],dataIdx) for mIdx,dataIdx in enumerate(dataIndex) for m,X in enumerate(data) if dataIdx in X.index.values ]
                        coords = pd.DataFrame(coords, columns = ["x","y","intID","idx"])
                        sortedDataIndex = coords["idx"].values
                        scatterColors = [propsData.loc[idx,"color"] for idx in sortedDataIndex]
                        scatterSizes = [propsData.loc[idx,"size"] for idx in sortedDataIndex]
                        self.quickSelectScatterDataIdx[ax] = {"idx":sortedDataIndex,"coords":coords}
                        self.updateQuickSelectScatter(ax,coords,scatterColors,scatterSizes)

            except Exception as e:
               
                print(e)

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
        if hasattr(self,"backgrounds"):
            for n, ax in self.axisDict.items():
                if n in self.data["hoverData"] and ax in self.backgrounds:
                    data = self.data["hoverData"][n]["x"]
                    coords = np.array([(self.data["positions"][n][m], X.loc[dataIdx]) for dataIdx in dataIndex for m,X in enumerate(data) if dataIdx in X.index.values ])
                    self.p.f.canvas.restore_region(self.backgrounds[ax])                  
                    if coords.size > 0:
                        
                        self.setHoverScatterData(coords,ax)
                    else:
                        self.p.f.canvas.blit(ax.bbox)
            
    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        data = self.data
        self.setAxisLabels({axisID:targetAx},data["axisLabels"],onlyForID=axisID)
    
        self.addTitles(onlyForID = axisID, targetAx = targetAx)
        self.initBoxenplots(onlyForID=axisID,targetAx=targetAx)
        self.mirrorStats(targetAx,axisID)
        self.setXTicksForAxes({axisID:targetAx},data["tickPositions"],data["tickLabels"], onlyForID = axisID, rotation=90)          
        #self.addSwarm("", [], [], onlyForID=axisID,targetAx=targetAx)
        self.addVerticalLines(axisID,targetAx)


     
       

