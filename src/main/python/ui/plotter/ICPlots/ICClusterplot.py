 
from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex
import numpy as np 
import pandas as pd
from ...custom.tableviews.ICDataTable import PandaTableDialog


class ICClusterplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICClusterplot,self).__init__(*args,**kwargs)
        self.boxplotItems = {} 
    
    def addGraphSpecActions(self,menus):
        ""
        menus["main"].addAction("Show data in cluster", self.displayClusterSpecificData)
    
    def initClusters(self,onlyForID = None, targetAx = None):
        ""
        if self.getParam("clusterplot.type") == "boxplot": 
            for n, boxplotProps in self.data["plotData"].items():
                if n in self.axisDict and onlyForID is None:
                    self.boxplotItems[n] = self.axisDict[n].boxplot(**boxplotProps)
                elif n == onlyForID and targetAx is not None:
                    self.targetBoxplotItems = dict()
                    self.targetBoxplotItems[n] = targetAx.boxplot(**boxplotProps)

        elif self.getParam("clusterplot.type") == "lineplot":
            self.colorGroupArtists = OrderedDict()
            self.groupColor = OrderedDict() 
            for n, lineProps in self.data["plotData"].items():
                
                if n in self.axisDict and onlyForID is None:
                    lineCollection = LineCollection(
                                            **lineProps,
                                            colors=self.data["facecolors"][n], 
                                            linewidths = self.getParam("clusterplot.linewidth"),
                                            alpha = self.getParam("alpha")
                                        )
             
                    if not self.getParam("clusterplot.lineplot.color.distance"):
                        intID = self.data["dataColorGroups"]["internalID"].iloc[n]
                        self.colorGroupArtists[intID] = [lineCollection]
                        self.groupColor[intID] = self.data["facecolors"][n] if isinstance(self.data["facecolors"][n],str) else to_hex(self.data["facecolors"][n][0])
                    
                    self.axisDict[n].add_collection(lineCollection)

                elif onlyForID is not None and targetAx is not None and onlyForID == n:
                    lineCollection = LineCollection(
                                        **lineProps,
                                        colors=self.data["facecolors"][n], 
                                        linewidths = self.getParam("clusterplot.linewidth"),
                                        alpha = self.getParam("alpha")
                                        )
                    targetAx.add_collection(lineCollection)                    
        
    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            
            self.initAxes(data["axisPositions"])
            
            if "tickPositions" in data and "tickLabels" in data:
                self.setXTicksForAxes(self.axisDict,data["tickPositions"],data["tickLabels"],rotation=90, onlyLastRow=True)
            
            if "axisLimits" in data:
                for n,ax in self.axisDict.items():
                    if n in data["axisLimits"]:
                        self.setAxisLimits(ax,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
            
            if "axisLabels" in data:
                self.setAxisLabels(self.axisDict,data["axisLabels"], onlyLastRowForX=True, onlyFirstColumnForY=True)
         
            self.addTitles()
            self.initClusters()
            self.savePatches()
            self.addExtraLines(self.axisDict,self.data["extraLines"])
           
            if self.interactive:
                for ax in self.axisDict.values():
                    self.addHoverScatter(ax) 
                #adda qucik select hover
                self.addQuickSelectHoverScatter()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.checkForQuickSelectDataAndUpdateFigure()

        except Exception as e:
            print(e)


    def displayClusterSpecificData(self):
        ""
        if hasattr(self,"menuClickedInAxis"):
            axisID = self.getAxisID(self.menuClickedInAxis)
            columnName = self.data["clusterLabels"].columns[0]
            selectedCluster = self.data["clusterLabels"][columnName].unique()[axisID]
            clusterDataIndex = self.data["clusterLabels"].loc[self.data["clusterLabels"][columnName] == selectedCluster].index
            dataFrame = self.mC.data.dfs[self.data["dataID"]].loc[clusterDataIndex]
            dlg = PandaTableDialog(mainController = self.mC ,df = dataFrame, parent=self.mC, ignoreChanges=True)
            dlg.exec_()
            

            

    def getClusterIDsByDataIndex(self):
        ""
        return self.data["clusterLabels"], self.data["dataID"]
    

    def savePatches(self):
        ""
        if self.getParam("clusterplot.type") == "boxplot":
            colorGroupData = self.data["dataColorGroups"]
            self.colorGroupArtists = OrderedDict([(intID,[]) for intID in colorGroupData["internalID"].values])
            self.groupColor = dict() 
            self.setFacecolors(colorGroupData)

    def setFacecolors(self, colorGroupData = None, onlyForID = None):
        ""
        if onlyForID is not None and hasattr(self,"targetBoxplotItems"):
            for n, boxprops in self.targetBoxplotItems.items():
                plottedBoxProps = self.boxplotItems[onlyForID] #get boxes from plotted items
                for artist, plottedArtist in zip(boxprops["boxes"],plottedBoxProps["boxes"]):
                    artist.set_facecolor(plottedArtist.get_facecolor())
        else:
            for n, boxprops in self.boxplotItems.items():
                for artist, fc in zip(boxprops["boxes"],self.data["facecolors"][n]):   
                    artist.set_facecolor(fc)

                    if colorGroupData is not None and hasattr(self,"groupColor"):
                        idx = colorGroupData.index[colorGroupData["color"] == fc]
                        
                        intID = colorGroupData.loc[idx,"internalID"].iloc[0]
                        self.colorGroupArtists[intID].append(artist)
                        if intID not in self.groupColor:
                            self.groupColor[intID] = fc


    def updateGroupColors(self,colorGroup,changedCategory=None):
        ""
        
        for color, _ , intID in colorGroup.values:
            if intID in self.colorGroupArtists:
                if self.groupColor[intID] != color:
                    artists = self.colorGroupArtists[intID]
                    if self.getParam("clusterplot.type") == "lineplot":
                        #Line collection at list position 0
                        artists[0].set_colors(color)
                    else:
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
                        if n in self.data["quickSelect"] and ax in self.backgrounds and ax in self.quickSelectScatter:                        
                            data = self.data["quickSelect"][n]["x"]
                            coords = [(self.data["quickSelect"][n]["positions"][m], X.loc[dataIdx], intIDMatch[mIdx],dataIdx) for mIdx,dataIdx in enumerate(dataIndex) for m,X in enumerate(data) if dataIdx in X.index.values ]
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
                #if self.getParam("clusterplot.type") == "boxplot":
                if n in self.data["quickSelect"] and ax in self.backgrounds:
                    data = self.data["quickSelect"][n]["x"]
                    coords = np.array([(self.data["quickSelect"][n]["positions"][m], X.loc[dataIdx]) for dataIdx in dataIndex for m,X in enumerate(data) if dataIdx in X.index.values ])
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
        self.initClusters(onlyForID=axisID,targetAx=targetAx)
        if self.getParam("clusterplot.type") == "boxplot":
            self.setFacecolors(onlyForID=axisID)
        
        self.addExtraLines(self.axisDict,self.data["extraLines"],axisID,targetAx)
        self.setXTicksForAxes({axisID:targetAx},data["tickPositions"],data["tickLabels"], onlyForID = axisID, rotation=90)


