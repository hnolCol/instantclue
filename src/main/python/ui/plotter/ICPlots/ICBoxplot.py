 
from .ICChart import ICChart
from collections import OrderedDict
import numpy as np 
import pandas as pd
class ICBoxplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICBoxplot,self).__init__(*args,**kwargs)

        self.boxplotItems = dict() 
        
        
    def initBoxplots(self,onlyForID = None, targetAx = None):
        ""
        for n, boxplotProps in self.data["plotData"].items():
            if n in self.axisDict and onlyForID is None:
                
                if len(boxplotProps["x"]) == 1 and boxplotProps["x"][0].size == 1:
                    self.axisDict[n].plot(boxplotProps["positions"],boxplotProps["x"][0],
                        marker = self.getParam("boxplot.flierprops.marker"),
                        markeredgewidth = self.getParam("boxplot.flierprops.markeredgewidth"),
                        markeredgecolor = self.getParam("boxplot.flierprops.markeredgecolor"),
                        markersize = self.getParam("boxplot.flierprops.markersize"),
                        color="black")

                else:
                    self.boxplotItems[n] = self.axisDict[n].boxplot(**boxplotProps)
                
            elif n == onlyForID and targetAx is not None:
                self.targetBoxplotItems = dict()
                self.targetBoxplotItems[n] = targetAx.boxplot(**boxplotProps)
           
    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            self.initAxes(data["axisPositions"])
            
            self.initBoxplots()
          
            self.setXTicksForAxes(self.axisDict,data["tickPositions"],data["tickLabels"],rotation=90)
     
            for n,ax in self.axisDict.items():
                if n in data["axisLimits"]:
                    self.setAxisLimits(ax,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
     
            self.setAxisLabels(self.axisDict,data["axisLabels"])
         
            self.addTitles()
            #self.setHoverItemGroups(hoverGroupItems)

            self.addVerticalLines()
                
            self.savePatches()
            if self.interactive:
                for ax in self.axisDict.values():
                    self.addHoverScatter(ax) 
                #adda qucik select hover
                self.addQuickSelectHoverScatter()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])

            self.checkForQuickSelectDataAndUpdateFigure()
        except Exception as e:
        
            print(e)
        

    def reorderBoxplotItemsForHover(self):
        ""
        try:
            hoverGroupItems = {}
            for n, boxprops in self.boxplotItems.items():
                artists = dict([(m,box) for m,box in enumerate(boxprops["boxes"])])
                colors = dict([(m,box.get_facecolor()) for m,box in enumerate(boxprops["boxes"])])
                texts = dict([(m,self.data["tooltipsTexts"][n][m]) for m,box in enumerate(boxprops["boxes"])])
                hoverGroupItems[self.axisDict[n]] = {"artists":artists,"texts":texts,"colors":colors}# [[box,*boxprops["whiskers"][n*2:n*2+2],*boxprops["caps"][n*2:n*2+2]] for n,box in enumerate(boxprops["boxes"])]
        except Exception as e:
            print(e)
        return hoverGroupItems

    def savePatches(self):
        ""
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
        data = self.data
        self.setAxisLabels({axisID:targetAx},data["axisLabels"],onlyForID=axisID)
    
        self.addTitles(onlyForID = axisID, targetAx = targetAx)
        self.initBoxplots(onlyForID=axisID,targetAx=targetAx)
        self.setFacecolors(onlyForID=axisID)
        self.mirrorStats(targetAx,axisID)
        self.setXTicksForAxes({axisID:targetAx},data["tickPositions"],data["tickLabels"], onlyForID = axisID, rotation=90)          
        self.addSwarm("", [], [], onlyForID=axisID,targetAx=targetAx)
        self.addVerticalLines(axisID,targetAx)


