 
from .ICChart import ICChart
from collections import OrderedDict
import matplotlib.patches as patches
import numpy as np 
import pandas as pd
class ICViolinplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICViolinplot,self).__init__(*args,**kwargs)

        self.violinItems = dict() 
        
    
    def addMedianScatter(self, onlyForID = None, targetAx = None):
        ""
        scatKwargs = self.getScatterKwargs()
        scatKwargs["facecolor"] = "white"
        scatKwargs["alpha"] = 1
        scatKwargs["zorder"] = 4
        for n, scatterProps in self.data["medianData"].items():
            if onlyForID is not None and targetAx is not None:
                targetAx.scatter(**scatterProps, **scatKwargs)
            elif n in self.axisDict and onlyForID is None:
                self.axisDict[n].scatter(**scatterProps, **scatKwargs)
    
    def addMinMaxLine(self, onlyForID = None, targetAx = None):
        ""
        for n, lProps in self.data["minMaxLine"].items():
            if onlyForID is not None and targetAx is not None and n == onlyForID: #check if chart shuld be exported to main figure
                for x,ymin,ymax in lProps:
                    targetAx.vlines(x,ymin,ymax,color="black",lw=0.5,ls = "-")
            elif onlyForID is None and n in self.axisDict:
                for x,ymin,ymax in lProps:
                    self.axisDict[n].vlines(x,ymin,ymax,color="black",lw=0.5,ls = "-")
        

    def addQuantileLine(self, onlyForID = None, targetAx = None):
        ""
        showQuantiles = self.getParam("violin.show.quantiles")
        if not showQuantiles:
            return 
        edgeColor = self.getParam("violin.show.quantiles.box.edgecolor")
        edgeWidth = self.getParam("violin.show.quantiles.box.linewidth")
        faceColor = self.getParam("violin.show.quantiles.box.facecolor")
        alpha = self.getParam("violin.show.quantiles.box.alpha")
        showBox = self.getParam("violin.show.quantiles.box")
        for n, lProps in self.data["quantileLine"].items():
            if onlyForID is not None and targetAx is not None and n == onlyForID:
                ax = targetAx
            elif onlyForID is not None:
                continue
            elif n not in self.axisDict:
                continue
            else:
                ax = self.axisDict[n]

            for x,ymin,ymax in lProps:
                if  showBox:
                    r = patches.Rectangle(xy=(x-0.1,ymin), width=0.2, height = ymax-ymin, ec = edgeColor, linewidth=edgeWidth, facecolor=faceColor, alpha=alpha)
                    ax.add_patch(r)
                else:
                    ax.vlines(x,ymin,ymax,color=edgeColor,lw=2,ls = "-")
                        

    def initViolinplots(self, onlyForID = None, targetAx = None):
        ""
        for n, violinProps in self.data["plotData"].items():
        
            if onlyForID is not None and targetAx is not None and onlyForID == n:
                self.targetViolinItems = targetAx.violinplot(**violinProps)
            elif onlyForID is not None:
                continue
            else:
                if n in self.axisDict:
                    self.violinItems[n] = self.axisDict[n].violinplot(**violinProps)
               
    def onDataLoad(self, data):
        ""
        try:
           
            self.data = data
            self.initAxes(data["axisPositions"])
            self.initViolinplots()
            self.setFaceColors()
            self.addQuantileLine()
            self.addMinMaxLine()
            self.addMedianScatter()
            self.setXTicksForAxes(self.axisDict,data["tickPositions"],data["tickLabels"],rotation=90)
            self.setAxisLabels(self.axisDict,data["axisLabels"])
            self.addTitles()
            self.addVerticalLines()
            #set limits
            for n,ax in self.axisDict.items():
                if n in data["axisLimits"]:
                    self.setAxisLimits(ax,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            if self.interactive:
                for ax in self.axisDict.values():
                    self.addHoverScatter(ax) 
                self.addQuickSelectHoverScatter()
            self.checkForQuickSelectDataAndUpdateFigure()
            
        except Exception as e:
            
            print(e)
        


    def setFaceColors(self, onlyForID = None, targetAx = None):
        if onlyForID is not None:
            violinProps = self.violinItems[onlyForID]
            for pc,pcMirror in zip(violinProps["bodies"],self.targetViolinItems["bodies"]):
                pcMirror.set_edgecolor("black")
                pcMirror.set_alpha(1) 
                pcMirror.set_facecolor(pc.get_facecolor())
        else:
            self.groupColor = dict()  
            if "facecolors" in self.data and "dataColorGroups" in self.data:
                #get internal ID, to make color changes
                colorGroupData = self.data["dataColorGroups"]
                self.colorGroupArtists = OrderedDict([(intID,[]) for intID in colorGroupData["internalID"].values])

                for n, violinProps in self.violinItems.items():
                    if n in self.data["facecolors"]:
                        
                        for pc,fc in zip(violinProps['bodies'],self.data["facecolors"][n]):
                            #get internal ID
                            idx = colorGroupData.index[colorGroupData["color"] == fc]
                            intID = colorGroupData.loc[idx,"internalID"].iloc[0]
                            #save pathcollection
                            self.colorGroupArtists[intID].append(pc)
                            if intID not in self.groupColor:
                                self.groupColor[intID] = fc
                            pc.set_facecolor(fc)
                            pc.set_edgecolor('black')
                            pc.set_alpha(1)

                

    def savePatches(self):
        ""
        colorGroupData = self.data["dataColorGroups"]
        self.colorGroupArtists = OrderedDict([(group,[]) for group in colorGroupData["group"].values])
        self.groupColor = dict() 
        try:
            for n, boxprops in self.boxplotItems.items():
                for artist, color in zip(boxprops["boxes"],self.data["facecolors"][n]):
                    artist.set_facecolor(color)
                    
                    idx = colorGroupData.index[colorGroupData["color"] == color]
                    groupName = colorGroupData.loc[idx,"group"].iloc[0]
                    self.colorGroupArtists[groupName].append(artist)
                    if groupName not in self.groupColor:
                        self.groupColor[groupName] = color

        except Exception as e:
            print(e)



    def highlightGroupByColor(self,colorGroup,highlightCategory):
        """
        highlightCategory = None -> reset
        """
        nanColor = self.getParam("nanColor")
        for color, _ , intID in colorGroup.values:
            if intID in self.colorGroupArtists:
                artists = self.colorGroupArtists[intID]
                if intID != highlightCategory and highlightCategory is not None:
                    for artist in artists:
                        artist.set_facecolor(nanColor)
                else:
                    for artist in artists:
                        artist.set_facecolor(color)
        self.updateFigure.emit() 


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
        "Update backgrounds for the blitting of hover data (triggered by the QuickSelect hover)."
        if not hasattr(self,"backgrounds"):
            self.backgrounds = {}
        self.backgrounds.clear() 
        for ax in self.axisDict.values():
            self.backgrounds[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)


    def setHoverData(self,dataIndex):
        "Show the hover data. This is the hover of the QuickSelect Widget."
        if hasattr(self,"backgrounds"):
            for n, ax in self.axisDict.items():
                if n in self.data["hoverData"] and ax in self.backgrounds:
                    #print(self.data["plotData"][n])
                    data = self.data["hoverData"][n]["x"]
                    coords = np.array([(self.data["plotData"][n]["positions"][m], X.loc[dataIdx]) for dataIdx in dataIndex for m,X in enumerate(data) if dataIdx in X.index.values ])
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
                        
                        coords =  [(self.data["plotData"][n]["positions"][m], X.loc[dataIdx], intIDMatch[mIdx], dataIdx) for mIdx,dataIdx in enumerate(dataIndex) for m,X in enumerate(data) if dataIdx in X.index.values ]
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
        self.setAxisLabels({axisID:targetAx},data["axisLabels"],onlyForID=axisID)
    
       # self.addTitles()
        self.initViolinplots(onlyForID=axisID,targetAx=targetAx)
        self.addQuantileLine(onlyForID=axisID,targetAx=targetAx)
        self.setFaceColors(onlyForID=axisID,targetAx=targetAx)
        self.addMedianScatter(onlyForID=axisID,targetAx=targetAx)
        self.addMinMaxLine(onlyForID=axisID,targetAx=targetAx)
        #self.setFacecolors(onlyForID=axisID)
        self.mirrorStats(targetAx,axisID)
        self.setXTicksForAxes({axisID:targetAx},data["tickPositions"],data["tickLabels"], onlyForID = axisID, rotation=90)        
        self.addVerticalLines(axisID,targetAx)  
        self.addSwarm("", [], [], onlyForID=axisID,targetAx=targetAx)