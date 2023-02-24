


from .ICChart import ICChart
from ...dialogs.ICAUCCalculation import ICDAUCDialog
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd 

from backend.utils.stringOperations import getRandomString

class ICXYPlot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICXYPlot,self).__init__(*args,**kwargs)

        self.xyplotItems = dict() 

    def addGraphSpecActions(self,menus):
        ""
        menus["main"].addAction("Area under curve", self.openAUCCalcDialog)

    def initXYPlot(self, onlyForID = None, targetAx = None):
        ""
        
        hoverGroups = dict() 
        if onlyForID is None and targetAx is None:
            for n,ax in self.axisDict.items():
                hoverGroups[ax] = {'colors': {}, 'artists' : {}, 'texts' : {}, "internalID" : {}}
                for m,l in enumerate(self.data["lines"][n]):
                    artistID = getRandomString()
                    intID = self.data["lineKwargs"][n][m]["ID"]

                    if isinstance(l,Line2D):
                        ax.add_line(l)
                    elif isinstance(l,LineCollection):
                        ax.add_collection(l)

                    #extract tooltip information
                    boolIdx = self.data['dataColorGroups']["internalID"] == intID
                    groupLabel = self.data['dataColorGroups'].loc[boolIdx,"group"].values[0]
                    
                    hoverGroups[ax]["artists"][artistID] = l
                    hoverGroups[ax]["colors"][artistID] = l.get_color()
                    hoverGroups[ax]["texts"][artistID] = groupLabel

                    if intID not in hoverGroups[ax]["internalID"]:
                            hoverGroups[ax]["internalID"][intID] = []
                    hoverGroups[ax]["internalID"][intID].append(artistID)
                    
                if n in self.data["markerLines"]:
                    for mLine in self.data["markerLines"][n]:
                        if mLine is not None:
                            ax.add_line(mLine)
                
        #handle export to main figure (onlyForID -> axis number)
        elif onlyForID in self.data["lines"]:
            for m,l in enumerate(self.data["lines"][onlyForID]):
                if isinstance(l,Line2D):
                    targetAx.add_line(Line2D(**self.data["lineKwargs"][onlyForID][m]["props"]))
                elif isinstance(l,LineCollection):
                    targetAx.add_collection(LineCollection(**self.data["lineKwargs"][onlyForID][m]["props"]))

            if onlyForID in self.data["markerLines"]:
                for markerKwrags in self.data["markerKwargs"][onlyForID]:
                    if "xdata" in markerKwrags["props"] and "ydata" in markerKwrags["props"]:
                        
                        targetAx.add_line(Line2D(**markerKwrags["props"]))
        return hoverGroups 

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            self.initAxes(data["axisPositions"])
            self.setAxisLabels(self.axisDict,self.data["axisLabels"])
            hoverGroups = self.initXYPlot()
            for n,ax in self.axisDict.items():
                if n in self.data["axisLimits"]:
                    self.setAxisLimits(ax,
                            self.data["axisLimits"][n]["xLimit"],
                            self.data["axisLimits"][n]["yLimit"])

            if self.interactive:
                for ax in self.axisDict.values():
                    self.addHoverScatter(ax) 

                self.addQuickSelectHoverScatter()

            #self.addTitles()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.setHoverItemGroups(hoverGroups)
            self.checkForQuickSelectDataAndUpdateFigure()
           
        except Exception as e:
            print(e)
        

    def setHoverData(self,dataIndex, showText = False):
        ""
        #find matching idx 
        idx = self.data["hoverData"].index.intersection(dataIndex)
        
        if hasattr(self,"backgrounds"):
            for n, ax in self.axisDict.items():
                if ax in self.backgrounds:
                    numericPair = self.data["numericColumnPairs"][n]
                    coords = self.data["hoverData"].loc[idx,numericPair].values
                    self.p.f.canvas.restore_region(self.backgrounds[ax])
                    if coords.size > 0:
                        self.setHoverScatterData(coords,ax)
                    else:
                        self.p.f.canvas.blit(ax.bbox)
    

    def openAUCCalcDialog(self,e=None):
        ""
        dlg = ICDAUCDialog(self.mC,self.data["dataID"],self.data["numericColumnPairs"],self.data["hoverData"])
        dlg.exec()

    def updateQuickSelectItems(self,propsData=None):
        
       # colorData = self.getQuickSelectData()
        dataIndex = self.getDataIndexOfQuickSelectSelection()
        idx = self.data["hoverData"].index.intersection(dataIndex)
        if not hasattr(self,"backgrounds"):
            self.updateBackgrounds()
        
        if hasattr(self,"quickSelectScatter"):
            try:
                for n,ax in self.axisDict.items():
                    if ax in self.backgrounds and ax in self.quickSelectScatter:                        
                        numericPair = self.data["numericColumnPairs"][n]
                        coords = self.data["hoverData"].loc[idx,numericPair].values

                        scatterColors = propsData.loc[idx,"color"]
                        scatterSizes = propsData.loc[idx,"size"]
                        self.quickSelectScatterDataIdx[ax] = idx
                        self.updateQuickSelectScatter(ax,coords,scatterColors,scatterSizes)
            
            except Exception as e:
                print(e)

    def updateGroupColors(self,colorGroup,changedCategory=None):
        "changed category is encoded in a internalID"
       
        if changedCategory is not None:
            changedCategories = [changedCategory]
        elif "internalID" in colorGroup.columns:
            changedCategories = colorGroup["internalID"].values.tolist() 
        else:
            return 

        if "linesByInternalID" in self.data:
            for changedCategory in changedCategories:
                l = self.data["linesByInternalID"][changedCategory]
                changedColor = colorGroup.loc[colorGroup["internalID"] == changedCategory]["color"].values[0]
                if hasattr(l,"set_color"):
                    l.set_color(changedColor)
                if isinstance(l,list):
                    for line in l:
                        if hasattr(line,"set_markerfacecolor"):
                            line.set_markerfacecolor(changedColor)
                        line.set_color(changedColor)
                #change color in props for export
                for lineKwargs in self.data["lineKwargs"].values():
                    for kwg in lineKwargs:
                        if kwg["ID"] == changedCategory:
                            kwg["props"]["color"] = changedColor
                
                for markerKwargs in self.data["markerKwargs"].values():
                    for kwg in markerKwargs:
                        if kwg["ID"] == changedCategory:
                            kwg["props"]["markerfacecolor"] = changedColor
                self.adjustColorsInTooltip(changedCategory,changedColor)

        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)
        self.updateFigure.emit()

    def updateBackgrounds(self):
        "Update Background for blitting"
        if not hasattr(self,"backgrounds"):
            self.backgrounds = {}
        self.backgrounds.clear() 
        for ax in self.axisDict.values():
            self.backgrounds[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)

    
    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        
        self.initXYPlot(onlyForID=axisID,targetAx=targetAx)
        self.setAxisLabels({axisID:targetAx},self.data["axisLabels"],onlyForID=axisID)