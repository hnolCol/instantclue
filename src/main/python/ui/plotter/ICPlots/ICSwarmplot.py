 
from .ICChart import ICChart
from .charts.scatter_plotter import scatterPlot
from collections import OrderedDict
import numpy as np 
class ICSwarmplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICSwarmplot,self).__init__(*args,**kwargs)

        self.scatterPlots = dict() 

    def addTooltip(self, tooltipColumnNames,dataID):
        "Tooltips in scatter plots use the natural hover bindings (no need to disconnect)."
        data = self.mC.data.getDataByColumnNames(dataID,tooltipColumnNames)["fnKwargs"]["data"]
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.addTooltip(data )
        

    def disconnectBindings(self):
        ""
        super().disconnectBindings()
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.disconnectBindings()

    def initScatterPlots(self, onlyForID = None, targetAx = None):
        ""
        if onlyForID is None:
            #clear saved scatters
            self.scatterPlots.clear()
            #init scatters
            self.scatterKwargs = self.getScatterKwargs()
            for n,ax in self.axisDict.items():
                columnPair = self.data["columnPairs"][n]
                self.scatterPlots[n] = scatterPlot(
                                        self,
                                        data = self.data["plotData"],
                                        plotter = self.p,
                                        ax = ax,
                                        numericColumns = list(columnPair),
                                        dataID = self.data["dataID"],
                                        scatterKwargs = self.scatterKwargs,
                                        hoverKwargs = self.getHoverKwargs(),
                                        multiScatter = True,
                                        multiScatterKwargs = self.data["multiScatterKwargs"][n]
                                        )
        else:
            columnPair = self.data["columnPairs"][onlyForID]
            scatterPlot(
                        self,
                        data = self.data["plotData"],
                        plotter = self.p,
                        ax = targetAx,
                        numericColumns = list(columnPair),
                        dataID = self.data["dataID"],
                        scatterKwargs = self.scatterKwargs,
                        hoverKwargs = self.getHoverKwargs(),
                        multiScatter = True,
                        multiScatterKwargs = self.data["multiScatterKwargs"][onlyForID],
                        interactive = False
                        )

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            self.initAxes(data["axisPositions"])
            self.setAxisLabels(self.axisDict,self.data["axisLabels"])
            self.setXTicksForAxes(self.axisDict,data["tickPositions"],data["tickLabels"],rotation=90)
            self.setColorCategoryIndexMatch(self.data["colorCategoryIndexMatch"])
            self.setSizeCategoryIndexMatch(self.data["sizeCategoryIndexMatch"])
            self.initScatterPlots()
            self.addTitles()
            self.adjustAxisLimits(self.axisDict,data["axisLimits"])
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.setDataInSizeTable(self.data["dataSizeGroups"],title= self.data["colorCategoricalColumn"])
            self.checkForQuickSelectDataAndUpdateFigure()
            
        except Exception as e:
            print(e)
    
    def adjustAxisLimits(self,axisDict,axisLimits):
        "Adjust axis limits, axisDict and axisLimits must have same keys"
        for n,ax in axisDict.items():
            if n in axisLimits:
                self.setAxisLimits(ax,yLimit=axisLimits[n]["yLimit"],xLimit=axisLimits[n]["xLimit"])
    
    def setHoverData(self,dataIndex, sender = None):
        "Sets hover data in scatter plots"
        for scatterPlot in self.scatterPlots.values():
            if sender is None:
                scatterPlot.setHoverData(dataIndex)
            elif sender != scatterPlot:
                scatterPlot.setHoverData(dataIndex)


    def updateQuickSelectItems(self,propsData):
        ""
        if self.isQuickSelectActive():
            if self.isQuickSelectModeUnique():
                dataIndex = np.concatenate([idx for idx in self.quickSelectCategoryIndexMatch.values()])
            else:
                dataIndex = self.getDataIndexOfQuickSelectSelection()

            for scatterPlot in self.scatterPlots.values():
                if hasattr(scatterPlot,"quickSelectScatter"):
                    scatterPlot.setQuickSelectScatterData(dataIndex,propsData.loc[dataIndex])
                    self.quickSelectScatterDataIdx[scatterPlot.ax] = dataIndex

        self.updateFigure.emit()


    def updateQuickSelectData(self,quickSelectGroup,changedCategory=None):
        ""
        for scatterPlot in self.scatterPlots.values():
            ax = scatterPlot.ax
            
            if self.isQuickSelectModeUnique():

                scatterSizes, scatterColors, _ = self.getQuickSelectScatterProps(quickSelectGroup)

            else:
                if ax in self.quickSelectScatterDataIdx:
                    dataIdx = self.quickSelectScatterDataIdx[ax]
                    scatterSizes = [quickSelectGroup["size"].loc[idx] for idx in dataIdx]	
                    scatterColors = [quickSelectGroup["color"].loc[idx] for idx in dataIdx]
                
            scatterPlot.updateQuickSelectScatter(scatterColors,scatterSizes)

        self.updateFigure.emit()

    def updateBackgrounds(self, redraw = False):
        "Update backgrouns for hover / bit capabilities"
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.updateBackground(redraw=redraw)                


    def changedCategoryIsInternalID(self, changedCategory ):
        ""
        if "interalIDColumnPairs" in self.data:
            internalIDFound = [(n,columnPairMatches) for n,columnPairMatches in self.data["interalIDColumnPairs"].items() if changedCategory in columnPairMatches]
            return len(internalIDFound) > 0
        return False

    def updateGroupColors(self,colorGroup, changedCategory = None):
        "changed category == internal id!"
        if self.colorCategoryIndexMatch is not None:
            
            if changedCategory is not None and changedCategory in self.colorCategoryIndexMatch:
                idx = self.colorCategoryIndexMatch[changedCategory]
                dataBool = colorGroup["internalID"] == changedCategory 
                color = colorGroup.loc[dataBool,"color"].values[0]
                #is the internalID part of the initial coloring (special to swarm plots?)
                if self.changedCategoryIsInternalID(changedCategory):
                    #why iterating through? check
                    for n, columnPairMatches in self.data["interalIDColumnPairs"].items():
                        if changedCategory in columnPairMatches:
                            columnPairs = columnPairMatches[changedCategory]
                            for columnPair in columnPairs:
                                self.scatterPlots[n].setColorForMultiScatter(columnPair,idx,color)
                else: #indicates that it is not about the original colors, but rather a category
                    for scatterplot in self.scatterPlots.values():
                        scatterplot.updateColorDataByIndex(idx,color)

            elif changedCategory is None and "internalID" in colorGroup.columns:
                
                    for internalID, color in colorGroup[["internalID","color"]].values:
                        idx = self.colorCategoryIndexMatch[internalID]
                        
                        if self.changedCategoryIsInternalID(changedCategory):
                            for n, columnPairMatches in self.data["interalIDColumnPairs"].items():
                                if internalID in columnPairMatches:
                                    columnPairs = columnPairMatches[internalID]
                                    for columnPair in columnPairs:
                                        self.scatterPlots[n].setColorForMultiScatter(columnPair,idx,color)

                        else: ##all changed
                            for scatterplot in self.scatterPlots.values():
                                    scatterplot.updateColorDataByIndex(idx,color)


        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)

        self.updateFigure.emit()
    
    def updateGroupSizes(self,sizeGroup, changedCategory = None):
        "changed category == internal id!"
        if self.sizeCategoryIndexMatch is not None:
            
            if changedCategory in self.sizeCategoryIndexMatch:
                try:
                    idx = self.sizeCategoryIndexMatch[changedCategory]
                    dataBool = sizeGroup["internalID"] == changedCategory 
                    size = sizeGroup.loc[dataBool,"size"].values[0]
                   
                    for n, columnPairMatches in self.data["interalIDColumnPairs"].items():
                        if changedCategory in columnPairMatches:
                            columnPairs = columnPairMatches[changedCategory]
                            for columnPair in columnPairs:
                                self.scatterPlots[n].setSizesForMultiScatter(columnPair,idx,size)
                    #self.updateScatterPropSection(idx,color,"color")
                except Exception as e:
                    print(e)
            #if hasattr(self,"colorLegend"):
             #   self.addColorLegendToGraph(colorGroup,update=False)
            self.updateFigure.emit()


    def mirrorAxisContent(self,axisID,targetAx,*args,**kwargs):
        "Mirror axis content to another axis"
        self.initScatterPlots(axisID,targetAx)
        self.setAxisLabels({axisID:targetAx},self.data["axisLabels"],onlyForID=axisID)
        self.setXTicksForAxes({axisID:targetAx},self.data["tickPositions"],self.data["tickLabels"], onlyForID = axisID, rotation=90)     
