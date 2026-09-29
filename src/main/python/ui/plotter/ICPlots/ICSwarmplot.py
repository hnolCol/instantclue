 
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
            # Export/mirror to the main figure must read the CURRENT, live
            # per-axis scatterPlot's own data (as ICScatter.py's precedent
            # does: exportData = self.scatterPlots[columnPair].data), not
            # self.data["plotData"]. The chart-level plotData is a one-time
            # snapshot from the initial onDataLoad() and is never updated by
            # runtime color changes (updateGroupColors/setNaNColor mutate
            # only each scatterPlot instance's own .data) - using it here
            # silently reverted any color changes (including "Remove Color
            # Encoding") on every export to the main figure.
            exportData = self.scatterPlots[onlyForID].data
            # self.data["multiScatterKwargs"][onlyForID] is the ONE-TIME dict built
            # by the backend at initial chart construction (each columnPair -> its
            # own {"color":..}), used by scatterPlot.replot()'s per-collection branch
            # to color each internalID/group independently on the SAME axis. Runtime
            # per-group recoloring (updateGroupColors -> setColorForMultiScatter)
            # only ever (a) pushes the new color onto the already-drawn collection via
            # set_facecolor(), and (b) records it in a "color(x:y)" column on the live
            # scatterPlot's own .data - it never writes back into this chart-level
            # dict. Passing that stale dict straight through on export made replot()
            # rebuild every group's collection with its ORIGINAL color, silently
            # discarding any live per-group recoloring (independent of the item 38
            # fix above, which only covers "Remove Color Encoding" / single-collection
            # staleness, not this per-group multi-collection path).
            multiScatterKwargsForExport = self.getLiveMultiScatterKwargs(
                exportData, self.data["multiScatterKwargs"][onlyForID]
                )
            scatterPlot(
                        self,
                        data = exportData,
                        plotter = self.p,
                        ax = targetAx,
                        numericColumns = list(columnPair),
                        dataID = self.data["dataID"],
                        scatterKwargs = self.scatterKwargs,
                        hoverKwargs = self.getHoverKwargs(),
                        multiScatter = True,
                        multiScatterKwargs = multiScatterKwargsForExport,
                        interactive = False
                        )

    def getLiveMultiScatterKwargs(self, exportData, staleMultiScatterKwargs):
        "\n        Rebuild a per-columnPair scatterKwargs dict for export, overriding\n        each pair's stale initial color with whatever color is CURRENTLY\n        recorded for that pair (i.e. reflecting any setColorForMultiScatter/\n        setSizesForMultiScatter runtime updates), read back from the live\n        scatterPlot's own 'color(x:y)' data column. Rows belonging to other\n        column pairs are NaN in that column (their x/y is NaN for this pair\n        too, so they never render) and are filled with a placeholder purely\n        to keep the color array's length equal to the data's full length,\n        which matplotlib's scatter() requires even for masked points.\n        "
        exportKwargs = {}
        for columnPair, kwargs in staleMultiScatterKwargs.items():
            pairKwargs = dict(kwargs)
            colorColumnName = "color({}:{})".format(*columnPair)
            if colorColumnName in exportData.columns:
                colorSeries = exportData[colorColumnName]
                if colorSeries.notna().any():
                    fillValue = pairKwargs.get("color", colorSeries.dropna().iloc[0])
                    pairKwargs["color"] = colorSeries.fillna(fillValue).values
            exportKwargs[columnPair] = pairKwargs
        return exportKwargs

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
                else:
                    #item 60 (B6): same as ICScatter.updateQuickSelectData - skip panels
                    #without quick-selected points instead of reusing stale colours
                    continue
                
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

    def setNaNColor(self):
        ""
        # "Remove Color Encoding" (ICColorTable.removeFromGraph) calls
        # graph.setNaNColor() for any chart where hasScatters() is True.
        # ICSwarmplot never overrode this - it fell back to ICChart's
        # no-op base implementation, so removing colors on a swarm plot
        # never actually reset the drawn points (on the original axis or
        # on export), unlike ICScatter's own setNaNColor (item 36). Mirror
        # that fix here: reset every underlying scatterPlot's points to the
        # NaN color and invalidate any cached categorical color legend.
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.setNaNColorToCollection()
        self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
        if hasattr(self,"colorLegend"):
            try:
                self.colorLegend.remove()
            except Exception as e:
                print(e)
            del self.colorLegend
        if hasattr(self,"colorLegendKwargs"):
            del self.colorLegendKwargs
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
