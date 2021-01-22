 
from .ICChart import ICChart
import numpy as np
from backend.color.data import colorParameterRange
from matplotlib.pyplot import colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns


class ICClustermap(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICClustermap,self).__init__(*args,**kwargs)
        self.meshGridKwargs = dict()
        self.movingMaxDLine = False
        self.forceLabels = False


    def addAnnotations(self, labelColumnNames,dataID):
        ""
        try:
            self.labelData = self.mC.data.getDataByColumnNames(dataID,labelColumnNames)["fnKwargs"]["data"]
            labelData = pd.DataFrame() 
            labelData["columnName"] = labelColumnNames
            self.setDataInLabelTable(labelData,title="Annotation Labels")
            self.onClusterYLimChange()
        except Exception as e:
            print(e)


    def addGraphSpecActions(self,menus):
        ""
        if "clusterRectangles" in self.data and len(self.data["clusterRectangles"]) != 0 and self.data["clusterRectangles"][0].get_visible():
            menus["main"].addAction("Remove Clusters", self.setClusterInvisible)
        elif "clusterRectangles" in self.data:
            menus["main"].addAction("Show Clusters", self.setClusterVisible)
        if hasattr(self, "labelData"):
            menus["main"].addAction("Show Labels", self.showLabels)
        for cMap in colorParameterRange:
            menus["Color Map (Cluster)"].addAction(cMap,self.updateColorMapOfClusterMesh)
        menus["main"].addAction("Export cluster ID", self.mC.mainFrames["right"].addClusterLabel)
        if self.plotType == "hclust":
            menus["main"].addAction("To Excel File", self.mC.mainFrames["right"].exportHClustToExcel)

            
    def addTooltip(self, tooltipColumnNames,dataID):
        ""
        self.annotationColumns = tooltipColumnNames.values.tolist()
        self.tooltipData = self.mC.data.getDataByColumnNames(dataID,tooltipColumnNames)["fnKwargs"]["data"]
        labelData = pd.DataFrame()    
        labelData["columnName"] = tooltipColumnNames.values.tolist()
        self.setDataInTooltipTable(labelData,title="Tooltip Data")
        self.tooltipActive = True
        self.updateBackgrounds()

    

    def removeTooltip(self):
        ""
        if hasattr(self,"tooltip"):
            self.tooltipActive = False
            self.tooltip.set_visible(False)

    def removeColumnNameFromTooltip(self,columnName):
        ""
        self.annotationColumns = [x for x in self.annotationColumns if x != columnName]

    def showLabels(self):
        ""
        if hasattr(self,"labelData"):
            self.forceLabels = True
            self.onClusterYLimChange()


    def addDendrogram(self,ax,lineCollection):
        ""
        if lineCollection is not None:
            ax.add_collection(lineCollection)

    def addColorMap(self,*args,**kwargs):
        ""
        colorbar(*args,**kwargs)

    def addColorMesh(self,ax,data, cmap = None, paramName = "twoColorMap", norm = None, addLineKwargs = True, colorMeshLimits = None):
        ""
        ax.clear()
        if cmap is None:
            cmap = self.mC.colorManager.get_max_colors_from_pallete(self.mC.config.getParam(paramName))
            cmap.set_bad(self.mC.config.getParam("nanColor"))

        elif isinstance(cmap,ScalarMappable):
            #camp is scalarmappable
            norm = cmap.norm
            cmap = cmap.get_cmap()
            
        if addLineKwargs and data.shape[0] < self.getParam("quad.linewidth.rowLimit") and data.shape[1] < self.getParam("quad.linewidth.columnLimit"):

            colorMeshLineKwargs = dict(
                    linewidth = 0.01, 
                    linestyle = '-',
					edgecolor = 'k')
        else:
            colorMeshLineKwargs = {}

        if colorMeshLimits is not None:
            vmin, vmax = colorMeshLimits
            valueLimitKwargs = {"vmin":vmin,"vmax":vmax}
            
        else:
            valueLimitKwargs = {}

        colorMesh = ax.pcolormesh(data, 
					  cmap = cmap,
                      norm = norm,
					  **colorMeshLineKwargs, **valueLimitKwargs)
        return colorMesh
    
    def updateColorMesh(self,mesh,dataShape):
        ""
        if dataShape[0] < self.getParam("quad.linewidth.rowLimit") and \
            dataShape[1] < self.getParam("quad.linewidth.columnLimit"):

            colorMeshLineKwargs = dict(
                    linewidth = 0.01, 
                    linestyle = '-',
					edgecolor = 'k')
        else:
            
            colorMeshLineKwargs = dict(
                    linewidth = 0, 
                    linestyle = '-',
					edgecolor = 'k')
        
        mesh.update(colorMeshLineKwargs)

    def updateColorMapOfClusterMesh(self,event = None, cmapName = None):
        ""
        if hasattr(self,"colorMesh"):
            if cmapName is None:
                cmapName = self.sender().text()
            
            #get map
            cmap = self.mC.colorManager.get_max_colors_from_pallete(cmapName)
            cmap.set_bad(self.mC.config.getParam("nanColor"))
            self.colorMesh.set_cmap(cmap)
            self.updateFigure.emit() 

    def addClusters(self):
        "Add cluster rectangles to dendrogram."
        self.axisDict["axRowDendro"].tick_params(which='minor', length=0)
        if "tickLabels" in self.data and "rowDendrogram" in self.data["tickLabels"]:
            self.addTicksToRowDendro()
            for rect in self.data["clusterRectangles"]:
                self.axisDict["axRowDendro"].add_patch(rect)

    def addTicksToRowDendro(self):
        ""
        tickLabels = self.data["tickLabels"]["rowDendrogram"]["tickLabels"]
        tickPosition = self.data["tickLabels"]["rowDendrogram"]["tickPosition"]
        if len(tickLabels) > 50: #if there are too many clusters, IC crashes
            tickLabels, tickPosition = [], []
        self.setYTicks(self.axisDict["axRowDendro"],tickPosition,tickLabels,tickwargs = {"minor":False})

    def hideTicksForRowDendro(self):
        ""
        self.setYTicks(self.axisDict["axRowDendro"],[],[])

    def refreshCluster(self,rowMaxD = None):
        "Refreshes clusters based on rowmaxd value. This happens on main thread"
        if rowMaxD is None:
            rowMaxD = self.data["rowMaxD"]
        self.setClusterRectanglesInvisible()
        rowClustNumber = self.mC.plotterBrain.getClusterNumber(self.data["rowLinkage"],rowMaxD)
        ytickPosition, ytickLabels, rectangles, clusterColorMap = self.mC.plotterBrain.getClusterRectangles(self.data["plotData"],rowMaxD,rowClustNumber,self.data["Z_row"])
        self.data["tickLabels"]["rowDendrogram"]["tickLabels"] = ytickLabels
        self.data["tickLabels"]["rowDendrogram"]["tickPosition"] = ytickPosition
        self.data["clusterRectangles"] = rectangles
        self.data["rowClustNumber"] = rowClustNumber
        self.data["clusterColorMap"] = clusterColorMap

        self.addClusters()
        yLim = self.getYlim(self.axisDict["axClusterMap"],mult=10)
        #addjust ylim 
        self.axisDict["axRowDendro"].set_ylim(yLim)
        # self.p.f.canvas.draw()
        self.updateFigure.emit()

    def setClusterInvisible(self,event=None):
        ""
        self.setClusterRectanglesInvisible()
        self.hideTicksForRowDendro()
        self.updateFigure.emit()

    def setClusterVisible(self,event=None):
        ""
        self.setClusterRectanglesVisible()
        self.addTicksToRowDendro()
        self.updateFigure.emit()

    def setClusterRectanglesInvisible(self):
        "Set cluster rectangles invisible. Mainly used to update the clusters"
        self.setClusterRectanglesVisibilty(False)

    def setClusterRectanglesVisible(self):
        "Set cluster rectangles visible."
        self.setClusterRectanglesVisibilty(True)
        
    def setClusterRectanglesVisibilty(self, visible = False):
        "Set cluster rectangles invisible. Mainly used to update the clusters"
        for rect in self.data["clusterRectangles"]:
            rect.set_visible(visible)

    def addClusterDistanceLine(self):
        ""
        if self.data["rowMaxD"] is not None and isinstance(self.data["rowMaxD"],float):

            self.rowClusterLine = self.axisDict["axRowDendro"].axvline(self.data["rowMaxD"] , linewidth=1.3, color = 'k')#'#1f77b4')

    def addHoverText(self):
        ""
        self.hoverTextProps = self.getStdTextProps()
        self.hoverText = self.axisDict["axLabelColor"].text(s = "", **self.hoverTextProps )

   
    def adjustColorMapLimits(self):
        ""
        limitProps = self.mC.config.getParam("colorMapLimits")

        if limitProps == "raw values":

            self.meshGridKwargs["vmin"] = np.nanmin(self.data["plotData"].values)
            self.meshGridKwargs["vmax"] = np.nanmax(self.data["plotData"].values)

        elif limitProps == "center 0":

            maxValue = np.nanmax(np.abs(self.data["plotData"].values))
            self.meshGridKwargs["vmin"] = -maxValue
            self.meshGridKwargs["vmax"] = maxValue
        
        elif limitProps == 'min = -1, max = 1':
					
            self.meshGridKwargs['vmin'] = -1
            self.meshGridKwargs['vmax'] = 1
        
        elif limitProps == "custom":
            
            minValue = self.mC.config.getParam("colorMapLimits.min")
            maxValue = self.mC.config.getParam("colorMapLimits.max")
            self.meshGridKwargs["vmin"] = minValue
            self.meshGridKwargs["vmax"] = maxValue
        

    def onDataLoad(self, data):
        ""
        self.data = data
    
        try:
            self.adjustColorMapLimits()
            
            self.axisDict = self.initAbsAxes(data["absoluteAxisPositions"])
            #add row dendro
            if "axRowDendro" in self.axisDict:
                self.addDendrogram(self.axisDict["axRowDendro"],
                                data["dendrograms"]["row"])
                
                self.setAxisLimits(self.axisDict["axRowDendro"],
                              data["axisLimits"]["rowDendrogram"]["x"],
                              data["axisLimits"]["rowDendrogram"]["y"])
            #add column dendro
            if "axColumnDendro" in self.axisDict:
                self.addDendrogram(self.axisDict["axColumnDendro"],
                                data["dendrograms"]["col"])
            
                self.setAxisLimits(self.axisDict["axColumnDendro"],
                              data["axisLimits"]["columnDendrogram"]["x"],
                              data["axisLimits"]["columnDendrogram"]["y"])
            #add colormesh
            self.colorMesh = self.addColorMesh(self.axisDict["axClusterMap"],
                            data["plotData"].values)
            #set yticks of label
            self.setAxisLimits(self.axisDict["axLabelColor"],yLimit=(0,self.data["plotData"].values.shape[0]))
            #set yaxis right
            self.setYTicksToRight(self.axisDict["axLabelColor"])
            self.setYTicksToRight(self.axisDict["axClusterMap"])
            #hide axis ticks
            if self.mC.getPlotType() == "corrmatrix" and "axRowDendro" in self.axisDict:
                self.setTicksOff(self.axisDict["axRowDendro"])
            else:  
                self.setYTicks(self.axisDict["axLabelColor"],[],[])
            
            #handle ticks
            if "axRowDendro" in self.axisDict:
                self.setXTicks(self.axisDict["axRowDendro"],[],[])
                #invert row dendro
                self.axisDict["axRowDendro"].invert_xaxis()
            #set all ticks off
            if "axColumnDendro" in self.axisDict:
                self.setTicksOff(self.axisDict["axColumnDendro"])
            
            #set all ticks off
            if "axLabelColor" in self.axisDict:
                self.axisDict["axLabelColor"].set_xlim((0,10))
                self.axisDict["axLabelColor"].set_axis_off()
                self.setTicksOff(self.axisDict["axLabelColor"])
            #set xticks on cluster map
            numColumns = self.data["plotData"].values.shape[1]
            if numColumns < 50:
                self.setXTicks(ax = self.axisDict["axClusterMap"], 
                            ticks = np.linspace(0.5,numColumns-0.5,num=numColumns),
                            labels = self.data["columnNames"], rotation=90)
            else:
                self.setXTicks(self.axisDict["axClusterMap"], [], [])
            self.setYTicks(self.axisDict["axClusterMap"],[],[])

            #add colorbar
            self.addColorMap(mappable=self.colorMesh,cax=self.axisDict["axColormap"], label="")
            #set title
            self.setAxisTitle(ax = self.axisDict["axColormap"],
                            title = "{}\nn = {}".format(
                                            self.mC.config.getParam("colorMatrixMethod") if self.mC.getPlotType() == "corrmatrix" else "Color mapping",
                                            data["plotData"].index.size))
            #set colormap limits
            self.setClims()
            if "axRowDendro" in self.axisDict:
                #add cluster line
                self.addClusterDistanceLine()
                self.addClusters() 
            if self.interactive:
                
                self.addHoverText()
                self.addHoverScatter(ax = self.axisDict["axLabelColor"])

                self.addHoverBinding()
                #add potential tooltip 
                self.ax = self.axisDict["axClusterMap"]
                self.extractAxisProps()
                self.defineBbox()
                self.defineText()
                self.buildTooltip()
                self.addYLimChangeEvent(ax= self.axisDict["axClusterMap"],
                                        callbackFn = self.onClusterYLimChange)

            if self.mC.groupingActive() and self.mC.getPlotType() == "corrmatrix":
                colorData = self.mC.plotterBrain.getColorGroupingForCorrmatrix(
                                self.data["plotData"].columns,
                                self.mC.grouping.getCurrentGrouping(),
                                self.mC.grouping.getCurrentGroupingName(),
                                self.mC.grouping.getCurrentCmap())
                self.updateHclustColor(colorMaPParamName= "colorMap", **colorData)
            qsData = self.getQuickSelectData()
            if qsData is not None:

                self.mC.quickSelectTrigger.emit()

            else:
                self.updateFigure.emit()
        except Exception as e:
            print(e)
        
        
    def getClusterIDsByDataIndex(self):
        "Returns data frame with data orginal index to export to source data."
        if "rawIndex" in self.data and "rowClustNumber" in self.data:  
            fillNaN = self.getParam("Object Replace String")
            clustLabels = ["C ({})".format(x) if not np.isnan(x) else fillNaN for x in self.data["rowClustNumber"]]
            df = pd.DataFrame(clustLabels,
                                        columns=["Cluster(ID)"],
                                        index = self.data["rawIndex"])

            dataID = self.data["dataID"]    

            return df, dataID
            
        else:
            return None, None 

    def getColorArray(self):
        ""
        return self.colorMesh.get_facecolors()

    def getClusteredData(self, reverseRows = True):
        ""
       
        return self.data["plotData"]

    def getClusterLabelsAndColor(self):
        ""
        df, _ = self.getClusterIDsByDataIndex()
  
        if df is not None and "clusterColorMap" in self.data:
            return df, self.data["clusterColorMap"]
        return None, None


    def getGraphSpecMenus(self):
        ""
        return ["Color Map (Cluster)"]

    def onClusterYLimChange(self,ax = None):
        ""
        if ax is None:
            ax = self.axisDict["axClusterMap"]
        currentYLim = ax.get_ylim() 
        #row dendro scale is 10 * 
        rowDendroYLim = [currentYLim[0] * 10,currentYLim[1] * 10]

        self.updateColorMesh(self.colorMesh,
                            (currentYLim[1] - currentYLim[0],self.data["plotData"].values.shape[1]))

        if hasattr(self, "colorLabelMesh"):

            self.updateColorMesh(self.colorLabelMesh,
                            (currentYLim[1] - currentYLim[0],self.data["plotData"].values.shape[1]))
        if hasattr(self, "labelColumnLimits"):
            
            axLabelXLimits = self.axisDict["axLabelColor"].set_xlim(self.labelColumnLimits)
        else:
            axLabelXLimits = None

        if self.getParam("keep.cluster.xaxis.fixed"):
            self.setAxisLimits(ax,xLimit = (0,len(self.data["columnNames"])))
        
        if "axRowDendro" in self.axisDict:
            self.setAxisLimits(self.axisDict["axRowDendro"],yLimit=rowDendroYLim)
        self.setAxisLimits(self.axisDict["axLabelColor"],yLimit=currentYLim, xLimit=axLabelXLimits)
        self.updateLabels(int(currentYLim[0]),int(currentYLim[1] + 0.5))
        self.updateFigure.emit()

    def updateLabels(self, idxMin, idxMax):
        ""
        if hasattr(self,"labelData"):
            if idxMin < 0:
                idxMin = 0
            if idxMax > self.data["plotData"].index.size:
                idxMax = self.data["plotData"].index.size
            if self.forceLabels or idxMax - idxMin < self.getParam("cluster.label.limit"):
                #set tick length to zero.
                self.axisDict["axLabelColor"].tick_params(which='minor', length=0)
                idxData = self.data["plotData"].index[idxMin:idxMax]
                tickPositions = np.arange(idxMin,idxMax) + 0.5
                tickLabels = [";".join(x) for x in self.labelData.loc[idxData].values]
                self.setYTicks(self.axisDict["axLabelColor"],tickPositions,tickLabels)
            else:
                self.setYTicks(self.axisDict["axLabelColor"],[],[])
        else:
            self.setYTicks(self.axisDict["axLabelColor"],[],[])

    def setRowClusterLineData(self, xPositions, ax):
        ""
        self.p.f.canvas.restore_region(self.rowDendroBackground)
        self.rowClusterLine.set_xdata([xPositions,xPositions])
        self.rowClusterLine.set_visible(True)
        self.axisDict["axRowDendro"].draw_artist(self.rowClusterLine)
        self.p.f.canvas.blit(self.axisDict["axRowDendro"].bbox)

    def onHover(self,event=None):
        ""
        if self.movingMaxDLine and event.inaxes != self.axisDict["axRowDendro"]:
            self.setRowClusterLineData(self.data["rowMaxD"],self.axisDict["axRowDendro"])
            self.movingMaxDLine = False

        if hasattr(self,"ax") and self.tooltipActive and event.inaxes != self.ax:
            self.tooltip.set_visible(False)
            self.drawTooltip(self.clusterMapBackground)
            return
        elif "axRowDendro" in self.axisDict and event.inaxes == self.axisDict["axRowDendro"]:
            if event.button is None and self.movingMaxDLine:
                newMaxD = self.rowClusterLine.get_xdata()[0]
                self.data["rowMaxD"] = newMaxD
                self.refreshCluster(newMaxD)
                self.movingMaxDLine = False

            elif event.button == 1 and self.rowClusterLine.contains(event)[0]:
                #print("beatufiul")
                #print(event.xdata)
                #print(self.rowClusterLine.get_xdata())
                self.setRowClusterLineData(event.xdata,self.axisDict["axRowDendro"])
                self.movingMaxDLine = True


        #on hover qick select only works for hierarchical clustering
        elif (self.isQuickSelectActive() or self.isLiveGraphActive() or self.tooltipActive) and self.mC.getPlotType() != "corrmatrix":
            idxData = None
            if event.inaxes == self.axisDict["axClusterMap"]:
                yDataEvent = int(event.ydata)
                fracSel = float(self.mC.config.getParam("selectionRectangleSize"))
                if fracSel > 0:
                    dataIndex = self.data["plotData"].index
                    nIdx = dataIndex.size 
                    idxWindow = int(fracSel * 100 / 2)
                    idxMin = 0 if yDataEvent - idxWindow < 0 else yDataEvent - idxWindow
                    idxMax = nIdx if yDataEvent + idxWindow > nIdx else yDataEvent + idxWindow
                    idxData = self.data["plotData"].index[idxMin:idxMax]
                else:
                    idxData = self.data["plotData"].index[yDataEvent]
            #show tooltip
            if self.tooltipActive and idxData is not None:
                #reduce number of tooltips to 14
                nAnnotColumns = len(self.annotationColumns)
                idxT = idxData if idxData.size * nAnnotColumns <= 14 else idxData.values[:int(14/nAnnotColumns)]
                self.updateTooltipPosition(event,"\n".join([str(x) if len(str(x)) < 20 else "{}..".format(str(x)[:20]) for x in self.tooltipData.loc[idxT,self.annotationColumns].values.flatten()[::-1]]))
                self.drawTooltip(self.clusterMapBackground)
            #send data to QuickSelect Widget
            if self.isQuickSelectActive():
                self.sendIndexToQuickSelectWidget(idxData)

            #send data to live graph
            if self.isLiveGraphActive():
                self.sendIndexToLiveGraph(idxData)

    def updateColorMap(self):
        ""
        self.colorMesh.set_cmap(
                self.mC.colorManager.get_max_colors_from_pallete(self.mC.config.getParam("twoColorMap")))
        
        self.updateFigure.emit()

    def updateClim(self):
        ""
        if hasattr(self,"data"):
            self.adjustColorMapLimits()
            self.setClims()
            self.updateFigure.emit()

    def updateGroupColors(self,colorGroup,changedCategory=None):
        ""
        if hasattr(self,"quickSelectScatter") and hasattr(self,"quickSelectProps") and changedCategory in self.colorCategoryIndexMatch:
            idx = self.colorCategoryIndexMatch[changedCategory]

            changedColor = colorGroup.loc[colorGroup["internalID"] == changedCategory,"color"].values[0]
            self.quickSelectProps.loc[idx,"color"] = changedColor
            if isinstance(idx,list):
                dataIdx = np.concatenate([np.array(v) for v in self.colorCategoryIndexMatch.values()])
            else:
                dataIdx = np.concatenate([v.values for v in self.colorCategoryIndexMatch.values()])
            idxPosition, dataIndexInClust = self.getPositionFromDataIndex(dataIdx)
            inv = self.axisDict["axLabelColor"].transLimits.inverted()
            xOffset,_= inv.transform((0.02, 0.25))
            
            coords = self.getOffsets(idxPosition,xOffset)
            
            self.quickSelectScatter.set_offsets(coords)
            self.quickSelectScatter.set_visible(True)
            self.quickSelectScatter.set_facecolor(self.quickSelectProps.loc[dataIndexInClust,"color"])
            self.quickSelectScatter.set_sizes(self.quickSelectProps.loc[dataIndexInClust,"size"])

            self.updateFigure.emit()
        
        else:
    
            cmap = ListedColormap(colorGroup["color"].values)
            self.colorLabelMesh.set_cmap(cmap)
            self.updateFigure.emit()

    def setClims(self,vmin=None,vmax=None):
        ""
        if "vmin" in self.meshGridKwargs and "vmax" in self.meshGridKwargs:
            self.colorMesh.set_clim(self.meshGridKwargs["vmin"],self.meshGridKwargs["vmax"])

    def setHoverObjectsInvisible(self):
        ""
        if hasattr(self,"hoverText"):
            self.hoverText.set_visible(False)
        if hasattr(self,"hoverScatter"):
            if self.axisDict["axLabelColor"] in self.hoverScatter:
                self.hoverScatter[self.axisDict["axLabelColor"]].set_visible(False)
        if hasattr(self,"tooltip"):
            self.tooltip.set_visible(False)
        if hasattr(self,"rowClusterLine"):
            self.rowClusterLine.set_visible(False)
        
       
    def setHoverData(self,dataIndex, showText = False):
        ""
       # print(dataIndex)
       # if dataIndex in self.data["plotData"].index:
        self.p.f.canvas.restore_region(self.colorLabelBackground)
        inv = self.axisDict["axLabelColor"].transLimits.inverted()
        xOffsetText,_= inv.transform((0.04, 0.25))
        xOffsetScatter,_= inv.transform((0.02, 0.25))
        idxPosition = [self.data["plotData"].index.get_loc(idx) + 0.5 for idx in dataIndex if idx in self.data["plotData"].index]
        if hasattr(self,"labelData"):
            if len(idxPosition) == 0:
                self.hoverTextProps["visible"] = False
                self.hoverTextProps["text"] = ""
            else:
                self.hoverTextProps["y"] = idxPosition[0]
                self.hoverTextProps["x"] = xOffsetText
                self.hoverTextProps["visible"] = True
                self.hoverTextProps["text"] = ";".join(self.labelData.loc[dataIndex].values[0,:])
                self.hoverTextProps["va"] = "center"
            self.hoverText.update(self.hoverTextProps)
            self.axisDict["axLabelColor"].draw_artist(self.hoverText)
        
        #create numpy array with scatter offsets
        coords = self.getOffsets(idxPosition, xOffsetScatter)

        self.setHoverScatterData(coords,self.axisDict["axLabelColor"])

    
    def updateHclustSize(self,sizeData):
        ""
        self.addColorMesh(self.axisDict["axLabelColor"],sizeData.loc[self.data["plotData"].index].values,paramName="hclustSizeColorMap")
        self.onClusterYLimChange()

    def updateHclustColor(self,colorData, colorGroupData, cmap=None, title="",colorMaPParamName = "hclustLabelColorMap"):
        ""
        #print(colorData.loc[self.data["plotData"].index].values)
        colorColumnNames = colorData.columns.values
        colorData = colorData.loc[self.data["plotData"].index].astype(np.float64).values
        self.colorLabelMesh = self.addColorMesh(
                        self.axisDict["axLabelColor"],
                        colorData,
                        cmap= cmap,
                        paramName=colorMaPParamName
                        )

        self.setDataInColorTable(colorGroupData, title = title)
        self.updateXlimForLabelColor(colorData.shape, colorColumnNames)
        self.onClusterYLimChange()
        
        
    def updateXlimForLabelColor(self,dataShape, labelColumnNames, addRectangleAndLabels = True):
        "Updates the xlim of the label color in a way that it matches the width of the clustermap"

        nColumns = self.axisDict["axClusterMap"].get_xlim()[1]
        clusterAxisWidth = self.axisDict["axClusterMap"].get_window_extent().width
        labelAxisWidth = self.axisDict["axLabelColor"].get_window_extent().width
        widthPerColumn = clusterAxisWidth/nColumns
        labelAxisColumns = labelAxisWidth/widthPerColumn
        self.labelColumnLimits = (0,labelAxisColumns)
        self.labeColumnNames = labelColumnNames
        self.axisDict["axLabelColor"].set_xlim(self.labelColumnLimits)
        self.axisDict["axLabelColor"].set_axis_off()

        if addRectangleAndLabels:
            # get 0.5% offset
            yOffset = self.axisDict["axLabelColor"].get_ylim()[1] * 0.005
            
            for n, labelColumnName in enumerate(labelColumnNames):
                self.axisDict["axLabelColor"].text(
                            x = 0.5+n,
                            y = dataShape[0] + yOffset, 
                            s = labelColumnName, 
                            rotation=90, 
                            ha = "center",
                            va = "bottom", 
                            fontproperties = self.getStdFontProps())

            for rN in range(dataShape[1]):
                p = patches.Rectangle((rN,0),width=1,height=dataShape[0], edgecolor = "black", linewidth = 0.6, fill=False)
                self.axisDict["axLabelColor"].add_patch(p)
        
        
    def updateBackgrounds(self):
        "Update Background for blitting"
        self.colorLabelBackground = self.p.f.canvas.copy_from_bbox(self.axisDict["axLabelColor"].bbox)	
        if "axRowDendro" in self.axisDict:
            self.rowDendroBackground = self.p.f.canvas.copy_from_bbox(self.axisDict["axRowDendro"].bbox)	
        if self.tooltipActive:
            self.clusterMapBackground = self.p.f.canvas.copy_from_bbox(self.axisDict["axClusterMap"].bbox)	
        
    def updateQuickSelectItems(self,propsData=None):
        ""
        self.quickSelectProps = propsData
        dataIndex = self.getDataIndexOfQuickSelectSelection()
        idxPosition, dataIndexInClust = self.getPositionFromDataIndex(dataIndex)
       
        
        if len(idxPosition) == 0:
            return
        if not hasattr(self,"quickSelectScatter"):
            self.quickSelectScatter = self.axisDict["axLabelColor"].scatter(x=[],y=[],**self.getScatterKwargs())
        
        inv = self.axisDict["axLabelColor"].transLimits.inverted()
        xOffset,_= inv.transform((0.02, 0.25))
        coords = self.getOffsets(idxPosition,xOffset)
        
        self.quickSelectScatter.set_offsets(coords)
        self.quickSelectScatter.set_visible(True)
        self.quickSelectScatter.set_facecolor(propsData.loc[dataIndexInClust,"color"])
        self.quickSelectScatter.set_sizes(propsData.loc[dataIndexInClust,"size"])


    def getPositionFromDataIndex(self,dataIndex):
        ""
        if isinstance(dataIndex,list):
            dataIndex = np.array(dataIndex)
        dataIndexInClust = self.data["plotData"].index.intersection(dataIndex)
        if not dataIndexInClust.size == 0:
            idxPosition = [self.data["plotData"].index.get_loc(idx) + 0.5 for idx in dataIndexInClust]
            return idxPosition, dataIndexInClust
        else:
            return [], []

    def getOffsets(self,idxPosition,xOffset = -0.15):
        ""
        coords = np.zeros(shape=(len(idxPosition),2))
        coords[:,1] = idxPosition
        coords[:,0] += xOffset
        return coords

    def setNaNColor(self):
        ""
        self.axisDict["axLabelColor"].clear()
        self.axisDict["axLabelColor"].set_xlim(0,10)
        self.axisDict["axLabelColor"].set_axis_off()
        # delete colorLabelMesh 
        del self.colorLabelMesh

        self.updateFigure.emit()