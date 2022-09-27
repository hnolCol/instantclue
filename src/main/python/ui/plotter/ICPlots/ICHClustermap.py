
from matplotlib.collections import LineCollection
from .ICChart import ICChart
import numpy as np
from backend.color.data import colorParameterRange
from matplotlib.pyplot import colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.backends.backend_pdf import FigureCanvasPdf
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns


class ICClustermap(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICClustermap,self).__init__(*args,**kwargs)

        self.meshGridKwargs = dict()
        self.axisTextLabels = dict()
        self.movingMaxDLine = False
        self.forceLabels = False
        self.numOfColorColumns = 0


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
        if "clusterRectangles" in self.data and len(self.data["clusterRectangles"]) != 0 and len(self.clusterRectangles) >0 and self.clusterRectangles[0].get_visible():
            menus["main"].addAction("Remove Clusters", self.setClusterInvisible)
        elif "clusterRectangles" in self.data:
            menus["main"].addAction("Show Clusters", self.setClusterVisible)
        if hasattr(self, "labelData"):
            menus["main"].addAction("Show Labels", self.showLabels)
        for cMap in colorParameterRange:
            menus["Color Map (Cluster)"].addAction(cMap,self.updateColorMapOfClusterMesh)
            if self.plotType == "hclust":
                menus["Color Map (Color column)"].addAction(cMap,self.updateColorMapOfColorColumns)
        if self.groupingExists():
            menus["main"].addAction("Add Grouping Legend", self.addGroupingLegend)
        menus["main"].addAction("Export cluster ID", self.mC.mainFrames["right"].addClusterLabel)
        if self.plotType == "hclust":
            menus["main"].addAction("To Excel File", self.mC.mainFrames["right"].exportHClustToExcel)
            menus["main"].addAction("Share graph", self.shareGraph)
            
    def groupingExists(self):
        return "axColumnGrouping" in self.axisDict

    def addGroupingLegend(self):
        ""
       # print(self.data)
        if self.groupingExists():
            groupingNames = self.data["tickLabels"]["axColumnGrouping"]["tickLabels"]
            columnNames = self.data["columnNames"]
        
        ax = self.axisDict["axColumnDendro"] if "axColumnDendro" in self.axisDict else self.axisDict["axColumnGrouping"]
        legendData = self.mC.grouping.getDataForLegend(groupingNames,columnNames)
        self.addGroupingLegendToGraph(legendData,ax, legendKwargs={"loc":"lower left", "bbox_to_anchor":(0.00, 1.02), "ncol":len(legendData)})

    def addTooltip(self, tooltipColumnNames,dataID):
        ""
        self.annotationColumns = tooltipColumnNames.values.tolist()
        self.tooltipData = self.mC.data.getDataByColumnNames(dataID,tooltipColumnNames)["fnKwargs"]["data"]
        labelData = pd.DataFrame()    
        labelData["columnName"] = tooltipColumnNames.values.tolist()
        self.setDataInTooltipTable(labelData,title="Tooltip Data")
        self.tooltipActive = True
        self.updateBackgrounds()

    def hasTooltip(self):
        if hasattr(self,"toolTipsActve"):
            return self.toolTipsActive
        else:
            return False

    def disconnectAnnotations(self):
        ""
        if hasattr(self,"labelData"):
            del self.labelData
        

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

        return hasattr(self,"labelData")

    def addDendrogram(self,ax,lineCollection):
        ""
        if lineCollection is not None:
            if isinstance(lineCollection,dict):
                lc = LineCollection(**lineCollection)
                ax.add_collection(lc)

    def annotateDataByIndex(self,dataIndex,annotationColumn):
        ""
        if isinstance(annotationColumn,str):
            annotationColumn = pd.Series([annotationColumn])
        self.addAnnotations(annotationColumn, self.mC.getDataID())

    def addColorMap(self,*args,**kwargs):
        ""
        colorbar(*args,**kwargs)

    def addColorMesh(self,ax,data, cmap = None, paramName = "twoColorMap", norm = None, addLineKwargs = True, colorMeshLimits = None, clearAxis = True):
        ""
        if clearAxis:
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
                    linewidth = 0.005, 
                    linestyle = '-',
					edgecolor = 'k')
        else:
            colorMeshLineKwargs = {}

        
        if colorMeshLimits is not None:
            if isinstance(colorMeshLimits,dict) and "vmin" in colorMeshLimits and "vmax" in colorMeshLimits and len(colorMeshLimits) == 2:
                valueLimitKwargs = colorMeshLimits.copy()
            elif isinstance(colorMeshLimits,dict) and "vmin" in colorMeshLimits and "vmax" in colorMeshLimits and len(colorMeshLimits) != 2:
                valueLimitKwargs = {"vmin":colorMeshLimits["vmin"],"vmax":colorMeshLimits["vmax"]}
            elif isinstance(colorMeshLimits,tuple) or  isinstance(colorMeshLimits,list):
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
                    linewidth = 0.005, 
                    linestyle = '-',
					edgecolor = 'k')
        else:
            
            colorMeshLineKwargs = dict(
                    linewidth = 0, 
                    linestyle = '-',
					edgecolor = 'k')
        
        mesh.update(colorMeshLineKwargs)

    def updateColorMapOfColorColumns(self,event = None,cmapName=None):
        ""
        self.updateColorMapOfColorMesh("colorLabelMesh",cmapName=cmapName)


    def updateColorMapOfClusterMesh(self,event = None, cmapName = None):
        ""
        self.updateColorMapOfColorMesh(cmapName=cmapName)


    def updateColorMapOfColorMesh(self,attrName = "colorMesh",cmapName = None):
        ""
        if hasattr(self,attrName):
            if cmapName is None:
                if hasattr(self,"sender") and hasattr(self.sender(),"text"):
                    cmapName = self.sender().text()
                else:
                    return
            colorMesh = getattr(self,attrName)
            self.updateColorMapOfMesh(colorMesh,cmapName)

    def updateColorMapOfMesh(self,colorMesh,cmapName):
        ""
        cmap = self.mC.colorManager.get_max_colors_from_pallete(cmapName)
        cmap.set_bad(self.mC.config.getParam("nanColor"))
        colorMesh.set_cmap(cmap)
        self.updateFigure.emit() 

    def addClusters(self):
        "Add cluster rectangles to dendrogram."
        self.axisDict["axRowDendro"].tick_params(which='minor', length=0)
        if "tickLabels" in self.data and "rowDendrogram" in self.data["tickLabels"]:
            self.addTicksToRowDendro()
            self.clusterRectangles = []
            for rectProps in self.data["clusterRectangles"]:
                rect = patches.Rectangle(**rectProps)
                self.clusterRectangles.append(rect)
                self.axisDict["axRowDendro"].add_patch(rect)

    def addTicksToGroupingAxis(self):
        ""
        if "axColumnGrouping" in self.data["tickLabels"] and "tickPosition" in self.data["tickLabels"]["axColumnGrouping"]:
            if len(self.data["tickLabels"]["axColumnGrouping"]["tickPosition"]) > 0:
                self.setYTicks(
                    self.axisDict["axColumnGrouping"], 
                    self.data["tickLabels"]["axColumnGrouping"]["tickPosition"],
                    self.data["tickLabels"]["axColumnGrouping"]["tickLabels"]
                    )
                self.setXTicks(self.axisDict["axColumnGrouping"],[],[])
                self.axisDict["axColumnGrouping"].tick_params(which='minor', length=0)

    def addGroupingRectangles(self):
        if "axColumnGrouping" in self.axisDict:
            for rect in self.data["groupingRectangles"]:
                self.axisDict["axColumnGrouping"].add_patch(rect)

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
        if hasattr(self,"clusterRectangles") and len(self.clusterRectangles) > 0:
            for rect in self.clusterRectangles:
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
                
                self.axisDict["axRowDendro"].set_navigate(False)
            #add column dendro
            if "axColumnDendro" in self.axisDict:
                self.addDendrogram(self.axisDict["axColumnDendro"],
                                data["dendrograms"]["col"])
            
                self.setAxisLimits(self.axisDict["axColumnDendro"],
                              data["axisLimits"]["columnDendrogram"]["x"],
                              data["axisLimits"]["columnDendrogram"]["y"])
                self.axisDict["axColumnDendro"].set_navigate(False)
            #add colormesh
            self.colorMesh = self.addColorMesh(self.axisDict["axClusterMap"],
                            data["plotData"].values)
            #set yticks of label
            self.setAxisLimits(self.axisDict["axLabelColor"],yLimit=(0,self.data["plotData"].values.shape[0]))
            #set yaxis right
            self.setYTicksToRight(self.axisDict["axLabelColor"])
            self.setYTicksToRight(self.axisDict["axClusterMap"])
            if "axColumnGrouping" in self.axisDict:
                self.setYTicksToRight(self.axisDict["axColumnGrouping"])
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
            #grouping 
            if "axColumnGrouping" in self.axisDict and "axColumnGrouping" in self.data["tickLabels"] and "axColumnGrouping" in self.data["axisLimits"]:
                
                self.setAxisLimits(self.axisDict["axColumnGrouping"],
                              data["axisLimits"]["axColumnGrouping"]["x"],
                              data["axisLimits"]["axColumnGrouping"]["y"])
                self.addTicksToGroupingAxis()
                self.addGroupingRectangles()
                self.axisDict["axColumnGrouping"].set_navigate(False)
            #set all ticks off
            if "axLabelColor" in self.axisDict:
                self.axisDict["axLabelColor"].set_xlim((0,10))
                self.axisDict["axLabelColor"].set_axis_off()
                self.setTicksOff(self.axisDict["axLabelColor"])
            #set xticks on cluster map
            numColumns = self.data["plotData"].values.shape[1]
            if numColumns < self.getParam("cluster.label.limit"):
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
                                            data["plotData"].index.size)
                            )
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
                if self.mC.getPlotType() == "corrmatrix" and self.getParam("corrmatrix.show.tooltip"):
                    self.tooltipActive = True

            # if self.mC.groupingActive() and self.mC.getPlotType() == "corrmatrix":
            #     colorData = self.mC.plotterBrain.getColorGroupingForCorrmatrix(
            #                     self.data["plotData"].columns,
            #                     self.mC.grouping.getCurrentGrouping(),
            #                     self.mC.grouping.getCurrentGroupingName(),
            #                     self.mC.grouping.getCurrentCmap())
            #     self.updateHclustColor(colorMaPParamName= "colorMap", **colorData)
                self.checkForQuickSelectDataAndUpdateFigure()
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


    def getColorDataArray(self):
        ""
        if hasattr(self,"colorLabelMesh") and hasattr(self.colorLabelMesh,"get_facecolors"):
            return self.colorLabelMesh.get_facecolors()

    def getColorData(self):
        ""
        if hasattr(self,"colorData"):
            return self.colorData

    def getColorColumnNames(self):
        ""
        if hasattr(self,"colorColumnNames"):
            return self.colorColumnNames.tolist() 
        else:
            return []
	
    def getHeatmapColorArray(self):
        "Return facecolors"
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

    def getDataForWebApp(self):
        ""
         
        numberRows = self.data["plotData"].index.size
        numberColumns = self.data["plotData"].columns.size
        values = self.data["plotData"].values.tolist() 

        self.meshGridKwargs["vmin"] 
        self.meshGridKwargs["vmax"] 

        colorValues = np.linspace(self.meshGridKwargs["vmin"],self.meshGridKwargs["vmax"],num=12)
        colorValuesRGBA,  _ = self.mC.colorManager.matchColorsToValues(colorValues,self.mC.config.getParam("twoColorMap"),vmin=self.meshGridKwargs["vmin"], vmax = self.meshGridKwargs["vmax"])
        colorPalette = [to_hex(c) for c in colorValuesRGBA ]

        if hasattr(self, "labelData"):
            
            rowLabels = self.labelData.loc[self.data["plotData"].index].values.tolist()
        else:
            rowLabels = np.arange(numberRows).astype(str).tolist()
        data = {
            "nRows":numberRows,
            "nCols":numberColumns,
            "values":values,
            "colorValues": colorValues.tolist(),
            "rowLabels":rowLabels,
            "colLabels":self.data["plotData"].columns.values.tolist(),
            "extraValues" : [],
            "extraColorValues" : None,
            "colLabelColors" : {}
            }
        return data, colorPalette
       

    def getGraphSpecMenus(self):
        ""
        return ["Color Map (Cluster)","Color Map (Color column)"]

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
            axLabelXLimits = self.labelColumnLimits
        else:
            axLabelXLimits = None

        if self.getParam("keep.cluster.xaxis.fixed"):
            self.setAxisLimits(ax,xLimit = (0,len(self.data["columnNames"])))
        
        if "axRowDendro" in self.axisDict:
            self.setAxisLimits(self.axisDict["axRowDendro"],yLimit=rowDendroYLim)

        self.updateLabels(int(currentYLim[0]),int(currentYLim[1] + 0.5))
        self.setAxisLimits(ax=self.axisDict["axLabelColor"],yLimit=currentYLim, xLimit=axLabelXLimits)
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
                idxData = self.data["plotData"].index[idxMin:idxMax]
                idxPosition, _ = self.getPositionFromDataIndex(idxData.values)
                tickLabels = [";".join(x) for x in self.labelData.loc[idxData].astype(str).values]
                
                inv = self.axisDict["axLabelColor"].transLimits.inverted()
                xOffset,_= inv.transform((0.012, 0.25))
                coords = self.getOffsets(idxPosition,xOffset + self.getColorMeshXOffset())
                for n,(x,y) in enumerate(coords):
                    labelStr = tickLabels[n]
                    if idxPosition[n] in self.axisTextLabels:
                        prevX, _ = self.axisTextLabels[idxPosition[n]].get_position()
                        prevText = self.axisTextLabels[idxPosition[n]].get_text()
                        if prevX != x or prevText != labelStr: #check if position changed (changing pos didnt work)
                            self.axisTextLabels[idxPosition[n]].remove()
                        else:
                            #if text object is present - just set it visible
                            self.axisTextLabels[idxPosition[n]].set_visible(True)
                            continue

                    t = self.axisDict["axLabelColor"].text(
                                                        x = x, 
                                                        y = y, 
                                                        s = labelStr, 
                                                        fontproperties = self.getStdFontProps(), 
                                                        verticalalignment='center',
                                                        zorder = 1e6)
                    self.axisTextLabels[idxPosition[n]] = t
                    

                _ = [v.set_visible(False) for k,v in self.axisTextLabels.items() if k not in idxPosition]
              
            else:
                _ = [v.set_visible(False) for k,v in self.axisTextLabels.items()]
        else:
            _ = [v.set_visible(False) for k,v in self.axisTextLabels.items()]

    def setRowClusterLineData(self, xPositions, ax):
        ""
        self.p.f.canvas.restore_region(self.rowDendroBackground)
        self.rowClusterLine.set_xdata([xPositions,xPositions])
        self.rowClusterLine.set_visible(True)
        self.axisDict["axRowDendro"].draw_artist(self.rowClusterLine)
        self.p.f.canvas.blit(self.axisDict["axRowDendro"].bbox)

    def onHover(self,event=None):
        ""
        #check if cluster line is moved
        if self.movingMaxDLine and event.inaxes != self.axisDict["axRowDendro"]:
            self.setRowClusterLineData(self.data["rowMaxD"],self.axisDict["axRowDendro"])
            self.movingMaxDLine = False
        #check if mouse left main axis
        if hasattr(self,"ax") and self.tooltipActive and event.inaxes != self.ax:
                if self.tooltip.get_visible():
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
                self.setRowClusterLineData(event.xdata,self.axisDict["axRowDendro"])
                self.movingMaxDLine = True
        #handle corrmatrix tooltip (auto on)
        elif self.tooltipActive and self.mC.getPlotType() == "corrmatrix":
            yDataEvent = int(event.ydata)
            xDataEvent = int(event.xdata)
            
            r = self.data["plotData"].iloc[xDataEvent,yDataEvent]
            yName = self.data["plotData"].index[yDataEvent]
            xName = self.data["plotData"].columns[xDataEvent]
            self.updateTooltipPosition(event,"{}↓\n{}→\ncoeff = {}".format(xName,yName,round(r,2)))
            self.drawTooltip(self.clusterMapBackground)
        #on hover qick select/ Live graph only works for hierarchical clustering not on corrmatrix (columns are summarized)
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
        if hasattr(self,"colorLabelMesh"):
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
        if hasattr(self,"colorLabelBackground"):
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
                    self.hoverTextProps["x"] = xOffsetText + self.getColorMeshXOffset()
                    self.hoverTextProps["visible"] = True
                    self.hoverTextProps["text"] = ";".join(self.labelData.loc[dataIndex].values[0,:])
                    self.hoverTextProps["va"] = "center"
                self.hoverText.update(self.hoverTextProps)
                self.axisDict["axLabelColor"].draw_artist(self.hoverText)
            
            #create numpy array with scatter offsets
            coords = self.getOffsets(idxPosition, xOffsetScatter + self.getColorMeshXOffset())

            self.setHoverScatterData(coords,self.axisDict["axLabelColor"])

    
    def updateHclustSize(self,sizeData):
        ""
        self.addColorMesh(self.axisDict["axLabelColor"],sizeData.loc[self.data["plotData"].index].values,paramName="hclustSizeColorMap")
        self.onClusterYLimChange()

    def updateHclustColor(self,colorData, colorGroupData, cmap=None, title="",colorMaPParamName = "hclustLabelColorMap", colorMeshLimits=None):
        ""
        #print(colorData,colorMeshLimits)
        #print(colorData.loc[self.data["plotData"].index].values)
        colorColumnNames = colorData.columns.values
        colorFloats = colorData.loc[self.data["plotData"].index].astype(np.float64)# prevent integer error
        #resort color data by index (as identified by clustering)
        colorValues = colorFloats.values 
        
        self.resetColorGroupElements()
       
        self.colorLabelMesh = self.addColorMesh(
                        self.axisDict["axLabelColor"],
                        colorValues,
                        cmap= cmap,
                        paramName=colorMaPParamName,
                        clearAxis = False,
                        colorMeshLimits = (0,np.nanmax(colorValues)) if colorMeshLimits is None else colorMeshLimits
                        )
        
        self.numOfColorColumns = colorValues.shape[1]
        self.setDataInColorTable(colorGroupData, title = title, isEditable=self.data["isEditable"] if "isEditable" in self.data else False)
        self.updateXlimForLabelColor(colorValues.shape, colorColumnNames)
        self.updateQuickSelectItemsCoords()
        self.onClusterYLimChange()

        self.colorData = colorFloats
        self.colorColumnNames = colorColumnNames
        
        
        
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
        labelPosition = self.getParam("hclust.color.column.label.position")
        showLabel = self.getParam("hclust.color.column.show")
        if addRectangleAndLabels:
            # get 0.5% offset
            self.rectangleAndLabels = []
            if showLabel:
                yOffset = self.axisDict["axLabelColor"].get_ylim()[1] * 0.0075
                for n, labelColumnName in enumerate(labelColumnNames):
                    if labelPosition == "top":
                        t = self.axisDict["axLabelColor"].text(
                                    x = 0.5+n,
                                    y = dataShape[0] + yOffset, 
                                    s = labelColumnName, 
                                    rotation=90, 
                                    ha = "center",
                                    va = "bottom", 
                                    fontproperties = self.getStdFontProps())
                    else:
                        t = self.axisDict["axLabelColor"].text(
                                    x = 0.5+n,
                                    y = 0-yOffset, 
                                    s = labelColumnName, 
                                    rotation=90, 
                                    ha = "center",
                                    va = "top", 
                                    fontproperties = self.getStdFontProps())
                self.rectangleAndLabels.append(t)
            for rN in range(dataShape[1]):
                p = patches.Rectangle((rN,0),width=1,height=dataShape[0], edgecolor = "black", linewidth = 0.6, fill=False)
                self.axisDict["axLabelColor"].add_patch(p)
                self.rectangleAndLabels.append(p)
        
        
    def updateBackgrounds(self):
        "Update Background for blitting"
        if isinstance(self.p.f.canvas,FigureCanvasPdf):
            return 
        if "axLabelColor" in self.axisDict:
            self.colorLabelBackground = self.p.f.canvas.copy_from_bbox(self.axisDict["axLabelColor"].bbox)	
        if "axRowDendro" in self.axisDict:
            self.rowDendroBackground = self.p.f.canvas.copy_from_bbox(self.axisDict["axRowDendro"].bbox)	
        if self.tooltipActive:
            self.clusterMapBackground = self.p.f.canvas.copy_from_bbox(self.axisDict["axClusterMap"].bbox)	

    def getQuickSelectDataCoords(self): 

        if self.isQuickSelectModeUnique() and hasattr(self,"quickSelectCategoryIndexMatch"):
            dataIndex = np.concatenate([idx for idx in self.quickSelectCategoryIndexMatch.values()])
            intIDMatch = np.concatenate([np.full(idx.size,intID) for intID,idx in self.quickSelectCategoryIndexMatch.items()]).flatten()
        else:
            dataIndex = self.getDataIndexOfQuickSelectSelection()
            intIDMatch = np.array(list(self.quickSelectCategoryIndexMatch.keys()))
        
        return dataIndex, intIDMatch

    def updateQuickSelectItems(self,propsData=None):
        ""
        
        self.quickSelectProps = propsData
        dataIndex, intIDMatch = self.getQuickSelectDataCoords()
        idxPosition, dataIndexInClust = self.getPositionFromDataIndex(dataIndex)
        #subset for indices that are actually in the clustering
        intIDMatch = pd.DataFrame(intIDMatch,index=dataIndex, columns = ["intID"]).loc[dataIndexInClust,"intID"].values.flatten()
        #print(dataIndexInClust,dataIndex)
        ax = self.axisDict["axLabelColor"]
        if len(idxPosition) == 0:
            return
        if not hasattr(self,"quickSelectScatter"):
            self.quickSelectScatter = dict()
            self.quickSelectScatter[ax] = self.axisDict["axLabelColor"].scatter(x=[],y=[],**self.getScatterKwargs())
        
        inv = self.axisDict["axLabelColor"].transLimits.inverted()
        xOffset,_= inv.transform((0.02, 0.25))
        coords = self.getOffsets(idxPosition,xOffset + self.getColorMeshXOffset())
        
        self.quickSelectScatter[ax].set_offsets(coords)
        self.quickSelectScatter[ax].set_visible(True)
        self.quickSelectScatter[ax].set_facecolor(propsData.loc[dataIndexInClust,"color"])
        self.quickSelectScatter[ax].set_sizes(propsData.loc[dataIndexInClust,"size"])
        df = pd.DataFrame(intIDMatch,columns=["intID"])
        df["idx"] = dataIndexInClust
        df["x"] = coords[:,0]
        df["y"] = coords[:,1]
        self.quickSelectScatterDataIdx[ax] = {
                                            "idxPosition":idxPosition,
                                            "dataIndexInClust":dataIndexInClust,
                                            "coords":df,
                                            "idx":dataIndexInClust}

    def updateQuickSelectItemsCoords(self):
        ""
        ax = self.axisDict["axLabelColor"]
        if hasattr(self,"quickSelectScatter") and ax in self.quickSelectScatterDataIdx:
            dataIndex, intIDMatch = self.getQuickSelectDataCoords()
            idxPosition, dataIndexInClust = self.getPositionFromDataIndex(dataIndex)
            inv = self.axisDict["axLabelColor"].transLimits.inverted()
            xOffset,_= inv.transform((0.02, 0.25))
            coords = self.getOffsets(idxPosition,xOffset + self.getColorMeshXOffset())
            self.quickSelectScatter[ax].set_offsets(coords)
            df = pd.DataFrame(intIDMatch,columns=["intID"])
            df["idx"] = dataIndexInClust
            df["x"] = coords[:,0]
            df["y"] = coords[:,1]
            self.quickSelectScatterDataIdx[ax] = {
                                            "idxPosition":idxPosition,
                                            "dataIndexInClust":dataIndexInClust,
                                            "coords":df,
                                            "idx":dataIndexInClust}

    def getQuickSelectDataIdxForExcelExport(self):
        ""
        ax = self.axisDict["axLabelColor"]
        if hasattr(self,"quickSelectScatterDataIdx") and ax in self.quickSelectScatterDataIdx:
            return [self.quickSelectScatterDataIdx[ax], self.quickSelectScatter[ax].get_facecolor()]
        


    def updateQuickSelectData(self,quickSelectGroup,changedCategory=None):
        ""
        if hasattr(self,"quickSelectScatter"):
            
            ax = self.axisDict["axLabelColor"]
            
            
            coords = self.quickSelectScatterDataIdx[ax]["coords"][["x","y"]].values
            scatterSizes, scatterColors, _ = self.getQuickSelectScatterProps(ax,quickSelectGroup)
            
            self.quickSelectScatter[ax].set_offsets(coords)
            self.quickSelectScatter[ax].set_sizes(scatterSizes)	
            self.quickSelectScatter[ax].set_facecolor(scatterColors)

           #self.updateQuickSelectScatter(ax, scatterColors = scatterColors, scatterSizes = scatterSizes)
            self.updateFigure.emit()
        

    def getColorMeshXOffset(self):
        ""
        if not hasattr(self,"colorLabelMesh"):
            return 0 
        else:
            return self.numOfColorColumns


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
        
        self.resetColorGroupElements()
        self.mC.resetGroupColorTable.emit()
        self.updateFigure.emit()
    
    def resetColorGroupElements(self):
        ""
        if hasattr(self,"colorLabelMesh"):
            self.colorLabelMesh.remove() 
            del self.colorLabelMesh
        if hasattr(self,"colorData"):
            del  self.colorData
        self.colorColumnNames= []
        self.numOfColorColumns = 0
        if hasattr(self , "rectangleAndLabels"):
            for n in range(len(self.rectangleAndLabels)):
                self.rectangleAndLabels[n].set_visible(False)
            del self.rectangleAndLabels