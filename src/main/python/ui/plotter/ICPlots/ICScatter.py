#
from matplotlib.pyplot import axis, scatter
from .ICChart import ICChart
from .charts.scatter_plotter import scatterPlot
from .ICScatterAnnotations import ICScatterAnnotations
from ...dialogs.OmicsTools.ICVolcanoStyling import ICVolcanoPlotStyling
import pandas as pd
import numpy as np

class ICScatterPlot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICScatterPlot,self).__init__(*args,**kwargs)
        self.scatterPlots = dict()
        self.requiredKwargs = ["columnPairs","axisPositions","dataColorGroups","dataSizeGroups"]

    def addAnnotations(self, labelColumnNames,dataID):
        ""
        try:
            if not hasattr(self,"annotations"):
                self.annotations = dict()
            else:
                self.disconnectAnnotations()
            
            for n,ax in self.axisDict.items():
                columnPair = self.data["columnPairs"][n]
              
                numericColumns = pd.Series(list(columnPair))
                #if columns present, no need to add them to the data.
                labelColumnNamesNotInData = [colName for colName in labelColumnNames if colName not in self.scatterPlots[columnPair].data.columns]
                #columnNames = labelColumnNames.append(numericColumns,ignore_index=True)
                if len(labelColumnNamesNotInData) > 0:
                    data = self.mC.data.getDataByColumnNames(dataID,labelColumnNamesNotInData)["fnKwargs"]["data"]
                    plotData = self.scatterPlots[columnPair].data
                    data = plotData.join(data)
                else:
                    data = self.scatterPlots[columnPair].data
                self.annotations[columnPair] = ICScatterAnnotations(
                                parent = self,
                                plotter = self.p,
                                ax = ax,
                                data = data,
                                labelColumns = labelColumnNames,
                                numericColumns = numericColumns,
                                scatterPlots = self.scatterPlots,
                                labelInAllPlots = self.getParam("annotate.in.all.plots")
                                )
            self.initAnnotations()
            labelData = pd.DataFrame() 
            labelData["columnName"] = labelColumnNames
            self.setDataInLabelTable(labelData,title="Annotation Labels")
        except Exception as e:
            print("error")
            print(e)

    def disconnectAnnotations(self):
        ""
        if hasattr(self,"annotations") and isinstance(self.annotations,dict) and len(self.annotations) > 0:
            for annotation in self.annotations.values():
                annotation.disconnectEventBindings()
            self.annotations.clear()

    def disconnectBindings(self):
        ""
        super().disconnectBindings()
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.disconnectBindings()
        if hasattr(self,"annotations"):
            for annotation in self.annotations.values():
                annotation.disconnectEventBindings()

    def getGraphSpecMenus(self):
        ""
        return []

    def addGraphSpecActions(self,menus):
        ""
        #menus["main"].addAction("Share graph", self.shareGraph)
        menus["main"].addAction("Enable volcano plot style",self.enableVolcanoPlotStyling)
        if self.preventQuickSelectCapture:
            menus["main"].addAction("Enable QuickSelect Capture", self.startQuickSelectCapture)
        else:
            menus["main"].addAction("Stop QuickSelect Capture", self.stopQuickSelectCapture)
        
        menus["main"].addAction("Connect nearest neighbors",self.getNearestNeighborLines)


    def addTooltip(self, tooltipColumnNames,dataID):
        ""
        try:
            self.tooltipColumnNames = tooltipColumnNames
            data = self.mC.data.getDataByColumnNames(dataID,tooltipColumnNames)["fnKwargs"]["data"]
            for scatterPlot in self.scatterPlots.values():
                scatterPlot.addTooltip(data )
            labelData = pd.DataFrame() 
            labelData["columnName"] = tooltipColumnNames
            self.setDataInTooltipTable(labelData,title="Tooltip Labels")
        except Exception as e:
            print(e)
    
    def enableVolcanoPlotStyling(self):
        ""
        
        dataID = self.data["dataID"]
        
        setattr(self,"volcanoPlotStyleActivate",True)
        
        columnPairs = list(self.scatterPlots.keys())
        numericColumns = pd.Series(np.array(columnPairs).flatten())
        categoricalColumns = self.mC.data.getCategoricalColumns(dataID )
        dlg = ICVolcanoPlotStyling(self.mC,dataID, numericColumns,categoricalColumns)
        if dlg.exec(): #returns true if accept() ran 
            significantColumns = dlg.getSignificantColumns()
            colorColumns = dlg.getColorColumns()
            self.centerXToZero(update=False) #should be moved into the response from the backend.
            
            funcProps = {
                "key" : "plotter:getScatterColorGroupsForVolcano",
                "kwargs" : {
                    "dataID" : dataID,
                    "significantColumns" : significantColumns,
                    "numericColumns" : numericColumns,
                    "colorColumns" : colorColumns,
                    "columnPairs" : columnPairs
                }
            }
            self.mC.sendRequestToThread(funcProps)


    
        
    def initScatterPlots(self, onlyForID = None, targetAx = None, scaleFactor = None):
        ""
        if onlyForID is None:
            #clear saved scatters
            self.scatterPlots.clear()
            #init scatters
            self.scatterKwargs = self.getScatterKwargs()
            for n,ax in self.axisDict.items():
                columnPair = self.data["columnPairs"][n]
                self.scatterPlots[columnPair] = scatterPlot(
                                        self,
                                        data = self.data["plotData"],
                                        plotter = self.p,
                                        ax = ax,
                                        numericColumns = list(columnPair),
                                        dataID = self.data["dataID"],
                                        scatterKwargs = self.scatterKwargs,
                                        hoverKwargs = self.getHoverKwargs()
                                        )
        else:
            columnPair = self.data["columnPairs"][onlyForID]
            if scaleFactor is not None and "size" in self.scatterPlots[columnPair].data:
                self.scatterPlots[columnPair].data["size"] = scaleFactor * self.scatterPlots[columnPair].data["size"]
                

            scatterPlot(self,
                        data = self.scatterPlots[columnPair].data,
                        plotter = self.p,
                        ax = targetAx,
                        numericColumns = list(columnPair),
                        dataID = self.data["dataID"],
                        scatterKwargs = self.scatterKwargs,
                        hoverKwargs = self.getHoverKwargs(),
                        interactive = False
                        )
        
        
    def onDataLoad(self, data):
        ""
    
        if not all(kwarg in data for kwarg in self.requiredKwargs):
            return
            
        self.data = data
        self.initAxes(data["axisPositions"])
        self.initScatterPlots()

        if "axisLabels" in self.data:
            self.setAxisLabels(self.axisDict,self.data["axisLabels"])

        if "axisTitles" in self.data:
            self.addTitles(data["axisTitles"])

        if self.getParam("scatter.equal.axis.limits"):
            self.alignLimitsOfAllAxes(updateFigure=False)

        if "dataColorGroups" in self.data and isinstance(self.data["dataColorGroups"], pd.DataFrame):
            self.setDataInColorTable(self.data["dataColorGroups"], 
                                    title = "Scatter Points")

        if "dataSizeGroups" in self.data and isinstance(self.data["dataSizeGroups"], pd.DataFrame):
            self.setDataInSizeTable(self.data["dataSizeGroups"],
                                    title="Scatter Points")

        if "colorCategoryIndexMatch" in self.data and self.data["colorCategoryIndexMatch"] is not None:
            self.setColorCategoryIndexMatch(self.data["colorCategoryIndexMatch"])

        if "sizeCategoryIndexMatch" in self.data and self.data["sizeCategoryIndexMatch"] is not None:
            self.setSizeCategoryIndexMatch(self.data["sizeCategoryIndexMatch"])           

        #annotate data that are selected by user in QuickSelect widget
        self.checkForQuickSelectDataAndUpdateFigure()

    def setHoverData(self,dataIndex, sender = None):
        "Sets hover data in scatter plots"
        for scatterPlot in self.scatterPlots.values():
            if sender is None:
                scatterPlot.setHoverData(dataIndex)
            elif sender != scatterPlot:
                scatterPlot.setHoverData(dataIndex)

    def setMask(self,dataIndex):
        "Sets a mask on the data to hide some"
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.setMask(dataIndex)
        #self.p.redraw()

    def setResizeTrigger(self,resize=True):
        ""
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.setResizeTrigger(resize)

    def resetMask(self):
        "Resets mask"
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.resetMask()
    
    def prepareGroupColor(self):
        ""
        data = pd.DataFrame()
        scatterColors = self.scatterKwargs["color"]
        if isinstance(scatterColors,str):

            data["color"] = [scatterColors]
            data["group"] = [""]

        self.setDataInColorTable(data, title = "Scatter Points")

    def updateBackgrounds(self, redraw = False):
        ""
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.updateBackground(redraw=redraw)
    
    def updateGroupSizes(self,sizeGroup,changedCategory=None):
        ""
        
        if len(sizeGroup.index) == 1 and sizeGroup.iloc[0,1] == "":
            for scatterPlot in self.scatterPlots.values():
                scatterPlot.updateSizeData(sizeGroup["size"].values[0])
            self.updateFigure.emit()
        
        elif self.sizeCategoryIndexMatch is not None:

            if changedCategory is not None and changedCategory in self.sizeCategoryIndexMatch:

                idx = self.sizeCategoryIndexMatch[changedCategory]
                
                dataBool = sizeGroup["internalID"] == changedCategory
                size = sizeGroup.loc[dataBool,"size"].values[0]
                self.updateScatterPropSection(idx,size,"size")
                self.updateFigure.emit()

            
    def updateGroupColors(self,colorGroup, changedCategory = None):
        ""
                
        if len(colorGroup.index) == 1 and colorGroup.iloc[0,1] == "":
            for scatterPlot in self.scatterPlots.values():
                scatterPlot.updateColorData(colorGroup["color"].values[0])
            self.updateFigure.emit()
        
        elif self.colorCategoryIndexMatch is not None:
            
            if changedCategory is None:
                #very slow for large data sets
                propsData = pd.DataFrame(columns=["color"])
                for k,idx in self.colorCategoryIndexMatch.items():
                    dataBool = colorGroup["internalID"] == k 
                    color = colorGroup.loc[dataBool,"color"].values[0]
                    df = pd.DataFrame([color]*idx.size, index=idx,columns=["color"])
                    propsData = propsData.append(df)
                self.updateScatterProps(propsData)
            else:
                
                if self.isVolcanoPlotStylingActive():
                    for columnPairs, scatterPlot in self.scatterPlots.items():
                        if columnPairs in self.colorCategoryIndexMatch and changedCategory in self.colorCategoryIndexMatch[columnPairs]:
                            idx = self.colorCategoryIndexMatch[columnPairs][changedCategory]
                            dataBool = colorGroup["internalID"] == changedCategory 
                            color = colorGroup.loc[dataBool,"color"].values[0]
                            self.updateScatterPropSectionByScatterplot(scatterPlot,idx,color,"color")
                elif changedCategory in self.colorCategoryIndexMatch:
                    try:
                        idx = self.colorCategoryIndexMatch[changedCategory]
                        dataBool = colorGroup["internalID"] == changedCategory 
                        color = colorGroup.loc[dataBool,"color"].values[0]
                        self.updateScatterPropSection(idx,color,"color")
                    except Exception as e:
                        print(e)
                
            if hasattr(self,"colorLegend"):
                self.addColorLegendToGraph(colorGroup,update=False, title = self.getTitleOfColorTable())
                
            self.updateFigure.emit()

    def updateQuickSelectItems(self,propsData):
        ""
        if self.isQuickSelectModeUnique() and hasattr(self,"quickSelectCategoryIndexMatch"):
            if len(self.quickSelectCategoryIndexMatch) == 0:
                return
            dataIndex = np.concatenate([idx for idx in self.quickSelectCategoryIndexMatch.values()])
            intIDMatch = np.concatenate([np.full(idx.size,intID) for intID,idx in self.quickSelectCategoryIndexMatch.items()]).flatten()
            
        else:
            dataIndex = self.getDataIndexOfQuickSelectSelection()
            intIDMatch = np.array(list(self.quickSelectCategoryIndexMatch.keys()))
        
        for scatterPlot in self.scatterPlots.values():
            if hasattr(scatterPlot,"quickSelectScatter"):
                scatterPlot.setQuickSelectScatterData(dataIndex,propsData.loc[dataIndex])
                self.quickSelectScatterDataIdx[scatterPlot.ax] = dataIndex
                self.quickSelectScatterDataIdx[scatterPlot.ax] = {"idx":dataIndex,"coords":pd.DataFrame(intIDMatch,columns=["intID"])}
        self.updateFigure.emit()
            
    def updateQuickSelectData(self,quickSelectGroup,changedCategory=None):
        ""
        for scatterPlot in self.scatterPlots.values():
            ax = scatterPlot.ax
            if self.isQuickSelectModeUnique():
                scatterSizes, scatterColors, _ = self.getQuickSelectScatterProps(ax,quickSelectGroup)
            else:
                if ax in self.quickSelectScatterDataIdx:
                    dataIdx = self.quickSelectScatterDataIdx[ax]["idx"]
                    scatterSizes = [quickSelectGroup["size"].loc[idx] for idx in dataIdx]	
                    scatterColors = [quickSelectGroup["color"].loc[idx] for idx in dataIdx]
            scatterPlot.updateQuickSelectScatter(scatterColors,scatterSizes)

        self.updateFigure.emit()
    
    def resetQuickSelectArtists(self):
        ""
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.setQuickSelectScatterInivisible()
       

    def mirrorQuickSelectArtists(self,axisID,targetAx):
        ""
        if axisID in self.axisDict:
            sourceAx = self.axisDict[axisID]
            for scatterPlot in self.scatterPlots.values():
                ax = scatterPlot.ax
                if ax == sourceAx:
                    coords,scatterColors,scatterSizes = scatterPlot.getQuickSelectScatterPropsForExport()
                    #add props to standard kwargs
                    kwargs = self.getScatterKwargs()
                    kwargs["zorder"] = 1e9
                    kwargs["s"] = scatterSizes
                    kwargs["color"] = scatterColors
                    targetAx.scatter(x = coords[:,0], y = coords[:,1], **kwargs)

            
    def setHoverObjectsInvisible(self):
        ""
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.setHoverObjectsInvisible(update=False)

    def resetSize(self): 
        ""
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.updateSizeData(self.getParam("scatterSize"))
        self.setDataInSizeTable(self.data["dataSizeGroups"],title="Scatter Points")
        self.updateFigure.emit()

    def setNaNColor(self):
        ""
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.setNaNColorToCollection()
        self.setDataInColorTable(self.data["dataColorGroups"], title = "Scatter Points")
        self.updateFigure.emit()
            

    def removeTooltip(self):
        ""
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.removeTooltip()
        self.setDataInLabelTable()


    def mirrorAxisContent(self,axisID,targetAx,*args,**kwargs):
        ""

        self.initScatterPlots(axisID,targetAx,*args,**kwargs)
        sourceAx = self.axisDict[axisID]
        self.mirrorAnnotations(sourceAx,targetAx,*args,**kwargs) 
        if len(self.textAnnotations) > 0:
            self.addTexts(self.textAnnotations,True,axisID,targetAx)
        self.setAxisLabels({axisID:targetAx},self.data["axisLabels"],onlyForID=axisID)

    def setLabelInAllPlots(self):
        ""
        self.mC.config.toggleParam("annotate.in.all.plots")

    def annotateInAllPlots(self,idx,sender):
        ""
        for annotation in self.annotations.values():
            if annotation != sender:
                annotation.addAnnotations(idx)

    def annotateInAxByDataIndex(self,axisID,idx):
        ""
        if hasattr(self,"selectedLabels"):
        #axisID = self.getAxisID(ax) 
            ax = self.axisDict[axisID]
            if idx in self.selectedLabels[ax]:

                annotObjects = self.getAnnotationTextObjs(ax)
                annotObj = annotObjects[idx]
                annotObj.remove()
                self.deleteAnnotation(ax,idx)
                self.updateFigure.emit()
                return
        
        columnPair = self.data["columnPairs"][axisID]
        self.annotations[columnPair].addAnnotations([idx]) #expect a list/pd.Series

    def initAnnotations(self):
        ""
        if not hasattr(self,"selectedLabels"):
            self.selectedLabels = dict() 
        if not hasattr(self,"addedAnnotations"):
            self.addedAnnotations = dict() 
        if not hasattr(self,"annotationProps"):
            self.annotationProps = dict()
        if not hasattr(self,"annotationBbox"):
            self.annotationBbox = dict()

        for ax in self.axisDict.values():
            self.checkAxInAnnotationDicts(ax)
    
    def checkAxInAnnotationDicts(self,ax):
        ""
        if ax not in self.selectedLabels:
            self.selectedLabels[ax] = []
        if ax not in self.addedAnnotations:
            self.addedAnnotations[ax] = {}
        if ax not in self.annotationProps:
            self.annotationProps[ax] = {}
        if ax not in self.annotationBbox:
            self.annotationBbox[ax] = {} 

    def saveAnnotations(self,ax,idx,annotationText,annotationProps):
        ""
        
        self.checkAxInAnnotationDicts(ax)
        self.selectedLabels[ax].append(idx)
        self.addedAnnotations[ax][idx] = annotationText
        self.annotationProps[ax][idx] = annotationProps
        self.annotationBbox[ax][idx] = annotationText.get_window_extent().bounds

    def getAnnotationIndices(self):
        ""
        if not hasattr(self,"selectedLabels"):
            return dict() 
        return self.selectedLabels
    
    def deleteAnnotation(self,ax,idxToDelete):
        ""
        if hasattr(self,"selectedLabels"):
            labelIdx = self.selectedLabels[ax]
            self.selectedLabels[ax] = [idx for idx in labelIdx if idx != idxToDelete]
            if ax in self.addedAnnotations and idxToDelete in self.addedAnnotations[ax]:
                del self.addedAnnotations[ax][idxToDelete]
            if ax in self.annotationProps and idxToDelete in self.annotationProps[ax]:
                del self.annotationProps[ax][idxToDelete]
            if ax in self.annotationBbox and idxToDelete in self.annotationBbox[ax]:
                del self.annotationBbox[ax][idxToDelete]

    def removeSavedAnnotations(self):
        ""
        if hasattr(self,"selectedLabels"):
            self.selectedLabels.clear()

        if hasattr(self,"addedAnnotations"):
            self.addedAnnotations.clear()

        if hasattr(self,"annotationProps"):
            self.annotationProps.clear() 

    def getAnnotatedLabels(self,ax):
        ""
        if hasattr(self,"selectedLabels"):
            if ax in self.selectedLabels:
                return self.selectedLabels[ax]
    
    def getAnnotationTextProps(self,ax):
        ""
        if hasattr(self,"annotationProps"):
            if ax in self.annotationProps:
                return self.annotationProps[ax]
    
    def getAnnotationBbox(self,ax):
        ""
        if hasattr(self,"annotationBbox"):
            if ax in self.annotationBbox:
                return self.annotationBbox[ax]
       

    def getAnnotationTextObjs(self,ax):
        ""
        if hasattr(self,"addedAnnotations"):
            if ax in self.addedAnnotations:
                return self.addedAnnotations[ax]
    
    def getDataForWebApp(self):
        ""
        if hasattr(self,"menuClickedInAxis"):

            sourceAx = self.menuClickedInAxis
            axisID = self.getAxisID(sourceAx)
            
            columnPair = self.data["columnPairs"][axisID]
            if columnPair in self.scatterPlots:
                scatterData = self.scatterPlots[columnPair].data.dropna(subset=columnPair)
                scatterData["idx"] = scatterData.index
                if "size" not in scatterData.columns:
                    scatterData["size"] = [self.getParam("scatterSize")] * scatterData.index.size
                scatterData["size"] = np.sqrt(scatterData["size"])
                if "color" not in scatterData.columns:
                    scatterData["color"] = [self.getParam("nanColor")] * scatterData.index.size

                xLimit = self.getAxisLimit(sourceAx,which="x")
                yLimit = list(self.getAxisLimit(sourceAx,which="y"))
                yLimit.reverse() #reverse limits for svg based positions(d3)

                annotatedIdx = self.getAnnotatedLabels(sourceAx)
                annotationProps = self.getAnnotationTextProps(sourceAx)

                return (scatterData, 
                            list(columnPair), 
                            {"xDomain":list(xLimit),"yDomain":yLimit},
                            annotatedIdx,
                            annotationProps
                            )
        

    def removeAnnotationsFromGraph(self, update = True):
        ""
       
        if hasattr(self,"annotations") and isinstance(self.annotations,dict) and len(self.annotations) > 0:
            for annotation in self.annotations.values():
                annotation.removeAnnotations()

        self.removeSavedAnnotations()
        if update:
            self.updateFigure.emit()
        

    def updateAnnotationPosition(self,ax,keyClosest,xyRectangle):
        ""
        if hasattr(self,"annotationProps"):
            if ax in self.annotationProps and keyClosest in self.annotationProps[ax]:
                self.annotationProps[ax][keyClosest]['xytext'] = xyRectangle
                #update bbox
                self.annotationBbox[ax][keyClosest] = self.addedAnnotations[ax][keyClosest].get_window_extent().bounds
    
    def isAnnotationInAllPlotsEnabled(self):
        ""
        return self.mC.config.getParam("annotate.in.all.plots")

    def mirrorAnnotations(self,sourceAx,targetAx,*args,**kwargs):
        ""
        if hasattr(self,"annotations") and isinstance(self.annotations,dict) and len(self.annotations) > 0:
            for annotation in self.annotations.values():
                annotation.mirrorAnnotationsToTargetAxis(sourceAx,targetAx,*args,**kwargs)

    def stopQuickSelectCapture(self,event=None):
        ""
        self.preventQuickSelectCapture = True
    
    def startQuickSelectCapture(self):
        ""
        self.preventQuickSelectCapture = False



class ICMultiScatterPlot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICMultiScatterPlot,self).__init__(*args,**kwargs)

        self.scatterPlots = dict()
        self.requiredKwargs = ["axisPositions","dataColorGroups","dataSizeGroups","scatterColumnPairs"]

    def addLinearRegression(self):
        ""
        if len(self.data["linRegFit"]) > 0:
            
            for axisID, lineData in self.data["linRegFit"].items():
                if axisID in self.axisDict:
                    self.addLine(
                            ax = self.axisDict[axisID],
                            xdata = lineData[0],
                            ydata = lineData[1],
                            name = "linRegress_{}".format(axisID)
                            )
    def addLowessLine(self):
        ""
        if len(self.data["lowessFit"]) > 0:
            
            for axisID, lineData in self.data["lowessFit"].items():
                if axisID in self.axisDict:
                    self.addLine(
                            ax = self.axisDict[axisID],
                            xdata = lineData[:,0],
                            ydata = lineData[:,1],
                            name = "lowessRegress_{}".format(axisID)
                            )


    def disconnectBindings(self):
        ""
        super().disconnectBindings()
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.disconnectBindings()
        if hasattr(self,"annotations"):
            for annotation in self.annotations.values():
                annotation.disconnectEventBindings()


    def initScatterPlots(self, onlyForID = None, targetAx = None, scaleFactor = None):
        ""
        if len(self.data["scatterColumnPairs"]) > 0:
            if onlyForID is None:
                #clear saved scatters
                self.scatterPlots.clear()
                #init scatters
                self.scatterKwargs = self.getScatterKwargs()
                for n,ax in self.axisDict.items():
                    if n in self.data["scatterColumnPairs"]:
                        columnPair = self.data["scatterColumnPairs"][n]
                        data = self.data["plotData"]
                        #print(self.data)
                        if "propsData" in self.data:
                            data = data.join(self.data["propsData"])
                        self.scatterPlots[columnPair] = scatterPlot(
                                                self,
                                                data = data,
                                                plotter = self.p,
                                                ax = ax,
                                                numericColumns = list(columnPair),
                                                dataID = self.data["dataID"],
                                                scatterKwargs = self.scatterKwargs,
                                                hoverKwargs = self.getHoverKwargs(),
                                                interactive = self.getParam("multi.scatter.interactive")
                                                )
    def initBackgrounds(self):
        ""
        if len(self.data["backgroundColors"]) > 0:
            for  nA, backgroundColor in self.data["backgroundColors"].items():
                if nA in self.axisDict:
                    ax = self.axisDict[nA]
                    ax.set_facecolor(backgroundColor)

    def initLabels(self):
        ""
        
        if len(self.data["labelData"]) > 0:
            for nAx, labelData in self.data["labelData"].items():
                ax = self.axisDict[nAx]
                self.addText(ax,axisTransform=True,stdTextFont=True,**labelData)
                
    def initKdePlots(self):
        ""
        if len(self.data["kdeData"]) > 0:
            for n,kdeData in self.data["kdeData"].items():
                self.addLine(self.axisDict[n],kdeData["xx"],kdeData["yKde"],"kdeLine")
                self.setAxisLimits(self.axisDict[n],kdeData["xLimit"],kdeData["yLimit"])

    def init2DHistograms(self):
        "Creates density plots on"
        if len(self.data["histogramData"]) > 0:
            for n, data in self.data["histogramData"].items():
                self.axisDict[n].pcolormesh(data["meshX"], data["meshY"], data["H"],cmap=self.getParam("multi.scatter.2D.histogram.cmap"))

    def onDataLoad(self, data):
        if not all(kwarg in data for kwarg in self.requiredKwargs):
            return
        self.data = data
        #create axis and adjust borders
        self.initAxes(data["axisPositions"])
        self.initBackgrounds()
        self.initScatterPlots()
        self.initKdePlots()
        self.initLabels()
        self.init2DHistograms()

        self.addLinearRegression()
        self.addLowessLine()
        #check quick select
        qsData = self.getQuickSelectData()
        if "categoryIndexMatch" in self.data and self.data["categoryIndexMatch"] is not None:
            self.setColorCategoryIndexMatch(self.data["categoryIndexMatch"])
        if "dataColorGroups" in self.data:
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorTitle"])
        if "dataSizeGroups" in self.data:
            self.setDataInSizeTable(self.data["dataSizeGroups"],title="Scatter Points")

        if qsData is not None:
            self.mC.quickSelectTrigger.emit()
        else:
            self.updateFigure.emit()
#
    def setHoverData(self,dataIndex, sender = None):
        "Sets hover data in scatter plots"
        for scatterPlot in self.scatterPlots.values():
            if sender is None:
                scatterPlot.setHoverData(dataIndex)
            elif sender != scatterPlot:
                scatterPlot.setHoverData(dataIndex)
        
    def updateBackgrounds(self, redraw = False):
        "Updates backgrounds in scatter plot. Required to enabled blitting"
        for scatterPlot in self.scatterPlots.values():
            scatterPlot.updateBackground(redraw=redraw)

    def updateGroupColors(self,colorGroup, changedCategory = None):
        "Update color by changes by the user in the color table."      
        if len(colorGroup.index) == 1 and colorGroup.iloc[0,1] == "":
            for scatterPlot in self.scatterPlots.values():
                scatterPlot.updateColorData(colorGroup["color"].values[0])
            self.updateFigure.emit()
        
        elif self.colorCategoryIndexMatch is not None:
            
            if changedCategory in self.colorCategoryIndexMatch:
                    
                        idx = self.colorCategoryIndexMatch[changedCategory]
                        dataBool = colorGroup["internalID"] == changedCategory 
                        color = colorGroup.loc[dataBool,"color"].values[0]
                        self.updateScatterPropSection(idx,color,"color")
                    
            if hasattr(self,"colorLegend"):
                self.addColorLegendToGraph(colorGroup,update=False, title = self.getTitleOfColorTable())
                
            self.updateFigure.emit()