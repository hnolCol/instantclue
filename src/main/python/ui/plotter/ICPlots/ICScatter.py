#
from PyQt5.QtCore import QObject, pyqtSignal
from .ICChart import ICChart
from .charts.scatter_plotter import scatterPlot
from .ICScatterAnnotations import ICScatterAnnotations
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
                labelColumnNamesNotInData = [colName for colName in labelColumnNames if colName not in labelColumnNames]
                #columnNames = labelColumnNames.append(numericColumns,ignore_index=True)
                if len(labelColumnNamesNotInData) > 0:
                    data = self.mC.data.getDataByColumnNames(dataID,labelColumnNamesNotInData)["fnKwargs"]["data"]
                    plotData = self.scatterPlots[columnPair].data
                    data = plotData.join(data)
               
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


    def getGraphSpecMenus(self):
        ""
        return ["Axis limits .."]

    def addGraphSpecActions(self,menus):
        ""
        menus["Axis limits .."].addAction("Raw limits", self.rawAxesLimits)
        menus["Axis limits .."].addAction("Center x to 0", self.centerXToZero)
        menus["Axis limits .."].addAction("Set equal axes limits", self.alignLimitsOfAllAxes)
        menus["Axis limits .."].addAction("Set x- and y-axis limits equal", self.alignLimitsOfXY)
        if self.preventQuickSelectCapture:
            menus["main"].addAction("Enable QuickSelect Capture", self.startQuickSelectCapture)
        else:
            menus["main"].addAction("Stop QuickSelect Capture", self.stopQuickSelectCapture)

    def addTooltip(self, tooltipColumnNames,dataID):
        ""
        try:
            data = self.mC.data.getDataByColumnNames(dataID,tooltipColumnNames)["fnKwargs"]["data"]
            for scatterPlot in self.scatterPlots.values():
                scatterPlot.addTooltip(data )
            labelData = pd.DataFrame() 
            labelData["columnName"] = tooltipColumnNames
            self.setDataInTooltipTable(labelData,title="Tooltip Labels")
        except Exception as e:
            print(e)

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

        qsData = self.getQuickSelectData()
        if self.getParam("scatter.equal.axis.limits"):
            self.alignLimitsOfAllAxes(updateFigure=False)
        if qsData is not None:
            self.mC.quickSelectTrigger.emit()
        else:
            self.setDataInColorTable(self.data["dataColorGroups"], title = "Scatter Points")
            self.setDataInSizeTable(self.data["dataSizeGroups"],title="Scatter Points")
            self.updateFigure.emit()
    
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
                if changedCategory in self.colorCategoryIndexMatch:
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
        self.mirrorAnnotations(sourceAx,targetAx,*args,**kwargs) #bad! ,testing
        self.setAxisLabels({axisID:targetAx},self.data["axisLabels"],onlyForID=axisID)

    def setLabelInAllPlots(self):
        ""
        self.mC.config.toggleParam("annotate.in.all.plots")

    def annotateInAllPlots(self,idx,sender):
        ""
        for annotation in self.annotations.values():
            if annotation != sender:
                annotation.addAnnotations(idx)
    
    def saveAnnotations(self,ax,idx,annotationText,annotationProps):
        ""
        if not hasattr(self,"selectedLabels"):
            self.selectedLabels = dict() 
        if not hasattr(self,"addedAnnotations"):
            self.addedAnnotations = dict() 
        if not hasattr(self,"annotationProps"):
            self.annotationProps = dict()

        if not ax in self.selectedLabels:
            self.selectedLabels[ax] = []
            self.addedAnnotations[ax] = {}
            self.annotationProps[ax] = {}
        
        self.selectedLabels[ax].append(idx)
        self.addedAnnotations[ax][idx] = annotationText
        self.annotationProps[ax][idx] = annotationProps
    
    def deleteAnnotation(self,ax,idxToDelete):
        ""
        if hasattr(self,"selectedLabels"):
            labelIdx = self.selectedLabels[ax]
            self.selectedLabels[ax] = [idx for idx in labelIdx if idx != idxToDelete]
            if ax in self.addedAnnotations and idxToDelete in self.addedAnnotations[ax]:
                del self.addedAnnotations[ax][idxToDelete]
            if ax in self.annotationProps and idxToDelete in self.annotationProps[ax]:
                del self.annotationProps[ax][idxToDelete]

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

    def getAnnotationTextObjs(self,ax):
        ""
        if hasattr(self,"addedAnnotations"):
            if ax in self.addedAnnotations:
                return self.addedAnnotations[ax]

    def removeAnnotationsFromGraph(self):
        ""
        try:
            if hasattr(self,"annotations") and isinstance(self.annotations,dict) and len(self.annotations) > 0:
                for annotation in self.annotations.values():
                    annotation.removeAnnotations()

            self.removeSavedAnnotations()
            self.updateFigure.emit()
        except Exception as e:
            print(e)

    def updateAnnotationPosition(self,ax,keyClosest,xyRectangle):
        ""
        if hasattr(self,"annotationProps"):
            if ax in self.annotationProps and keyClosest in self.annotationProps[ax]:
                self.annotationProps[ax][keyClosest]['xytext'] = xyRectangle
    
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
