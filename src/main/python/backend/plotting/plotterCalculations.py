
import numpy as np 
import pandas as pd 
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor, QBrush
from matplotlib.colors import to_hex
from matplotlib.pyplot import boxplot
#backend imports
from backend.utils.stringOperations import getReadableNumber
from ..utils.stringOperations import getMessageProps, getReadableNumber, getRandomString, mergeListToString
from ..utils.misc import scaleBetween, replaceKeyInDict
from .postionCalculator import calculatePositions, getAxisPostistion

from threadpoolctl import threadpool_limits

#cluster
import fastcluster
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.colors import to_hex
from matplotlib.pyplot import hist
import matplotlib.cm as cm
from matplotlib import rcParams

from collections import OrderedDict
import time
import seaborn as sns 

import numpy as np, scipy.stats as st

def CI(a, ci = 0.95):
    a = a[~np.isnan(a)]
    if a.size == 0:
        return 0
    mean = np.mean(a)
    minCI, maxCI =  st.t.interval(ci, len(a)-1, loc=mean, scale=st.sem(a))
    return maxCI - mean

plotFnDict = {
    "boxplot":"getBoxplotProps",
    "barplot":"getBarplotProps",
    "pointplot":"getPointplotProps",
    "swarmplot":"getSwarmplotProps",
    "scatter":"getScatterProps",
    "hclust":"getHeatmapProps",
    "corrmatrix":"getCorrmatrixProps",
    "violinplot":"getViolinProps",
    "lineplot":"getLineplotProps",
    "histogram":"getHistogramProps",
    "countplot":"getCountplotProps",
    "x-ys-plot":"getXYPlotProps",
    "dim-red-plot":"getDimRedProps"
}

line_kwargs = dict(linewidths=.45, colors='k')

transformDict = {"log2" : np.log2, "log10" : np.log10, "ln" : np.log}

class PlotterBrain(object):
    ""
    def __init__(self, sourceData):
        ""
        self.sourceData = sourceData
        self.axisBoxColor = "#F0F0F0"
        self.boxplotCapsLineWidth = 0.5
        self.axisBorderLineWidth = 1
        self.axisLineWidth = 1
        self.corrMatrixMethod = "pearson"
        self.maxColumns = 4
        self.colorColumn = None
        self.colorColumnType = "Categories"
        self.sizeColumn = None
        self.sizeColumnType = "Categories"
        self.scatterSize = 50
        self.maxScatterSize = 120
        self.minScatterSize = 20
        self.aggMethod = "mean"
        self.plotAgainstIndex = False
        self.indexSort = "ascending"
        self.barplotError = "CI (95%)"
        self.barplotMetric = "mean"
        self.histogramBins = 15
        self.histogramCumulative = False
        self.histogramDensity = False
        self.histogramHisttype = "bar"
        self.histogramLog = False
        self.histogramLinewidth = 1
        self.histogramSortCategories = False
        self.countTransform = "none"

    def findPositions(self,numericColumns,categoricalColumns, plotType):
        ""
        nNumCol = len(numericColumns)
        if len(categoricalColumns) == 0:

            if plotType == "boxplot":
                width = 0.8
                x0 = np.linspace(0,nNumCol-1, num = nNumCol) - width/2
                return x0,width
            
    def getPlotProps(self, dataID, numericColumns, categoricalColumns = None, plotType = "Scatter", **kwargs):
        ""

        self.numericColumns = numericColumns
        self.categoricalColumns = categoricalColumns
        self.plotType = plotType
        with threadpool_limits(limits=1, user_api='blas'): 
            return getattr(self,plotFnDict[plotType])(dataID,numericColumns,categoricalColumns,**kwargs)


    def getCountplotProps(self,dataID, numericColumns, categoricalColumns):
        ""
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        subplotBorders = dict(wspace=0.15, hspace = 0.0,bottom=0.15,right=0.95,top=0.95)
        axisDict = getAxisPostistion(2, maxCol = 1)
        groupbyCatColumns = data.groupby(by=categoricalColumns, sort=False)
        colors = self.sourceData.colorManager.getNColorsByCurrentColorMap(len(categoricalColumns),"countplotLabelColorMap")
        groupSizes = groupbyCatColumns.size().sort_values(ascending=False).reset_index(name='counts')
        #
      #  print(groupSizes)
        colorGroups["group"] = categoricalColumns
        colorGroups["color"] = colors
        colorGroups["internalID"] = [getRandomString() for _ in categoricalColumns]
        uniqueValues = OrderedDict([(catColumn, groupSizes[catColumn].unique()) for catColumn in categoricalColumns])
        tickColors = np.concatenate([[colors[n]] * v.size for n,v in enumerate(uniqueValues.values())])

        numUniqueValues = np.sum([v.size for v in uniqueValues.values()])
        uniqueValueList = np.concatenate([v for v in uniqueValues.values()])
        xLimitMax = groupSizes.index.size
        xCountValues = np.arange(groupSizes.index.size)
        rawCounts = groupSizes["counts"]
        if self.countTransform != "none" and self.countTransform in transformDict:
            groupSizes["counts"] = transformDict[self.countTransform](groupSizes["counts"])
        yCountValues = groupSizes["counts"].values 
        maxCountValue = np.max(yCountValues)
        yTicksPoints = np.arange(numUniqueValues)
        #get facotrs for unique cats
        factors = OrderedDict() 
        for n, catCol in enumerate(categoricalColumns):
            codes, _ = groupSizes[catCol].factorize()
            if n == 0:
                factors[catCol] = codes 
            else:
                maxFactors = np.max([v for v in factors.values()]) + 1
                codes = codes + maxFactors 
                factors[catCol] = codes 
        lines = {}
        hoverData = {}
        for idx in groupSizes.index:

            groupValues = groupSizes.loc[idx,categoricalColumns].values
            if len(categoricalColumns) > 1:

                hoverData[idx] = groupbyCatColumns.get_group(tuple(groupValues)).index
            else:
                hoverData[idx] = groupbyCatColumns.get_group(groupValues[0]).index

            vs = np.array([(idx, factors[categoricalColumns[n]][idx]) for n,x in enumerate(groupValues)])
            lines[idx] = {"xdata" :vs[:,0],"ydata" : vs[:,1]}
            #lines.append(l)

        
        
        return {"data":{
            "axisPositions":axisDict,
            "subplotBorders":subplotBorders,
            "rawCounts" : rawCounts,
            "tickColors" : tickColors,
            "plotData" : {"bar" : 
                            {"x":xCountValues,"height" : yCountValues},
                          "lineplot": lines},
            "axisLimits" : {0:{"xLimit" : [-0.5,xLimitMax-0.5], "yLimit" : [0,maxCountValue+0.01*maxCountValue]},
                            1:{"xLimit" : [-0.5,xLimitMax-0.5], "yLimit" : [-0.5,numUniqueValues]}},
            "tickPositions" : {1:{"y":yTicksPoints,"x":np.arange(xLimitMax)}},
            "tickLabels" : {1:{"y":uniqueValueList,"x": [""] * xLimitMax}},
            "barLabels" : yCountValues,
            "hoverData" : hoverData,
            "colorCategoricalColumn" : "Categorical Columns",
            "dataColorGroups": colorGroups

            
        }}



    def getBarplotProps(self, dataID, numericColumns, categoricalColumns):
        ""
        subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        if len(categoricalColumns) > 3:
            splitByCats = False
        else:
            splitByCats = self.sourceData.parent.config.getParam("boxplot.split.data.on.category")
        plotData, \
        axisPositions, \
        boxPositions, \
        tickPositions, \
        tickLabels, \
        colorGroups, \
        faceColors, \
        colorCategoricalColumn, \
        xWidth, axisLabels, axisLimits, axisTitles, groupNames, verticalLines = calculatePositions(dataID,self.sourceData,numericColumns,categoricalColumns,self.maxColumns,splitByCategories= splitByCats)
            

        filteredData = OrderedDict()
        hoverData = OrderedDict() 
        for n,plotData in plotData.items():
            data = plotData["x"]
            hoverData[n] = {"x" : data}
            plotData["height"] = [np.mean(x) for x in data]
            plotData["width"] = xWidth
            plotData["color"] = faceColors[n]
            if "CI" in self.barplotError:
                plotData["yerr"] = [CI(a.values) for a in data]
            elif self.barplotError == "Std":
                plotData["yerr"] = [np.std(a.values) for a in data]
            plotData["error_kw"] = {"capsize":rcParams["errorbar.capsize"],"elinewidth":0.5,"markeredgewidth":0.5,"zorder":1e6}

            plotData["x"] = boxPositions[n] 
            filteredData[n] = plotData

        return {"data":{
                "plotData":filteredData,#"
                "facecolors" : faceColors,
                "axisPositions":axisPositions,
                "hoverData" : hoverData,
                "tickLabels": tickLabels,
                "tickPositions": tickPositions,
                "axisLabels" : axisLabels,
                "axisLimits" : axisLimits,
                "axisTitles" : axisTitles,
                "groupNames" : groupNames,
                "dataColorGroups": colorGroups,
                "subplotBorders":subplotBorders,
                "verticalLines" : verticalLines,
                "colorCategoricalColumn" : colorCategoricalColumn,
                #"tooltipsTexts" : texts,
                "dataID":dataID}}

    def getLineplotProps(self,dataID,numericColumns,categoricalColumns):
        ""
        minQ = np.inf
        maxQ = -np.inf
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        axisTitles = {}

        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        if len(categoricalColumns) == 0:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPostistion(1,maxCol=self.maxColumns)
            quantiles = np.nanquantile(data.values,[0,0.25,0.5,0.75,1],axis=0)
            xValues = np.arange(len(numericColumns))
            plotData = {0:[{"quantiles":quantiles,"xValues":xValues,"color":self.sourceData.colorManager.nanColor}]}
            minQ, maxQ = np.min(quantiles[0,:]),np.max(quantiles[-1,:])
            marginY = np.sqrt(minQ**2 + maxQ**2) * 0.05
            nNumCol = len(numericColumns)
            marginX = 0.05*(nNumCol-1)
            axisLimits = {0:{"xLimit":[-marginX,len(numericColumns)-(1-marginX)],"yLimit":[minQ-marginY,maxQ+marginY]}}
            colorCategoricalColumn = ""
            colorGroups["color"] = [self.sourceData.colorManager.nanColor]
            colorGroups["group"] = [""]
            colorGroups["internalID"] = [getRandomString()]
            hoverData = {0:data}
            tickPositions = {0:np.arange(nNumCol)}
            tickLabels = {0:numericColumns}


        elif len(categoricalColumns) == 1:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPostistion(1,maxCol=self.maxColumns)
            plotData = {0:[]}
            colorCategoricalColumn = categoricalColumns[0]
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = colorCategoricalColumn)            
           # colorList = self.sourceData.colorManager.getNColorsByCurrentColorMap(N=len(groupBy.keys()))
            colors,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            for groupName, groupData in data.groupby(by=categoricalColumns[0],sort=False):
                quantiles = np.nanquantile(groupData[numericColumns],[0,0.25,0.5,0.75,1],axis=0)
                xValues = np.arange(len(numericColumns))
                plotData[0].append({"quantiles":quantiles,"xValues":xValues,"color":colors[groupName]})
                minQGroup, maxQGroup = np.min(quantiles[0,:]),np.max(quantiles[-1,:])
                if minQGroup < minQ:
                    minQ = minQGroup
                if maxQGroup > maxQ:
                    maxQ = maxQGroup
            nNumCol = len(numericColumns)
            marginY = np.sqrt(minQ**2 + maxQ**2) * 0.05  
            marginX = 0.05*(nNumCol-1)  
            axisLimits = {0:{"xLimit":[-marginX,len(numericColumns)-(1-marginX)],"yLimit":[minQ-marginY,maxQ+marginY]}}
            tickPositions = {0:np.arange(nNumCol)}
            tickLabels = {0:numericColumns}

            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(colorCategories.size)]

            hoverData = {0:data}

        elif len(categoricalColumns) == 2:
            subplotBorders = dict(wspace=0.10, hspace = 0.1,bottom=0.05,right=0.95,top=0.95)
            splitCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1]) 
            axisPositions = getAxisPostistion(splitCategories.size,maxCol=self.maxColumns)
            plotData = dict([(n,[]) for n in axisPositions.keys()])

            colorCategoricalColumn = categoricalColumns[0]
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = colorCategoricalColumn)            
           # colorList = self.sourceData.colorManager.getNColorsByCurrentColorMap(N=len(groupBy.keys()))
            colors,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            hoverData = {} 
            
            for n, (axisName,axisData) in enumerate(data.groupby(by=categoricalColumns[1],sort=False)):
                #n = axis number
                hoverData[n] = axisData
                for groupName, groupData in axisData.groupby(by=categoricalColumns[0],sort=False):
                    quantiles = np.nanquantile(groupData[numericColumns],[0,0.25,0.5,0.75,1],axis=0)
                    xValues = np.arange(len(numericColumns))
                    plotData[n].append({"quantiles":quantiles,"xValues":xValues,"color":colors[groupName]})
                    minQGroup, maxQGroup = np.min(quantiles[0,:]),np.max(quantiles[-1,:])
                    if minQGroup < minQ:
                        minQ = minQGroup
                    if maxQGroup > maxQ:
                        maxQ = maxQGroup

                nNumCol = len(numericColumns)
                marginY = np.sqrt(minQ**2 + maxQ**2) * 0.05  
                marginX = 0.05*(nNumCol-1) 
                 
                axisLimits = dict([(n,{"xLimit":[-marginX,len(numericColumns)-(1-marginX)],
                                       "yLimit":[minQ-marginY,maxQ+marginY]}) for n in axisPositions.keys()])
                tickPositions = dict([(n,np.arange(nNumCol)) for n in axisPositions.keys()])
                tickLabels = dict([(n, numericColumns) for n in axisPositions.keys()])
                axisTitles[n] = "{}\n{}".format(categoricalColumns[1],axisName)
                colorGroups["color"] = colors.values()
                colorGroups["group"] = colorCategories
                colorGroups["internalID"] = [getRandomString() for n in range(colorCategories.size)]

                




        return {"data":{
                "plotData":plotData,
                "hoverData" : hoverData,
                "axisPositions":axisPositions,
                "axisTitles" : axisTitles,
                "subplotBorders":subplotBorders,
                "dataColorGroups": colorGroups,
                "axisLimits":axisLimits,
                "tickPositions" : tickPositions,
                "tickLabels" : tickLabels,
                "numericColumns":numericColumns,
                "colorCategoricalColumn" : colorCategoricalColumn}
                }

    def getHistogramProps(self,dataID,numericColumns,categoricalColumns):
        ""
        #get raw data
        patches = {}
        axisLimits = {}
        axisLabels = {}
        hoverData = {}

        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        subplotBorders = dict(wspace=0.175, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        axisPositions = getAxisPostistion(len(numericColumns),maxCol=self.maxColumns)
        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        
        if len(categoricalColumns) == 0:

            colors = self.sourceData.colorManager.getNColorsByCurrentColorMap(len(numericColumns))
            internalIDs = [getRandomString() for n in range(len(colors))]
            for n,numCol in enumerate(numericColumns):
                if self.histogramLog:
                    histData = np.log2(data[numCol]).replace([np.inf, -np.inf],np.nan).dropna()
                else:
                    histData = data[numCol].replace([np.inf, -np.inf], np.nan).dropna()
                if histData.empty:
                        continue
                hist, bins = np.histogram(histData.values, 
                                            bins=self.histogramBins, 
                                            density=self.histogramDensity)
                width = 0.95 * (bins[1] - bins[0])
                
               # center = (bins[:-1] + bins[1:]) / 2
                if self.histogramCumulative:
                    hist = np.cumsum(hist)
                axisLabels[n] = {"x":numCol if not self.histogramLog else "log2({})".format(numCol)}
                hoverData[n] = histData
                if self.histogramHisttype == "bar":
                    patches[n] = [{
                            "p":dict(
                                xy = (binStart,0), 
                                width= width, 
                                height= hist[m], 
                                facecolor = colors[n],
                                linewidth = 0.2,
                                edgecolor = "black"),
                            "type": "Rectangle",
                            "internalID" : internalIDs[n]
                                                    } for m,binStart in enumerate(bins[:-1])]
                elif self.histogramHisttype == "step":

                    xy = self.createSteppedHistoValues(bins,hist)
                    patches[n] = [{"p":dict(
                                        xy = xy,
                                        edgecolor = colors[n],
                                        linewidth = self.histogramLinewidth,
                                        closed= False,
                                        facecolor = "None"
                                        ),
                                    "type":"Polygon",
                                    "internalID" : internalIDs[n]}]
                
                colorGroups["color"] = colors
                colorGroups["group"] = numericColumns
                colorGroups["internalID"] = internalIDs 
                colorCategoricalColumn = "Numeric Columns"
                minHist, maxHist = np.min(hist), np.max(hist)
                xMargin = np.sqrt(bins[-1]**2 + bins[0]**2) * 0.02
                yMargin = np.sqrt(minHist**2 + maxHist**2) * 0.02
                axisLimits[n] = {
                    "xLimit":(bins[0]-xMargin ,bins[-1]+xMargin),
                    "yLimit":(0,maxHist + yMargin)}
                
        elif len(categoricalColumns) > 0:
            colorCategoricalColumn = "\n".join(categoricalColumns)
            groupby = data.groupby(by=categoricalColumns, sort=self.histogramSortCategories)
            colorCategories = list(groupby.groups.keys()) #self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = colorCategoricalColumn)
            internalIDs = OrderedDict([(colorCat,getRandomString()) for colorCat in colorCategories])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, 
                                                            as_hex=True, 
                                                            addNaNLevels=[(self.sourceData.replaceObjectNan,)*len(categoricalColumns)],)
            
            for n,numCol in enumerate(numericColumns):
                minHist, maxHist = np.inf, -np.inf
                if self.histogramLog:
                    histRangeMin, histRangeMax = np.nanmin(np.log2(data[numCol].values)), np.nanmax(np.log2(data[numCol].values))
                else:
                    histRangeMin, histRangeMax = np.nanmin(data[numCol].values), np.nanmax(data[numCol].values)

                for groupName, groupData in groupby:

                    internalID = internalIDs[groupName]
                    if self.histogramLog:
                        histData = np.log2(groupData[numCol]).replace([np.inf, -np.inf],np.nan).dropna()
                    else:
                        histData = groupData[numCol].dropna()
                    hoverData[n] = histData
                    if histData.empty:
                        continue
                    hist, bins = np.histogram(
                                histData.values, 
                                bins=self.histogramBins, 
                                density=self.histogramDensity,
                                range = (histRangeMin, histRangeMax))
                    width = 0.95 * (bins[1] - bins[0])
                # center = (bins[:-1] + bins[1:]) / 2
                    if self.histogramCumulative:
                        hist = np.cumsum(hist)
                    axisLabels[n] = {"x": numCol if not self.histogramLog else "log2({})".format(numCol)}
                    if self.histogramHisttype == "bar":

                        histPatches = [{
                                "p":dict(
                                    xy = (binStart,0), 
                                    width= width, 
                                    height= hist[m], 
                                    facecolor = colors[groupName],
                                    linewidth = 0.2,
                                    edgecolor = "black",
                                    alpha = .5),
                                "type" : "Rectangle",
                                "internalID" : internalID
                                                        } for m,binStart in enumerate(bins[:-1])]

                    elif self.histogramHisttype == "step":

                        xy = self.createSteppedHistoValues(bins,hist)
                        histPatches = [{"p":dict(
                                            xy = xy,
                                            edgecolor = colors[groupName],
                                            linewidth = self.histogramLinewidth,
                                            closed= False,
                                            facecolor = "None"
                                            ),
                                        "type" : "Polygon",
                                        "internalID" : internalIDs[groupName]}]    
                    if n in patches:
                        p = patches[n]
                        patches[n] = p + histPatches
                    else:
                        patches[n] = histPatches

                    minHistCol, maxHistCol = np.nanmin(hist), np.nanmax(hist)

                    if minHistCol < minHist:
                        minHist = minHistCol
                    if maxHistCol > maxHist:
                        maxHist = maxHistCol
                    
                    xMargin = np.sqrt(bins[-1]**2 + bins[0]**2) * 0.02
                    yMargin = np.sqrt(minHist**2 + maxHist**2) * 0.02

                    axisLimits[n] = {
                        "xLimit":(bins[0]-xMargin ,bins[-1]+xMargin),
                        "yLimit":(0,maxHist + yMargin)}
                    

            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories 
            colorGroups["internalID"] = internalIDs.values()



        return {"data":{
            "patches" : patches,
            "axisLabels" : axisLabels,
            "axisPositions" : axisPositions,
            "axisLimits" : axisLimits,
            "subplotBorders" : subplotBorders,
            "dataColorGroups": colorGroups,
            "colorCategoricalColumn" : colorCategoricalColumn,
            "hoverData" : hoverData
        }}
                        

    def createSteppedHistoValues(self, bins, hist):
        ""
        x = np.repeat(bins[1:],2)
        x = np.append([bins[0]],x).reshape(-1,1)
        y = np.repeat(hist,2)
        y = np.append(y,[hist[-1]]).reshape(-1,1)
        
        return np.append(x,y,axis=1)

    def getViolinProps(self,dataID,numericColumns, categoricalColumns):
        ""    
        # axisPostions = dict([(n,[1,1,n+1]) for n in range(1)])
        subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)

        if len(categoricalColumns) > 3:
            splitByCats = False
        else:
            splitByCats = self.sourceData.parent.config.getParam("boxplot.split.data.on.category")
        #data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        #colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
        plotData, \
        axisPositions, \
        violinPositions, \
        tickPositions, \
        tickLabels, \
        colorGroups, \
        faceColors, \
        colorCategoricalColumn, \
        xWidth, axisLabels, axisLimits, axisTitles, groupNames, verticalLines = calculatePositions(
                                                                                dataID,
                                                                                self.sourceData,
                                                                                numericColumns,
                                                                                categoricalColumns,
                                                                                self.maxColumns,
                                                                                splitByCategories= splitByCats)        
        # xLabel = ""

        filteredData = OrderedDict()
        medianData = OrderedDict() 
        minMaxLine = OrderedDict() 
        quantileLine = OrderedDict()
        hoverData = OrderedDict() 

        for n,plotData in plotData.items():
            data = plotData["x"]
            plotData["widths"] = xWidth
            plotData["dataset"] = [x.values for x in data]
            
            plotData["showmedians"] = False
            plotData["positions"] = violinPositions[n]
            plotData["showextrema"] = False
            plotData["points"] = 200 
            
            hoverData[n] = {"x" : data }
            del plotData["x"]
            filteredData[n] = plotData

            medianData[n] = {}
            medianData[n]["x"] = violinPositions[n]
            medianData[n]["y"] = [np.median(yData) for yData in plotData["dataset"]]

            minMaxLine[n] = [np.append(np.array([x]),np.nanquantile(yData,q = [0,1])) for x,yData in zip(violinPositions[n],plotData["dataset"])]
            quantileLine[n] = [np.append(np.array([x]),np.nanquantile(yData,q = [0.25,0.75])) for x,yData in zip(violinPositions[n],plotData["dataset"])]

        return {"data":{
                "plotData":filteredData,#"
                "facecolors" : faceColors,
                "medianData" : medianData,
                "minMaxLine" :  minMaxLine,
                "quantileLine" : quantileLine,
                "axisPositions":axisPositions,
                "tickLabels": tickLabels,
                "axisTitles" : axisTitles,
                "tickPositions": tickPositions,
                "axisLabels" : axisLabels,
                "axisLimits" : axisLimits,
                "groupNames" : groupNames,
                "hoverData" : hoverData,
                "dataColorGroups": colorGroups,
                "subplotBorders":subplotBorders,
                "verticalLines" : verticalLines,
                "colorCategoricalColumn" : colorCategoricalColumn,
                "dataID":dataID}}

    def getBoxplotProps(self, dataID, numericColumns, categoricalColumns):
        ""
        
        subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        if len(categoricalColumns) > 3:
            splitByCats = False
        else:
            splitByCats = self.sourceData.parent.config.getParam("boxplot.split.data.on.category")

        plotCalcData, axisPositions, boxPositions, tickPositions, tickLabels, colorGroups, \
            faceColors, colorCategoricalColumn, xWidth, axisLabels, axisLimits, \
                axisTitles, groupNames, verticalLines = calculatePositions(dataID,
                                                                    self.sourceData,
                                                                    numericColumns,
                                                                    categoricalColumns,
                                                                    self.maxColumns, 
                                                                    splitByCategories = splitByCats)


        
        filteredData = OrderedDict()
        for n,plotData in plotCalcData.items():
            plotData["widths"] = xWidth
            plotData["patch_artist"] = True
            plotData["positions"] = boxPositions[n] 
            plotData["capprops"] = {"linewidth":self.boxplotCapsLineWidth}
            filteredData[n] = plotData

        
        #print(axisPositions)
        
    
        return {"data":{
                "plotData":filteredData,#"
                "facecolors" : faceColors,
                "axisPositions":axisPositions,
                "axisLimits" : axisLimits,
                "tickLabels": tickLabels,
                "tickPositions": tickPositions,
                "axisLabels" : axisLabels,
                "axisTitles" : axisTitles,
                "groupNames" : groupNames,
                "dataColorGroups": colorGroups,
                "subplotBorders":subplotBorders,
                "colorCategoricalColumn" : colorCategoricalColumn,
                "verticalLines" : verticalLines,
                #"tooltipsTexts" : texts,
                "dataID":dataID}}
        
    def getPointplotProps(self,dataID,numericColumns,categoricalColumns):
        """
        If categoricalColmns == 0 
        


        return general  line2Ds
        """

        scaleXAxis = self.sourceData.parent.config.getParam("scale.numeric.x.axis")
        splitString = self.sourceData.parent.config.getParam("split.string.x.category")
        splitIndex = self.sourceData.parent.config.getParam("split.string.index")
        axisLimits = {}
        tickLabels = {}
        tickPositions = {}
        axisLabels = {}
        #
        lineKwargs = OrderedDict()
        errorKwargs = OrderedDict()
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        
        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        if len(categoricalColumns) == 0:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPostistion(1,maxCol=self.maxColumns)
            colorList = self.sourceData.colorManager.getNColorsByCurrentColorMap(N=len(numericColumns))
            
            for n in axisPositions.keys():
                columnMeans = rawData[numericColumns].mean().values
                errorValues = [CI(rawData[columnName].dropna()) for columnName in numericColumns]
                maxErrorValue = np.nanmax(errorValues)
                minValue, maxValue = np.nanmin(columnMeans), np.nanmax(columnMeans)
                if np.isnan(maxErrorValue):
                    maxErrorValue = 0.05*maxValue
                tickLabels[n] = numericColumns
                tickPositions[n] = np.arange(len(numericColumns))

                line2DKwargs = []
                line2DErrorKwargs = []
                for m in np.arange(len(numericColumns)):
                    line2D = {}
                    error2D = {}
                    line2D["marker"] = "o"
                    line2D["markerfacecolor"] = colorList[m]
                    line2D["markeredgecolor"] = rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = np.sqrt(50)
                    line2DKwargs.append(line2D)

                    error2D["x"] = [m]
                    error2D["y"] = [columnMeans[m]]
                    error2D["yerr"] = [errorValues[m]]
                    error2D["elinewidth"] = rcParams["patch.linewidth"]
                    error2D["ecolor"] = rcParams["patch.edgecolor"]
                    line2DErrorKwargs.append(error2D)
                lineKwargs[n] = line2DKwargs
                errorKwargs[n] = line2DErrorKwargs
                #define axis limits
                axisLimits[n] = {
                        "xLimit": (-0.5,len(numericColumns)-0.5),
                        "yLimit" : (minValue-3*maxErrorValue,maxValue+3*maxErrorValue)
                        }
                
                colorGroups["color"] = colorList
                colorGroups["group"] = numericColumns
                colorGroups["internalID"] = [getRandomString() for n in range(len(numericColumns))]
                axisLabels[n] = {"x":"","y":"Value"}

            colorCategoricalColumn = "Numeric Columns"
        
        elif len(categoricalColumns) == 1 and len(numericColumns) == 1:
            
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPostistion(1,maxCol=self.maxColumns)
            #get unique categories
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(nColorCats)]
            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns)
            colorCategoricalColumn = categoricalColumns[0]
            for n in axisPositions.keys():
                columnMeans = [data[numericColumns[0]].mean() for groupName, data in groupByCatColumn]
                errorValues = [CI(data[numericColumns[0]].dropna()) for groupName, data in groupByCatColumn]
                maxErrorValue = np.nanmax(errorValues)
                minValue, maxValue = np.nanmin(columnMeans), np.nanmax(columnMeans)
                if np.isnan(maxErrorValue):
                    maxErrorValue = 0.05*maxValue
                tickLabels[n] = colorCategories
                if scaleXAxis:
                    try:
                        tickPos = [float(x.split(splitString)[splitIndex]) for x in colorCategories]
                    except: 
                        tickPos = np.arange(nColorCats)
                else:
                    tickPos = np.arange(nColorCats)
                    
                tickPositions[n] = tickPos

                line2DKwargs = []
                line2DErrorKwargs = []
                for m, category in enumerate(colorCategories):
                    line2D = {}
                    error2D = {}
                    line2D["marker"] = "o"
                    line2D["markerfacecolor"] = colors[category]
                    line2D["markeredgecolor"] = rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = np.sqrt(50)
                    line2DKwargs.append(line2D)

                    error2D["x"] = [tickPos[m]]
                    error2D["y"] = [columnMeans[m]]
                    error2D["yerr"] = [errorValues[m]]
                    error2D["elinewidth"] = rcParams["patch.linewidth"]
                    error2D["ecolor"] = rcParams["patch.edgecolor"]
                    line2DErrorKwargs.append(error2D)
                lineKwargs[n] = line2DKwargs
                errorKwargs[n] = line2DErrorKwargs
                #define axis limits
                minX, maxX = np.min(tickPos), np.max(tickPos)
                distance = np.sqrt(minX**2 + maxX**2) * 0.05
                axisLimits[n] = {
                        "xLimit": (minX-distance,maxX+distance),
                        "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
                        }
                axisLabels[n] = {"x":"colorCategoricalColumn","y":"value"}
        
        elif len(categoricalColumns) == 1 and len(numericColumns) > 1:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPostistion(1,maxCol=self.maxColumns)

            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(nColorCats)]

            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns,as_index=False)

            meanErrorData, minValue, maxValue, maxErrorValue = self.getCIForGroupby(groupByCatColumn,numericColumns)
            if np.isnan(maxErrorValue):
                maxErrorValue = 0.05*maxValue
            colorCategoricalColumn = categoricalColumns[0]
            
            for n in axisPositions.keys():
                tickLabels[n] = numericColumns

                tickPositions[n] = np.arange(len(numericColumns))
                line2DKwargs = []
                line2DErrorKwargs = []
                for m,category in enumerate(colorCategories):
                    
                    line2D = {}
                    error2D = {}
                    line2D["marker"] = "o"
                    line2D["color"] = "black"
                    line2D["linestyle"] = "-"
                    line2D["linewidth"] = 0.5
                    line2D["aa"] = True
                    line2D["markerfacecolor"] = colors[category]
                    line2D["markeredgecolor"] = rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = np.sqrt(self.sourceData.parent.config.getParam("scatterSize"))
                    line2DKwargs.append(line2D)

                    error2D["x"] = np.arange(len(numericColumns))#line2D["xdata"]
                    error2D["y"] = np.array([meanErrorData[category][numColumn]["value"] for numColumn in numericColumns])#groupMeans.loc[category ,:].values#line2D["ydata"]
                    error2D["yerr"] = np.array([meanErrorData[category][numColumn]["error"] for numColumn in numericColumns])
                    error2D["elinewidth"] = rcParams["patch.linewidth"]
                    error2D["ecolor"] = rcParams["patch.edgecolor"]
                    line2DErrorKwargs.append(error2D)
                lineKwargs[n] = line2DKwargs
                errorKwargs[n] = line2DErrorKwargs
                axisLimits[n] = {
                        "xLimit": (-0.5,len(numericColumns)-0.5),
                        "yLimit" : (minValue-3*maxErrorValue,maxValue+3*maxErrorValue)
                        }
                axisLabels[n] = {"x":"","y":"value"}
        
        elif len(categoricalColumns) == 2 and len(numericColumns) >= 1:
            
            nNumCols = len(numericColumns)
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPostistion(nNumCols,maxCol=self.maxColumns)

            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            tickCats = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(nColorCats)]
            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns,as_index=False)
            meanErrorData, minValue, maxValue, maxErrorValue = self.getCIForGroupby(groupByCatColumn,numericColumns)
            if np.isnan(maxErrorValue):
                maxErrorValue = 0.05*maxValue
        
            
           
            for n in axisPositions.keys():
                tickLabels[n] = tickCats
                if scaleXAxis:
                    try:
                        tickPos = [float(x.split(splitString)[splitIndex]) for x in tickCats]
                    except: 
                        tickPos = np.arange(tickCats.size)
                else:
                    tickPos = np.arange(tickCats.size)
                tickPositions[n] = tickPos 
                xValues = dict([(catName,tickPos[nCat]) for nCat,catName in enumerate(tickCats)])
                numColumn = numericColumns[n]
                
                minX, maxX = np.min(tickPos), np.max(tickPos)
                distance = np.sqrt(minX**2 + maxX**2) * 0.05
                axisLimits[n] = {
                        "xLimit": (minX-distance,maxX+distance),
                        "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
                        }
           
                line2DKwargs = []
                line2DErrorKwargs = []
                    
                for m,category in enumerate(colorCategories):
                    
                    x = []
                    y = []
                    e = []

                    for tCat in tickCats:
                        
                        groupName = (tCat,category)
                        if groupName in meanErrorData:
                            if tCat in xValues:
                                x.append(xValues[tCat])
                            else:
                                x.append(np.nan)
                            y.append(meanErrorData[(tCat,category)][numColumn]["value"])# groupMeans.loc[(tCat,category),numColumn])
                            e.append(meanErrorData[(tCat,category)][numColumn]["error"])#groupErrors.loc[(tCat,category),numColumn])
                        else:
                            continue
                    line2D = {}
                    error2D = {}

                    line2D["marker"] = "o"
                    line2D["color"] = "black"
                    line2D["linestyle"] = "-"
                    line2D["linewidth"] = 0.5
                    line2D["markerfacecolor"] = colors[category]
                    line2D["markeredgecolor"] = rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = np.sqrt(self.sourceData.parent.config.getParam("scatterSize"))
                    line2DKwargs.append(line2D)

                    error2D["x"] = x
                    error2D["y"] = y
                    error2D["yerr"] = e
                    error2D["elinewidth"] = rcParams["patch.linewidth"]
                    error2D["ecolor"] = rcParams["patch.edgecolor"]
                    line2DErrorKwargs.append(error2D)
                lineKwargs[n] = line2DKwargs
                errorKwargs[n] = line2DErrorKwargs
                axisLabels[n] = {"x":categoricalColumns[0],"y":numColumn}
            colorCategoricalColumn = categoricalColumns[1]



        elif len(categoricalColumns) == 3 and len(numericColumns) >= 1:
            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
            #catgeory dividend into different axes
            axisCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[2])
            #first category splis data on x axis
            xAxisCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            #second category is color coded
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            #number of numeric columns
            NNumCol = len(numericColumns)
            #create axis
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPostistion(n = axisCategories.size *  NNumCol, maxCol = axisCategories.size)
            nColorCats = colorCategories.size
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            #get tick categories
            tickCats = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in colors.values()]
            #some calculataions for axis limits
            globalMin, globalMax = np.nanquantile(rawData[numericColumns].values, q = [0,1])
            yMargin = np.sqrt(globalMax**2 + globalMin**2)*0.05
            #create groupby 
            colorGroupby = rawData.groupby(categoricalColumns[0],sort=False)
            xAxisGroupby = rawData.groupby(categoricalColumns[1],sort=False)
            axisGroupby = rawData.groupby(categoricalColumns[2],sort=False)
            
            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns,as_index=False)
            meanErrorData, minValue, maxValue, maxErrorValue = self.getCIForGroupby(groupByCatColumn,numericColumns)
            print(meanErrorData)
            if np.isnan(maxErrorValue):
                maxErrorValue = 0.05*maxValue
        
            xValues = dict([(catName,x) for x,catName in enumerate(tickCats)])
            nAxis = -1
            for numColumn in numericColumns:

                for axisCat, axisCatData in axisGroupby:
                    line2DKwargs = []
                    line2DErrorKwargs = []
                    nAxis += 1
                    tickLabels[nAxis] = tickCats
                    if scaleXAxis:
                        try:
                            tickPos = [float(x.split(splitString)[splitIndex]) for x in tickCats]
                        except: 
                            tickPos = np.arange(tickCats.size)
                    else:
                        tickPos = np.arange(tickCats.size)
                        
                    tickPositions[n] = tickPos
                    minX, maxX = np.min(tickPos), np.max(tickPos)
                    distance = np.sqrt(minX**2 + maxX**2) * 0.05
                    
                    axisLimits[nAxis] = {
                        "xLimit": (minX-distance,maxX+distance),
                        "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
                        }
           
                    for m,colorCategory in enumerate(colorCategories):

                            x = []
                            y = []
                            e = []
                            for xAxisCat in tickCats:
                                group = (colorCategory,xAxisCat,axisCat)
                                if group in meanErrorData:
                                    if xAxisCat in xValues:
                                        x.append(xValues[xAxisCat])
                                    else:
                                        x.append(np.nan)
                                    y.append(meanErrorData[group][numColumn]["value"])
                                    e.append(meanErrorData[group][numColumn]["error"])
                                else:
                                    continue
                            line2D = {}
                            error2D = {}

                            line2D["marker"] = "o"
                            line2D["color"] = "black"
                            line2D["linestyle"] = "-"
                            line2D["linewidth"] = 0.5
                            line2D["markerfacecolor"] = colors[colorCategory]
                            line2D["markeredgecolor"] = rcParams["patch.edgecolor"]
                            line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                            line2D["markersize"] = np.sqrt(self.sourceData.parent.config.getParam("scatterSize"))
                            line2DKwargs.append(line2D)

                            error2D["x"] = x
                            error2D["y"] = y
                            error2D["yerr"] = e
                            error2D["elinewidth"] = rcParams["patch.linewidth"]
                            error2D["ecolor"] = rcParams["patch.edgecolor"]
                            line2DErrorKwargs.append(error2D)
                    lineKwargs[nAxis] = line2DKwargs
                    errorKwargs[nAxis] = line2DErrorKwargs
                    axisLabels[nAxis] = {"x":categoricalColumns[1],"y":numColumn}
            colorCategoricalColumn = categoricalColumns[0]
# for n in axisPositions.keys():
#                 tickLabels[n] = tickCats
#                 tickPositions[n] = np.arange(tickCats.size)
#                 numColumn = numericColumns[n]
#                 axisLimits[n] = {
#                         "xLimit": (-0.5,tickCats.size-0.5),
#                         "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
#                         }
           
#                 line2DKwargs = []
#                 line2DErrorKwargs = []




        return {"data":{
                "plotData":lineKwargs,#"
                "errorData":errorKwargs,
                "axisPositions":axisPositions,
                "tickLabels": tickLabels,
                "tickPositions": tickPositions,
                "axisLabels" : axisLabels,
                "axisLimits" : axisLimits,
                "dataColorGroups": colorGroups,
                "subplotBorders":subplotBorders,
                "colorCategoricalColumn" : colorCategoricalColumn,
                #"tooltipsTexts" : texts,
                "dataID":dataID}}


    def getCIForGroupby(self,groupby, numericColumns):
        ""
        meanErrorData = OrderedDict() 
        es = []
        maxValue = np.nanmax(groupby.max()[numericColumns].values)
        minValue = np.nanmin(groupby.min()[numericColumns].values)

        for groupName, data in groupby:
            meanErrorData[groupName] = {}
            for numColumn in numericColumns:
                X = data[numColumn].dropna()
                if not X.empty:
                    errorValue = CI(X.values)
                    es.append(errorValue)
                    meanErrorData[groupName][numColumn] = {"value":np.mean(X.values),
                                                           "error":errorValue}
                else:
                    meanErrorData[groupName][numColumn] = {"value":np.nan,
                                                           "error":np.nan}
        maxError = np.nanmax(errorValue)
        return meanErrorData, minValue, maxValue, maxError




    def _getBoxplotTooltips(self,data,numericColumns):
        ""
        tooltipsStr = {}
        desc = data[numericColumns].describe()
        for columnName in numericColumns:
            baseStr = ""
            for idx in desc.index:
                if baseStr != "":
                    baseStr += "\n"
                baseStr += "{} : {}".format(idx,getReadableNumber(desc.loc[idx,columnName]))
            tooltipsStr[columnName] = baseStr
        return tooltipsStr
            

    def getBoxplotProps2(self, dataID, numericColumns, categoricalColumns):
        ""
        boxplotProps = []
        xAxisLabels = []
        outliers = np.array([]).reshape(-1,2)

        if categoricalColumns is None or len(categoricalColumns) == 0:
            colorMap  = self.getColorMapper(numericColumns)
            data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]
            quantiles, IQR, upperBound, lowerBound = self._getBoxProps(data.values)

            x0, width = self.findPositions(numericColumns,categoricalColumns,self.plotType)

            for n,numericColumn in enumerate(numericColumns):

                outlierBool = (data[numericColumn] > upperBound[n]) | (data[numericColumn] < lowerBound[n])
                limits = [data.loc[outlierBool == False,numericColumn].min(), data.loc[outlierBool == False,numericColumn].max()]
                xCenter = x0[n] + width/2
                boxplotProp = {"IQR":IQR[n],"median":quantiles[1,n],"limits":limits,"y0": quantiles[0,n],"x0":x0[n],"width":width,"color":colorMap[numericColumn]}
                boxplotProps.append(boxplotProp)

                nOutliers = np.sum(outlierBool)
                outlierPositions = np.full(shape=(nOutliers,2),fill_value=xCenter)
                outlierPositions[:,1] = data.loc[outlierBool,numericColumn].values

                outliers = np.append(outliers,outlierPositions,axis=0)
                xAxisLabels.append((xCenter,numericColumn))






            # minValue, maxValue = boxplotData["limits"]
            # IQR = boxplotData["IQR"]
            # width = boxplotData["width"]
            # median = boxplotData["median"]
            # x0 = boxplotData["x0"]
            # y0 = boxplotData["y0"]
            # color = boxplotData["color"]
            # xCenter = x0+width/2


        return {"newPlot":True,"data":{"plotData":boxplotProps,"outliers":outliers,"xAxisLabels":[xAxisLabels],"plotType":"boxplot"}}


    def _getBoxProps(self,arr):
        ""
        quantiles = np.nanquantile(arr,q=[0.25, 0.5, 0.75], axis=0)
        IQR = quantiles[2,:] - quantiles[0,:]
        upperBound = quantiles[2,:] + 1.5 * IQR
        lowerBound = quantiles[0,:] - 1.5 * IQR

        return quantiles, IQR, upperBound, lowerBound

    def getCorrmatrixProps(self,dataID,numericColumns, categoricalColumns):
        ""
        return self.getHeatmapProps(dataID,numericColumns,categoricalColumns,True)

    def getHeatmapProps(self,dataID, numericColumns, categoricalColumns, corrMatrix = False):
        ""
        rowMaxD = None
        colMaxD = None
        rowClustNumber = None
        rowLinkage = None
        ytickLabels = []
        ytickPosition = []
        rowLineCollection = []
        colLineCollection = []
        rectangles = []
        clusterColorMap = {}
        
        #cluster rows 
        rowMetric = self.sourceData.statCenter.rowMetric
        rowMethod = self.sourceData.statCenter.rowMethod
         #cluster columns
        columnMetric = self.sourceData.statCenter.columnMetric
        columnMethod = self.sourceData.statCenter.columnMethod

        try:
            if corrMatrix:
                data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"].corr(method = self.corrMatrixMethod)
            else:
                if rowMetric == "nanEuclidean":
                    nanThreshold = self.sourceData.parent.config.getParam("min.required.valid.values")
                    if nanThreshold > len(numericColumns):
                        nanThreshold = len(numericColumns)
                    data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"].dropna(thresh=nanThreshold)
                    rowMetric = "nanEuclidean" 
                    if columnMetric != "None":
                        columnMetric = "nanEuclidean"
                else:
                    data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"].dropna()
                
            #remove no deviation data (Same value)
            data = data.loc[data.std(axis=1) != 0,:]
        

            #nRows, nCols = data.shape
            rawIndex = data.index
            rowXLimit, rowYLimit, rowLineCollection = None, None, None
            colXLimit, colYLimit, colLineCollection = None, None, None


            axisDict = self.getClusterAxes(numericColumns, corrMatrix=corrMatrix)
           # print(axisDict)
            

            if data.shape[0] > 1 and rowMetric != "None" and rowMethod != "None":
                rowLinkage, rowMaxD = self.sourceData.statCenter.clusterData(data,rowMetric,rowMethod)

                Z_row = sch.dendrogram(rowLinkage, orientation='left', color_threshold= rowMaxD, 
                                    leaf_rotation=90, ax = None, no_plot=True)
                
                rowXLimit, rowYLimit, rowLineCollection = self.addDendrogram(Z_row,True)

                rowClustNumber = self.getClusterNumber(rowLinkage,rowMaxD)
                ytickPosition, ytickLabels, rectangles, clusterColorMap = self.getClusterRectangles(data,rowMaxD,rowClustNumber,Z_row)
                data = data.iloc[Z_row['leaves']]
                
            else:
                #print("deleting")
                del axisDict["axRowDendro"]
                Z_row = None

            if data.shape[1] > 1 and columnMetric != "None" and columnMethod != "None":
                
                columnLinkage, colMaxD = self.sourceData.statCenter.clusterData(np.transpose(data.values),columnMetric,columnMethod)

                Z_col = sch.dendrogram(columnLinkage, orientation='top', color_threshold = colMaxD, 
                                    leaf_rotation=90, ax = None, no_plot=True)
                
                data = data.iloc[:,Z_col['leaves']]
                numericColumns = [numericColumns[idx] for idx in Z_col['leaves']]

                colXLimit, colYLimit, colLineCollection = self.addDendrogram(Z_col,False)

            else:
                del axisDict["axColumnDendro"]
           
        except Exception as e:
            print(e)
            return {}
       

       

        return {"newPlot":True,
            "data":{"plotData":data,
                "rowMaxD" : rowMaxD,
                "dataID" : dataID,
                "rawIndex": rawIndex,
                "rowClustNumber" : rowClustNumber,
                "clusterColorMap" : clusterColorMap,
                "rowLinkage" : rowLinkage,
                "Z_row": Z_row,
                "dendrograms":{
                    "row":rowLineCollection,
                    "col":colLineCollection
                },
                "axisLimits":{
                    "rowDendrogram":{"x":rowXLimit,"y":rowYLimit},
                    "columnDendrogram":{"x":colXLimit,"y":colYLimit},
                    },
                "tickLabels" : {"rowDendrogram":{"tickLabels": ytickLabels,"tickPosition": ytickPosition}},
                "absoluteAxisPositions" : axisDict,
                "clusterRectangles": rectangles,
                "dataID":dataID,
                "columnNames":numericColumns}
                }
    
    def addDendrogram(self,dendrogram,rotate):
        '''
        Idea is from the seaborn package.
        '''
        dependent_coord = dendrogram['dcoord']
        independent_coord = dendrogram['icoord']
        max_dependent_coord = max(map(max, dependent_coord))

        if rotate:
            lines = LineCollection([list(zip(x, y))
                                            for x, y in zip(dependent_coord,
                                            independent_coord)],
                                            **line_kwargs)
            yLimit = len(dendrogram['leaves']) * 10
            xLimit = max_dependent_coord * 1.05
            #ax.set_ylim(0, self.yLimitRow)
            #ax.set_xlim(0, self.xLimitRow)
            #ax.invert_xaxis()

        else:
            lines = LineCollection([list(zip(x, y))
                                            for x, y in zip(independent_coord,
                                            dependent_coord)],
                                            **line_kwargs)	
            xLimit =  len(dendrogram['leaves']) * 10
            yLimit =  max_dependent_coord * 1.05  
            #ax.set_xlim(0, self.xLimitCol)
            #ax.set_ylim(0, self.yLimitCol)

        return (0,xLimit), (0,yLimit), lines
    

    def getClusterRectangles(self,data,rowMaxD,rowClustNumber,Z_row):
        ""
        annotationDf = pd.DataFrame(rowClustNumber,columns=['labels'],index = data.index)
        sortedDataIndex =  data.index[Z_row['leaves']]
        uniqueCluster = annotationDf.loc[sortedDataIndex]['labels'].unique()
        valueCounts = annotationDf.loc[sortedDataIndex]['labels'].value_counts(sort=False)
        countsClust = [valueCounts.loc[clustLabel] for clustLabel in uniqueCluster]
        ytickPosition = [(sum(countsClust[0:n+1])-valueCounts.loc[x]/2)*10 for n,x in enumerate(uniqueCluster)]
        ytickLabels = ['C ({})'.format(cluster) for cluster in uniqueCluster]
        clusterColors = self.sourceData.colorManager.getNColorsByCurrentColorMap(uniqueCluster.size,"hclustClusterColorMap")
        clusterColorMap = dict([(k,v) for k,v in zip(ytickLabels,clusterColors)])
        #clusterColors = sns.color_palette(self.clusterColorMap,uniqueCluster.size)
        rectangles = [Rectangle(
                        xy = (0,n if n == 0 else sum(countsClust[:n] * 10)),
                        width = rowMaxD, 
                        height = yLimit * 10,
                        alpha = 0.75,
                        facecolor = clusterColors[n]) for n, yLimit in enumerate(countsClust)]

        return ytickPosition, ytickLabels, rectangles, clusterColorMap

    def getClusterNumber(self,linkage,maxD):
        '''
        Returns cluster numbers
        '''
        return sch.fcluster(linkage,maxD,'distance')	

    def getClusterAxes(self, numericColumns, corrMatrix=False):
        ""
        x0,y0 = 0.15,0.15
        x1,y1 = 0.95,0.95
        width = x1-x0
        height = y1-y0
        multWidth = 0.4
        correctHeight = 1
        # emperically determined to give almost equal width independent of number of columns 
			
        addFactorMainWidth =  -0.15+len(numericColumns) * 0.008 
		
        clusterMapWidth = width*multWidth+addFactorMainWidth
        rowDendroWidth = width * 0.13
        if clusterMapWidth > 0.75:
            clusterMapWidth = 0.75
				
        if corrMatrix:
            ## to produce a corr matrix in the topleft corner of the graph
            heightMain = height * 0.5
            y0 = 0.4
            height = y1-y0
            
        else:
            heightMain = height * 0.8 
            
        axisDict = dict() 

        axisDict["axRowDendro"] = [x0,
                                    y0,
                                    rowDendroWidth,
                                    heightMain]
                                    
        axisDict["axColumnDendro"] = [x0 + rowDendroWidth, 
                                    y0+heightMain,
                                    clusterMapWidth,
                                    (width* 0.13)*correctHeight]

        axisDict["axClusterMap"] = [x0+width*0.13,
                                    y0,
                                    clusterMapWidth,
                                    heightMain]
        
        axisDict["axLabelColor"] =  [x0+rowDendroWidth+clusterMapWidth+width*0.02, #add margin
                                    y0,
                                    width-clusterMapWidth-rowDendroWidth,
                                    heightMain]		
        
        axisDict["axColormap"] =    [x0,
                                    y0+height*0.84,
                                    width*0.025,
                                    height*0.12]	
        
        return axisDict

    def getDimRedProps(self,dataID,numericColumns,categoricalColumns):
        ""
        return {"data":{}}

    def getPCAProps(self,dataID,numericColumns,categoricalColumns):
        ""
        subplotBorders = dict(wspace=0.30, hspace = 0.30,bottom=0.15,right=0.95,top=0.95)
        #data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]
        checkPassed, driverResult, eigVectors  = self.sourceData.statCenter.runPCA(dataID,numericColumns, initGraph = True, n_components = 3)
        if not checkPassed:
            return getMessageProps("Error ..","Filtering resulted in an invalid data frame.")

        columnPairs = [("Component_01","Component_02"), ("Component_02","Component_03")]
        #print(result)
        print(eigVectors)
        axisPostions = dict([(n,[2,2,n+1]) for n in range(4)])
    

        return {"data":{"plotData":{"projection":driverResult,"eigV":eigVectors},
                "axisPositions":axisPostions,
                "numericColumns":numericColumns,
                "subplotBorders":subplotBorders,
                "columnPairs":columnPairs,
                "dataID":dataID}}

    def getScatterProps(self,dataID, numericColumns, categoricalColumns):
        ""
        
        try:
            axisTitles = {} 

            if len(categoricalColumns) == 0:
                data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]

                if not self.plotAgainstIndex and len(numericColumns) > 1:
                    numericColumnPairs = list(zip(numericColumns[0::2], numericColumns[1::2]))

                else:
                    numericColumnPairs = [("Index ({:02d})".format(n+1),numColumn) for n,numColumn in enumerate(numericColumns)]
                    for indexName, numColumn in numericColumnPairs:
                    
                        
                        index = data.sort_values(by=numColumn, ascending = self.indexSort == "ascending").index
                        data = data.join(pd.Series(np.arange(index.size),index=index, name = indexName))
                        
                nrows,ncols,subplotBorders = self._findScatterSubplotProps(numericColumnPairs)
                axisPositions = dict([(n,[nrows,ncols,n+1]) for n in range(len(numericColumnPairs))])
                axisLabels = dict([(n,{"x":x1,"y":x2}) for n, (x1,x2) in enumerate(numericColumnPairs)])
                 # {0:{"x":xLabel,"y":"value"}}

            else:
                
                subplotBorders = dict(wspace=0, hspace = 0.2, bottom=0.15,right=0.95,top=0.95)
                data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
                uniqueValuesCat1 = data[categoricalColumns[0]].unique() 
                numUniqueCat = uniqueValuesCat1.size 
                
               
                axisLabels = {}
                firstAxisRow = True
                axisID = 0
                plotNumericPairs = []

                if not self.plotAgainstIndex and len(numericColumns) > 1:
                    numericColumnPairs = list(zip(numericColumns[0::2], numericColumns[1::2]))

                if len(categoricalColumns) == 1:
                    numOfAxis = len(numericColumnPairs) * numUniqueCat
                    axisPositions = getAxisPostistion(numOfAxis,maxCol=numUniqueCat)
                    requiredColumns = []
                    for numCols in numericColumnPairs:
                        for uniqueValue in uniqueValuesCat1:
                            requiredColumns.append("{}:{}:({})".format(uniqueValue,categoricalColumns[0],numCols[0]))
                            requiredColumns.append("{}:{}:({})".format(uniqueValue,categoricalColumns[0],numCols[1]))

                    plotData = pd.DataFrame(
                                index=data.index, 
                                columns = requiredColumns
                                )
                    
                    
                    for numCols in numericColumnPairs:
                        for groupName, groupData in data.groupby(categoricalColumns[0]):

                            if axisID == numUniqueCat:
                                firstAxisRow = False
                            elif firstAxisRow:
                                axisTitles[axisID] = {"title":"{}\n{}".format(categoricalColumns[0],groupName),
                                                                "appendWhere":"top",
                                                                "textRotation" : 0}
                            pair = []
                            for columnName in numCols:
                                columnKey = "{}:{}:({})".format(groupName,categoricalColumns[0],columnName)
                                plotData.loc[groupData.index,columnKey] = groupData[columnName]
                                pair.append(columnKey)
                            plotNumericPairs.append(tuple(pair))

                            axisLabels[axisID] = {"x":numCols[0],"y":numCols[1]}  
                            axisID += 1  
                    
                   # axisPositions = getAxisPostistion(len(numericColumnPairs),nCols=numUniqueCat)
                elif len(categoricalColumns) == 2:
                    uniqueValuesCat2 = data[categoricalColumns[1]].unique() 
                    numUniqueCat2 = uniqueValuesCat2.size

                    numOfAxis = numUniqueCat * numUniqueCat2 * len(numericColumnPairs)
                    axisPositions = getAxisPostistion(numOfAxis,maxCol=numUniqueCat)
                    requiredColumns = []
                    for numCols in numericColumnPairs:
                        for uniqueValueCat2 in uniqueValuesCat2:
                            for uniqueValueCat1 in uniqueValuesCat1:

                                requiredColumns.append("{}:{}:{}:({})".format(uniqueValueCat1,
                                                                            uniqueValueCat2,
                                                                            categoricalColumns[0],
                                                                            numCols[0]))
                                requiredColumns.append("{}:{}:{}:({})".format(uniqueValueCat1,
                                                                            uniqueValueCat2,
                                                                            categoricalColumns[0],
                                                                            numCols[1]))
                    
                    plotData = pd.DataFrame(
                                index=data.index, 
                                columns = requiredColumns
                                )
                    
                    
                    for numCols in numericColumnPairs:
                        for uniqueValueCat2, cat2data in data.groupby(categoricalColumns[1],sort=False):
                            
                            for groupName, groupData in cat2data.groupby(categoricalColumns[0], sort=False):
                                    if axisID == numUniqueCat:
                                        firstAxisRow = False
                                    elif firstAxisRow:
                                        axisTitles[axisID] = {"title":"{}:{}".format(categoricalColumns[0],groupName),
                                                                "appendWhere":"top",
                                                                "textRotation" : 0}
                                    if groupName == uniqueValueCat1[-1]:
                                        titleProps = {
                                                        "title":"{}:{}".format(categoricalColumns[1],uniqueValueCat2),
                                                        "appendWhere":"right",
                                                        "textRotation" : 90
                                                    }
                                                    
                                        if axisID in axisTitles:
                                            addTitles = [axisTitles[axisID],titleProps]
                                            axisTitles[axisID] = addTitles
                                        else:
                                            axisTitles[axisID] = titleProps
                                    
                                    pair = []
                                    for columnName in numCols:
                                        columnKey = "{}:{}:{}:({})".format(groupName,uniqueValueCat2,categoricalColumns[0],columnName)
                                        plotData.loc[groupData.index,columnKey] = groupData[columnName]
                                        pair.append(columnKey)
                                    plotNumericPairs.append(tuple(pair))

                                    axisLabels[axisID] = {"x":numCols[0],"y":numCols[1]}  
                                    axisID += 1  

                print(plotData)
                
                numericColumnPairs = plotNumericPairs
                print(numericColumnPairs)

                print(axisTitles)
                data = plotData
                



        except Exception as e:
                print(e)

        
        colorGroupsData = pd.DataFrame() 
        colorGroupsData["color"] = [self.sourceData.colorManager.nanColor]
        colorGroupsData["group"] = [""]

        sizeGroupsData = pd.DataFrame() 
        sizeGroupsData["size"] = [self.scatterSize]
        sizeGroupsData["group"] = [""]

        return {"data":{
            "plotData":data,
            "axisPositions":axisPositions, 
            "axisTitles": axisTitles,
            "columnPairs":numericColumnPairs,
            "dataColorGroups": colorGroupsData,
            "dataSizeGroups" : sizeGroupsData,
            "axisLabels" : axisLabels,
            "subplotBorders":subplotBorders,
            "dataID":dataID}}
        
        #{"plotData":data.values,"xLabel":numericColumns[0],"yLabel":numericColumns[1],"plotType":"scatter"}

    def getMarkerGroupsForScatter(self,dataID,markerColumn,markerColumnType):
        ""
        categoryIndexMatch = None
        markerGroupData = pd.DataFrame(columns=["marker","group","internalID"])
        markerColumnName = markerColumn.values[0]
        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,markerColumn)["fnKwargs"]["data"]
        if markerColumnType == "Integers":
            markerColumnType, rawData = self._checkIntegerColumn(markerColumn,rawData)

        if markerColumnType == "Categories":
            if markerColumn.index.size > 1:
                rawMarkerData = rawData.apply(tuple,axis=1)
            else:
                rawMarkerData = rawData[markerColumnName]
            markerCategories = rawMarkerData.unique()
            
            availableMarkers = ["o","^","D","s","P","v","p","+","."]

            markerMap = OrderedDict([(cat,availableMarkers[n % len(availableMarkers)]) for n,cat in enumerate(markerCategories)])
  
            markerData = rawMarkerData.map(markerMap)
            markerGroupData["marker"] = markerMap.values()
            markerGroupData["group"] = markerCategories
            markerGroupData["internalID"] = [getRandomString() for n in markerGroupData.index]
        
        propsData = pd.DataFrame(markerData.values,columns=["marker"], index=rawData.index)
        title = mergeListToString(markerColumn.values,"\n")      

        categoryIndexMatch = dict([(intID,rawData[rawData.values == category].index) for category, intID in zip(markerGroupData["group"].values,
                                                                                                                markerGroupData["internalID"].values)])

        return {"propsData":propsData, 
                "title":title, 
                "markerGroupData":markerGroupData,
                "categoryEncoded":"marker",
                "categoryIndexMatch":categoryIndexMatch}

    def _checkIntegerColumn(self,columnName,rawData):
        ""
        uniqueValues = np.unique(rawData[columnName].values).size
        if uniqueValues > rawData.index.size * 0.75 / columnName.size:
            columnType = "Numeric Floats"
            rawData = rawData.astype(float)
        else:
            columnType= "Categories"
            rawData = rawData.astype(str)

        return columnType, rawData

    def getSizeGroupsForScatter(self,dataID, sizeColumn = None, sizeColumnType = None):
        ""
        #at the moment similiar to columnColor - merge
        if sizeColumn is None:
            sizeColumn = self.sizeColumn
        if sizeColumnType is None:
            sizeColumnType = self.sizeColumnType
        sizeColumnName = sizeColumn.values[0]
        sizeGroupData = pd.DataFrame(columns=["size","group"])

        categoryIndexMatch = None
        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,sizeColumn)["fnKwargs"]["data"]

        if sizeColumnType == "Integers":
            sizeColumnType, rawData = self._checkIntegerColumn(sizeColumn,rawData)

        if sizeColumnType == "Categories":
            if sizeColumn.index.size > 1:
                rawSizeData = rawData.apply(tuple,axis=1)
            else:
                rawSizeData = rawData[sizeColumnName]

            sizeCategories = rawSizeData.unique()
            nCategories = sizeCategories.size
            scaleSizes = np.linspace(0.2,1,num=nCategories,endpoint=True)
            sizeMap = dict(zip(sizeCategories,scaleSizes))
            sizeMap = replaceKeyInDict(self.sourceData.replaceObjectNan,sizeMap,0.1)

            scaledSizedata = rawSizeData.map(sizeMap)
            
            sizeData = scaledSizedata.values * (self.maxScatterSize-self.minScatterSize) + self.minScatterSize

            sizeGroupData["size"] = scaleSizes * (self.maxScatterSize-self.minScatterSize) + self.minScatterSize
            sizeGroupData["group"] = sizeCategories
            sizeGroupData["internalID"] = [getRandomString() for n in sizeGroupData.index]


            categoryIndexMatch = dict([(intID,rawData[rawData.values == category].index) for category, intID in zip(sizeGroupData["group"].values,
                                                                                                                    sizeGroupData["internalID"].values)])
          #  print(sizeGroupData)  
           # print(propsData)  
        elif sizeColumnType == "Numeric Floats":
            transformFuncs = {"log2":np.log2,"ln":np.log,"log10":np.log10}
            if sizeColumn.size > 1:
                if self.aggMethod == "mean":
                    rawData = rawData.mean(axis=1)
                elif self.aggMethod == "sum":
                    rawData = rawData.sum(axis=1)
            fakeIndex = np.arange(rawData.index.size)
            nanIndex = fakeIndex[np.isnan(rawData.values).flatten()]
            sizeTransform = self.sourceData.parent.config.getParam("transform.numeric.size.columns")
            if sizeTransform in transformFuncs:
                rawValuesZeroNan = rawData.replace(0,np.nan)
                rawValues = transformFuncs[sizeTransform](rawValuesZeroNan.values)

            else:
                rawValues = rawData.values
            #get quanitles for normalization
            minV, q25, median, q75, maxV = np.nanquantile(rawValues, q = [0,0.25,0.5,0.75,1])
            #scale size
            sizeData = (rawValues.flatten() - minV) / (maxV - minV) * (self.maxScatterSize-self.minScatterSize) + self.minScatterSize
            #set nan Size
            sizeData[nanIndex] = 0.1 * self.minScatterSize

            groupNames = ["Max ({})".format(getReadableNumber(maxV)),
                            "75% Quantile ({})".format(getReadableNumber(q75)),
                            "Median ({})".format(getReadableNumber(median)),
                            "25% Quantile ({})".format(getReadableNumber(q25)),
                            "Min ({})".format(getReadableNumber(minV)),
                            "NaN"]

            sizeGroupData["group"] = groupNames
            sizeGroupData["size"] = [(x-minV)/(maxV-minV) * (self.maxScatterSize-self.minScatterSize) + self.minScatterSize \
                                                                            for x in  [maxV,q75,median,q25,minV,0]] #0.1 == Nan Size

            sizeGroupData["internalID"] = [getRandomString() for n in sizeGroupData.index]
            sizeGroupData.loc[sizeGroupData["group"] == "NaN","size"] = 0.1 * self.minScatterSize
           

        propsData = pd.DataFrame(sizeData,columns=["size"], index=rawData.index)
        title = mergeListToString(sizeColumn.values,"\n")                                                                   

        
        return {"sizeGroupData":sizeGroupData,"propsData":propsData,"title":title,"categoryIndexMatch":categoryIndexMatch,"categoryEncoded":"size"}


    def getLinearRegression(self,dataID,numericColumnPairs):
        ""
        lineData = {}
        for n,numColumns in enumerate(numericColumnPairs):
            
            xList,yList,slope,intercept,rValue,pValue, stdErrorSlope  = self.sourceData.statCenter.runLinearRegression(dataID,list(numColumns))
            lineKwargs = {"xdata":xList,"ydata":yList}
            lineData[n] = lineKwargs
            
        funcProps = getMessageProps("Done..","Linear regression line added.")
        funcProps["lineData"] = lineData
        return funcProps

    def getLowessLine(self,dataID,numericColumnPairs):
        ""
        lineData = {}
        for n,numColumns in enumerate(numericColumnPairs):

            lowessFit = self.sourceData.statCenter.runLowess(dataID,list(numColumns))

            lineKwargs = {"xdata":lowessFit[:,0],"ydata":lowessFit[:,1]}
            lineData[n] = lineKwargs
        funcProps = getMessageProps("Done..","Lowess line added.")
        funcProps["lineData"] = lineData
        return funcProps

    def getColorGroupsDataForScatter(self,dataID, colorColumn = None, colorColumnType = None, colorGroupData = None):
        ""
     
            
        if colorColumn is None:
            if self.colorColumn is None:
                return {}
            colorColumn = self.colorColumn
        if colorColumnType is None:
            colorColumnType = self.colorColumnType
        colorColumnName = colorColumn.values[0]
        
        
        categoryIndexMatch = None

        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,colorColumn)["fnKwargs"]["data"]
        if colorColumnType == "Integers":
            uniqueValues = np.unique(rawData[colorColumn].values).size
            if uniqueValues > rawData.index.size * 0.75 / colorColumn.size:
                colorColumnType = "Numeric Floats"
                rawData = rawData.astype(float)
            else:
                colorColumnType = "Categories"
                rawData = rawData.astype(str)

        if colorColumnType == "Categories":

            if colorColumn.index.size > 1:
                rawColorData = rawData.apply(tuple,axis=1)
            else:
                rawColorData = rawData[colorColumnName]
            
            if colorGroupData is None:
                colorCategories = rawColorData.unique()
            else:
                colorCategories = colorGroupData["group"].values

            colors, layerMap = self.sourceData.colorManager.createColorMapDict(colorCategories, 
                                                                        addNaNLevels=[(self.sourceData.replaceObjectNan,)*colorColumn.index.size],
                                                                        as_hex=True)
            colorData = pd.DataFrame(rawColorData.map(colors).values,
                                        columns=["color"],
                                        index=rawData.index)
            #add color layer props
            colorData["layer"] = colorData["color"].map(layerMap)
            
            if colorGroupData is None:
                colorGroupData = pd.DataFrame(columns=["color","group"])
                colorGroupData["group"] = colorCategories
                colorGroupData["color"] = [colors[k] for k in colorCategories]
                #map color data and save as df
                
                
                colorGroupData["internalID"] = [getRandomString() for n in colorGroupData.index]
               # categoryIndexMatch = dict([(intID,rawData[rawColorData.values == category].index) for category, intID in zip(colorGroupData["group"].values,
                #                                                                                                            colorGroupData["internalID"].values)])
            else:
                
                colorGroupData["color"] = [colors[k] for k in colorCategories]
            
            categoryIndexMatch = dict([(intID,rawData.index[rawColorData == category]) for category, intID in zip(colorGroupData["group"].values,
                                                                                                                colorGroupData["internalID"].values)])


        elif colorColumnType == "Numeric Floats":

            twoColorMap = self.sourceData.colorManager.colorMap
            if colorColumn.size > 1:
                if self.aggMethod == "mean":
                    rawData = rawData.mean(axis=1)
                elif self.aggMethod == "sum":
                    rawData = rawData.sum(axis=1)

            nanIndex = rawData.index[np.isnan(rawData.values).flatten()]
            
            minV, q25, median, q75, maxV = np.nanquantile(rawData.values,q = [0,0.25,0.5,0.75,1])
            scaledColorValues = (rawData.values.flatten() - minV) / (maxV - minV)
            
            cmap = self.sourceData.colorManager.get_max_colors_from_pallete(colorMap = twoColorMap)
            
            colorData =  pd.DataFrame([to_hex(c) for c in cmap(scaledColorValues)],
                                    columns=["color"],
                                    index=rawData.index)
            colorData.loc[nanIndex,"color"] = self.sourceData.colorManager.nanColor
            
            #save colors for legend
            scaledColorVs = [to_hex(cmap( (x - minV) / (maxV - minV))) for x in [maxV,q75,median,q25,minV]]
            
            colorLimitValues = scaledColorVs + [self.sourceData.colorManager.nanColor]

            if colorGroupData is None:
                colorGroupData = pd.DataFrame(columns=["color","group"])

                groupNames = ["Max ({})".format(getReadableNumber(maxV)),
                            "75% Quantile ({})".format(getReadableNumber(q75)),
                            "Median ({})".format(getReadableNumber(median)),
                            "25% Quantile ({})".format(getReadableNumber(q25)),
                            "Min ({})".format(getReadableNumber(minV)),
                            "NaN"]
                colorGroupData["color"] = colorLimitValues
                colorGroupData["group"] = groupNames
            else:
                colorGroupData["color"] = colorLimitValues
        
        tableTitle = mergeListToString(colorColumn.values,"\n") 
        #save data to enable fast update 
        self.colorColumn = colorColumn
        self.colorColumnType = colorColumnType
       # print({"colorGroupData":colorGroupData,"propsData":colorData,"title":tableTitle,"categoryIndexMatch":categoryIndexMatch,"categoryEncoded":"color"})
        
        return {"colorGroupData":colorGroupData,"propsData":colorData,"title":tableTitle,"categoryIndexMatch":categoryIndexMatch,"categoryEncoded":"color","isEditable":colorColumnType == "Categories"}



    def getSwarmplotProps(self,dataID,numericColumns,categoricalColumns):
        ""
        #subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        multiScatterKwargs = {}
        colorCategoryIndexMatch = {}
        interalIDColumnPairs = {}
        axisTitles = {}
        axisLimits = {}
        tickPositions = {}
        numericColumnPairs = {}
        axisLabels = {}
        tickLabels = {}

        colorGroupsData = pd.DataFrame() 
        sizeGroupsData = pd.DataFrame()
        numCatColumns = len(categoricalColumns)
        numNumColumns = len(numericColumns)
        if numCatColumns == 0:
            #get raw data
            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]
            numericColumnPairs = []
            widthBox = 0.75
            tickPositions = {0:np.arange(numNumColumns) + widthBox}
            positions = np.arange(numNumColumns)
            colorDict,_ = self.sourceData.colorManager.createColorMapDict(numericColumns, as_hex=True)
            plotData = pd.DataFrame(rawData, index = rawData.index) 
            columnNames = []
            multiScatterKwargs[0] = dict() #0 = axis id
            interalIDColumnPairs[0] = dict() #0 = axis id
            colorGroupsData["color"] = list(colorDict.values())
            colorGroupsData["group"] = numericColumns
            colorGroupsData["internalID"] = [getRandomString() for _ in range(numNumColumns)]

            sizeGroupsData["size"] = [self.sourceData.parent.config.getParam("scatterSize")] * numNumColumns
            sizeGroupsData["group"] = numericColumns
            sizeGroupsData["internalID"] = colorGroupsData["internalID"].values

            for n,numColumn in enumerate(numericColumns):
                xName = "x({})".format(numColumn)
                groupData = rawData[[numColumn]].dropna()
                if groupData.index.size == 1:
                    kdeData = np.array([0]) +  positions[n]
                    data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                else:
                    #get kernel data
                    kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                    #get random x position around 0 to spread data
                    kdeData = np.array([np.random.uniform(-x*0.5,x*0.5) for x in kdeData])
                    kdeData = kdeData + tickPositions[0][n]
                    #save data
                    data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])
                plotData = plotData.join(data)
                columnNames.extend([xName,numColumn])
                multiScatterKwargs[0][(xName,numColumn)] = {"color": colorDict[numColumn]}
                internalID = colorGroupsData.loc[colorGroupsData["group"] == numColumn]["internalID"].values[0]
                colorCategoryIndexMatch[internalID] = kdeIndex
                if internalID not in interalIDColumnPairs[0]:
                    interalIDColumnPairs[0][internalID] = [(xName,numColumn)]
                else:
                    interalIDColumnPairs[0][internalID].append((xName,numColumn))

            
            numericColumnPairs = {0:tuple(columnNames)}
            nrows,ncols,subplotBorders = self._findScatterSubplotProps(numericColumnPairs)
            axisPostions = dict([(n,[nrows,ncols,n+1]) for n in range(len(numericColumnPairs))])
            axisLabels = dict([(n,{"x":"Numeric Column(s)","y":"Value"}) for n in range(1)])
            tickLabels = dict([(n,numericColumns) for n in range(1)])
            #tickPositions = dict([(n,np.arange(len(numericColumns))) for n in range(1)])
            colorCategoricalColumn = "Numeric Columns"

        elif numCatColumns == 1:

            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
            numericColumnPairs = []
            
            plotData = pd.DataFrame(rawData, index = rawData.index) 
            columnNames = []
            border = 1/5
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            nColorCats = colorCategories.size
            colorDict,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            colorCategoricalColumn = categoricalColumns[0]
            catGroupby = rawData.groupby(categoricalColumns[0], sort=False)
            multiScatterKwargs[0] = dict()
            interalIDColumnPairs[0] = dict() #0 = axis ID

            colorGroupsData["color"] = colorDict.values() 
            colorGroupsData["group"] = colorCategories
            colorGroupsData["internalID"] = [getRandomString() for _ in range(colorCategories.size)]

            sizeGroupsData["size"] = [self.sourceData.parent.config.getParam("scatterSize")] * nColorCats
            sizeGroupsData["group"] = colorCategories
            sizeGroupsData["internalID"] = colorGroupsData["internalID"].values
            
            tickPositions = []
            tickLabels = []
            widthBox= 1/(nColorCats)
            
            for m,numColumn in enumerate(numericColumns):

                startPos = m if m == 0 else m + (widthBox/3 * m) #add border
                endPos = startPos + widthBox * (nColorCats-1)
                positions = np.linspace(startPos,endPos,num=nColorCats)
                tickPos = np.median(positions)
                tickPositions.append(tickPos)
                tickLabels.append(numColumn)

                for nColCat, colCat in enumerate(colorCategories):
                    #remove nan for kernel estimation
                    groupData = catGroupby.get_group(colCat).dropna(subset=[numColumn])
                    if groupData.index.size > 0:
                        xName = "x({}:{})".format(numColumn,colCat)
                        #get kernel data
                        if groupData.index.size == 1:
                            kdeData = np.array([0])+  positions[nColCat]
                            data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                        else:
                            kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                            #get random x position around 0 to spread data between - and + kdeData
                            kdeData = np.array([np.random.uniform(-x * 0.3 , x * 0.3) for x in kdeData])
                            kdeData = kdeData + positions[nColCat]
                            data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])
                        plotData = plotData.join(data)
                        columnNames.extend([xName,numColumn])
                        multiScatterKwargs[0][(xName,numColumn)] = {"color": colorDict[colCat]}
                        internalID = colorGroupsData.loc[colorGroupsData["group"] == colCat]["internalID"].values[0]
                        #saving internal id and color matches with pandas index (to allow easy manipulation)
                        if internalID not in colorCategoryIndexMatch:
                            colorCategoryIndexMatch[internalID] = kdeIndex
                        else:
                            #because of nan removal, we have to join indices
                            colorCategoryIndexMatch[internalID] = colorCategoryIndexMatch[internalID].join(kdeIndex, how="outer")

                        if internalID not in interalIDColumnPairs[0]:
                            interalIDColumnPairs[0][internalID] = [(xName,numColumn)]
                        else:
                            interalIDColumnPairs[0][internalID].append((xName,numColumn))
                    
            

            numericColumnPairs = {0:tuple(columnNames)}
            
            nrows,ncols,subplotBorders = self._findScatterSubplotProps(numericColumnPairs)
            axisPostions = dict([(n,[nrows,ncols,n+1]) for n in range(len(numericColumnPairs))])
            axisLabels = dict([(n,{"x":"Numeric Column(s)","y":"Value"}) for n in range(1)])
            tickLabels = {0:tickLabels}
            tickPositions = {0:tickPositions}
            #print(tickPositions,tickLabels)

        elif numCatColumns == 2:

            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            globalMin, globalMax = np.nanquantile(rawData[numericColumns].values, q = [0,1])
            yMargin = np.sqrt(globalMax**2 + globalMin**2)*0.05
           
            #get color cats
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            #matching colors to categories (dict)
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            tickCats = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            nXAxisCats = tickCats.size
            colorCategoricalColumn = categoricalColumns[0]
            #save colorGroups data
            colorGroupsData["color"] = colors.values()
            colorGroupsData["group"] = colorCategories
            colorGroupsData["internalID"] = [getRandomString() for n in range(nColorCats)]
            
            sizeGroupsData["size"] = [self.sourceData.parent.config.getParam("scatterSize")] * nColorCats
            sizeGroupsData["group"] = colorCategories
            sizeGroupsData["internalID"] = colorGroupsData["internalID"].values

            axisPostions = getAxisPostistion(len(numericColumns),maxCol=self.maxColumns)
            catGroupby = rawData.groupby(categoricalColumns, sort=False)

            #plot data
            plotData = pd.DataFrame(rawData, index = rawData.index) 
            widthBox= 1/(nColorCats)
            border = widthBox / 3

            for nAxis,numColumn in enumerate(numericColumns):
                multiScatterKwargs[nAxis] = dict()
                interalIDColumnPairs[nAxis] = dict() #nAxis = axis ID
                catTickPositions = []
                catTickLabels = []
                columnNames = []

                for nTickCat, tickCat in enumerate(tickCats):
                    startPos = nTickCat if nTickCat == 0 else nTickCat + (border * nTickCat) #add border
                    endPos = startPos + widthBox * nColorCats - widthBox
                    positions = np.linspace(startPos,endPos,num=nColorCats)
                    catTickPositions.append(np.median(positions))
                    catTickLabels.append(tickCat)
                    for nColCat, colCat in enumerate(colorCategories):
                        groupName = (colCat,tickCat)
                        if groupName not in catGroupby.groups:
                            continue
                        groupData = catGroupby.get_group(groupName).dropna(subset=[numColumn])
                        if groupData.index.size > 0:
                            xName = "x({}:{}:{})".format(numColumn,colCat,tickCat)
                            if groupData.index.size == 1:
                                kdeData = np.array([0]) +  positions[nColCat]
                                data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                            else:
                                #get kernel data
                                
                                kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.8)
                                #get random x position around 0 to spread data between - and + kdeData
                                kdeData = np.array([np.random.uniform(-x*0.5,x*0.5) for x in kdeData])
                                kdeData = kdeData + positions[nColCat]
                                data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])
                            plotData = plotData.join(data)
                            columnNames.extend([xName,numColumn])
                            multiScatterKwargs[nAxis][(xName,numColumn)] = {"color": colors[colCat]}
                            #saving internal id and color matches with pandas index (to allow easy manipulation)
                            internalID = colorGroupsData.loc[colorGroupsData["group"] == colCat]["internalID"].values[0]
                            if internalID not in colorCategoryIndexMatch:
                                colorCategoryIndexMatch[internalID] = kdeIndex
                            else:
                                #because of nan removal, we have to join indices
                                colorCategoryIndexMatch[internalID] = colorCategoryIndexMatch[internalID].join(kdeIndex, how="outer")

                            if internalID not in interalIDColumnPairs[nAxis]:
                                interalIDColumnPairs[nAxis][internalID] = [(xName,numColumn)]
                            else:
                                interalIDColumnPairs[nAxis][internalID].append((xName,numColumn))

                tickPositions[nAxis] = catTickPositions
                numericColumnPairs[nAxis] = tuple(columnNames)
                axisLabels[nAxis] = {"x":categoricalColumns[1],"y":numColumn}
                tickLabels[nAxis] = catTickLabels
                axisLimits[nAxis] = {"yLimit":[globalMin-yMargin,globalMax+yMargin],"xLimit":[0-widthBox/2-border,nXAxisCats+widthBox/2+border]}
                #saving internal id and color matches with pandas index (to allow easy manipulation)
                
        elif numCatColumns == 3:

            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            globalMin, globalMax = np.nanquantile(rawData[numericColumns].values, q = [0,1])
            yMargin = np.sqrt(globalMax**2 + globalMin**2)*0.05
           
            #get color cats
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            #matching colors to categories (dict)
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            tickCats = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            axisCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[2])
            nXAxisCats = tickCats.size
            colorCategoricalColumn = categoricalColumns[0]
            #save colorGroups data
            colorGroupsData["color"] = colors.values()
            colorGroupsData["group"] = colorCategories
            colorGroupsData["internalID"] = [getRandomString() for n in range(nColorCats)]
            
            sizeGroupsData["size"] = [self.sourceData.parent.config.getParam("scatterSize")] * nColorCats
            sizeGroupsData["group"] = colorCategories
            sizeGroupsData["internalID"] = colorGroupsData["internalID"].values

            axisPostions = getAxisPostistion(n = axisCategories.size *  numNumColumns, maxCol = axisCategories.size)
            catGroupby = rawData.groupby(categoricalColumns, sort=False)

            #plot data
            plotData = pd.DataFrame(rawData, index = rawData.index) 
            widthBox= 1/(nColorCats)
            border = widthBox / 3
            

            tickPositions = {}
            numericColumnPairs = {}
            axisLabels = {}
            tickLabels = {}
            
            nAxis = -1

            for n,numColumn in enumerate(numericColumns):

                for nAxisCat, axisCat in enumerate(axisCategories):
                    nAxis +=1 
                    multiScatterKwargs[nAxis] = dict()
                    interalIDColumnPairs[nAxis] = dict() #nAxis = axis ID
                    catTickPositions = []
                    catTickLabels = []
                    columnNames = []

                    for nTickCat, tickCat in enumerate(tickCats):

                        startPos = nTickCat if nTickCat == 0 else nTickCat + (border * nTickCat) #add border
                        endPos = startPos + widthBox * nColorCats - widthBox
                        positions = np.linspace(startPos,endPos,num=nColorCats)
                        tickPos = np.median(positions)
                        catTickPositions.append(tickPos)
                        catTickLabels.append(tickCat)

                        for nColCat, colCat in enumerate(colorCategories):
                            groupName = (colCat,tickCat,axisCat)
                            if groupName not in catGroupby.groups:
                                continue
                            groupData = catGroupby.get_group(groupName).dropna(subset=[numColumn])
                            if groupData.index.size > 0:
                                xName = "x({}:{}:{}:{})".format(numColumn,colCat,tickCat,axisCat)
                                #get kernel data
                                if groupData.index.size == 1:
                                    kdeData = np.array([0])+  positions[nColCat]
                                    data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                                else:
                                    kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.8)
                                    #get random x position around 0 to spread data between - and + kdeData
                                    kdeData = np.array([np.random.uniform(-x*0.5,x*0.5) for x in kdeData])
                                    kdeData = kdeData + positions[nColCat]
                                    data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])
                                plotData = plotData.join(data)
                                columnNames.extend([xName,numColumn])
                                multiScatterKwargs[nAxis][(xName,numColumn)] = {"color": colors[colCat]}
                                #saving internal id and color matches with pandas index (to allow easy manipulation)
                                internalID = colorGroupsData.loc[colorGroupsData["group"] == colCat]["internalID"].values[0]
                                if internalID not in colorCategoryIndexMatch:
                                    colorCategoryIndexMatch[internalID] = kdeIndex
                                else:
                                    #because of nan removal, we have to join indices
                                    colorCategoryIndexMatch[internalID] = colorCategoryIndexMatch[internalID].join(kdeIndex, how="outer")

                                if internalID not in interalIDColumnPairs[nAxis]:
                                    interalIDColumnPairs[nAxis][internalID] = [(xName,numColumn)]
                                else:
                                    interalIDColumnPairs[nAxis][internalID].append((xName,numColumn))

                    tickPositions[nAxis] = catTickPositions
                    numericColumnPairs[nAxis] = tuple(columnNames)
                    axisLabels[nAxis] = {"x":categoricalColumns[1],"y":numColumn}
                    tickLabels[nAxis] = catTickLabels
                    axisLimits[nAxis] = {"yLimit":[globalMin-yMargin,globalMax+yMargin],"xLimit":[0-widthBox/2-border,nXAxisCats+widthBox/2+border]}
                    if not nAxisCat in axisTitles:   
                        axisTitles[nAxisCat] = "{}\n{}".format(categoricalColumns[2],axisCat)
                #saving internal id and color matches with pandas index (to allow easy manipulation)
        

                    

            
            
            

            #tickPositions = dict([(n,np.arange(len(numericColumns))) for n in range(1)])
        #             index = data.sort_values(by=numColumn, ascending = self.indexSort == "ascending").index
        #            # index = data.index
        #             data = data.join(pd.Series(np.arange(index.size),index=index, name = indexName))
        #           #  print(kdeData)

        #  #kdeData, index = self.sourceData.getKernelDensity(dataID,pd.Series(numColumn),justData = True)
        #             #kdeData = [np.random.uniform(-x,x) for x in kdeData]
        #             #data.loc[index,indexName] = pd.Series(kdeData,index = index)


        return {"data":{"plotData":plotData,
            "axisPositions":axisPostions, 
            "columnPairs":numericColumnPairs,
            "tickLabels": tickLabels,
            "tickPositions" : tickPositions,
            "axisTitles" : axisTitles,
            "dataColorGroups": colorGroupsData,
            "dataSizeGroups" : sizeGroupsData,
            "axisLimits" : axisLimits,
            #"dataSizeGroups" : sizeGroupsData,
            "axisLabels" : axisLabels,
            "subplotBorders":subplotBorders,
            "colorCategoryIndexMatch" : colorCategoryIndexMatch,
            "sizeCategoryIndexMatch" : colorCategoryIndexMatch,
            "multiScatterKwargs": multiScatterKwargs,
            "interalIDColumnPairs" : interalIDColumnPairs,
            "colorCategoricalColumn" : colorCategoricalColumn,
            "sizeCategoricalColumn" : colorCategoricalColumn,
            "dataID":dataID}}    

    def getColorQuadMeshForHeatmap(self,dataID, colorColumn = None, colorColumnType = None):
        ""    
        if colorColumn is None:
            colorColumn = self.colorColumn
        if colorColumnType is None:
            colorColumnType = self.colorColumnType
        colorColumnNames = colorColumn.values

        rawData = self.sourceData.getDataByColumnNames(dataID,colorColumn)["fnKwargs"]["data"]
        #unique values. add nan value first -> by default light grey
        uniqueValuesList = self.sourceData.getUniqueValues(dataID,colorColumn.values.tolist(), forceListOutput=True)
        #aüüend replaceObjectNan - should be first item!
        uniqueValuesTotal = pd.Series([self.sourceData.replaceObjectNan]).append(
                            pd.Series(np.concatenate(uniqueValuesList)) , ignore_index=True).unique()
        
        factorMapper = OrderedDict(zip(uniqueValuesTotal,np.arange(uniqueValuesTotal.size)))
        
        #ensure -1 is first in color chooser
       # factorMapper = OrderedDict(sorted(factorMapper.items(), key=lambda x:x[1]))
        

        colorData = pd.DataFrame(columns = colorColumn, index = rawData.index)

        for columnName in colorColumn.values:
            colorData.loc[rawData.index,columnName] = rawData[columnName].map(factorMapper)


        colorValues = sns.color_palette("Paired",len(factorMapper)).as_hex()
        colorValues[0] = self.sourceData.colorManager.nanColor
        cmap = ListedColormap(colorValues)
        colorGroupData = pd.DataFrame(columns=["color","group","internalID"])
        colorGroupData["color"] = colorValues
        colorGroupData["group"] = list(factorMapper.keys())
        colorGroupData["internalID"] = [getRandomString() for n in colorGroupData.index]

	

        return {"colorData":colorData,"colorGroupData":colorGroupData,"cmap":cmap,"title":mergeListToString(colorColumnNames,"\n"),"isEditable":False}
        

    def getSizeQuadMeshForHeatmap(self,dataID,sizeColumn = None,sizeColumnType = None):
        ""
        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,sizeColumn)["fnKwargs"]["data"]

        if sizeColumnType == "Numeric Floats":

            quantiles = np.nanquantile(rawData.values, q = np.linspace(0,1,num=20))

            repData = np.repeat(rawData.values,quantiles.size,axis=1)
            repData[repData < quantiles] = np.nan
            sizeData = pd.DataFrame(repData,index=rawData.index)
        
        return {"sizeData":sizeData}

	
    def getColorGroupingForCorrmatrix(self, columnNames, grouping, groupingName, groupColorMapper):

        #turn values and keys around
        matchDict = {k: oldk for oldk, oldv in grouping.items() for k in oldv}
        groupNames = pd.Series(columnNames.map(matchDict))

        groupFactors, levels = pd.factorize(groupNames)
        groupMapper = dict([(levels[factor],factor) for factor in groupFactors if factor >= 0])
       
        colorData = pd.DataFrame(groupFactors, columns=["Grouping"], index = columnNames.values)
        colorData["Grouping"] = colorData["Grouping"].replace(-1,np.nan)
    
        title = groupingName

        colorGroupData = pd.DataFrame(columns=["color","group","internalID"])
        colorGroupData["color"] = [to_hex(groupColorMapper.to_rgba(x)) for x in groupMapper.values()] + [self.sourceData.colorManager.nanColor]
        colorGroupData["group"] = list(groupMapper.keys())  + [self.sourceData.replaceObjectNan]
        colorGroupData["internalID"] = [getRandomString() for n in colorGroupData.index + 1]

        return {"colorData":colorData,"colorGroupData":colorGroupData,"cmap":groupColorMapper,"title":title}

    def _findScatterSubplotProps(self,numericPairs):
        ""
        subplotProps = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        n_pairs = len(numericPairs)
        if  n_pairs == 1:
            return 1,1,subplotProps
        
        subplotProps["wspace"]=0.30
        subplotProps["hspace"]=0.30

        if n_pairs < 5:
            return 2,2, subplotProps
        else: 
            rows = np.ceil(n_pairs/3)
            columns = 3 
            return rows, columns, subplotProps

    def getColorMapper(self,uniqueValues):
        ""
        colorMap, _ = self.sourceData.colorManager.createColorMapDict(uniqueValues, as_hex=True)
        return colorMap



    def exportFigure(self,filePath, exporter):
        ""
        try:
            exporter.export(filePath)
            completeKwargs = getMessageProps("Exported..","Figure saved.")
        except Exception as e:
            print(e)
        
        return completeKwargs

    def getXYPlotProps(self,dataID,numericColumns,categoricalColumns):
        "Returns plot properties for a XY plot"
        colorGroupsData = pd.DataFrame() 
        axisLabels = {}
        axisLimits = {}
        linesByInternalID = {}

        rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        config =  self.sourceData.parent.config
        
        
        if config.getParam("xy.plot.against.index") or len(numericColumns) == 1:
            idxColumnName = "ICIndex_{}".format(getRandomString())#ensures that there are no duplicates
            rawData[idxColumnName] = np.arange(rawData.index.size)
            numericColumnPairs = [(idxColumnName,columnName) for columnName in numericColumns]
        elif config.getParam("xy.plot.single.x"): #use only first numeric column
            numericColumnPairs = [(numericColumns[0],columnName) for columnName in numericColumns[1:]]
        else:
            numericColumnPairs = list(zip(numericColumns[0::2], numericColumns[1::2]))
        separatePairs = config.getParam("xy.plot.separate.column.pairs")
        axisPostions = getAxisPostistion(1 if not separatePairs else len(numericColumnPairs))

        colorValues = self.sourceData.colorManager.getNColorsByCurrentColorMap(len(numericColumnPairs))
        #line2D 
        colorGroupsData["color"] = colorValues
        colorGroupsData["group"] = ["{}:{}".format(*columnPair) if "ICIndex" not in columnPair[0] else "{}".format(columnPair[1]) for columnPair in numericColumnPairs]
        colorGroupsData["internalID"] = [getRandomString() for n in colorValues]
        #
        lines = {}
        lineKwargs = {}
        for n,columnPair in enumerate(numericColumnPairs):
            internalID = colorGroupsData["internalID"].iloc[n]
            lineProps = dict(
                    xdata = rawData[columnPair[0]].values,
                    ydata = rawData[columnPair[1]].values,
                    color = colorValues[n],
                    markeredgecolor = "darkgrey",
                    markeredgewidth = config.getParam("xy.plot.marker.edge.width"),
                    alpha = config.getParam("xy.plot.alpha"), 
                    markersize = config.getParam("xy.plot.marker.size"),
                    linewidth = config.getParam("xy.plot.linewidth"), 
                    marker = config.getParam("xy.plot.marker") if config.getParam("xy.plot.show.marker") else "")

            l = Line2D(**lineProps)
            
            linesByInternalID[internalID] = l
            if not separatePairs:
                if 0 not in lines:    
                    lines[0] = []
                    lineKwargs[0] = []
                lines[0].append(l)
                lineKwargs[0].append(lineProps)
            else:
                lines[n] = [l]   
                lineKwargs[n] = [lineProps]   
        
    
        if not separatePairs:
            xAxisColumns = np.unique([columnPair[0] for columnPair in numericColumnPairs])
            yAxisColumns = np.unique([columnPair[1] for columnPair in numericColumnPairs])
            xAxisMin, xAxisMax, yAxisMin, yAxisMax = self._getXYLimits(X = rawData[xAxisColumns].values.flatten(),
                                    Y  = rawData[yAxisColumns].values.flatten())
                
            axisLimits[0] = {"xLimit": (xAxisMin,xAxisMax), "yLimit": (yAxisMin,yAxisMax)}
            
            yAxisLabel = numericColumnPairs[0][1] if not (config.getParam("xy.plot.against.index") or len(numericColumns) == 1) else "Index"
            axisLabels[0] = {"x":numericColumnPairs[0][0],
                            "y":numericColumnPairs[0][1]} if len(numericColumnPairs) == 1 else {"x":"Index","y":"y-value"}
        else:
            
            for n, columnPair in enumerate(numericColumnPairs):
                xColumn, yColumn = columnPair
                xAxisMin, xAxisMax, yAxisMin, yAxisMax = self._getXYLimits(
                                                    X = rawData[xColumn].values.flatten(),
                                                    Y  = rawData[yColumn].values.flatten())
                xAxisLabel = xColumn if not (config.getParam("xy.plot.against.index") or len(numericColumns) == 1) else "Index"
                axisLimits[n] = {"xLimit": (xAxisMin,xAxisMax), "yLimit": (yAxisMin,yAxisMax)}
                axisLabels[n] = {"x":xAxisLabel,"y":yColumn}

        return {"data":{
                    "lines":lines,
                    "lineKwargs" : lineKwargs,
                    "axisLabels" : axisLabels,
                    "axisPositions":axisPostions,
                    "dataColorGroups": colorGroupsData,
                    "colorCategoricalColumn": "Lines",
                    "axisLimits" : axisLimits,
                    "dataID" : dataID,
                    "linesByInternalID": linesByInternalID
                    }}
        
    def _getXYLimits(self,X,Y,marginFrac = 0.02):
        ""
        minX, maxX = np.nanquantile(X,q = [0,1])
        marginX = np.sqrt(maxX**2 + minX**2)
        xAxisMin,  xAxisMax = minX - marginX * marginFrac, maxX + marginX * marginFrac

        minY, maxY = np.nanquantile(Y,q = [0,1])
        marginY = np.sqrt(maxY**2 + minY**2)
        yAxisMin,  yAxisMax = minY - marginY * marginFrac , maxY + marginY * marginFrac
        return xAxisMin, xAxisMax, yAxisMin, yAxisMax
#  return {"data":{"plotData":plotData,
#             "axisPositions":axisPostions, 
#             "columnPairs":numericColumnPairs,
#             "tickLabels": tickLabels,
#             "tickPositions" : tickPositions,
#             "axisTitles" : axisTitles,
#             "dataColorGroups": colorGroupsData,
#             "dataSizeGroups" : sizeGroupsData,
#             "axisLimits" : axisLimits,
#             #"dataSizeGroups" : sizeGroupsData,
#             "axisLabels" : axisLabels,
#             "subplotBorders":subplotBorders,
#             "colorCategoryIndexMatch" : colorCategoryIndexMatch,
#             "sizeCategoryIndexMatch" : colorCategoryIndexMatch,
#             "multiScatterKwargs": multiScatterKwargs,
#             "interalIDColumnPairs" : interalIDColumnPairs,
#             "colorCategoricalColumn" : colorCategoricalColumn,
#             "sizeCategoricalColumn" : colorCategoricalColumn,
#             "dataID":dataID}}    