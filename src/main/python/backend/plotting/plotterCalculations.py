
from multiprocessing import Pool
from numba.np.ufunc import parallel
import numpy as np
from numpy.core import numeric
from numpy.lib.histograms import histogram 
import pandas as pd 

from matplotlib.colors import to_hex
#backend imports
from backend.utils.stringOperations import getReadableNumber
from backend.statistics.statistics import loess_fit

from ..utils.stringOperations import getMessageProps, getReadableNumber, getRandomString, mergeListToString
from ..utils.misc import scaleBetween, replaceKeyInDict
from .postionCalculator import calculatePositions, getAxisPosition

from threadpoolctl import threadpool_limits
from statsmodels.stats.contingency_tables import Table2x2
#cluster
import scipy.cluster.hierarchy as sch
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.colors import to_hex
from matplotlib.pyplot import get, hist, sca
import matplotlib.cm as cm
from matplotlib import rcParams

from collections import OrderedDict
import seaborn as sns 
import itertools
import numpy as np
import scipy.stats as st
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import GridSearchCV
from numba import jit, njit, prange
from wordcloud import WordCloud
import io
import time
import pickle
from scipy.stats import linregress

@njit(parallel=True)
def spreadSwarm(kde):
    ""
    A = np.zeros(shape=(kde.size,1))
    for i in prange(kde.size):
        A[i,0] = np.random.uniform(-kde[i]*0.85,kde[i]*0.85) 
    return A

def buildScatterPairs(axisInts,columnPairs,scatterColumnPairs):
    ""
    for nA in axisInts.flatten():
        scatterColumnPairs[nA] = columnPairs[nA]
    return scatterColumnPairs

def lowessFit(X,it=None,frac=None,*args,**kwargs):
        '''
        Calculates lowess line from dataFrame input
        '''
        X.dropna(inplace=True)
        data = X.sort_values(by = X.columns.values[0])
        x = data.iloc[:,0].values
        y = data.iloc[:,1].values

        if it is None and frac is None:
            lenX = x.size
            if lenX > 1000:
                it = 3
                frac = 0.65
            else:
                it = 1
                frac = 0.3
        yfit,ymin,ymax = loess_fit(x,y,span=frac)
        #lowessLine = lowess(y,x, it=it, frac=frac,*args,**kwargs)
        lowessLine = np.empty(shape=(x.size,4))
        lowessLine[:,0] = x 
        lowessLine[:,1] = yfit
        lowessLine[:,2] = ymin
        lowessLine[:,3] = ymax
        return lowessLine 

def buildLabelData(axisInts, numericColumns, groupColorDict, labelData):
    ""
    for numericColumn, nAx in zip(numericColumns,axisInts):
        
        tKwargs = {
            "color" : groupColorDict[numericColumn] if numericColumn in groupColorDict else "black",
            "s" : numericColumn,
            "horizontalalignment" : 'center',
            "verticalalignment" : 'center',
            "x" : 0.5,
            "y" : 0.5
            }
        
        labelData[nAx] = tKwargs
        
    
    return labelData

def build2DHistogramData(axisInts, columnPairs, data, histogramData):
        #get pcolormesh for column pairs
        ""
        for nA in axisInts.flatten():
            numericPairs = list(columnPairs[nA])
            
            XY  = data[numericPairs].dropna().values
            
            H, xedges, yedges = np.histogram2d(XY[:,0], XY[:,1], bins=(25, 25))
            meshX, meshY = np.meshgrid(xedges, yedges)
            histogramData[nA] = {
                        "H":H,
                        "xedges":xedges,
                        "yedges":yedges,
                        "meshX":meshX,
                        "meshY":meshY
                        }
        return histogramData


def buildCorrelationLabelData(axisInts, columnPairs, labelData, corrMatrix  = None, spearmanCorrMatrix=None):
    ""
    if corrMatrix is None and spearmanCorrMatrix is None:
        return labelData

    for nA in axisInts.flatten():
        colA, colB = columnPairs[nA]
        pearCorr = "" if corrMatrix is None else "r = {}\n".format(round(corrMatrix.loc[colA, colB],2))
        spearCorr = "" if spearmanCorrMatrix is None else "rho = {}".format(round(spearmanCorrMatrix.loc[colA, colB],2))

        tKwargs = {
            "color" : "black",
            "s" : pearCorr+spearCorr,
            "horizontalalignment" : 'left',
            "verticalalignment" : 'top',
            "x" : 0.02,
            "y" : 0.95
            }
        labelData[nA] = tKwargs

    return labelData

def buildScatterMatrix(data,
                    numericColumns,
                    plotType,
                    axisInts,
                    backgroundColorHex,
                    scatterColumnPairs,
                    colorBackground,
                    addLinReg,
                    addLowess,
                    columnPairs,
                    kdeKwargs,
                    groupColorDict,
                    addToPlotType,
                    addPearson,
                    addSpearman,
                    corrmatrix,
                    spearmanCorrMatrix,
                    lowessKwargs):
    ""
    backgroundColors = {} 
    histogramData = {}
    linregressFit = {}
    lowessData = {}
    labelData = {}
    kdeData = {}
    if plotType == "scatter":
        scatterColumnPairs = buildScatterPairs(axisInts,columnPairs,scatterColumnPairs)
        if colorBackground and backgroundColorHex is not None:
            for nA, columnPair in scatterColumnPairs.items():
                colA, colB = columnPair

                backgroundColors[nA] = backgroundColorHex.loc[colA,colB]
        if addLinReg or addLowess:
            for nA in axisInts.flatten():
                numericPairs = list(columnPairs[nA])
                if numericPairs[0] !=  numericPairs[1]:
                    if addLinReg:
                        linregressFit[nA] = linRegress(data[numericPairs])
                    if addLowess:
                        lowessData[nA] = lowessFit(data[numericPairs],**lowessKwargs)
    elif plotType == "2D-Histogram":
        histogramData =  build2DHistogramData(axisInts,columnPairs,data,histogramData)
    elif plotType == "kde" or plotType == "label-kde":
       kdeData = buildKdeData(axisInts,data,numericColumns,kdeData,**kdeKwargs)

    if "label" in plotType:
        labelData = buildLabelData(axisInts,numericColumns,groupColorDict,labelData)

    if (addPearson or addSpearman) and (addToPlotType == "all types" or addToPlotType == plotType) and plotType not in ["label-kde","label","kde"]:            
                
            labelData = buildCorrelationLabelData(axisInts,
                            columnPairs,
                            labelData,
                            corrMatrix = None if not addPearson else corrmatrix,
                            spearmanCorrMatrix = spearmanCorrMatrix)

    return (scatterColumnPairs,backgroundColors,linregressFit,histogramData,kdeData,labelData,lowessData)


def _findBwByGridSearch(x,minBW,maxBW,numCrossValidations=5,kernel="gaussian"):
    ""
    grid = GridSearchCV(KernelDensity(kernel=kernel,algorithm='ball_tree'),
                {'bandwidth': np.linspace(minBW, maxBW, 30)},
                cv=numCrossValidations) # 5-fold cross-validation
    grid.fit(x)
    
    return grid.best_params_["bandwidth"]

def buildKdeData(axisInts,data,numericColumns,kdeData,bw,kernel,bwGridSearch,logDensity,gridMin,gridMax,numCVs):
    ""

    minMaxValues = np.nanquantile(data[numericColumns].values, q = [0.0,1.0],axis=0)
    
    for n,nAx in enumerate(axisInts):
        x = data[numericColumns[n]].dropna().values.reshape(-1,1)
    
        #find x limits
        minValue,maxValue = minMaxValues[:,n]
        offSet = (maxValue - minValue) * 0.1
        minValue -= offSet
        maxValue += offSet

        #caluclate kde and get y values
        xx = np.linspace(minValue,maxValue,num=500).reshape(-1,1)
        if bwGridSearch:
            
            bw = _findBwByGridSearch(x,gridMin,gridMax,numCVs)
        
        
        kde = KernelDensity(bandwidth=bw, kernel=kernel, algorithm='ball_tree').fit(x)
        y = kde.score_samples(xx)
        if not logDensity:
            y = np.exp(y)
        
        #find y limits
        yMax = np.max(y)
        yMin = np.min(y)
        yMax += 0.15*np.sqrt((yMax-yMin)**2)
        #add kdea data - key is axis int
        kdeData[nAx] = {"xx" : xx, "x" : x, "yKde" : y ,"xLimit":(minValue,maxValue),"yLimit":(yMin,yMax)}

    return kdeData

def linRegress(X):
    X = X.dropna()
    
    x = X.iloc[:,0].values
    y = X.iloc[:,1].values

    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    x1, x2 = x.min(), x.max()
    y1, y2 = slope*x1+intercept, slope*x2+intercept

    return [x1,x2],[y1,y2],slope, intercept, r_value, p_value, std_err

def kdeCalc(X, bw = None, kernel = "gaussian", widthBox=1, addToX=0, numColumn = "Col", defaultPos = 0):
    xName = "x({})".format(numColumn)
    if X.index.size > 2:
        
        if bw is None:
            bw = X.index.size**(-1/(X.columns.size+4)) 
        kde = KernelDensity(bandwidth=bw,
            kernel=kernel, algorithm='ball_tree')
        kde.fit(X.values) 
        kdeData = np.exp(kde.score_samples(X.values))

        allSame = np.all(kdeData == kdeData[0])
        if allSame:
            kdeData = np.zeros(shape=kdeData.size)
        else:
            kdeData = scaleBetween(kdeData,(0,widthBox/2))
        #kdeData = spreadSwarm(kdeData.flatten())
        kdeData = np.array([np.random.uniform(-x*0.85,x*0.85) for x in kdeData])
        #print(time.time()-t1,"numpy")
        kdeData = kdeData + addToX
        #save data
        data = pd.DataFrame(kdeData, index = X.index, columns=[xName])
    else:

        
        kdeData = np.array([0]) +  defaultPos
        data = pd.DataFrame(kdeData ,index=X.index, columns = [xName])

    return (X.index,data,(xName,numColumn))

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
    "dim-red-plot":"getDimRedProps",
    "forestplot" : "getForestplotProps",
    "wordcloud"  : "getWordCloud",
    "clusterplot" : "getClusterProps",
    "mulitscatter" : "getScatterCorrMatrix",
    "proteinpeptideplot" : "getProteinPeptideProps"
}

line_kwargs = dict(linewidths=.2, colors='k')

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
        

    def figToClipboard(self,figure):
        ""
       # print(figure)
        buf = io.BytesIO()
        #pickle.dump(figure,buf)
        dpi = self.sourceData.parent.config.getParam("copy.to.clipboard.dpi")
        transparent = self.sourceData.parent.config.getParam("copy.to.clipboard.background.transparent")
        figure.savefig(buf,format="png",dpi=dpi,transparent=transparent)
        funcProps = getMessageProps("Done..","Figure saved to clipboard.")
        funcProps["buf"] = buf
        return funcProps
        #QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
        #self.figure.savefig(buf,format="png")
        
        # buf.close()


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
            graphData = getattr(self,plotFnDict[plotType])(dataID,numericColumns,categoricalColumns,**kwargs)
            graphData["plotType"] = plotType
            return graphData

    def getCountplotProps(self,dataID, numericColumns, categoricalColumns,*args,**kwargs):
        ""
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        subplotBorders = dict(wspace=0.15, hspace = 0.0,bottom=0.15,right=0.95,top=0.95)
        axisDict = getAxisPosition(2, maxCol = 1)
        groupbyCatColumns = data.groupby(by=categoricalColumns, sort=False)
        colors = self.sourceData.colorManager.getNColorsByCurrentColorMap(len(categoricalColumns),"countplotLabelColorMap")
        groupSizes = groupbyCatColumns.size().sort_values(ascending=False).reset_index(name='counts')
        if groupbyCatColumns.ngroups > 50:
            return getMessageProps("Error..","More than 50 unique categories found. This plot type is not appropiate for so many categories.")
      
    
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
            "dataColorGroups": colorGroups,
            "chartData" : groupSizes
 
            
        }}

    def getClusterLineplots(self,clusterLabels, data, numericColumns, columnName,*args,**kwargs):
        ""
        plotData = {}
        quickSelect = {}
        xValues = np.arange(len(numericColumns))
        groupby = clusterLabels.groupby(columnName,sort=False)
        for n, (clusterLabel, clusterData) in enumerate(groupby):
            plotData[n] = {}
            quickSelect[n] = {}
            plotData[n]["segments"] = [list(zip(xValues,data.loc[idx,:].values.flatten())) for idx in clusterData.index] 
            quickSelect[n]["positions"] = np.arange(len(numericColumns))
            quickSelect[n]["x"] = [data.loc[clusterData.index,colName] for colName in data.columns]
        return plotData, quickSelect, groupby

    def getClusterBoxplots(self,clusterLabels, data, numericColumns, columnName,*args,**kwargs):
        ""
        plotData = {}
        quickSelect = {}
        for n, (clusterLabel, clusterData) in enumerate(clusterLabels.groupby(columnName,sort=False)):
            plotData[n] = {}
            plotData[n]["x"] = [data.loc[clusterData.index,colName] for colName in data.columns] 
            plotData[n]["patch_artist"] = True
            plotData[n]["positions"] = np.arange(len(numericColumns))
        quickSelect = plotData.copy()
        return plotData, quickSelect
           
            
    def getNumericColorMappingGroupNames(self, maxV,q75,median,q25,minV):
        ""
        groupNames = ["Max ({})".format(getReadableNumber(maxV)),
                            "75% Quantile ({})".format(getReadableNumber(q75)),
                            "Median ({})".format(getReadableNumber(median)),
                            "25% Quantile ({})".format(getReadableNumber(q25)),
                            "Min ({})".format(getReadableNumber(minV))]
        return groupNames

    def getClusterProps(self, dataID, numericColumns, categoricalColumns,*args,**kwargs):
        ""
        clusterCenters = None
        #get config 
        config = self.sourceData.parent.config
        #get cluster method
        method = config.getParam("clusterplot.method")
        plottype = config.getParam("clusterplot.type")
        #color for distance

        # color group file
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])

        clusterLabels, data, model = self.sourceData.statCenter.runCluster(dataID,numericColumns,method,True)
        clusterLabels = clusterLabels.sort_values("Labels")
        columnName = "C({})".format(method)

        clusterLabels = pd.DataFrame(["C({})".format(x) for x in clusterLabels.values.flatten()], index=clusterLabels.index,columns=[columnName])
        


        if method in ["kmeans","Birch"] and config.getParam("clusterplot.show.cluster.center"):

            if hasattr(model,"cluster_centers_"):
                clusterCenters = model.cluster_centers_
            elif hasattr(model,"subcluster_centers_"):
                clusterCenters = model.subcluster_centers_

            extraLines = dict([(n,{
                                "xdata":np.arange(len(numericColumns)),
                                "ydata":clusterCenter.flatten(),
                                "linewidth":0.75,
                                "color":"black"}) for n,clusterCenter in enumerate(clusterCenters)])
        else:
            extraLines = {}
        #print(clusterLabels)
        uniqueClusters = clusterLabels[columnName].unique()
        nClusters = uniqueClusters.size 
        displayDistanceByColor = config.getParam("clusterplot.lineplot.color.distance") and method in ["kmeans","Birch"] and plottype == "lineplot"
        #print(uniqueClusters)
        #print(nClusters)
        colorMap, _ = self.sourceData.colorManager.createColorMapDict(uniqueClusters,addNaNLevels=[-1],as_hex = True)
        #print(colorMap)
        if plottype == "boxplot":
            plotData,qSData = self.getClusterBoxplots(clusterLabels,data,numericColumns,columnName)
            faceColors = dict([(n,[colorMap[uniqueCluster]] * len(numericColumns)) for n,uniqueCluster in enumerate(uniqueClusters)])

        elif plottype == "lineplot":
            
            plotData, qSData, groupby = self.getClusterLineplots(clusterLabels,data,numericColumns,columnName)
            if displayDistanceByColor:
                faceCs = []
                distanceMeasures = pd.DataFrame(model.transform(data.loc[clusterLabels.index]),index=clusterLabels.index)
               # cmap = self.sourceData.colorManager.get_max_colors_from_pallete(config.getParam("twoColorMap"))
                #cmap,colors = self.sourceData.colorManager.matchColorsToValues(distanceMeasures)
                distanceValues = np.concatenate([distanceMeasures.loc[clusterData.index,n].values.flatten() for n, (_, clusterData) in enumerate(groupby)])
                qs = np.quantile(distanceValues,q=[0,0.25,0.5,0.75,1])
                minValue,q25,median,q75, maxValue = qs
                groupNamesForLegend = self.getNumericColorMappingGroupNames(maxValue,q75,median,q25,minValue)
                colorArrayForLegend, _  = self.sourceData.colorManager.matchColorsToValues(np.flip(qs.flatten()),"Spectral",vmin=minValue,vmax=maxValue)
                colorArrayForLegendHex = [to_hex(c) for c in colorArrayForLegend]
                for n, (_, clusterData) in enumerate(groupby):
                    X = distanceMeasures.loc[clusterData.index,n].values
                    colorArray, _  = self.sourceData.colorManager.matchColorsToValues(X,"Spectral",vmin=minValue,vmax=maxValue)
                    faceCs.append((n,colorArray))
                # for n,uniqueCluster in enumerate(uniqueClusters):
                #     boolIdx = clusterLabels[columnName] == "C({})".format(n)
                #     X = distanceMeasures[boolIdx.index,n] #get distance for cluster
                #     print(X)
                #     colorArray, _  = self.sourceData.colorManager.matchColorsToValues(X,"Blues_r")
                #     faceCs.append((n,colorArray))
                faceColors = dict(faceCs)#dict([(n,colorMap[uniqueCluster]) ])

            else:

                faceColors = dict([(n,colorMap[uniqueCluster]) for n,uniqueCluster in enumerate(uniqueClusters)])
        
        axisPositions = getAxisPosition(nClusters,maxCol=self.maxColumns)
        axisLabels = dict([(n,{"x":"","y":"Value"}) for n in range(nClusters)])#
        axisTitles = dict([(n,"{} n:{}".format(uniqueCluster,np.sum(clusterLabels[columnName] == uniqueCluster))) for n,uniqueCluster in enumerate(uniqueClusters)])
        tickPositions = dict([(n,np.arange(len(numericColumns))) for n in range(nClusters)])#
        #tickLabels = dict([(n,[str(x) for x in np.arange(len(numericColumns))]) for n in range(nClusters)])#
        tickLabels = dict([(n,numericColumns) for n in range(nClusters)])#



        if displayDistanceByColor:
            colorGroups["group"] = groupNamesForLegend
            colorGroups["color"] = colorArrayForLegendHex
            colorGroups["internalID"] = [getRandomString() for _ in range(len(groupNamesForLegend))]
        else:

            colorGroups["group"] = list(colorMap.keys())
            colorGroups["color"] = list(colorMap.values())
            colorGroups["internalID"] = [getRandomString() for _ in range(len(colorMap))]

        
       

        return {"data":{
            "plotType" : plottype,
            "plotData": plotData,
            "axisPositions" : axisPositions,
            "axisLabels"    :   axisLabels,
            "tickPositions": tickPositions,
            "tickLabels": tickLabels,
            "axisTitles" : axisTitles,
            "facecolors" : faceColors,
            "dataColorGroups" : colorGroups,
            "colorCategoricalColumn" : "Cluster Labels" if not displayDistanceByColor else "Cluster Distance",
            "clusterLabels" : clusterLabels,
            "dataID" : dataID,
            "extraLines" : extraLines,
            "quickSelect" : qSData
        }}

    def getBarplotProps(self, dataID, numericColumns, categoricalColumns,*args,**kwargs):
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
        xWidth, axisLabels, axisLimits, axisTitles, groupNames, verticalLines, groupedPlotData  = calculatePositions(dataID,self.sourceData,numericColumns,categoricalColumns,self.maxColumns,splitByCategories= splitByCats)
            

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

    def getLineplotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        minQ = np.inf
        maxQ = -np.inf
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        axisTitles = {}

        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        if len(categoricalColumns) == 0:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)
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
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)
            plotData = {0:[]}
            colorCategoricalColumn = categoricalColumns[0]
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = colorCategoricalColumn)            
           # colorList = self.sourceData.colorManager.getNColorsByCurrentColorMap(N=len(groupBy.keys()))
            colors,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            for groupName, groupData in data.groupby(by=categoricalColumns[0],sort=False):
                quantiles = np.nanquantile(groupData[numericColumns],[0,0.25,0.5,0.75,1],axis=0)
                xValues = np.arange(len(numericColumns))
                plotData[0].append({"quantiles":quantiles,"xValues":xValues,"color":colors[groupName]})
                minQGroup, maxQGroup = np.nanmin(quantiles[0,:]),np.nanmax(quantiles[-1,:])
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
            axisPositions = getAxisPosition(splitCategories.size,maxCol=self.maxColumns)
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
                    minQGroup, maxQGroup = np.nanmin(quantiles[0,:]),np.nanmax(quantiles[-1,:])
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
                axisTitles[n] = "{}: {}".format(categoricalColumns[1],axisName)
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

    def getHistogramProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        #get raw data
        patches = {}
        axisLimits = {}
        axisLabels = {}
        hoverData = {}

        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        subplotBorders = dict(wspace=0.175, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        axisPositions = getAxisPosition(len(numericColumns),maxCol=self.maxColumns)
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

    def getViolinProps(self,dataID,numericColumns, categoricalColumns,*args,**kwargs):
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
        xWidth, axisLabels, axisLimits, axisTitles, groupNames, verticalLines, groupedPlotData  = calculatePositions(
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
            
            plotData["showmedians"] = self.sourceData.parent.config.getParam("violin.show.means")
            plotData["positions"] = violinPositions[n]
            plotData["showextrema"] = self.sourceData.parent.config.getParam("violin.show.extrema")
            plotData["points"] = self.sourceData.parent.config.getParam("violin.points")
            #plotData["quantiles"] = [0,0.1,0.2,0.5,0.7,0.9]
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

    def getBoxplotProps(self, dataID, numericColumns, categoricalColumns,*args,**kwargs):
        ""
        
        subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        if len(categoricalColumns) > 3:
            splitByCats = False
        else:
            splitByCats = self.sourceData.parent.config.getParam("boxplot.split.data.on.category")

        plotCalcData, axisPositions, boxPositions, tickPositions, tickLabels, colorGroups, \
            faceColors, colorCategoricalColumn, xWidth, axisLabels, axisLimits, \
                axisTitles, groupNames, verticalLines, groupedPlotData  = calculatePositions(dataID,
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
            plotData["capprops"] = {"linewidth":rcParams["boxplot.whiskerprops.linewidth"]} 
            filteredData[n] = plotData

        
        #get data to display



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
                "groupedPlotData" : groupedPlotData,
                #"tooltipsTexts" : texts,
                "dataID":dataID}}
        
    def getPointplotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        """
        If categoricalColmns == 0 
        


        return general  line2Ds
        """

        scaleXAxis = self.sourceData.parent.config.getParam("scale.numeric.x.axis")
        splitString = self.sourceData.parent.config.getParam("split.string.x.category")
        splitIndex = self.sourceData.parent.config.getParam("split.string.index")
        faceAndLineColorSame = self.sourceData.parent.config.getParam("pointplot.line.marker.same.color")
        lineWidth = self.sourceData.parent.config.getParam("pointplot.line.width")
        errorColorAsLineColor = self.sourceData.parent.config.getParam("pointplot.error.bar.color.as.line")
        errorEdgeColorAsLineColor = self.sourceData.parent.config.getParam("pointplot.edgecolor.as.line")
        errorLineWidth = self.sourceData.parent.config.getParam("pointplot.error.line.width")
        axisLimits = {}
        tickLabels = {}
        tickPositions = {}
        axisLabels = {}
        hoverData = {}
        #
        lineKwargs = OrderedDict()
        errorKwargs = OrderedDict()
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        markerSize = self.sourceData.parent.config.getParam("pointplot.marker.size")

        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        if len(categoricalColumns) == 0:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)
            colorList = self.sourceData.colorManager.getNColorsByCurrentColorMap(N=len(numericColumns))
            
            hoverData[0] = rawData

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
                    line2D["color"] = colorList[m] if faceAndLineColorSame else "black"
                    line2D["linewidth"] = lineWidth
                    line2D["markerfacecolor"] = colorList[m]
                    line2D["markeredgecolor"] = colorList[m] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = markerSize
                    line2DKwargs.append(line2D)

                    error2D["x"] = [m]
                    error2D["y"] = [columnMeans[m]]
                    error2D["yerr"] = [errorValues[m]]
                    error2D["elinewidth"] = errorLineWidth
                    
                    error2D["ecolor"] = colorList[m] if errorColorAsLineColor else rcParams["patch.edgecolor"]
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
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)
            #get unique categories
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(nColorCats)]
            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns)
            colorCategoricalColumn = categoricalColumns[0]
            hoverData[0] = rawData 
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
                    line2D["color"] = colors[category] if faceAndLineColorSame else "black"
                    line2D["linewidth"] = lineWidth
                    line2D["markerfacecolor"] = colors[category]
                    line2D["markeredgecolor"] = colors[category] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = markerSize
                    line2DKwargs.append(line2D)

                    error2D["x"] = [tickPos[m]]
                    error2D["y"] = [columnMeans[m]]
                    error2D["yerr"] = [errorValues[m]]
                    error2D["elinewidth"] = errorLineWidth
                    
                    error2D["ecolor"] = colors[category] if errorColorAsLineColor else rcParams["patch.edgecolor"]
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
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)

            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(nColorCats)]

            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns,as_index=False)

            meanErrorData, minValue, maxValue, maxErrorValue = self.getCIForGroupby(groupByCatColumn,numericColumns)
            hoverData[0] = rawData

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
                    line2D["color"] = colors[category] if faceAndLineColorSame else "black"
                    line2D["linestyle"] = "-"
                    line2D["linewidth"] = lineWidth
                    line2D["aa"] = True
                    line2D["markerfacecolor"] = colors[category]
                    line2D["markeredgecolor"] = colors[category] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = markerSize
                    line2DKwargs.append(line2D)

                    error2D["x"] = np.arange(len(numericColumns))#line2D["xdata"]
                    error2D["y"] = np.array([meanErrorData[category][numColumn]["value"] for numColumn in numericColumns])#groupMeans.loc[category ,:].values#line2D["ydata"]
                    error2D["yerr"] = np.array([meanErrorData[category][numColumn]["error"] for numColumn in numericColumns])
                    error2D["elinewidth"] = errorLineWidth
                    
                    error2D["ecolor"] = colors[category] if errorColorAsLineColor else rcParams["patch.edgecolor"]
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
            axisPositions = getAxisPosition(nNumCols,maxCol=self.maxColumns)

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
                    line2D["color"] = colors[category] if faceAndLineColorSame else "black"
                    line2D["linestyle"] = "-"
                    line2D["linewidth"] = lineWidth
                    line2D["markerfacecolor"] = colors[category]
                    line2D["markeredgecolor"] = colors[category] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = markerSize
                    line2DKwargs.append(line2D)

                    error2D["x"] = x
                    error2D["y"] = y
                    error2D["yerr"] = e
                    error2D["elinewidth"] = errorLineWidth
                    
                    error2D["ecolor"] = colors[category] if errorColorAsLineColor else rcParams["patch.edgecolor"]
                    line2DErrorKwargs.append(error2D)
                lineKwargs[n] = line2DKwargs
                errorKwargs[n] = line2DErrorKwargs
                axisLabels[n] = {"x":categoricalColumns[0],"y":numColumn}
                hoverData[n] = rawData[numColumn]
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
            axisPositions = getAxisPosition(n = axisCategories.size *  NNumCol, maxCol = axisCategories.size)
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
            
            axisGroupby = rawData.groupby(categoricalColumns[2],sort=False)
            
            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns,as_index=False)
            meanErrorData, minValue, maxValue, maxErrorValue = self.getCIForGroupby(groupByCatColumn,numericColumns)
            
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
                        
                    tickPositions[nAxis] = tickPos
                    minX, maxX = np.min(tickPos), np.max(tickPos)
                    distance = np.sqrt(minX**2 + maxX**2) * 0.05
                    hoverData[nAxis] = axisCatData
                    
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
                            line2D["color"] =  colors[colorCategory] if faceAndLineColorSame else "black"
                            line2D["linestyle"] = "-"
                            line2D["linewidth"] = lineWidth
                            line2D["markerfacecolor"] = colors[colorCategory]
                            line2D["markeredgecolor"] = colors[colorCategory] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                            line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                            line2D["markersize"] = np.sqrt(self.sourceData.parent.config.getParam("scatterSize"))
                            line2DKwargs.append(line2D)

                            error2D["x"] = x
                            error2D["y"] = y
                            error2D["yerr"] = e
                            error2D["elinewidth"] = errorLineWidth
                            
                            error2D["ecolor"] = colors[colorCategory] if errorColorAsLineColor else rcParams["patch.edgecolor"]
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
                "hoverData" : hoverData,
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
            

    def getProteinPeptideProps(self, proteinDataID, peptideDataID, proteinColumnNames, peptideColumnNames):
        ""



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



        return {"newPlot":True,"data":{"plotData":boxplotProps,"outliers":outliers,"xAxisLabels":[xAxisLabels],"plotType":"boxplot"}}


    def _getBoxProps(self,arr):
        ""
        quantiles = np.nanquantile(arr,q=[0.25, 0.5, 0.75], axis=0)
        IQR = quantiles[2,:] - quantiles[0,:]
        upperBound = quantiles[2,:] + 1.5 * IQR
        lowerBound = quantiles[0,:] - 1.5 * IQR

        return quantiles, IQR, upperBound, lowerBound

    def getCorrmatrixProps(self,dataID,numericColumns, categoricalColumns, figureSize = None, *args,**kwargs):
        ""
        return self.getHeatmapProps(dataID,numericColumns,categoricalColumns, True, figureSize=figureSize,*args,**kwargs)

    def getHeatmapProps(self,dataID, numericColumns, categoricalColumns, corrMatrix = False, groupingName = None,figureSize = None,*args,**kwargs):
        ""
        rowMaxD = None
        colMaxD = None
        rowClustNumber = None
        rowLinkage = None
        ytickLabels = []
        ytickPosition = []
        rowLineCollection = []
        colLineCollection = []
        groupingRectangles = []
        rectangles = []
        clusterColorMap = {}
        
        #display grouping setting
        displayGrouping = self.sourceData.parent.config.getParam("hclust.display.grouping")
        
        colorsByColumnNames = OrderedDict()
        colorsByColumnNamesFiltered = OrderedDict()

        if displayGrouping and groupingName is not None:
            if isinstance(groupingName,str):
                groupingName = [groupingName]
            
            for gN in groupingName:
               
                colorsByColumnNames[gN] = self.sourceData.parent.grouping.getColorsForGroupMembers(gN)

                colorsByColumnNamesFiltered[gN] = dict([(k,v) for k,v in colorsByColumnNames[gN].items() if k in numericColumns])
                if len(colorsByColumnNamesFiltered[gN]) == 0: #if no column in grouping - dont display grouping
                   del colorsByColumnNamesFiltered[gN]
               
        else:
          #  grouping = {}
            displayGrouping = False

        #cluster rows 
        rowMetric = self.sourceData.statCenter.rowMetric
        rowMethod = self.sourceData.statCenter.rowMethod
         #cluster columns
        columnMetric = self.sourceData.statCenter.columnMetric
        columnMethod = self.sourceData.statCenter.columnMethod

        try:
            if corrMatrix:

                data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"].corr(method = self.corrMatrixMethod)
                if data.dropna().empty:
                    return getMessageProps("Error..","Correlation matrix calculations resulted in a complete NaN matrix.")
            else:
                if rowMetric == "nanEuclidean":
                    nanThreshold = self.sourceData.parent.config.getParam("min.required.valid.values")
                    if nanThreshold > len(numericColumns):
                        nanThreshold = len(numericColumns)
                    data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"].dropna(thresh=nanThreshold)
                    #remove no deviation data (Same value)
                    data = data.loc[data.std(axis=1) != 0,:]
                    rowMetric = "nanEuclidean" 
                    if columnMetric != "None":
                        columnMetric = "nanEuclidean"
                else:
                    if (rowMetric != "None" and rowMethod != "None") or  (columnMetric != "None" and columnMethod != "None"):
                        data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"].dropna()
                        #remove no deviation data (Same value)
                        data = data.loc[data.std(axis=1) != 0,:]
                    else:
                        data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]
                        #if no clustering applied, we can keep all values (e.g. just display)
                        
        
            #nRows, nCols = data.shape
            rawIndex = data.index
            rowXLimit, rowYLimit, rowLineCollection = None, None, None
            colXLimit, colYLimit, colLineCollection = None, None, None


            axisDict = self.getClusterAxes(numericColumns, 
                    figureSize = figureSize,
                    corrMatrix=corrMatrix, 
                    rowOn = rowMetric != "None" and rowMethod != "None", 
                    columnOn =  columnMethod != "None" and columnMetric != "None",
                    numberOfGroups=len(colorsByColumnNamesFiltered),
                    displayGrouping = displayGrouping 
                    )
            

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
        
            groupingRectangles = []
            if groupingName is not None and displayGrouping and len(colorsByColumnNamesFiltered) > 0:
                for ii, (gN, colorsByColumnNamesForGn) in enumerate(colorsByColumnNamesFiltered.items()):
                    groupingGNRectangles = [
                        Rectangle(
                            xy = (0+10*n,-0.5+ii),
                            width = 10, 
                            height = 1,
                            faceColor =  colorsByColumnNamesForGn[columnName] if columnName in  colorsByColumnNamesForGn else self.sourceData.colorManager.nanColor,
                            alpha = 0.75) for n,columnName in enumerate(data.columns.values)]
                    groupingRectangles.extend(groupingGNRectangles)
        except Exception as e:
            print(e)
            return {}
       
     #   print(axisDict)
        
        groupingAxLabels = {"tickLabels":groupingName,"tickPosition":np.arange(len(groupingName))} if groupingName is not None else {}
        #print(groupingName)
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
                    "axColumnGrouping":{"x":[0,len(numericColumns)*10],"y":[-0.5,len(colorsByColumnNamesFiltered)-0.5]}
                    },
                "tickLabels" : {"rowDendrogram":{"tickLabels": ytickLabels,"tickPosition": ytickPosition},
                                "axColumnGrouping":groupingAxLabels
                                },
                "absoluteAxisPositions" : axisDict,
                "clusterRectangles": rectangles,
                "groupingRectangles" : groupingRectangles,
                "dataID":dataID,
                "columnNames":numericColumns,
                "params" : [
                    ("Type","Hclust" if not corrMatrix else "Corrmarix"),
                    ("rowMetric",rowMetric),
                    ("rowMethod",rowMethod),
                    ("columnMetric",columnMetric),
                    ("columnMethod",columnMethod),
                    ("nanFilter","all NaNs filtered" if rowMetric != "nanEuclidean" else "minValidValues = {}".format(nanThreshold))]
                    }
                }
    
    def addDendrogram(self,dendrogram,rotate,*args,**kwargs):
        '''
        Idea is from the seaborn package.
        '''
        line_kwargs = dict(linewidths=.4, colors='k')
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

    def getClusterAxes(self, numericColumns, figureSize, corrMatrix=False, rowOn = True,columnOn = True ,numberOfGroups = 0, displayGrouping = False):
        ""
        x0,y0 = 0.04,0.15
        x1,y1 = 0.95,0.95
        labelHeight = 0.15
        width = x1-x0
        if corrMatrix:
            height = y1-y0-labelHeight
        else:
            height = y1-y0
        
        #multWidth = 0.4
        pixelFigureWidth = figureSize["width"]
        pixelFigureHeight = figureSize["height"]
        pixelPerColumn = self.sourceData.parent.config.getParam("pixel.width.per.column")
        maxPixelForHeatmap = (width - 0.1) * pixelFigureWidth #0.1 = min margin
        maxPixelHeightHeatmap = height * pixelFigureHeight
        
        if not corrMatrix:
            heightMain = height * 0.8 
     
        
        if maxPixelForHeatmap < len(numericColumns) * pixelPerColumn:
            clusterMapWidth = 0.75
            widthInPixel = maxPixelForHeatmap
        else:
            widthInPixel = len(numericColumns) * pixelPerColumn
            clusterMapWidth = 0.75 * widthInPixel/maxPixelForHeatmap
            

        if corrMatrix:
            
            if widthInPixel <= maxPixelHeightHeatmap:
               
                heightInPixel = widthInPixel
                heightMain = height * widthInPixel/maxPixelHeightHeatmap
                
            else:
                widthInPixel = maxPixelHeightHeatmap 
                clusterMapWidth = 0.75 * widthInPixel/maxPixelForHeatmap
                heightMain = height * widthInPixel/maxPixelHeightHeatmap
        
           
          #  y0 = height - heightMain - 0.1# 0.1#(maxPixelHeightHeatmap - heightInPixel) / maxPixelHeightHeatmap
            
            y0 = 1 - heightMain - 0.22 #bit of margin
            
        #widthPer pixelWidth / len(numericColumns)
        #correctHeight = 1
        # emperically determined to give almost equal width independent of number of columns 
			
        
		
        #clusterMapWidth = 10 * len(numericColumns)
        rowDendroWidth = width * 0.08 if rowOn else 0
        if columnOn:
            if rowDendroWidth > 0:
                columnDendroHeight = rowDendroWidth
            else:
                columnDendroHeight = width * 0.13
        else: 
            columnDendroHeight = 0
       
        if numberOfGroups > 0 and displayGrouping:
           
           groupingAxHeight = width * 0.013 * numberOfGroups
           addSpacingForGroupAx = width * 0.013 * 0.75
           
        else:
            groupingAxHeight = 0
            addSpacingForGroupAx = 0
				

            
        axisDict = dict() 

        axisDict["axRowDendro"] = [x0,
                                    y0,
                                    rowDendroWidth,
                                    heightMain]
                                    
        axisDict["axColumnDendro"] = [x0 + rowDendroWidth, 
                                    y0+heightMain + groupingAxHeight + addSpacingForGroupAx,
                                    clusterMapWidth,
                                    columnDendroHeight ]

        if groupingAxHeight > 0 and displayGrouping:

            axisDict["axColumnGrouping"] = [x0 + rowDendroWidth, 
                                    y0+heightMain + addSpacingForGroupAx,
                                    clusterMapWidth,
                                    groupingAxHeight]

        axisDict["axClusterMap"] = [x0+rowDendroWidth,
                                    y0,
                                    clusterMapWidth,
                                    heightMain]
        
        axisDict["axLabelColor"] =  [x0+rowDendroWidth+clusterMapWidth+clusterMapWidth/2/len(numericColumns), #add margin
                                    y0,
                                    width-clusterMapWidth-rowDendroWidth if clusterMapWidth < 0.75 else 0.1,
                                    heightMain]		
        
        axisDict["axColormap"] =    [x0,
                                    y1-0.15,
                                    0.02,
                                    0.1]	
        
        return axisDict

    def getDimRedProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        return {"data":{}}


    def getForestplotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        tickLabels = {}
        tickPositions = {}
        axisLabels = {}
        axisLimits = {}

        axisLabelConverter = {
                            "oddsRatio":"Odds ratio",
                            "logOddsRatio":"log (Odds ratio)",
                            "riskRatio": "Risk ratio"
                            }

        if not (len(numericColumns) > 0 and len(categoricalColumns) > 0):
            return getMessageProps("Error ..","")
        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        plotData = OrderedDict() 

        faceColors = dict()
        useColorMap = self.sourceData.parent.config.getParam("forest.plot.use.colormap")
        defaultColor = self.sourceData.parent.config.getParam("forest.plot.marker.color")
        colorGroupsData = pd.DataFrame()
        internalIDs = []

        if  self.sourceData.parent.config.getParam("forest.plot.calculated.data"):
            axisPositions = getAxisPosition(1) 
            #categorical columns -> names of variables -> yticks
            categoricalData = data[categoricalColumns[0]].values.flatten()
            colors, _ = self.sourceData.colorManager.createColorMapDict(categoricalData, as_hex=True)
            
            if useColorMap:
                colorGroupsData["color"] = colors.values()
            else:
                colorGroupsData["color"] = [defaultColor] * categoricalData.size
            colorGroupsData["group"] = categoricalData
            tickLabels[0] = categoricalData
            tickPositions[0] = np.arange(categoricalData.size)
            axisLimits[0] = {"yLimit":[-0.5,categoricalData.size-0.5],"xLimit": None}
            axisLabels[0] = {
                        "x":numericColumns[0],
                        "y":"Variables"
                        }
            
            faceColors[0] = {}
            if len(numericColumns) == 3:
                    meanColumName,ci_lowerColumName, ci_upperColumName = numericColumns
            else:
                    meanColumName = numericColumns[0]
                    ci_lowerColumName = None
                    ci_upperColumName = None

            plotData[0] = OrderedDict() 
            for m,idx in enumerate(data.index):
                
                meanValue = data.loc[idx,meanColumName]  
                rowName = categoricalData[m]
                ci_lower = data.loc[idx,ci_lowerColumName]  if ci_lowerColumName is not None else np.nan
                ci_upper = data.loc[idx,ci_upperColumName] if ci_upperColumName is not None else np.nan
                internalID = getRandomString()
                plotData[0][rowName] = {
                            "ratio"  :   meanValue,
                            "ratioCI":   (ci_lower, ci_upper),
                            "yPosition":  m,
                            "internalID" : internalID
                        }
                internalIDs.append(internalID)
                if not useColorMap:
                    faceColors[0][rowName] = defaultColor
                else:
                    faceColors[0][rowName] = colors[rowName]
        else:

            axisPositions = getAxisPosition(len(numericColumns))
            
            colors, layer = self.sourceData.colorManager.createColorMapDict(categoricalColumns, as_hex=True)
            if useColorMap:
                colorGroupsData["color"] = colors.values()
            else:
                colorGroupsData["color"] = [defaultColor] * len(categoricalColumns)
            colorGroupsData["group"] = categoricalColumns
            for n,numericColumn in enumerate(numericColumns):
                plotData[n] = OrderedDict()
                for m,categoricalColumn in enumerate(categoricalColumns):

                    ct=pd.crosstab(data[numericColumn],data[categoricalColumn])
                    table = Table2x2(ct.values)
                    internalID = getRandomString()
                    
                    plotData[n][categoricalColumn] = {
                                        "oddsRatio"          :   table.oddsratio,
                                        "oddsRatioCI"        :   table.oddsratio_confint(),
                                        "oddsRatioPValue"    :   table.oddsratio_pvalue(),
                                        "logOddsRatio"       :   table.log_oddsratio,
                                        "logOddsRatioCI"     :   table.log_oddsratio_confint(),
                                        "logOddsRatioP"      :   table.log_oddsratio_pvalue(),
                                        "riskRatio"          :   table.riskratio,
                                        "riskRatioCI"        :   table.riskratio_confint(),
                                        "yPosition"          :   m,
                                        "internalID"         :   internalID
                                        }
                    internalIDs.append(internalID)
                tickLabels[n] = categoricalColumns
                axisLabels[n] = {
                        "x":axisLabelConverter[self.sourceData.parent.config.getParam("forest.plot.cont.table.ratio")],
                        "y":"Variables"
                        }
                tickPositions[n] = np.arange(len(categoricalColumns))
                axisLimits[n] = {"yLimit":[-0.5,len(categoricalColumns)-0.5],"xLimit": None}
                
                if not useColorMap:
                    faceColors[n][categoricalColumn] = defaultColor
                else:
                    faceColors[n][categoricalColumn] = colors[rowName]

        colorGroupsData["internalID"] = internalIDs

            
    
        return {"data":{
                    "plotData" : plotData,
                    "axisPositions" : axisPositions,
                    "tickLabels"    : tickLabels,
                    "axisLabels"    : axisLabels,
                    "tickPositions" : tickPositions,
                    "axisLimits"    : axisLimits,
                    "faceColors"    : faceColors,
                    "dataColorGroups": colorGroupsData
                    }
                }
        

    def getPCAProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        subplotBorders = dict(wspace=0.30, hspace = 0.30,bottom=0.15,right=0.95,top=0.95)
        #data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]
        checkPassed, driverResult, eigVectors  = self.sourceData.statCenter.runPCA(dataID,numericColumns, initGraph = True, n_components = 3)
        if not checkPassed:
            return getMessageProps("Error ..","Filtering resulted in an invalid data frame.")

        columnPairs = [("Component_01","Component_02"), ("Component_02","Component_03")]
        #print(result)
      #  print(eigVectors)
        axisPostions = dict([(n,[2,2,n+1]) for n in range(4)])
    

        return {"data":{"plotData":{"projection":driverResult,"eigV":eigVectors},
                "axisPositions":axisPostions,
                "numericColumns":numericColumns,
                "subplotBorders":subplotBorders,
                "columnPairs":columnPairs,
                "dataID":dataID}}

    def _addIndexToData(self,data,numericColumns):
        ""
        numericColumnPairs = [("Index ({:02d})".format(n+1),numColumn) for n,numColumn in enumerate(numericColumns)]
        for indexName, numColumn in numericColumnPairs:
                           
            index = data.sort_values(by=numColumn, ascending = self.indexSort == "ascending").index
            data = data.join(pd.Series(np.arange(index.size),index=index, name = indexName))

        return data,numericColumnPairs

    def getScatterProps(self,dataID, numericColumns, categoricalColumns,*args,**kwargs):
        ""
        
        try:
            axisTitles = {} 

            if len(categoricalColumns) == 0:
                data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]

                if not self.plotAgainstIndex and len(numericColumns) > 1:
                    numericColumnPairs = list(zip(numericColumns[0::2], numericColumns[1::2]))

                else:
                    data, numericColumnPairs = self._addIndexToData(data,numericColumns)
                        
                nrows,ncols,subplotBorders = self._findScatterSubplotProps(numericColumnPairs)
                axisPositions = dict([(n,[nrows,ncols,n+1]) for n in range(len(numericColumnPairs))])
                axisLabels = dict([(n,{"x":x1,"y":x2}) for n, (x1,x2) in enumerate(numericColumnPairs)])

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
                else:
                    data, numericColumnPairs = self._addIndexToData(data,numericColumns)

                if len(categoricalColumns) == 1:
                    numOfAxis = len(numericColumnPairs) * numUniqueCat
                    axisPositions = getAxisPosition(numOfAxis,maxCol=numUniqueCat)
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
                        for nGroup , (groupName, groupData) in enumerate(data.groupby(categoricalColumns[0],sort=False)):

                            if axisID == numUniqueCat:
                                firstAxisRow = False
                            elif firstAxisRow:
                                axisTitles[axisID] = {"title":"{}: {}".format(categoricalColumns[0],groupName),
                                                                "appendWhere":"top",
                                                                "textRotation" : 0}
                            pair = []
                            for columnName in numCols:
                                columnKey = "{}:{}:({})".format(groupName,categoricalColumns[0],columnName)
                                plotData.loc[groupData.index,columnKey] = groupData[columnName]
                                pair.append(columnKey)
                            plotNumericPairs.append(tuple(pair))

                            axisLabels[axisID] = {"x":numCols[0],"y":numCols[1] if nGroup == 0 else ""}  
                            axisID += 1  
                    
                   # axisPositions = getAxisPosition(len(numericColumnPairs),nCols=numUniqueCat)
                elif len(categoricalColumns) == 2:
                    uniqueValuesCat2 = data[categoricalColumns[1]].unique() 
                    numUniqueCat2 = uniqueValuesCat2.size

                    numOfAxis = numUniqueCat * numUniqueCat2 * len(numericColumnPairs)
                    axisPositions = getAxisPosition(numOfAxis,maxCol=numUniqueCat)
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
                            
                            for nGroup,  (groupName, groupData) in enumerate(cat2data.groupby(categoricalColumns[0], sort=False)):
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

                                    axisLabels[axisID] = {"x":numCols[0],"y":numCols[1] if nGroup == 0 else ""}  
                                    axisID += 1  
                else:
                    return getMessageProps("Error ..","For Scatter plots only two categorical columns are considered. You can use color, size and marker highlights.")
                
                numericColumnPairs = plotNumericPairs
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

    def _buildColumnPairs(self,numericColumns):
        ""
        numNumericColumns = len(numericColumns)
        #np.zeros(shape=(numNumericColumns,numNumericColumns))
        columnPairs = {}
        n = 0
        for nRow in range(numNumericColumns):
            for nCol in range(numNumericColumns):
                yValue = numericColumns[nRow]
                xValue = numericColumns[nCol]
                columnPairs[n] = tuple([xValue,yValue])
                n += 1

        return columnPairs
        
    def _build2DHistogramData(self, axisInts, columnPairs, data, histogramData):
        #get pcolormesh for column pairs
        ""
        for nA in axisInts.flatten():
            numericPairs = list(columnPairs[nA])
            
            XY  = data[numericPairs].dropna().values
            
            H, xedges, yedges = np.histogram2d(XY[:,0], XY[:,1], bins=(25, 25))
            meshX, meshY = np.meshgrid(xedges, yedges)
            histogramData[nA] = {
                        "H":H,
                        "xedges":xedges,
                        "yedges":yedges,
                        "meshX":meshX,
                        "meshY":meshY
                        }
        return histogramData

    def _buildScatterPairs(self,axisInts,columnPairs,scatterColumnPairs):
        ""
        for nA in axisInts.flatten():
            scatterColumnPairs[nA] = columnPairs[nA]
        return scatterColumnPairs

    def _findBwByGridSearch(self,x,minBW,maxBW,numCrossValidations=5,kernel="gaussian"):
        ""
        grid = GridSearchCV(KernelDensity(kernel=kernel,algorithm='ball_tree'),
                    {'bandwidth': np.linspace(minBW, maxBW, 30)},
                    cv=numCrossValidations) # 5-fold cross-validation
        grid.fit(x)
        
        return grid.best_params_["bandwidth"]

    def _buildKdeData(self,axisInts,data,numericColumns,kdeData):
        ""

        minMaxValues = np.nanquantile(data[numericColumns].values, q = [0.0,1.0],axis=0)
        bw = self.sourceData.parent.config.getParam("multi.scatter.kde.bandwidth")
        kernel = self.sourceData.parent.config.getParam("multi.scatter.kde.kernel")
        bwGridSearch = self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search")
        logDensity = self.sourceData.parent.config.getParam("multi.scatter.kde.log.density")
        gridMin = self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search.min")
        gridMax = self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search.max")
        numCVs = self.sourceData.parent.config.getParam("multi.scatter.kde.grid.n.cross.val")
        for n,nAx in enumerate(axisInts):
            x = data[numericColumns[n]].dropna().values.reshape(-1,1)
        
            #find x limits
            minValue,maxValue = minMaxValues[:,n]
            offSet = (maxValue - minValue) * 0.1
            minValue -= offSet
            maxValue += offSet

            #caluclate kde and get y values
            xx = np.linspace(minValue,maxValue,num=500).reshape(-1,1)
            if bwGridSearch:
                
                bw = self._findBwByGridSearch(x,gridMin,gridMax,numCVs)
           
           
            kde = KernelDensity(bandwidth=bw, kernel=kernel, algorithm='ball_tree').fit(x)
            y = kde.score_samples(xx)
            if not logDensity:
                y = np.exp(y)
           
            #find y limits
            yMax = np.max(y)
            yMin = np.min(y)
            yMax += 0.15*np.sqrt((yMax-yMin)**2)
            #add kdea data - key is axis int
            kdeData[nAx] = {"xx" : xx, "x" : x, "yKde" : y ,"xLimit":(minValue,maxValue),"yLimit":(yMin,yMax)}

        return kdeData

    def _buildLabelData(self,axisInts, numericColumns, groupColorDict, labelData):
        ""
        for numericColumn, nAx in zip(numericColumns,axisInts):
            
            tKwargs = {
                "color" : groupColorDict[numericColumn] if numericColumn in groupColorDict else "black",
                "s" : numericColumn,
                "horizontalalignment" : 'center',
                "verticalalignment" : 'center',
                "x" : 0.5,
                "y" : 0.5
                }
            
            labelData[nAx] = tKwargs
            
        
        return labelData

    def _buildCorrelationLabelData(self,axisInts, columnPairs, labelData, corrMatrix  = None, spearmanCorrMatrix=None):
        ""
        if corrMatrix is None and spearmanCorrMatrix is None:
            return labelData

        for nA in axisInts.flatten():
            colA, colB = columnPairs[nA]
            pearCorr = "" if corrMatrix is None else "r = {}\n".format(round(corrMatrix.loc[colA, colB],2))
            spearCorr = "" if spearmanCorrMatrix is None else "rho = {}".format(round(spearmanCorrMatrix.loc[colA, colB],2))

            tKwargs = {
                "color" : "black",
                "s" : pearCorr+spearCorr,
                "horizontalalignment" : 'left',
                "verticalalignment" : 'top',
                "x" : 0.02,
                "y" : 0.95
                }
            labelData[nA] = tKwargs

        return labelData

 
    def getScatterCorrMatrix(self,dataID, numericColumns, categoricalColumns,*args,**kwargs):
        """
        Provide required information to plot a scatter matrix. 
        Several options are taken from the config settings. 
        """
        #init dicts
        histogramData = dict() 
        kdeData = dict()
        scatterColumnPairs = dict()
        colorDict = dict() 
        groupColorDict = dict()
        labelData = dict()
        lowessFit,linregressFit = dict(), dict()
        backgroundColors = dict()
        backgroundColorHex = None
        categoryIndexMatch = None

        config = self.sourceData.parent.config

        topPlotType = config.getParam("multi.scatter.top.right.type")
        bottomPlotType = config.getParam("multi.scatter.bottom.left.type")
        diagPlotType = config.getParam("multi.scatter.diag.type")

        #add linear regression
        addLinReg = config.getParam("multi.scatter.add.linregress")
        addLowess = config.getParam("multi.scatter.add.lowess")
        lowessFrac = config.getParam("multi.scatter.lowess.frac")
        lowessIt = config.getParam("multi.scatter.lowess.it")
        lowessDelta = config.getParam("multi.scatter.lowess.delta")

        #correlation
        addToPlotType = config.getParam("multi.scatter.add.corr.coeff.to")
        addPearson = config.getParam("multi.scatter.add.pearson")
        addSpearman = config.getParam("multi.scatter.add.spearman")
        colorBackground = config.getParam("multi.scatter.background.by.pearson")
        backgroundCmap = config.getParam("multi.scatter.background.colormap")
        colorScale = config.getParam("multi.scatter.background.colorScale")
        #get data
        columnSorting = config.getParam("multi.scatter.sort.columns.by")

        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        corrmatrix = data[numericColumns].corr()
        spearmanCorrMatrix =  data[numericColumns].corr(method="spearman") if addSpearman else None
        nNumCols = len(numericColumns)
        
        if colorBackground:
            if colorScale == "dataMinMax":
                vmin, vmax = None, None
            else:
                vmin,vmax = [float(x) for x in colorScale.split(",")]
            backgroundColorMapper = self.sourceData.colorManager.matchColorsToValues(corrmatrix.values,backgroundCmap,vmin,vmax)
      
            backgroundColorHex = pd.DataFrame(
                    np.array([to_hex(c)  for cs in backgroundColorMapper[0] for c in cs ]).reshape((nNumCols,nNumCols)),
                    columns = numericColumns, index = numericColumns)
            
        if  columnSorting != "None":
            if columnSorting == "Grouping" and self.sourceData.parent.grouping.groupingExists():
                groupingName = self.sourceData.parent.grouping.getCurrentGroupingName()
                columnNames = self.sourceData.parent.grouping.getColumnNames(groupingName)
                columnsFoundInGrouping = [colName for colName in columnNames if colName in numericColumns]
                numericColumns = columnsFoundInGrouping + [colName for colName in numericColumns if colName not in columnsFoundInGrouping]
                groupColorDict = self.sourceData.parent.grouping.getColorsForGroupMembers(groupingName)
            if columnSorting == "Hierarch. Clustering":
                columnLinkage, colMaxD = self.sourceData.statCenter.clusterData(np.transpose(data[numericColumns].dropna(axis=0).values))
                if columnLinkage is None:
                    return getMessageProps("Error..","There was an error in clustering columns.")
                Z_col = sch.dendrogram(columnLinkage, orientation='top', color_threshold = colMaxD, 
                                    leaf_rotation=90, ax = None, no_plot=True)
                
                data = data.iloc[:,Z_col['leaves']]
                numericColumns = [numericColumns[idx] for idx in Z_col['leaves']]
                

        nAxis = int(nNumCols ** 2)
        axisPositions = getAxisPosition(nAxis, maxCol=nNumCols)
        subplotBorders = dict(wspace=0.1, hspace = 0.1, bottom=0.15,right=0.95,top=0.95)
        #numericColumnPairs =  list(itertools.product(numericColumns,2))  #list(zip(numericColumns[0::2], numericColumns[1::2]))
        columnPairs = self._buildColumnPairs(numericColumns)

        if len(categoricalColumns) > 0:
            colorDict = self.getColorGroupsDataForScatter(dataID,pd.Series(categoricalColumns))
            
            colorGroupsData = colorDict["colorGroupData"]
            colorTitle = colorDict["title"]
            propsData = colorDict["propsData"]
            categoryIndexMatch = colorDict["categoryIndexMatch"]
        else:
            colorGroupsData = pd.DataFrame()
            if len(categoricalColumns) == 0:
                colorGroupsData["color"] = [self.sourceData.colorManager.nanColor]
                colorGroupsData["group"] = [""]
            colorTitle = "Scatter points"
            propsData = pd.DataFrame(index = data.index, columns = ["color","layer"])
            propsData["color"] = self.sourceData.colorManager.nanColor
            propsData["layer"] = 0
            # else:
            #     colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            #     nColorCats = colorCategories.size
            #     colorDict,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)

            #     colorGroupsData["color"] = list(colorDict.values())
            #     colorGroupsData["group"] = list(colorDict.keys())
            #     colorGroupsData["internalID"] = [getRandomString() for _ in range(nColorCats)]

        axisInts = np.array(np.arange(nAxis)).reshape((nNumCols,nNumCols))
        axisIntsTop = np.triu(axisInts)
        axisIntsBottom = np.tril(axisInts)
        np.fill_diagonal(axisIntsTop,0)
        np.fill_diagonal(axisIntsBottom,0)

        #get ints for scatter by using the indicies found by tril/diagonal is zero 
        axisIntsTop = axisIntsTop.flatten()
        axisIntsTop = axisIntsTop[axisIntsTop > 0]

        axisIntsBottom = axisIntsBottom.flatten()
        axisIntsBottom = axisIntsBottom[axisIntsBottom > 0]

        axisIntsDiagonal = np.diag(axisInts)

        NProcesses = self.sourceData.parent.config.getParam("n.processes.multiprocessing")

        kdeKwargs = {"bw" :  self.sourceData.parent.config.getParam("multi.scatter.kde.bandwidth"),
                    "kernel" :  self.sourceData.parent.config.getParam("multi.scatter.kde.kernel"),
                    "bwGridSearch" :  self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search"),
                    "logDensity" :  self.sourceData.parent.config.getParam("multi.scatter.kde.log.density"),
                    "gridMin" :  self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search.min"),
                    "gridMax" :  self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search.max"),
                    "numCVs" : self.sourceData.parent.config.getParam("multi.scatter.kde.grid.n.cross.val")}
        lowessKwargs = {"frac":lowessFrac , "it" : lowessIt, "delta" : lowessDelta}
                            # if addLowess:
                            #     lowessFit[nA] = self.sourceData.statCenter.runLowess(dataID,numericPairs,frac=lowessFrac , it = lowessIt, delta = lowessDelta)
        #multi processing to calculate the axisInts (topRight,bottomleft,and diagonal simult.)      
        #
        # 

        if nNumCols > 5:  
            with Pool(NProcesses) as p:
                rPool = p.starmap(buildScatterMatrix,[(data,
                                            numericColumns,
                                            plotType, 
                                            axisInts,
                                            backgroundColorHex,
                                            scatterColumnPairs,
                                            colorBackground,
                                            addLinReg,
                                            addLowess,
                                            columnPairs,
                                            kdeKwargs,
                                            groupColorDict,
                                            addToPlotType,
                                            addPearson,
                                            addSpearman,
                                            corrmatrix,
                                            spearmanCorrMatrix,
                                            lowessKwargs) for plotType, axisInts in zip([topPlotType,bottomPlotType,diagPlotType],
                                                                                                    [axisIntsTop,axisIntsBottom,axisIntsDiagonal])])
           # print(rPool) #output of pool - merged:(scatterColumnPairs,backgroundColors,linregressFit,histogramData,kdeData,labelData,lowessData)
        else:
            rPool = [buildScatterMatrix(data,
                                            numericColumns,
                                            plotType, 
                                            axisInts,
                                            backgroundColorHex,
                                            scatterColumnPairs,
                                            colorBackground,
                                            addLinReg,
                                            addLowess,
                                            columnPairs,
                                            kdeKwargs,
                                            groupColorDict,
                                            addToPlotType,
                                            addPearson,
                                            addSpearman,
                                            corrmatrix,
                                            spearmanCorrMatrix,
                                            lowessKwargs) for plotType,axisInts in zip([topPlotType,bottomPlotType,diagPlotType],
                                                                                                    [axisIntsTop,axisIntsBottom,axisIntsDiagonal])]
        scatterColumnPairs = {k: v for d in rPool for k, v in d[0].items()}
        backgroundColors = {k: v for d in rPool for k, v in d[1].items()}
        linregressFit = {k: v for d in rPool for k, v in d[2].items()}
        histogramData = {k: v for d in rPool for k, v in d[3].items()}
        kdeData = {k: v for d in rPool for k, v in d[4].items()}
        labelData = {k: v for d in rPool for k, v in d[5].items()}
        lowessFit = {k: v for d in rPool for k, v in d[6].items()}
           
                

        #compile size groups
        sizeGroupsData = pd.DataFrame() 
        sizeGroupsData["size"] = [self.scatterSize]
        sizeGroupsData["group"] = [""]

        return {"data":{
                        "plotData":data,
                        "axisPositions":axisPositions, 
                        "scatterColumnPairs": scatterColumnPairs,
                        "dataColorGroups": colorGroupsData,
                        "dataSizeGroups" : sizeGroupsData,
                        "histogramData" : histogramData,
                        "kdeData" : kdeData,
                        "subplotBorders":subplotBorders,
                        "dataID":dataID,
                        "linRegFit" : linregressFit,
                        "lowessFit": lowessFit,
                        "propsData" : propsData,
                        "colorTitle" : colorTitle,
                        "labelData" : labelData,
                        "backgroundColors" : backgroundColors,
                        "categoryIndexMatch" : categoryIndexMatch
                        }
            }

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

    def getNearestNeighborConnections(self,dataID,numericColumnPairs,numberNearestNeighbors=3):
        ""
        # print(numericColumnPairs)
        # print(np.array(numericColumnPairs))
        columnNames = pd.Series(np.array(numericColumnPairs).flatten()).unique()
        # print(columnNames)
        rawData = self.sourceData.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"]
        # print(rawData)
        lineCollections = {}
        for n,columnPair in enumerate(numericColumnPairs):
            X = rawData[list(columnPair)].dropna().values
           # print(X)
            euclideanDistanceMatrix = squareform(pdist(X, 'euclidean'))

            # select the kNN for each datapoint
            neighbors = np.sort(np.argsort(euclideanDistanceMatrix, axis=1)[:, 0:numberNearestNeighbors])
           # print(neighbors)
            N = neighbors.shape[0]
            coordinates = np.zeros((N, numberNearestNeighbors, 2, 2))
            for i in np.arange(N):
                for j in np.arange(numberNearestNeighbors):
                    coordinates[i, j, :, 0] = np.array([X[i,:][0], X[neighbors[i, j], :][0]])
                    coordinates[i, j, :, 1] = np.array([X[i,:][1], X[neighbors[i, j], :][1]])
            lineCollections[n] = {"segments":coordinates.reshape((N*numberNearestNeighbors, 2, 2)), "color":'black', "linewidth" : 0.5,"zorder":0}

        returnKwargs = getMessageProps("Done..","Nearest neighbors calculated and added to graph.")
        returnKwargs["lineCollections"] = lineCollections
        return returnKwargs

    def getSizeGroupsForScatter(self,dataID, sizeColumn = None, sizeColumnType = None, sizesSetByUser = None):
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

            categoryIndexMatch = dict([(intID,rawData[rawSizeData == category].index) for category, intID in zip(sizeGroupData["group"].values,
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
            sizeGroupData["size"] = [round((x-minV)/(maxV-minV) * (self.maxScatterSize-self.minScatterSize) + self.minScatterSize,1) \
                                                                            for x in  [maxV,q75,median,q25,minV,0]] #0.1 == Nan Size
            
            sizeGroupData["internalID"] = [getRandomString() for n in sizeGroupData.index]
            
            sizeGroupData.loc[sizeGroupData["group"] == "NaN","size"] = round(0.1 * self.minScatterSize,1)
           

        propsData = pd.DataFrame(sizeData,columns=["size"], index=rawData.index)
        title = mergeListToString(sizeColumn.values,"\n")                                                                   

        
        return {"sizeGroupData":sizeGroupData,"propsData":propsData,"title":title,"categoryIndexMatch":categoryIndexMatch,"categoryEncoded":"size","isEditable":sizeColumnType == "Categories"}


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
        areaData = {}
        for n,numColumns in enumerate(numericColumnPairs):

            lowessFit = self.sourceData.statCenter.runLowess(dataID,list(numColumns))

            lineKwargs = {"xdata":lowessFit[:,0],"ydata":lowessFit[:,1]}
            lineData[n] = lineKwargs
            areaData[n] = {"x" : lowessFit[:,0], "y1": lowessFit[:,2], "y2":lowessFit[:,3], "facecolor":"lightgrey","edgecolor":"None","alpha":0.5}
        
        funcProps = getMessageProps("Done..","Lowess line added.")
        funcProps["lineData"] = lineData
        funcProps["areaData"] = areaData
        
        return funcProps

    def getColorGroupsDataForScatter(self,dataID, colorColumn = None, colorColumnType = None, colorGroupData = None, userMinMax = None):
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
            if userMinMax is not None:
                minV, maxV = userMinMax
                median = (maxV+minV) / 2 
                q25 = median - (median-minV)/2
                q75 = median + (maxV-median)/2

            
            colorValues, colorsInMapper = self.sourceData.colorManager.matchColorsToValues(arr = rawData.values.flatten(), colorMapName = twoColorMap, vmin = minV, vmax = maxV, asHex=True)
            colorData =  pd.DataFrame(colorValues,
                                    columns=["color"],
                                    index=rawData.index)
            colorData.loc[nanIndex,"color"] = self.sourceData.colorManager.nanColor
            
            #save colors for legend
            #scaledColorVs = [to_hex(cmap( (x - minV) / (maxV - minV))) for x in [maxV,q75,median,q25,minV]]
            legendColors, _ = self.sourceData.colorManager.matchColorsToValues(arr = [maxV,q75,median,q25,minV], colorMapName = twoColorMap, vmin = minV, vmax = maxV, asHex=True)
            colorLimitValues = legendColors.flatten().tolist() + [self.sourceData.colorManager.nanColor]
            
            colorGroupData = pd.DataFrame(columns=["color","group"])
            groupNames = ["Max ({})".format(getReadableNumber(maxV)),
                        "75% Quantile ({})".format(getReadableNumber(q75)),
                        "Median ({})".format(getReadableNumber(median)),
                        "25% Quantile ({})".format(getReadableNumber(q25)),
                        "Min ({})".format(getReadableNumber(minV)),
                        "NaN"]
            colorGroupData["color"] = colorLimitValues
            colorGroupData["group"] = groupNames
        
        tableTitle = mergeListToString(colorColumn.values,"\n") if userMinMax is None else "Custom Scale\n" + mergeListToString(colorColumn.values,"\n")
        #save data to enable fast update 
        self.colorColumn = colorColumn
        self.colorColumnType = colorColumnType
       # print({"colorGroupData":colorGroupData,"propsData":colorData,"title":tableTitle,"categoryIndexMatch":categoryIndexMatch,"categoryEncoded":"color"})
        
        return {"colorGroupData":colorGroupData,"propsData":colorData,"title":tableTitle,"categoryIndexMatch":categoryIndexMatch,"categoryEncoded":"color","isEditable":colorColumnType == "Categories"}



    def getSwarmplotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
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

        NProcesses = self.sourceData.parent.config.getParam("n.processes.multiprocessing")

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

            if rawData.size > 15000 and rawData.shape[1] > 4:
                #if big data, use multiprocessing, otherwise not
                with Pool(NProcesses) as p:
                    r = p.starmap(kdeCalc,[(rawData[[numColumn]].dropna(),0.75,"gaussian",widthBox,tickPositions[0][n],numColumn,n)  for n,numColumn in enumerate(numericColumns)])
                    kdeIndex, dataList, xNames = zip(*r)
                # print(kdeIndex, dataList, xNames)
                plotData = plotData.join(dataList)
                for n,(xName,numColumn) in enumerate(xNames):
                    columnNames.extend([xName,numColumn]) 
                    multiScatterKwargs[0][(xName,numColumn)] = {"color": colorDict[numColumn]}
                    internalID = colorGroupsData.loc[colorGroupsData["group"] == numColumn]["internalID"].values[0]
                    colorCategoryIndexMatch[internalID] = kdeIndex[n]
                    if internalID not in interalIDColumnPairs[0]:
                        interalIDColumnPairs[0][internalID] = [(xName,numColumn)]
                    else:
                        interalIDColumnPairs[0][internalID].append((xName,numColumn))
                
            else:

                for n,numColumn in enumerate(numericColumns):
                    xName = "x({})".format(numColumn)
                    groupData = rawData[[numColumn]].dropna()
                    if groupData.index.size == 1:
                        kdeData = np.array([0]) +  positions[n]
                        data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                        kdeIndex = data.index
                    else:
                        #get kernel data
                        kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                        #get random x position around 0 to spread data
                        allSame = np.all(kdeData == kdeData[0])
                        if allSame:
                            kdeData = np.zeros(shape=kdeData.size)
                        else:
                            kdeData = scaleBetween(kdeData,(0,widthBox/2))
                        kdeData = np.array([np.random.uniform(-x*0.85,x*0.85) for x in kdeData])
                        #print(time.time()-t1,"numpy")
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
                        if groupData.index.size < 5:
                            kdeData = np.array([0]*groupData.index.size) +  positions[nColCat]
                            data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                            kdeIndex = data.index
                        else:
                            kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                            #get random x position around 0 to spread data between - and + kdeData
                            kdeData = scaleBetween(kdeData,(0,widthBox/2)) 
                            kdeData = np.array([np.random.uniform(-x * 0.80 , x * 0.80) for x in kdeData])
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

            axisPostions = getAxisPosition(len(numericColumns),maxCol=self.maxColumns)
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
                                kdeIndex = data.index
                            else:
                                #get kernel data
                                
                                kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                                #get random x position around 0 to spread data between - and + kdeData
                                allSame = np.all(kdeData == kdeData[0])
                                if allSame:
                                    kdeData = np.zeros(shape=kdeData.size)
                                else:
                                    kdeData = scaleBetween(kdeData,(0,widthBox/2)) 
                                
                                kdeData = np.array([np.random.uniform(-x*0.85,x*0.85) for x in kdeData])
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

            axisPostions = getAxisPosition(n = axisCategories.size *  numNumColumns, maxCol = axisCategories.size)
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
                                    kdeIndex = data.index
                                else:
                                    kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                                    #get random x position around 0 to spread data between - and + kdeData
                                    allSame = np.all(kdeData == kdeData[0])
                                    if allSame:
                                        kdeData = np.zeros(shape=kdeData.size)
                                    else:
                                        kdeData = scaleBetween(kdeData,(0,widthBox/2))
                                    
                                    kdeData = np.array([np.random.uniform(-x*0.85,x*0.85) for x in kdeData])
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
                        axisTitles[nAxisCat] = "{}:{}".format(categoricalColumns[2],axisCat)
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
       # print(factorMapper)
        #ensure -1 is first in color chooser
       # factorMapper = OrderedDict(sorted(factorMapper.items(), key=lambda x:x[1]))
        

        colorData = pd.DataFrame(columns = colorColumn, index = rawData.index)

        for columnName in colorColumn.values:
            colorData.loc[rawData.index,columnName] = rawData.loc[rawData.index,columnName].map(factorMapper)


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
            rows = int(np.ceil(n_pairs/3))
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

    def getWordCloud(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        
        axisPostions = getAxisPosition(1)
        rawData = self.sourceData.getDataByColumnNames(dataID,categoricalColumns)["fnKwargs"]["data"]
        config =  self.sourceData.parent.config
        splitString = config.getParam("word.cloud.split_string")
        cmap = self.sourceData.colorManager.get_max_colors_from_pallete()
        
        #countedValues = rawData[categoricalColumns[0]].value_counts(normalize=True).to_dict()
        #print(splitData)
        
        #print(countedValues)
        
        wc = WordCloud(
            font_path = "Arial",
            max_font_size = config.getParam("word.cloud.max_font_size"),
            min_font_size = config.getParam("word.cloud.min_font_size"),
            normalize_plurals = config.getParam("word.cloud.normalize_plurals"),
            max_words = config.getParam("word.cloud.max_words"),
            colormap = cmap, 
            include_numbers = config.getParam("word.cloud.include_numbers"),
            background_color = config.getParam("word.cloud.background_color"),
            width = 1600, 
            height = 800)

        if config.getParam("word.cloud.categories_to_frequencies"):
            
            splitData = rawData[categoricalColumns[0]].astype("str").str.split(splitString, expand=True).values.flatten()
            countedValues = pd.Series(splitData).value_counts(normalize=True).to_dict()
            wordcloud = wc.generate_from_frequencies(countedValues)
        else:
            textInput = " ".join(rawData[categoricalColumns].values.flatten().astype(str))
            wordcloud = wc.generate(textInput)

        return {"data":{
                "cloud":wordcloud,
                "axisPositions":axisPostions}
                }

    def getXYPlotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        "Returns plot properties for a XY plot"
        colorGroupsData = pd.DataFrame() 
        axisLabels = {}
        axisLimits = {}
        linesByInternalID = {}
        #
        lines = {}
        lineKwargs = {}
        markerLines = {}
        markerKwargs = {}
        #
        markerLine = None
        markerProps = {}

        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        config =  self.sourceData.parent.config
        

        if config.getParam("xy.plot.stem.mode"):
            stemBottom = config.getParam("xy.plot.bottom.stem")
        
        if config.getParam("xy.plot.against.index") or len(numericColumns) == 1:
            idxColumnName = "ICIndex_{}".format(getRandomString())#ensures that there are no duplicates
            rawData[idxColumnName] = np.arange(rawData.index.size)
            numericColumnPairs = [(idxColumnName,columnName) for columnName in numericColumns]
        elif config.getParam("xy.plot.single.x"): #use only first numeric column
            numericColumnPairs = [(numericColumns[0],columnName) for columnName in numericColumns[1:]]
        else:
            numericColumnPairs = list(zip(numericColumns[0::2], numericColumns[1::2]))
        
        separatePairs = config.getParam("xy.plot.separate.column.pairs")
        axisPostions = getAxisPosition(1 if not separatePairs else len(numericColumnPairs))
        if len(categoricalColumns) == 0:
            colorValues = self.sourceData.colorManager.getNColorsByCurrentColorMap(len(numericColumnPairs))
            colorGroupsData["group"] = ["{}:{}".format(*columnPair) if "ICIndex" not in columnPair[0] else "{}".format(columnPair[1]) for columnPair in numericColumnPairs]
            colorCategories = ["None"]
        else:
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns)
            #print(colorCategories)
            numColorCategories = colorCategories.size
            colorGroupsData["group"] = colorCategories
            colorValues = self.sourceData.colorManager.getNColorsByCurrentColorMap(numColorCategories)
            colorGroupBy = rawData.groupby(by=categoricalColumns[0],sort=False)
        #line2D 
        colorGroupsData["color"] = colorValues
        colorGroupsData["internalID"] = [getRandomString() for n in colorValues]
        colorGroupsData = colorGroupsData[["color","group","internalID"]]

        for n,columnPair in enumerate(numericColumnPairs):

            for m,colorCategory in enumerate(colorCategories):

                if len(categoricalColumns) == 0:
                    colorValue = colorValues[n]
                    data = rawData
                    internalID = colorGroupsData["internalID"].iloc[n]
                else:
                    data = colorGroupBy.get_group(colorCategory)
                    internalID = colorGroupsData["internalID"].iloc[m]
                    colorValue = colorValues[m]
                
                if config.getParam("xy.plot.stem.mode"):
                    #create LineCollections 
                    lineSegments = [[(x,stemBottom),(x,y)] for x,y in data[list(columnPair)].values]
                    lineProps = dict(
                            linestyle = config.getParam("xy.plot.line.style"),
                            segments = lineSegments, 
                            color = colorValue,
                            linewidth = config.getParam("xy.plot.linewidth"))

                    l = LineCollection(**lineProps)
                    if config.getParam("xy.plot.show.marker"):

                        markerProps = dict(
                            xdata = data[columnPair[0]].values,
                            ydata = data[columnPair[1]].values,
                            color = "None",
                            markeredgecolor = "darkgrey",
                            markerfacecolor = colorValue,
                            markeredgewidth = config.getParam("xy.plot.marker.edge.width"),
                            alpha = config.getParam("xy.plot.alpha"), 
                            markersize = config.getParam("xy.plot.marker.size"),
                            linewidth = 0, 
                            marker = config.getParam("xy.plot.marker"))

                        markerLine = Line2D(**markerProps)
                    #markerline, stemlines, baseline = stem(rawData[columnPair[0]].values,rawData[columnPair[1]].values, use_line_collection = True)

                else:

                    lineProps = dict(
                            xdata = data[columnPair[0]].values,
                            ydata = data[columnPair[1]].values,
                            color = colorValue,
                            markeredgecolor = "darkgrey",
                            markeredgewidth = config.getParam("xy.plot.marker.edge.width"),
                            alpha = config.getParam("xy.plot.alpha"), 
                            markersize = config.getParam("xy.plot.marker.size"),
                            linewidth = config.getParam("xy.plot.linewidth"), 
                            marker = config.getParam("xy.plot.marker") if config.getParam("xy.plot.show.marker") else "")

                    l = Line2D(**lineProps)
            
                
                if markerLine is not None:
                    if internalID in linesByInternalID:
                        linesByInternalID[internalID].extend([l,markerLine])
                    else:
                        linesByInternalID[internalID] = [l,markerLine]
                else:
                    if internalID in linesByInternalID:
                        linesByInternalID[internalID].append(l)
                    else:
                        linesByInternalID[internalID] = [l]

                if not separatePairs:
                    if 0 not in lines:    
                        lines[0] = []
                        lineKwargs[0] = []
                        markerLines[0] = []
                        markerKwargs[0] = []
                    lines[0].append(l)
                    lineKwargs[0].append({"ID":internalID,"props":lineProps})
                    markerLines[0].append(markerLine)
                    markerKwargs[0].append({"ID":internalID,"props":markerProps})
                else:
                    if n not in lines:
                        lines[n] = [l]   
                        lineKwargs[n] = [{"ID":internalID,"props":lineProps}]
                        markerKwargs[n] = [{"ID":internalID,"props":markerProps}]
                        markerLines[n] = [markerLine]
                    else:
                        lines[n].append(l)
                        lineKwargs[n].append({"ID":internalID,"props":lineProps})
                        markerKwargs[n].append({"ID":internalID,"props":markerProps})
                        markerLines[n].append(markerLine)

            
    
        if not separatePairs:
            xAxisColumns = np.unique([columnPair[0] for columnPair in numericColumnPairs])
            yAxisColumns = np.unique([columnPair[1] for columnPair in numericColumnPairs])
            xAxisMin, xAxisMax, yAxisMin, yAxisMax = self._getXYLimits(X = rawData[xAxisColumns].values.flatten(),
                                    Y  = rawData[yAxisColumns].values.flatten())
                
            axisLimits[0] = {"xLimit": (xAxisMin,xAxisMax), "yLimit": (yAxisMin,yAxisMax)}
            
            axisLabels[0] = {"x":numericColumnPairs[0][0],
                            "y":numericColumnPairs[0][1]} if len(numericColumnPairs) == 1 else {"x":"Index","y":"Y-value"}
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
                    "colorCategoricalColumn": "Line Collection" if len(categoricalColumns) == 0 else categoricalColumns[0],
                    "axisLimits" : axisLimits,
                    "dataID" : dataID,
                    "linesByInternalID": linesByInternalID,
                    "markerKwargs": markerKwargs,
                    "markerLines": markerLines,
                    "numericColumnPairs" : numericColumnPairs,
                    "hoverData" : rawData
                    }}
        
    def _getXYLimits(self,X,Y,marginFrac = 0.1):
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