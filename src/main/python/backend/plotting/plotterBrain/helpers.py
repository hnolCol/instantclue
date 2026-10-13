
from ast import Or
from cmath import pi
from multiprocessing import Pool
from sys import intern
from tokenize import group
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
from .postionCalculator import calculatePositions, getAxisPosition, _width_functions, _lv_box_ends, _lv_outliers

from threadpoolctl import threadpool_limits
from statsmodels.stats.contingency_tables import Table2x2
#cluster
import scipy.cluster.hierarchy as sch
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib.colors import to_hex
from matplotlib.pyplot import box, get, hist, sca
import matplotlib.cm as cm
from matplotlib import rcParams

from collections import OrderedDict
import operator
import seaborn as sns
import itertools
import numpy as np
import scipy.stats as st
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import truncnorm
from sklearn.model_selection import GridSearchCV
from numba import jit, njit, prange
from wordcloud import WordCloud
import io
import re
import pickle
from scipy.stats import linregress

@njit(parallel=True)
def spreadSwarm(kde):
    ""
    A = np.zeros(shape=(kde.size,1))
    for i in prange(kde.size):
        A[i,0] = np.random.uniform(-kde[i]*0.85,kde[i]*0.85)
    return A

def extractCIFromstring(ciString):
    ""
    try:
        ci = float(re.findall( r'(\d{1,3})%',ciString)[0])/100
    except:
        ci = 0.95
    return ci

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

def distKDEData(kdeData):
    ""
    boolIdx = np.isnan(kdeData)
    kdeData[boolIdx] = 1e-6


    return  np.random.uniform(low=kdeData*(-0.75),high=kdeData*0.75,size=kdeData.size)


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
        kdeData = distKDEData(kdeData)
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
    "boxenplot" : "getBoxenProps",
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

