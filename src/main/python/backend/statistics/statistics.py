import itertools
from tokenize import group
from pandas.core.accessor import delegate_names
from pandas.core.reshape.melt import melt
from pingouin.correlation import corr
from scipy.sparse import data
from .dimensionalReduction.ICPCA import ICPCA
from .featureSelection.ICFeatureSelection import ICFeatureSelection
from ..utils.stringOperations import getMessageProps, getRandomString, mergeListToString
from backend.utils.stringOperations import getNumberFromTimeString
from backend.utils.misc import getKeyMatchInValuesFromDict
from backend.filter.utils import buildRegex
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd

from scipy.optimize import curve_fit
from scipy.stats import linregress, f_oneway, ttest_ind, mannwhitneyu, wilcoxon, fisher_exact, chi2_contingency

from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering, Birch, AffinityPropagation
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests
#import pingouin as pg
from pingouin import anova, mixed_anova


#mixed ANOVA import
import hdbscan
import umap
import re
import pandas as pd 
import numpy as np
#from cvae import cvae
import fastcluster
from collections import OrderedDict
#pycombat import
from combat.pycombat import pycombat
from itertools import chain

from joblib import Parallel, delayed, dump, load
import time
from multiprocessing import Pool, Process, Queue
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')
try:
    from skmisc.loess import loess
    useStatsmodelLoess = False
except:
    from statsmodels.nonparametric.smoothers_lowess import lowess as loess
    useStatsmodelLoess = True
def _matchRegExToPandasSeries(data,regEx,uniqueCategory):
    return (uniqueCategory, data.str.contains(regEx, case = True))
def _matchMultipleRegExToPandasSeries(data,regExs,uniqueCategories):
    ""
    return [(uniqueCategory, data.str.contains(regEx, case = True)) for regEx, uniqueCategory in zip(regExs,uniqueCategories)]

clusteringMethodNames = OrderedDict([
                            ("kmeans",KMeans),
                            ("Birch",Birch),
                            ("OPTICS",OPTICS),
                            ("HDBSCAN",hdbscan.HDBSCAN),
                            ("Affinity Propagation",AffinityPropagation),
                            ("Agglomerative Clustering",AgglomerativeClustering)
                            ])

manifoldFnName = {"Isomap":"runIsomap","MDS":"runMDS","TSNE":"runTSNE","LLE":"runLLE","SpecEmb":"runSE"}



def runCombat(data,batchIdx):
    
    return pycombat(data,batchIdx)

def expIncrease(x,A,k,b):
    ""
    return 1 - (A * np.exp(-x*k) + b)

def expDecrease(x,A,k,b):
    ""
    return A * np.exp(-x*k) + b 
    

def fitExponentialToChunk(fitType,xdata,Y,chunkIdx):
    
    """
    To Do:
    get is nan for complete Y and index by row.
    """
    fits = []
    if fitType == "decrease":
        for ydata in Y:
            try:
                fitResult = curve_fit(expDecrease,xdata[~np.isnan(ydata)],ydata[~np.isnan(ydata)], bounds=(0, [1., 1., 0.01])) 
            except:
                fitResult = (np.array([np.nan,np.nan,np.nan]), None)
            fits.append(fitResult)
        
        #fits =  [curve_fit(expDecrease,xdata[~np.isnan(ydata)],ydata[~np.isnan(ydata)]) for ydata in Y]
        residuals = [ydata[~np.isnan(ydata)] - expDecrease(xdata[~np.isnan(ydata)], *ff[0]) if ff[1] is not None else None for ydata,ff in zip(Y,fits)]
        
    else:
        for ydata in Y:
            try:
                fitResult = curve_fit(expIncrease,xdata[~np.isnan(ydata)],ydata[~np.isnan(ydata)], bounds=(0, [1., 1., 0.01]))
            except:
                fitResult = (None,None)
            fits.append(fitResult)
        
        residuals = [ydata[~np.isnan(ydata)] - expIncrease(xdata[~np.isnan(ydata)], *ff[0]) if ff[1] is not None else None for ydata,ff in zip(Y,fits)]

    ss_res = [np.sum(res**2) if res is not None else None for res in residuals]
    ss_tot = [np.nansum((ydata[~np.isnan(ydata)]-np.nanmean(ydata[~np.isnan(ydata)]))**2) for ydata in Y]
    r_squared = [1 - (ss_r / ss_t) if ss_r is not None else np.nan for ss_r, ss_t in zip(ss_res,ss_tot)]
    output = pd.DataFrame([np.append(ff[0],r_squared[n]) for n,ff in enumerate(fits)])
    return (chunkIdx,output)


def p_anova(data,dv,grouping,idx):
    try:
        aov = anova(data=data,dv=dv,between=grouping)
        if "p-unc" in aov.columns:
            return ((idx,aov["p-unc"].values.flatten()), aov["Source"])
    except:
        return 

def loess_fit(x, y, span=0.75):
    """
    loess fit and confidence intervals
    """
    if useStatsmodelLoess:
        # setup
        lo = loess(x, y, span=span)
        # fit
        lo.fit()
        # Predict
        prediction = lo.predict(x, stderror=True)
        # Compute confidence intervals
        ci = prediction.confidence(0.05)
        # Since we are wrapping the functionality in a function,
        # we need to make new arrays that are not tied to the
        # loess objects
        yfit = np.array(prediction.values)
        ymin = np.array(ci.lower)
        ymax = np.array(ci.upper)

    else:
        yfit = loess(y,x,frac=span)
        ymin = np.nan
        ymax = np.nan
    return yfit, ymin, ymax

class StatisticCenter(object):


    def __init__(self, sourceData, n_jobs = 4):
        ""

        self.sourceData = sourceData
        self.n_jobs = n_jobs
        self.featureSelection = ICFeatureSelection()

        #clustering
        self.rowMetric = "euclidean"
        self.columnMetric = "euclidean"
        
        self.rowMethod = "complete"
        self.columnMethod = "complete"

        #self.umap_n_neighbors = 15
        #self.umap_min_dist = 0.1
        #self.umap_metric = "euclidean"
        #self.umap_n_components = 2

        self.absCorrCoeff = 0.90
        self.corrMethod = "pearson"

        self.corrCoeffName = {"pearson":"r","spearman":"rho","kendall":"rho"}

    def calculateAUCFromGraph(self,dataID,numericColumnPairs,chartData, replicateColumn = "None", addAsDataFrame = True):
        ""
        R = pd.DataFrame(columns=["xColumn","yColumn","AUC"])
        for xColumn,yColumn in numericColumnPairs:
            X = chartData[[xColumn,yColumn]].dropna().values
            if X.shape[0] > 1:
                AUC = np.trapz(X[:,1],X[:,0])
                R = R.append({"xColumn":xColumn,"yColumn":yColumn,"AUC":AUC},ignore_index=True)
        
        fileName="AUCData ({})".format(self.sourceData.getFileNameByID(dataID))
        rows, columns = R.shape
        if addAsDataFrame:
            return self.sourceData.addDataFrame(R, fileName = fileName)
        else:
            return {"messageProps":
				{"title":"Data Frame Added {}".format(fileName),
				"message":"AUC results added.\nShape (rows x columns) is {} x {}".format(rows,columns)},
				"dataFrame" : R
			}

    def checkData(self,X):
        
        if X.empty:
            return False, "Filtering resulted in empty data matrix"

        if X.columns.size < 2:
            return False, "Need more than two numeric columns."

        if X.index.size < 3:
            return False, "Filtering resulted in data matrix with 2 rows."

        return True, "Check done.."

    def correleateColumnsOfTwoDfs(self,df1,df2):
        ""
        n = len(df1)
        v1, v2 = df1.values, df2.values
        sums = np.multiply.outer(np.nansum(v2,axis=0), np.nansum(v1,axis=0))
        stds = np.multiply.outer(np.nanstd(v2,axis=0), np.nanstd(v1,axis=0))
        return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n,
                            df2.columns, df1.columns)

    def getNJobs(self):
        ""
        return self.n_jobs

    def getData(self, dataID, columnNames):
        ""
        return self.sourceData.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"]

    def removeNan(self,X):
        ""
        if not isinstance(X, pd.DataFrame):
            self.X = pd.DataFrame(X, dtpye = np.float64)
        X = X.astype(np.float64).dropna()
        return X

    def prepareData(self, dataID, columnNames):
        ""
        X = self.getData(dataID,columnNames)
        X = self.removeNan(X)
        dataIndex = X.index

        dataPassed, msg  = self.checkData(X)

        return X,dataPassed,msg, dataIndex

    def runLDA(self,dataID,groupingName):
        ""
        config = self.sourceData.parent.config 
        grouping = self.sourceData.parent.grouping
        grouping.setCurrentGrouping(groupingName)
        groupingName = grouping.getCurrentGroupingName()
        columnNames = grouping.getColumnNames(groupingName)
        groupingAnnotation = grouping.getGroupNamesByColumnNames(columnNames)
        #print(groupingAnnotation)
        if len(columnNames) > 0:
            X = self.getData(dataID,columnNames.values).dropna()
            Y = groupingAnnotation.values.flatten()
            if config.getParam("lda.scale"):
                XT = X.values.T
                XT = StandardScaler().fit_transform(XT)
            else:
                XT = X.values.T

            LDA = LinearDiscriminantAnalysis(n_components=config.getParam("LDA.n.components"))
            LDA.fit(XT,Y)
            X_TRANFORMED = LDA.transform(XT)
            
            XX = pd.DataFrame(X_TRANFORMED, columns = ["Comp:{:02d}({:.2f}%)".format(n,perc*100) for n,perc in enumerate(LDA.explained_variance_ratio_.flatten())])
            XX["ColumnNames"] = columnNames
            XX[groupingName] = Y
            weightVectors = pd.DataFrame(LDA.coef_.T, index=X.index,columns=groupingAnnotation.unique().tolist())
            weightVectors = weightVectors.join(self.sourceData._getCategoricalData(dataID))
            self.sourceData.addDataFrame(weightVectors, fileName="LDA:WeightVectors({})".format(self.sourceData.getFileNameByID(dataID)))
            completedKwargs = self.sourceData.addDataFrame(XX,fileName="LDA:({})".format(self.sourceData.getFileNameByID(dataID)))
            completedKwargs['messageProps'] = getMessageProps("Done..","LDA calulcated. Dataframes were added. Weight vector and transformed values.")
            return completedKwargs
        else:
            return getMessageProps("Error..","No column names in grouping.")
    
    def runLinearRegression(self,dataID, columnNames):
        '''
        Calculates linear regression
        '''
        
        X = self.getData(dataID,columnNames)
        X = self.removeNan(X)
       
        x = X.iloc[:,0].values
        y = X.iloc[:,1].values

        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        x1, x2 = x.min(), x.max()
        y1, y2 = slope*x1+intercept, slope*x2+intercept

        return [x1,x2],[y1,y2],slope, intercept, r_value, p_value, std_err


    def runLowess(self,dataID,columnNames,it=None,frac=None,*args,**kwargs):
        '''
        Calculates lowess line from dataFrame input
        '''
        data = self.getData(dataID,columnNames)
        data.dropna(inplace=True)
        data = data.sort_values(by = data.columns.values[0])
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

    def runANOVA(self,dataID,betweenGroupings = [] ,withinGroupings = [], logPValues = True, subjectGrouping = []):
        ""
       # print(dataID)

        if len(betweenGroupings) == 0 and len(withinGroupings) == 0:
            return getMessageProps("Error..","No grouping found.")
        
        combinedGroupings = betweenGroupings + withinGroupings
        groupNames = [list(d["values"].values()) for d in combinedGroupings if d is not None]
        uniqueColumnNames = np.unique(groupNames)
        data = self.getData(dataID,uniqueColumnNames).dropna()

        rowIdentifier = getRandomString()
        data[rowIdentifier] = data.index
        meltedData = pd.melt(data,value_vars=uniqueColumnNames, id_vars=[rowIdentifier])
        for bGroup in betweenGroupings:
            mapDict = {}
            groupName = bGroup["name"]
            groupItems = bGroup["values"]
            for k,v in  groupItems.items(): #always length 1
                for colName in v.values:
                    mapDict[colName] = groupName + k 
            meltedData[groupName] = meltedData["variable"].map(mapDict)

       # print(meltedData)

        collectedData = None 
        
        for idx, idxData in meltedData.groupby(rowIdentifier):
            idxData = idxData.dropna(subset=["value"])
            idxData['Subject'] = np.arange(1,idxData.index.size+1)
           # print(idxData)
           # anova = Anova(
            #            idxData,
             #           dependentVariable = "value", 
              #          wFactors = [k["name"] for k in withinGroupings],
               #         bFactors = [k["name"] for k in betweenGroupings])
           # res#ults = anova.getResults().set_index("Source")
          #  print(results)
            r = anova(data = idxData, dv = "value" ,between = [k["name"] for k in betweenGroupings], detailed=True)
            results = r.set_index("Source")
            if collectedData is None:
                collectedData = pd.DataFrame(index=data.index, columns = ["p-value:{}".format(colName) for colName in results.index if colName != "Residual"])
           # print(results)
            for source in results.index:
                if source != "Residual":
                    colName = "p-value:{}".format(source)
                    collectedData.loc[idx,colName] = results.loc[source,"p-unc"]
           # print(collectedData)

            
           # intColName = " * ".join([k["name"] for k in betweenGroupings])
            #print(intColName)
           # collectedData.loc[idx,"p-value:interaction"] = np.float(results.loc[intColName,"p-value"])
        collectedData = collectedData.astype(float)
        return self.sourceData.joinDataFrame(dataID,collectedData)





    def runTSNE(self,X,*args,**kwargs):
        ""
        nComp = self.sourceData.parent.config.getParam("tsne.n.components")
        perplexity = self.sourceData.parent.config.getParam("tsne.perplexity")
        early_exaggeration = self.sourceData.parent.config.getParam("tsne.early_exaggeration")
        self.calcTSNE = TSNE(n_components=nComp,
                                perplexity=perplexity,
                                early_exaggeration = early_exaggeration,
                                *args,**kwargs)
        return self.calcTSNE.fit_transform(X)
            
            

    # def runCVAE(self,dataID,columnNames, transpose = False, *args,**kwargs):
    #     "Runs a Variational Autoencoder Dimensional Reduction"
    #     X, checkPassed, errMsg, dataIndex  = self.prepareData(dataID,columnNames)
    #     if checkPassed:
        
    #         embedder = cvae.CompressionVAE(X.values,)
    #         embedder.train() 
    #         embeddings = embedder.embed(X.values)
    #         compNames = ["Emb::CVAE_{:02d}".format(n) for n in range(2)]
    #         df = pd.DataFrame(embeddings.astype(np.float64),index=dataIndex,columns = compNames)
    #         msgProps = getMessageProps("Done..","CVAE (Variational Autoencoder) calculation performed.\nColumns were added to the tree view.")
                        
    #         result = {**self.sourceData.joinDataFrame(dataID,df),**msgProps}
    
    #         return result

    #     else:
    #             return getMessageProps("Error ..",errMsg)

    def runUMAP(self,dataID,columnNames, transpose = False, *args,**kwargs):
        
        with threadpool_limits(limits=1, user_api='blas'): #require to prevent crash (np.dot not thread safe)
            X, checkPassed, errMsg, dataIndex  = self.prepareData(dataID,columnNames)
            if checkPassed:
                    config = self.sourceData.parent.config
                    nN = config.getParam("umap.n.neighbors")
                    minDist = config.getParam("umap.min.dist")
                    metric = config.getParam("umap.metric")
                    nComp = config.getParam("umap.n.components")

                    self.calcUMAP = umap.UMAP(n_neighbors = nN,
                                                metric = metric,
                                                min_dist = minDist,
                                                n_components = nComp
                                                )
                    compNames = ["Comp::UMAP_{:02d}".format(n) for n in range(nComp)]
                    if transpose:
                        embedding = self.calcUMAP.fit_transform(X.T)
                 
                        df = pd.DataFrame(embedding, columns=compNames)
                        df[compNames] = df[compNames].astype(float)
                        df["ColumnNames"] = columnNames.values
                        return self.sourceData.addDataFrame(df,fileName = "UMAP:2Comp:BaseFile")
                    else:

                        embedding = self.calcUMAP.fit_transform(X)
                        
                        df = pd.DataFrame(embedding.astype(np.float64),index=dataIndex,columns = compNames)
                        msgProps = getMessageProps("Done..","UMAP calculation performed.\nColumns were added to the tree view.")
                        
                        result = {**self.sourceData.joinDataFrame(dataID,df),**msgProps}
    
                        return result
                
            else:
                return getMessageProps("Error ..",errMsg)
    
    def runPCA(self,dataID, columnNames, initGraph = False, liveGraph = False, returnProjections = False, *args,**kwargs):
        "Transform values."
        X, checkPassed, errMsg, dataIndex  = self.prepareData(dataID,columnNames)
        if checkPassed:
            
            config = self.sourceData.parent.config
            nComps = config.getParam("pca.n.components")
            scaleData = config.getParam("pca.scale")
            attachCurrentGroupingOnly = config.getParam("pca.add.current.grouping.only")
            
            with threadpool_limits(limits=1, user_api='blas'):
                self.calcPCA = ICPCA(X,n_components=nComps, scale=scaleData)
                embedding, eigV, explVariance = self.calcPCA.run()
            comColumns = ["PCA_Component_{:02d} ({:.2f})".format(n+1,explVariance[n]) for n in range(embedding.shape[1])]

            if returnProjections:
                
                df = pd.DataFrame(eigV, columns = ["Component {} ({:.2f}%)".format(n,explVariance[n]) for n in range(eigV.shape[1])])
                
                df["ColumnNames"] = columnNames.values.flatten()
                annotatedGroupings = self.sourceData.parent.grouping.getGroupingsByColumnNames(columnNames,attachCurrentGroupingOnly)
                if len(annotatedGroupings) > 0:
                    for groupingName, groupMatches in annotatedGroupings.items(): 
                        df[groupingName] = groupMatches.values
                baseFileName = self.sourceData.getFileNameByID(dataID)
                result = self.sourceData.addDataFrame(df, fileName = "PCA.T:({})".format(baseFileName))
            else:
                df = pd.DataFrame(embedding,columns = comColumns, index = dataIndex)
                df[comColumns] = df[comColumns].astype(float)
                result = self.sourceData.joinDataFrame(dataID,df)

            return result
        else:
                
            return getMessageProps("Error ..",errMsg)

    def runRowCorrelation(self,dataID,columnNames, indexColumn = ["Gene names"]):
        """
        Calculates correlation between all rows.
        Correlation coefficients will be filterd for NaN and threshold specified by
        the class attribute absCorrCoeff.
        """
        if columnNames.size < 3:
            return getMessageProps("Error..","Requires at least three columns.")
        #check index column
        if isinstance(indexColumn,str):
            indexColumn = [indexColumn]
        #get correlation data
        data = self.getData(dataID,columnNames)
        #remove rows that have less than 3 values (corr would be always 1)
        data = data.dropna(thresh = 3)
        #calculate corr matrix
        corrMatrix = data.T.corr(method = self.corrMethod)
        #get cateogrical daata
        catData = self.sourceData.getDataByColumnNames(dataID,indexColumn,rowIdx=data.index)["fnKwargs"]["data"][indexColumn[0]]
        #set index
        corrMatrix.index = catData.values
        corrMatrix.columns = catData.values
        #move to long format
        corrMatrixLong = corrMatrix.unstack().reset_index()
        #add prev index
        numberOfFeatures = corrMatrix.index.size
        longFormatSize = corrMatrixLong.index.size
        corrMatrixLong["Index"] = np.tile(data.index,int(longFormatSize/numberOfFeatures))
        #remove nana
        corrMatrixLong = corrMatrixLong.dropna()
        if self.corrMethod in self.corrCoeffName:
            coeffName = self.corrCoeffName[self.corrMethod]
        else:
            coeffName = "CorrCoeff"
        corrMatrixLong.columns = ["Level 0","Level 1",coeffName,"Index"]

        if self.absCorrCoeff > 0: 
            boolIdx = np.abs(corrMatrixLong["r"].values) > self.absCorrCoeff
            corrMatrixLong = corrMatrixLong.loc[boolIdx,]
        baseFileName = self.sourceData.getFileNameByID(dataID)

        return self.sourceData.addDataFrame(corrMatrixLong,fileName="corrMatrix::{}".format(baseFileName))

    def runFeatureSelection(self,dataID,columnNames, grouping, groupFactors, model = "Random Forest", createSubset=False, RFECV = False):
        "Selects features based on model"
        try:
            #set current settings
            config = self.sourceData.parent.config
            self.featureSelection.setMaxFeautres(config.getParam("feature.selection.num"))
            scaleData = config.getParam("feature.scale.data")
            data = self.getData(dataID,columnNames).dropna()
            groupingName = self.sourceData.parent.grouping.getCurrentGroupingName()
            if data.index.size > 2:
                Y =  np.array([groupFactors[colName] for colName in data.columns.values])
                X = data.values.T
                if model == "Random Forest":
                    #Y =  np.array([groupFactors[colName] for colName in data.columns.values])
                    #X = data.values.T
                    nTrees = config.getParam("feature.randomforest.n_estimators")
                    min_samples_split = config.getParam("feature.randomforest.min_samples_split")
                    if RFECV:

                        rowBools =  self.featureSelection.selectFeaturesByRFECV(X,Y,
                                    model="Random Forest",
                                    ModelKwargs={"n_estimators":nTrees,"min_samples_split":min_samples_split},
                                    scale=scaleData)
                    else:
                        rowBools, featureImportance = self.featureSelection.selectFeaturesByRF(X,Y,{"n_estimators":nTrees,"min_samples_split":min_samples_split},scale=scaleData)
                   # print(featureImportance)
                    idx = data.index[rowBools]
                    if createSubset:
                        subsetName = "FSelRF:({})".format(self.sourceData.getFileNameByID(dataID))
                        return self.sourceData.subsetDataByIndex(dataID,idx,subsetName)
                    else:
                        columnName = "FSelRF({})".format(groupingName)
                        self.sourceData.joinDataFrame(dataID,pd.DataFrame(featureImportance,index=data.index,columns = ["FeatureImportancenRF()"]))
                        return self.sourceData.addAnnotationColumnByIndex(dataID, idx, columnName)

                elif "SVM" in model:

                    kernel = re.search('\(([^)]+)', model).group(1)
                    C = config.getParam("feature.svm.c")
                    if RFECV:
                        rowBools =  self.featureSelection.selectFeaturesByRFECV(X,Y,
                                                            model="SVC",
                                                            ModelKwargs={"kernel":kernel,"C":C},
                                                            scale=scaleData)

                    else:
                        rowBools =  self.featureSelection.selectFeaturesBySVM(X,Y,{"kernel":kernel,"C":C},scale=scaleData)

                    idx = data.index[rowBools]
                    
                    if createSubset:
                        subsetName = "FSelSVM:({})".format(self.sourceData.getFileNameByID(dataID))
                        return self.sourceData.subsetDataByIndex(dataID,idx,subsetName)
                    else:
                        columnName = "FSelSVM({})".format(groupingName)
                        return self.sourceData.addAnnotationColumnByIndex(dataID, idx, columnName)
                
                elif model == "False Positive Rate":
                    alpha = config.getParam("feature.selection.alpha")
                    self.featureSelection.setAlpha(alpha)
                    rowBools = self.featureSelection.selectFeaturesByFpr(X,Y,scale=scaleData)
                    idx = data.index[rowBools]
                    if createSubset:
                        subsetName = "FSelFPR(a={}):({})".format(alpha,self.sourceData.getFileNameByID(dataID))
                        return self.sourceData.subsetDataByIndex(dataID,idx,subsetName)
                    else:
                        columnName = "FSelFPR(a={})({})".format(alpha,groupingName)
                        return self.sourceData.addAnnotationColumnByIndex(dataID, idx, columnName)
                
                elif model == "False Discovery Rate":
                    alpha = config.getParam("feature.selection.alpha")
                    self.featureSelection.setAlpha(alpha)
                    rowBools = self.featureSelection.selectFeaturesByFdr(X,Y,scale=scaleData)
                    idx = data.index[rowBools]
                    if createSubset:
                        subsetName = "FSelFDR(a={}):({})".format(alpha,self.sourceData.getFileNameByID(dataID))
                        return self.sourceData.subsetDataByIndex(dataID,idx,subsetName)
                    else:
                        columnName = "FSelFDR(a={})({})".format(alpha,groupingName)
                        return self.sourceData.addAnnotationColumnByIndex(dataID, idx, columnName)

                return getMessageProps("Error..","Feature model unknown.")

            else:
                return getMessageProps("Error..","NaN filtering resulted in less than 2 rows.")#
        except Exception as e:
            print(e)



    def runHDBSCAN(self,dataID,columnNames, attachToSource = False):
        ""
        data = self.getData(dataID,columnNames).dropna()
        model = hdbscan.HDBSCAN(
                min_cluster_size=self.sourceData.parent.config.getParam("hdbscan.min.cluster.size"), 
                min_samples=self.sourceData.parent.config.getParam("hdbscan.min.samples"),
                cluster_selection_epsilon=self.sourceData.parent.config.getParam("hdbscan.cluster.selection.epsilon"))
        clusterLabels = model.fit_predict(data.values)
        
        df = pd.DataFrame(["C({})".format(x) for x in clusterLabels], columns = ["HDBSCAN"], index = data.index)
        if not attachToSource:
            return df, data
        else:
            return self.sourceData.joinDataFrame(dataID,df)

    def fitModel(self, dataID, columnNames, timeGrouping, compGrouping, replicateGrouping = "None", model="First Order Kinetic", normalization = "None", transformation = "None", sortTimeGroup = True, addDataAUC = True, addFitAUC = True):
        "Fit Model to Data"
        
        def _calcLineRegress(row, xValues,addDataAUC, addFitAUC, addHalfLife = True):
            Y = row.values
            mask = np.isnan(Y)
            Y = Y[~mask]
            xValues = xValues[~mask]
            r = linregress(x = xValues, y=Y)
            lRegress = r._asdict()
            if addDataAUC:
                lRegress["dataAUC"] = np.trapz(Y,xValues)
            if addFitAUC:
                x = np.linspace(np.nanmin(xValues), np.nanmax(xValues), num = 200)
                y = x * r.slope + r.intercept
                lRegress["fitAUC"] = np.trapz(y,x)
            if addHalfLife:
                t12 = np.log(2)/r.slope * (-1)
                lRegress["halfLife"] = t12 if t12 > 0 else np.nan
            
            return pd.Series(lRegress)

        def _calcIncrease(row,xValues, corrK):
            ""
            def f(x,A,k):
                return A*(1 - np.exp(-(k+corrK)*x))
            #print(corrK)
            popt, pcov = curve_fit(f,xValues,row)
            r = {"A":popt[0],"k":popt[0]}
            return pd.Series(r)

        try:
            returnProps = {}
            xtime = [(getNumberFromTimeString(groupName),groupColumns.values) for groupName,groupColumns in timeGrouping.items()]
            if sortTimeGroup:
                xtime.sort(key=lambda x:x[0])
            xtime = OrderedDict(xtime)
            data = self.getData(dataID,np.unique(columnNames))
            #transform
            if replicateGrouping is None:

                replicateGrouping = {"None":pd.Series(columnNames)}
            #normalize

            if compGrouping == "None" or compGrouping is None:

                compGrouping = {"raw":pd.Series(columnNames)} 
           
            for groupNameComp, groupColumnsComp in compGrouping.items():
            
                for repID, replicateColumns in replicateGrouping.items():
                    groupColumnsCompPerReplicate = [colName for colName in groupColumnsComp.values if colName in replicateColumns.values]
                    timeValues = [getKeyMatchInValuesFromDict(groupColumn,xtime) for groupColumn in  groupColumnsCompPerReplicate]
                    #remove vcolumns where no time was submitted
                    timeValueColumns = OrderedDict([(groupColumnsCompPerReplicate[n],timeValue) for n,timeValue in enumerate(timeValues) if timeValue is not None])
                    #if len(timeValueColumns) < 2: #no regression performed with less than 2 timepoints
                    #   continue
                    
                    if normalization != "None":
                        if normalization in ["Divide by median of first timepoint"]:
                            firstTimePoint = list(xtime.keys())[0]
                            firstTimePointColumns = xtime[firstTimePoint]
                            firstTimePointColumnsPerGroup = [colName for colName in firstTimePointColumns if colName in groupColumnsCompPerReplicate]
                            firstTimePointMedian = np.nanmedian(data[firstTimePointColumnsPerGroup], axis=1).flatten()
                   
                    if transformation  != "None":
                        if transformation == "2^x":
                            dataSubset = 2 ** data[list(timeValueColumns.keys())]
                    else:
                        dataSubset = data[list(timeValueColumns.keys())]

                    if normalization != "None":

                        dataSubset = dataSubset[list(timeValueColumns.keys())].divide(2 ** firstTimePointMedian, axis="rows")

                    #dataSubset[list(timeValueColumns.keys())] = np.log(dataSubset[list(timeValueColumns.keys())])
                    nNaNInFit = self.sourceData.parent.config.getParam("fit.model.min.non.nan")
                    dataSubset = dataSubset.dropna(thresh=nNaNInFit)
                    #corrK = 0.0303 if groupNameComp == "C" else 0
                    addt12 = model == "First Order Kinetic"
                    xValues = np.array(list(timeValueColumns.values()))
                    X = dataSubset.apply(lambda row : _calcLineRegress(row,xValues,addDataAUC,addFitAUC,addHalfLife=True), axis=1)
                    ####X = dataSubset.apply(lambda row : _calcIncrease(row,xValues,0), axis=1)
                   
                    X.index = dataSubset.index
                    X.columns = ["fit:({}):{}".format(groupNameComp,colName) if repID == "None" and len(replicateGrouping) == 1 else "fit:({}):{}_{}".format(groupNameComp,colName,repID) for colName in X.columns.values]
                    
                    if normalization != "None":
                        dataSubset.columns = ["norm:fitModel:{}".format(colName) for colName in dataSubset.columns]
                        self.sourceData.joinDataFrame(dataID,dataSubset)
                    
                    returnProps = self.sourceData.joinDataFrame(dataID, X)

            returnProps["messageProps"] = getMessageProps("Done..","Model fitting performed. Columns added to the data frame.")["messageProps"]
            return returnProps

        except Exception as e:
            print(e)
            return getMessageProps("Error..","Time group could not be interpreted.")
          


    def runCategoricalFisherEnrichment(self,data,categoricalColumn, testColumns, splitString = ";", alternative = "two-sided"):
        ""
        # import time
        # print(dataID)
        # print(testColumns)
        # print(categoricalColumn)
        # columnNames = [categoricalColumn] + testColumns.values.flatten().tolist() 
        # print(columnNames)
       # data = self.getData(dataID,columnNames).copy()
        
        #freeze_support()
        groupedByData = data.groupby(by=categoricalColumn)
        minGroupSize = self.sourceData.parent.config.getParam("categorical.enrichment.min.group.size")
        adjPvalueCutoff = self.sourceData.parent.config.getParam("categorical.enrichment.adj.pvalue.cutoff")
        NProcesses = self.sourceData.parent.config.getParam("n.processes.multiprocessing")
        adjPvalueMethod = self.sourceData.parent.config.getParam("categorical.enrichment.multipletest.method")
        
        results = pd.DataFrame(columns=["p-value(Fisher)",
                                        "p-value(Chi2)",
                                        "{} p-value(Fisher)".format(adjPvalueMethod),
                                        "{} p-value(Chi2)".format(adjPvalueMethod),
                                        "oddsratio",
                                        "testColumn",
                                        "category",
                                        "groupName",
                                        "n_positive(group)",
                                        "n_positive(total)",
                                        "n_negative(group)",
                                        "n_negative(total)",
                                        "groupSize",
                                        "totalSize"])

        for testColumn in testColumns.values.flatten():
            #print(testColumn)
            #get unique categories for non nan groups (e.g. by default -)
            splitData = data.loc[data[testColumn] != self.sourceData.replaceObjectNan,testColumn].astype("str").str.split(splitString).values
            uniqueCategories = list(set(chain.from_iterable(splitData)))
            

            regExByCategory = dict([(uniqueCategory,buildRegex([uniqueCategory],True,splitString)) for uniqueCategory in uniqueCategories])
            #most beneift from using Parallel for this job. Others were not faster (fisher test etc, using "normal" protomics data (e.g. 4-10K rows))
            X = data[testColumn]

            A = np.array(list(regExByCategory.items()))
            As = np.array_split(A,NProcesses,axis=0)

            with Pool(NProcesses) as p:
                r = p.starmap(_matchMultipleRegExToPandasSeries,[(X, A[:,1], A[:,0]) for A in As])
                poolResult =list(chain.from_iterable(r))
               
            boolIdxByCategory = dict(poolResult)
      
            #t1 = time.time()
            # print("joblib started")
            # boolIdxByCategory = dict(Parallel(n_jobs=8,backend="multiprocessing")(delayed(_matchRegExToPandasSeries)(data[testColumn], regExp,uniqueCategory) for uniqueCategory, regExp in regExByCategory.items()))
            # print(time.time()-t1)
            for groupName, groupData in groupedByData:
                if groupName == self.sourceData.replaceObjectNan:
                    continue
                groupSize = groupData.index.size
                overallDataSize = data.index.size
                r = []

                for category,boolIdx in boolIdxByCategory.items():
                   # print(category,boolIdx)
                   # print(groupData)
                   
                    
                    categoryInGroup = np.sum(data[boolIdx.values].index.isin(groupData.index))
                    if categoryInGroup <= minGroupSize:
                        continue
                    categoryNotInGroup = groupSize-categoryInGroup

                    categoryInCompleteData = np.sum(boolIdx)
                    categoryNotInCompleteData = overallDataSize - categoryInCompleteData
                    
                    #table = np.array([[categoryInGroup,categoryNotInGroup],[categoryInCompleteData , categoryNotInCompleteData]])
                    table = np.array([[categoryInGroup,categoryInCompleteData],[groupSize , overallDataSize]])
                    
                    oddsratio, pvalue = fisher_exact(table,alternative=alternative)
                    chi2,chiPValue,_,_ = chi2_contingency(table)
                   
                    # print(category)
                    # print(np.sum(boolIdx))

                    # print(table)
                    r.append({"p-value(Fisher)":pvalue,
                                "p-value(Chi2)":chiPValue,
                                "oddsratio":oddsratio,
                                "testColumn":testColumn,
                                "category":category,
                                "groupName":groupName,
                                "n_positive(group)":categoryInGroup,
                                "n_positive(total)":categoryInCompleteData,
                                "n_negative(group)":categoryNotInGroup,
                                "n_negative(total)":categoryNotInCompleteData,
                                "groupSize":groupSize,
                                "totalSize":overallDataSize})
                
                if len(r) > 0:
                    r = pd.DataFrame(r)
                    
                    boolIdxFisher, adjFisherPValue, _, _  = multipletests(r["p-value(Fisher)"].values,method=adjPvalueMethod,alpha=adjPvalueCutoff)
                    boolIdxChi, adjChi2PValue, _, _  = multipletests(r["p-value(Chi2)"].values,method=adjPvalueMethod,alpha=adjPvalueCutoff)
                   
                    r["{} p-value(Fisher)".format(adjPvalueMethod)] = adjFisherPValue.astype(float)
                    r["{} p-value(Chi2)".format(adjPvalueMethod)] = adjChi2PValue.astype(float)
                    
                    boolIdxAdjPvalue = boolIdxFisher | boolIdxChi
                    r = r.loc[boolIdxAdjPvalue,:]
                    if r.index.size > 0:
                        results = results.append(r,ignore_index=True)
        if results.index.size == 0:
            return getMessageProps("Error","No significant hits found.")
        return self.sourceData.addDataFrame(results,fileName="FisherEnrichmentResults({})".format(categoricalColumn))
       
    def runBatchCorrection(self,dataID, groupingName,*args,**kwargs):
        ""
        columnNames = self.sourceData.parent.grouping.getColumnNames(groupingName)
        data = self.getData(dataID,columnNames).dropna() 
        factorizedColumns = self.sourceData.parent.grouping.getFactorizedColumns(groupingName)
        batchIdx = [factorizedColumns[colName] for colName in columnNames if colName in factorizedColumns]
        with Pool(1) as p:
                poolResult = p.starmap(runCombat,[(data,batchIdx)])
        XCorrected = poolResult[0]

        #attach all other data
        rawDf = self.sourceData.dfs[dataID]
        columnsToJoin = [colName for colName in rawDf.columns if colName not in XCorrected.columns]
        XCorrected = XCorrected.join(rawDf[columnsToJoin],how="inner")
        fileName = self.sourceData.getFileNameByID(dataID)
        
        return self.sourceData.addDataFrame(XCorrected,fileName = "combat({})".format(fileName))

    def runOneDEnrichment(self,dataID,columnNames,categoricalColumns, splitString = ";" , alternative="two-sided"):
        ""
        combinedColumnnames = columnNames.values.tolist() + categoricalColumns.tolist()
        data = self.getData(dataID,combinedColumnnames)
        data = data.dropna(subset=columnNames,how="all")
        minGroupSize = self.sourceData.parent.config.getParam("1D.enrichment.min.category.group.size")
        minGroupDifference = self.sourceData.parent.config.getParam("1D.enrichment.min.abs.group.difference")
        NProcesses = self.sourceData.parent.config.getParam("n.processes.multiprocessing")
        for categoricalColumn in categoricalColumns:
            
            splitData = data[categoricalColumn].astype("str").str.split(splitString).values
            uniqueCategories = list(set(chain.from_iterable(splitData)))
            regExByCategory = dict([(uniqueCategory,buildRegex([uniqueCategory],True,splitString)) for uniqueCategory in uniqueCategories])
            #boolIdxByCategory = dict([(uniqueCategory,data[categoricalColumn].str.contains(regExp, case = True)) for uniqueCategory, regExp in regExByCategory.items()])
            X = data[categoricalColumn]
            t1=time.time()
            with Pool(NProcesses) as p:
                poolResult = p.starmap(_matchRegExToPandasSeries,[(X, regExp, uniqueCategory) for uniqueCategory, regExp in regExByCategory.items()])
            boolIdxByCategory = dict(poolResult)
            
            
        resultDF  = pd.DataFrame(columns=["numericColumn","Category","p-value","U-statistic","categorySize","categorySize(noNaN)","mean","median","stdev","difference","-log10 p-value","adj. p-value"])
        r = []
        for numericColumn in columnNames.values:
            
            for uniqueCat, boolIdx in boolIdxByCategory.items():
                N = np.sum(boolIdx)
                if N >= minGroupSize: #check the group size
                    X = data.loc[boolIdx,numericColumn].dropna().values.flatten()
                    Y = data.loc[~boolIdx,numericColumn].dropna().values.flatten()
                    if X.size == 0 or Y.size == 0:
                        continue
                    uniqueCatMean = np.mean(X)
                    uniqueCatMedian = np.median(X)
                    stdev = np.std(X)
                    populationMean = np.mean(Y)
                    groupDifference = uniqueCatMean - populationMean
                    if minGroupDifference > 0 and abs(groupDifference) < minGroupDifference:
                        continue
                    
                    U, p = mannwhitneyu(
                                x = X,
                                y = Y,
                                alternative=alternative
                                )
                    
                    
                    r.append({"numericColumn":numericColumn,
                                "Category":uniqueCat,
                                "p-value":p,
                                "U-statistic":U,
                                "n":N,
                                "mean":uniqueCatMean,
                                "categorySize(noNaN)":X.size,
                                "median":uniqueCatMedian,
                                "categorySize":N,
                                "stdev":stdev,
                                "difference":groupDifference}
                                )
            
        resultDF = resultDF.append(r,ignore_index=True)
        pValues = resultDF["p-value"].values.flatten()
        resultDF.loc[:,"-log10 p-value"] = (-1)*np.log10(pValues)
        boolIdx, adjPValue, _, _  = multipletests(pValues,method="fdr_tsbh")
        resultDF.loc[:,"adj. p-value"] = adjPValue
        #filter results
        adjPCutoff = self.sourceData.parent.config.getParam("1D.enrichment.adj.p.value.cutoff")
        if adjPCutoff < 1:
            boolIdx = resultDF.loc[:,"adj. p-value"] < adjPCutoff
            resultDF = resultDF.loc[boolIdx,:]

        return self.sourceData.addDataFrame(
                resultDF,
                fileName="1DEnrichment:({})".format(self.sourceData.getFileNameByID(dataID))) 
            


    
    def runKMeansElbowMethod(self,dataID,columnNames,kMax = 20):
        ""
        with threadpool_limits(limits=1, user_api='blas'):
            data = self.getData(dataID,columnNames).dropna()
            ks = list(range(1,kMax+1))
            SSE = [KMeans(n_clusters=k).fit(data.values).inertia_ for k in ks]
            
            df = pd.DataFrame()
            df["k"] = ks
            df["SSE"] = SSE
            baseFileName = self.sourceData.getFileNameByID(dataID)
            return self.sourceData.addDataFrame(df,fileName="KMeansElbow::{}".format(baseFileName))
            #return getMessageProps("Error")


    def runKMeans(self,dataID,columnNames,k):
        ""
        with threadpool_limits(limits=1, user_api='blas'): #require to prevent crash (np.dot not thread safe)
            data = self.getData(dataID,columnNames).dropna()
            km = KMeans(n_clusters=k).fit(data.values)
            Y = km.transform(data.values)
            if self.sourceData.parent.config.getParam("report.distance.space"):
                df = pd.DataFrame(Y,index=data.index,columns = ["kM(k={}):CDist({})".format(k,n) for n in range(k)])
            else:
                df = pd.DataFrame()

            df["kM(k={}):ClusterID".format(k)] = ["C({})".format(n) for n in km.labels_]
            
            return self.sourceData.joinDataFrame(dataID,df)
           
    def runMultipleTestingCorrection(self,dataID,columnNames,method=None):
        ""
        config = self.sourceData.parent.config
        addCatColumn = config.getParam("mt.add.categorical.column")
        if method is None:
            method = config.getParam("mt.method")
        alpha = config.getParam("mt.alpha")
        with threadpool_limits(limits=1, user_api='blas'): #require to prevent crash (np.dot not thread safe)
            data = self.getData(dataID,columnNames)
            results = pd.DataFrame(index = data.index)
            for columnName in columnNames:
                noNanData = data[columnName].dropna()
                reject, corrPValues, sidakAlpha, bonfAlpha = multipletests(noNanData.values,alpha=alpha,method=method)
                results.loc[noNanData.index,"corr-p({},{}):{}".format(method,alpha,columnName)] = corrPValues
                if addCatColumn:
                    results.loc[noNanData.index,"sig({},{}):{}".format(method,alpha,columnName)] = ["+" if x else self.sourceData.replaceObjectNan for x in reject]
        return self.sourceData.joinDataFrame(dataID,results)
        #return getMessageProps("Done..","Multiple testing correction done.")

    def runRMOneTwoWayANOVA(self,dataID,withinGroupin1, withinGroupin2, subjectGrouping,*args,**kwargs):
        ""



    def runNWayANOVA(self,dataID,groupings):
        ""
        NProcesses = self.sourceData.parent.config.getParam("n.processes.multiprocessing")
        grouping = self.sourceData.parent.grouping
        columnNamesForGroupings  = [grouping.getColumnNames(groupingName = groupingName) for groupingName in groupings]
        
        if np.unique(groupings).size != groupings.size:
            return getMessageProps("Error..","Groupings must be unique and no duplicates are allowed.")
        if any(x.size != columnNamesForGroupings[0].size for x in columnNamesForGroupings):
            return getMessageProps("Error..","Groups must be of same length.")
        if groupings.size > 1:
            if not all(np.all(np.isin(columnNamesForGroupings[0],columnNamesForGrouping)) for columnNamesForGrouping in columnNamesForGroupings):
                return getMessageProps("Error..","All groupings must contain same column names..")

        dataFrame = self.getData(dataID, columnNamesForGroupings[0]).dropna(thresh=3)

        dataFrame["priorMeltIndex"] = dataFrame.index.values
        meltedDataFrame = dataFrame.melt(value_vars = columnNamesForGroupings[0], id_vars = "priorMeltIndex")
        
        columnNameMatchesByGrouping = grouping.getGroupingsByColumnNames(columnNamesForGroupings[0])
        r = None
        for groupName in groupings:
            #map groups to column names 
            mapDict = dict([(k,v) for k,v in zip(columnNamesForGroupings[0],columnNameMatchesByGrouping[groupName].values)])
            meltedDataFrame[groupName] = meltedDataFrame["variable"].map(mapDict)
        meltedDataFrame[groupings] = meltedDataFrame[groupings].astype(str)
        groupedDF = meltedDataFrame.groupby("priorMeltIndex")
        t1 = time.time()
        with Pool(NProcesses) as p:
            rPool = p.starmap(p_anova,[(indexDf,"value",groupings.tolist(),idx) for idx,indexDf in groupedDF])
           # print(rPool)
            idx, data = zip(*[r[0] for r in rPool if r is not None and r[0] is not None])
            sourceName = rPool[0][1]
            columnNames = ["p-unc ({})".format(idx) for idx in sourceName.values]
            #print(rPool)
        #print(rPool[0])
        if len(rPool) > 0:
            df = pd.DataFrame(data,index=idx,columns=columnNames)
            return self.sourceData.joinDataFrame(dataID,df)
           # print(df)
        else:
            return getMessageProps("Error..","All performed tests did not yield a valid result. No data attached.")
    
    
    def runMixedTwoWayANOVA(self,dataID,groupingWithin,groupingBetween,groupingSubject):
        ""
        
        
        grouping = self.sourceData.parent.grouping
        columnNamesWithin  = grouping.getColumnNames(groupingName = groupingWithin)
        columnNamesBetween  = grouping.getColumnNames(groupingName = groupingBetween)
        columnNamesSubject  = grouping.getColumnNames(groupingName = groupingSubject)

        if columnNamesWithin.size != columnNamesBetween.size and columnNamesWithin.size != columnNamesSubject.size:
            return getMessageProps("Error","Groupings must be of the same length and must contain same column names!")
        if not all(np.all(np.isin(columnNamesWithin.values,columnNamesForGrouping)) for columnNamesForGrouping in [columnNamesBetween.values,columnNamesSubject.values]):
            return getMessageProps("Error..","All groupings must contain the exact same column names..")

        dataFrame = self.getData(dataID, columnNamesWithin).dropna()
        dataFrame["priorMeltIndex"] = dataFrame.index.values
        
        meltedDataFrame = dataFrame.melt(value_vars = columnNamesWithin, id_vars = "priorMeltIndex")
        groupedDF = meltedDataFrame.groupby("priorMeltIndex")

        columnNameMatchesByGrouping = grouping.getGroupingsByColumnNames(columnNamesWithin)
        r = None
        for groupName in [groupingBetween,groupingWithin,groupingSubject]:
            mapDict = dict([(k,v) for k,v in zip(columnNamesWithin,columnNameMatchesByGrouping[groupName].values)])
            meltedDataFrame[groupName] = meltedDataFrame["variable"].map(mapDict)
        for index, indexDF in groupedDF:
            
                try:
                    aov = mixed_anova(dv="value", between=groupingBetween, within=groupingWithin, subject=groupingSubject, data=indexDF)
                    columnNames = ["p-unc ({})".format(idx) for idx in aov["Source"].values]
                    if r is None:
                        r = pd.DataFrame(index=dataFrame.index.values, columns = columnNames)
                    r.loc[index,columnNames] = aov["p-unc"].values.flatten()
                    
                except:
                    continue
        if r is None:
            return getMessageProps("Error..","All performed tests did not yield a valid result. No data attached.")
        r = r.astype(float)
        return self.sourceData.joinDataFrame(dataID,r)
       # return getMessageProps("Done","Mixed two way anova performed.")

    def runComparison(self,dataID,grouping,test,referenceGroup=None, logPValues = True):
        """Compare groups."""
        
        if test == "2W-ANOVA":
            return self.runANOVA(dataID, logPValues = logPValues, **grouping)
            
        #print(grouping)
        #print(test)
        colNameStart = {"euclidean":"eclD:","t-test":"tt:","Welch-test":"wt"}
  
        groupComps = self.sourceData.parent.grouping.getGroupPairs(referenceGroup)
        groupingName = self.sourceData.parent.grouping.getCurrentGroupingName()
        data = self.getData(dataID,self.sourceData.parent.grouping.getColumnNames())
        results = pd.DataFrame(index = data.index)
        if test == "1W-ANOVA":
            testGroupData = [data[columnNames].values for columnNames in grouping.values()]
            F,p = f_oneway(*testGroupData,axis=1)
            results["F({})".format(groupingName)] = F
            if logPValues:
                pValueColumn = "-log10-p-1WANOVA({})".format(groupingName)
                results[pValueColumn] = np.log10(p) * (-1)
                results[pValueColumn] = results[pValueColumn].replace(1.0,np.nan)
            else:
                results["p-1WANOVA({})".format(groupingName)] = p
        elif test == "2W-ANOVA":

            self.runANOVA(dataID)
        else:
            resultColumnNames = [colNameStart[test] + "({})_({})".format(group1,group0) for group0,group1 in groupComps]
            for n,(group0, group1) in enumerate(groupComps):
                columnNames1 = grouping[group0]
                columnNames2 = grouping[group1]
                X = data[columnNames1].values
                Y = data[columnNames2].values
                if test == "euclidean":

                    #dist
                    dist = X - Y 
                    #find complete nans and save them, otherwise sum returns 0 (e.g. zero distance)
                    nanBool = np.sum(np.isnan(X-Y),axis=1).reshape(-1,1) == columnNames1.size
                    # calculate euclidean distance
                    D = np.sqrt(np.nansum(dist ** 2, axis=1)).reshape(data.index.size,1)
                    #replace zero with nan
                    D[nanBool] = np.nan
                    results[resultColumnNames[n]] = D 
           
                elif test in ["t-test","Welch-test"]:

                    t, p = ttest_ind(X,Y,axis=1,nan_policy="omit",equal_var = test == "t-test")
                    if logPValues:
                        pValueColumn = "-log10-p-value:({})".format(resultColumnNames[n])
                        results[pValueColumn] = np.log10(p) * (-1)
                        results[pValueColumn] = results[pValueColumn].replace(1.0,np.nan)
                    else:
                        results["p-value:({})".format(resultColumnNames[n])] = p 
                    results["T-stat:({})".format(resultColumnNames[n])] = t
                    results["diff:({})".format(resultColumnNames[n])] = np.nanmean(Y,axis=1) - np.nanmean(X,axis=1) 

        return self.sourceData.joinDataFrame(dataID,results)

    def performPairwiseTest(self,XY,kind = "t-test", props = {}):
        X,Y = XY
        if X.values.size < 2 or Y.values.size < 3:
            return "Error, less than 2 values."
        if kind in ["t-test","Welch-test"]:
            return ttest_ind(X.values,Y.values,nan_policy="omit", equal_var=False if kind == "Welch-test" else True)
        elif kind == "(Whitney-Mann) U-test":
            
            return mannwhitneyu(X.values,Y.values)
        elif kind == "Wilcoxon":
            if X.values.size != Y.values.size:
                return "Error, samples must be of the same size (length)."
            return wilcoxon(X.values, Y.values)
        return
    
    def _getClusterKwargs(self,methodName):
        ""
        if methodName=="kmeans":
            return {
                "n_clusters":self.sourceData.parent.config.getParam("kmeans.default.number.clusters")
                }
        return {}

    def runCluster(self,dataID,columnNames,methodName,returnModel = False):
        "Clsuter analysis for cluster plot"
        if methodName in clusteringMethodNames:
            if methodName == "HDBSCAN":

                clusterLabels, data = self.runHDBSCAN(dataID,columnNames,attachToSource = False)
                clusterLabels.columns = ["Labels"]
            
            else:
                data = self.getData(dataID,columnNames).dropna()
                alg = clusteringMethodNames[methodName](**self._getClusterKwargs(methodName))
                alg.fit(data.values)
                if hasattr(alg,"labels_"):
                    clusterLabels = pd.DataFrame(alg.labels_, index=data.index,columns = ["Labels"])
            if returnModel:

                return clusterLabels,data,alg

            return clusterLabels, data

    def runManifoldEmbedding(self,dataID,columnNames,manifoldName,attachToSource=True):
        
        X, checkPassed, errMsg, dataIndex  = self.prepareData(dataID,columnNames)
        if checkPassed: 
    
            if manifoldName in manifoldFnName:

                embed = getattr(self,manifoldFnName[manifoldName])(X)
                if self.sourceData.parent.config.getParam("add.column.names.in.emb.name"):
                    compNames  = ["E({}):({}):_{}".format(manifoldName,mergeListToString(columnNames.values,","),n) for n in range(embed.shape[1])]  
                else:
                    compNames  = ["E({}):_{}".format(manifoldName,n) for n in range(embed.shape[1])]  
                df = pd.DataFrame(embed.astype(np.float64),index=dataIndex,columns = compNames)

                if not attachToSource:
                    return df, data
                else:
                    msgProps = getMessageProps("Done..","{} calculation performed.\nColumns were added to the tree view.".format(manifoldName))
                    result = {**self.sourceData.joinDataFrame(dataID,df),**msgProps}
                    return result

            return getMessageProps("Error..","Manifold unknown.")

        return getMessageProps("Error..", errMsg)
        
    def runIsomap(self,X):
        ""
        config  = self.sourceData.parent.config
        return Isomap(
                n_components = config.getParam("isomap.n.components"),
                n_neighbors = config.getParam("isomap.n.neighbors"),
                path_method= config.getParam("isomap.path.method")
                ).fit_transform(X)

    def runMDS(self,X):
        ""
        config  = self.sourceData.parent.config
        return MDS(
            n_components = config.getParam("mds.n.components"), 
            metric = config.getParam("mds.metric")).fit_transform(X)
        

    def runLLE(self,X,*args,**kwargs):
        ""
        config  = self.sourceData.parent.config
        return LocallyLinearEmbedding(
                    n_neighbors = config.getParam("locally.linear.n.neighbors"),
                    method = config.getParam("locally.linear.method"), 
                    n_components = config.getParam("locally.linear.n.components"),
                    neighbors_algorithm = config.getParam("locally.neighbors.algorithm"), *args, **kwargs).fit_transform(X)
                    
    def runSE(self,X,*args,**kwargs):
        ""
        config  = self.sourceData.parent.config
        return SpectralEmbedding(
                    n_components=config.getParam("spectral.embedding.n.components"), 
                    affinity=config.getParam("spectral.embedding.n.affinity"),
                    *args,**kwargs).fit_transform(X)


    def _nanEuclidianDistance(self,u,v):

        return np.sqrt(np.nansum((u - v) ** 2, axis=0))

    def runExponentialFit(self,dataID,fitType,timeGrouping,replicateGrouping = None,comparisonGrouping = None):
        ""
        groupings = [timeGrouping,replicateGrouping,comparisonGrouping]
        NProcesses = self.sourceData.parent.config.getParam("n.processes.multiprocessing")
        grouping = self.sourceData.parent.grouping
        timeGroupsByColumns = grouping.getGrouping(timeGrouping)
        columnNamesForGroupings  = [grouping.getColumnNames(groupingName = groupingName) for groupingName in groupings if groupingName is not None and groupingName != "None"]
        combinedGroupingColumns = pd.concat(columnNamesForGroupings)
        print(combinedGroupingColumns)

        #unique column names in groups
        columnNames = combinedGroupingColumns.unique().tolist()
        
        data = self.getData(dataID,columnNames)
        
        xtime = [(getNumberFromTimeString(groupName),groupColumns.values) for groupName,groupColumns in timeGroupsByColumns.items()]
        times = dict([(colName,t) for t, colNames in xtime for colName in colNames])
        

        #transform
        if replicateGrouping is None or replicateGrouping == "None":

            replicateGrouping = {"None":pd.Series(columnNames)}
            #normalize
        else:
            replicateGrouping = grouping.getGrouping(replicateGrouping)

        if comparisonGrouping == "None" or comparisonGrouping is None:

            comparisonGrouping = {"raw":pd.Series(columnNames)} 
        else:
            comparisonGrouping = grouping.getGrouping(comparisonGrouping)
        output = []
        for groupNameComp, groupColumnsComp in comparisonGrouping.items():
            for repID, replicateColumns in replicateGrouping.items():
                groupColumnsCompPerReplicate = pd.Series([colName for colName in groupColumnsComp.values if colName in replicateColumns.values and colName in times])
                dataCompRep = data[groupColumnsCompPerReplicate.values].dropna(thresh=3)# self.getData(dataID,groupColumnsCompPerReplicate).dropna(thresh=3)
                Ys = np.array_split(dataCompRep.values,NProcesses,axis=0)
                xdata = groupColumnsCompPerReplicate.map(times).values
                with Pool(NProcesses) as p:
                    
                    rs = p.starmap(fitExponentialToChunk,[(fitType,xdata,Y,n) for n,Y in enumerate(Ys)])
                    rs.sort(key=lambda x: x[0])  #sort by chunkIdx
                    
                    fitData = pd.concat([r[1] for r in rs])
                    fitData.index  = dataCompRep.index
                    fitData.columns = ["{}:_{}_{}".format(groupNameComp,colName,repID) for colName in ["A","k","b","r2"]]
                    output.append(fitData)
                    print(data)
		    #A = np.concatenate([r[1] for r in rs],axis=0)
        return self.sourceData.joinDataFrame(dataID,pd.concat(output,axis=1))

    def clusterData(self, data, metric = "euclidean", method = "complete"):
        '''
        Clusters the data for hierarchical clustering
        '''
        
        try:
            if not isinstance(data, np.ndarray):
                if isinstance(data, pd.DataFrame):
                    data = data.values
                else:
                    return None, None
            
            if  metric in  ['euclidean','correlation','ward','cosine']:   
                linkage = fastcluster.linkage(data, method = method, metric = metric)   
            
            elif metric == "nanEuclidean":
                i = data
                j = np.nansum((i - i[:, None]) ** 2, axis=2) ** .5
                linkage = fastcluster.linkage(j,method = method)
            elif metric == "nanCorrelation":
                i = data
                j = np.nansum((i - i[:, None]) ** 2, axis=2) ** .5
                linkage = fastcluster.linkage(j,method = method)
            else:
                distanceMatrix = scd.pdist(data, metric = metric)
                linkage = sch.linkage(distanceMatrix,method = method)
                        
        except:
                return None, None

        maxD = 0.7*max(linkage[:,2])
        return linkage, maxD 


    def setNJobs(self,n):
        ""
        if isinstance(n,int):
            self.n_jobs = n

    def _getCombinationsOfTwoLists(self,a,b):
        ""
        return [(x,y) for x in a for y in b]
        

    def runGroupCorrelations(self,dataID,groupingNames=[]):
        ""
        corrMethod = "pearson"
        columnNames = self.sourceData.parent.grouping.getUniqueGroupItemsByGroupingList(groupingNames)
        
        data = self.getData(dataID, columnNames = columnNames)
        corrMatrix = data.corr(method=corrMethod)
        #groupings = self.sourceData.parent.grouping.getGroupingsByList(groupingNames)
       #r = pd.DataFrame(columns=["Grouping","Group1","Group2","colName1","colName2","type","coeff"])
        rr = []
        for groupingName in groupingNames:
            #groupNameCombinations = self.sourceData.parent.grouping.getGroupPairsOfGrouping(groupingName)
            groupNameByColumn = self.sourceData.parent.grouping.getGroupNameByColumn(groupingName)
            columnNames = list(groupNameByColumn.keys())
            #withing groups 
            for colName1, colName2 in itertools.combinations(columnNames,r=2):
                if all(colName in corrMatrix.columns for colName in [colName1,colName2]):
                    r = corrMatrix.loc[colName1,colName2]
                    groupName1 = groupNameByColumn[colName1]
                    groupName2 = groupNameByColumn[colName2]
                    rr.append({
                        "Grouping":groupingName,
                        "Group1":groupName1,
                        "Group2":groupName2,
                        "colName1":colName1,
                        "colName2":colName2,
                        "coeff":r,
                        "type":"within" if groupName1 == groupName2 else "between"
                        })
        R = pd.DataFrame(rr)
        return self.sourceData.addDataFrame(R,fileName="GroupCorrelation") 
            




