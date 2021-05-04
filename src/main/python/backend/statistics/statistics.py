from .dimensionalReduction.ICPCA import ICPCA
from .featureSelection.ICFeatureSelection import ICFeatureSelection
from ..utils.stringOperations import getMessageProps, getRandomString, mergeListToString
from backend.utils.stringOperations import getNumberFromTimeString
from backend.utils.misc import getKeyMatchInValuesFromDict
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd

from scipy.optimize import curve_fit
from scipy.stats import linregress, f_oneway, ttest_ind, mannwhitneyu, wilcoxon

from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering, Birch, AffinityPropagation
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding

from threadpoolctl import threadpool_limits
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests
#import pingouin as pg
from pingouin import anova
#from .anova.anova import Anova

import hdbscan
import umap
import re
import pandas as pd 
import numpy as np
#from cvae import cvae
import fastcluster
from collections import OrderedDict

clusteringMethodNames = OrderedDict([
                            ("kmeans",KMeans),
                            ("Birch",Birch),
                            ("OPTICS",OPTICS),
                            ("HDBSCAN",hdbscan.HDBSCAN),
                            ("Affinity Propagation",AffinityPropagation),
                            ("Agglomerative Clustering",AgglomerativeClustering)
                            ])

manifoldFnName = {"Isomap":"runIsomap","MDS":"runMDS","TSNE":"runTSNE","LLE":"runLLE","SpecEmb":"runSE"}

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

    def checkData(self,X):
        
        if X.empty:
            return False, "Filtering resulted in empty data matrix"

        if X.columns.size < 2:
            return False, "Need more than two numeric columns."

        if X.index.size < 3:
            return False, "Filtering resulted in data matrix with 2 rows."

        return True, "Check done.."

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


    def runLowess(self,dataID,columnNames):
        '''
        Calculates lowess line from dataFrame input
        '''
        data = self.getData(dataID,columnNames)
        data.dropna(inplace=True)
        x = data.iloc[:,0].values
        y = data.iloc[:,1].values

        lenX = x.size
        if lenX > 1000:
            it = 3
            frac = 0.65
        else:
            it = 1
            frac = 0.3
            
        lowessLine = lowess(y,x, it=it, frac=frac)

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

    def runRowCorrelation(self,dataID,columnNames, indexColumn = ["T: PG.Genes"]):
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
        corrMatrix = corrMatrix.unstack().reset_index()
        #remove nana
        corrMatrix = corrMatrix.dropna()
        if self.corrMethod in self.corrCoeffName:
            coeffName = self.corrCoeffName[self.corrMethod]
        else:
            coeffName = "CorrCoeff"
        corrMatrix.columns = ["Level 0","Level 1",coeffName]

        if self.absCorrCoeff > 0: 
            boolIdx = np.abs(corrMatrix["r"].values) > self.absCorrCoeff
            corrMatrix = corrMatrix.loc[boolIdx,]
        baseFileName = self.sourceData.getFileNameByID(dataID)

        return self.sourceData.addDataFrame(corrMatrix,fileName="corrMatrix::{}".format(baseFileName))

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
                    print(featureImportance)
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
        
        def _calcLineRegress(row, xValues,addDataAUC, addFitAUC):
            r = linregress(x = xValues, y=row)
            lRegress = r._asdict()
            if addDataAUC:
                lRegress["dataAUC"] = np.trapz(row,xValues)
            if addFitAUC:
                x = np.linspace(np.nanmin(xValues), np.nanmax(xValues), num = 200)
                y = x * r.slope + r.intercept
                lRegress["fitAUC"] = np.trapz(y,x)
            t12 = np.log(2)/r.slope * (-1)
            lRegress["halfLife"] = t12 if t12 > 0 else np.nan
            
            return pd.Series(lRegress)

        def _calcIncrease(row,xValues, corrK):
            ""
            def f(x,k):
                return 1 - np.exp(-(k+corrK)*x)
            print(corrK)
            popt, pcov = curve_fit(f,xValues,row)
            r = {"k":popt[0]}
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

                compGrouping = {"selectedColumns":pd.Series(columnNames)} 
           
            for groupNameComp, groupColumnsComp in compGrouping.items():
            
                for repID, replicateColumns in replicateGrouping.items():
                    
                    groupColumnsCompPerReplicate = [colName for colName in groupColumnsComp.values if colName in replicateColumns.values]
                    timeValues = [getKeyMatchInValuesFromDict(groupColumn,xtime) for groupColumn in  groupColumnsCompPerReplicate]
                    #remove vcolumns where not time was submitted
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
                    dataSubset = dataSubset.dropna()
                    
                    #corrK = 0.0303 if groupNameComp == "C" else 0
                    X = dataSubset.apply(lambda row, xValues = list(timeValueColumns.values()) : _calcLineRegress(row,xValues,addDataAUC,addFitAUC), axis=1)
                    #X = dataSubset.apply(lambda row, xValues = list(timeValueColumns.values()) : _calcIncrease(row,xValues,corrK), axis=1)
                   #
                    #print(X)
                    X.index = dataSubset.index
                    
                    X.columns = ["fit:({}):{}".format(groupNameComp,colName) if repID == "None" else "fit:({}):{}_{}".format(groupNameComp,colName,repID) for colName in X.columns.values]
                    if normalization != "None":
                        dataSubset.columns = ["norm:fitModel:{}".format(colName) for colName in dataSubset.columns]
                        self.sourceData.joinDataFrame(dataID,dataSubset)
                    
                    returnProps = self.sourceData.joinDataFrame(dataID, X)

            returnProps["messageProps"] = getMessageProps("Done..","Model fitting performed. Columns added to the data frame.")["messageProps"]
            return returnProps

        except Exception as e:
            print(e)
            return getMessageProps("Error..","Time group could not be interpreted.")
          
       
        
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
            
            if  metric ==  'euclidean':   
                linkage = fastcluster.linkage(data, method = method, metric = metric)   
            
            elif metric == "nanEuclidean":
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


