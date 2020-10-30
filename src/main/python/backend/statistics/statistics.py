from .dimensionalReduction.ICPCA import ICPCA
from .featureSelection.ICFeatureSelection import ICFeatureSelection
from ..utils.stringOperations import getMessageProps

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd
from scipy.stats import linregress, f_oneway, ttest_ind, mannwhitneyu, wilcoxon

from sklearn.cluster import KMeans

from threadpoolctl import threadpool_limits
from statsmodels.nonparametric.smoothers_lowess import lowess
import hdbscan
import umap
import re
import pandas as pd 
import numpy as np
#from cvae import cvae
import fastcluster


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
            self.X = pd.DataFrame(X)
        X = X.dropna()
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
        
        data = self.getData(dataID,columnNames)
        data.dropna(inplace=True)
        x = data.iloc[:,0].values
        y = data.iloc[:,1].values

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


    def runTSNE(self,dataID,columnNames,transformGraph = True, *args,**kwargs):
        ""
        try:
            data, checkPassed, errMsg, dataIndex  = self.prepareData(dataID,columnNames)
            if checkPassed:
                self.calcTSNE = TSNE_CALC(data.values,*args,**kwargs)
                embedding = self.calcTSNE.run()
            

            if transformGraph:
                df = pd.DataFrame(embedding,index=dataIndex,columns = ["TSNE_01","TSNE_02"])
                completeKwargs = getMessageProps("Done..","TSNE calculated.")
                result = self.sourceData.joinDataFrame(dataID,df)
                return result
        except Exception as e:
            print(e)

    def runCVAE(self,dataID,columnNames, transpose = False, *args,**kwargs):
        "Runs a Variational Autoencoder Dimensional Reduction"
        X, checkPassed, errMsg, dataIndex  = self.prepareData(dataID,columnNames)
        if checkPassed:
        
            embedder = cvae.CompressionVAE(X.values,)
            embedder.train() 
            embeddings = embedder.embed(X.values)
            compNames = ["Emb::CVAE_{:02d}".format(n) for n in range(2)]
            df = pd.DataFrame(embeddings.astype(np.float64),index=dataIndex,columns = compNames)
            msgProps = getMessageProps("Done..","CVAE (Variational Autoencoder) calculation performed.\nColumns were added to the tree view.")
                        
            result = {**self.sourceData.joinDataFrame(dataID,df),**msgProps}
    
            return result

        else:
                return getMessageProps("Error ..",errMsg)

    def runUMAP(self,dataID,columnNames, transpose = False, *args,**kwargs):
        
        with threadpool_limits(limits=1, user_api='blas'): #require to prevent crash (np.dot not thread safe)
            X, checkPassed, errMsg, dataIndex  = self.prepareData(dataID,columnNames)
            if checkPassed:
                    nN = self.sourceData.parent.config.getParam("umap.n.neighbors")
                    minDist = self.sourceData.parent.config.getParam("umap.min.dist")
                    metric = self.sourceData.parent.config.getParam("umap.metric")
                    nComp = self.sourceData.parent.config.getParam("umap.n.components")

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
    
    def runPCA(self,dataID,columnNames,initGraph = False, liveGraph = False, returnProjections = False, *args,**kwargs):
        "Transform values."
        X, checkPassed, errMsg, dataIndex  = self.prepareData(dataID,columnNames)
        if checkPassed:
            
            config = self.sourceData.parent.config
            nComps = config.getParam("pca.n.components")
            scaleData = config.getParam("pca.scale")
            
            with threadpool_limits(limits=1, user_api='blas'):
                self.calcPCA = ICPCA(X,n_components=nComps, scale=scaleData)
                embedding, eigV, explVariance = self.calcPCA.run()
            comColumns = ["PCA_Component_{:02d} ({:.2f})".format(n+1,explVariance[n]) for n in range(embedding.shape[1])]

            if initGraph:
                return checkPassed, pd.DataFrame(embedding,index=dataIndex,columns = comColumns), pd.DataFrame(eigV,index=columnNames,columns = comColumns)
            elif liveGraph:
                funcProps =  getMessageProps("Updatep","live updated")
                funcProps["data"] = {"projection":pd.DataFrame(embedding,index=dataIndex,columns = comColumns),
                                    "eigV":pd.DataFrame(eigV,index=columnNames,columns = comColumns)}
                return funcProps
            else:
                
                
                if returnProjections:
                   
                    df = pd.DataFrame(eigV, columns = ["Component {} ({:.2f}%)".format(n,explVariance[n]) for n in range(eigV.shape[1])])
                    try:
                        df["ColumnNames"] = columnNames
                        baseFileName = self.sourceData.getFileNameByID(dataID)
                    except Exception as e:
                        print(e)
                    result = self.sourceData.addDataFrame(df, fileName = "PCA.T:({})".format(baseFileName))
                else:
                    df = pd.DataFrame(embedding,columns = comColumns)
                    df[comColumns] = df[comColumns].astype(float)
                    result = self.sourceData.joinDataFrame(dataID,df)

                return result
        else:
                
            return getMessageProps("Error ..",errMsg)

    def runRowCorrelation(self,dataID,columnNames, indexColumns = ["T: PG.Genes"]):
        """
        Calculates correlation between all rows.
        Correlation coefficients will be filterd for NaN and threshold specified by
        the class attribute absCorrCoeff.
        """
        if columnNames.size < 3:
            return getMessageProps("Error..","Requires at least three columns.")
        #get correlation data
        data = self.getData(dataID,columnNames)
        #remove rows that have less than 3 values (corr would be always 1)
        data = data.dropna(thresh = 3)
        #calculate corr matrix
        corrMatrix = data.T.corr(method = self.corrMethod)
        #get cateogrical daata
        catData = self.sourceData.getDataByColumnNames(dataID,indexColumns,rowIdx=data.index)["fnKwargs"]["data"]
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

    def runHDBSCAN(self,dataID,columnNames):
        ""
        data = self.getData(dataID,columnNames).dropna()
        model = hdbscan.HDBSCAN(
                min_cluster_size=self.sourceData.parent.config.getParam("hdbscan.min.cluster.size"), 
                min_samples=self.sourceData.parent.config.getParam("hdbscan.min.samples"),
                cluster_selection_epsilon=self.sourceData.parent.config.getParam("hdbscan.cluster.selection.epsilon"))
        clusterLabels = model.fit_predict(data.values)
        df = pd.DataFrame(["C({})".format(x) for x in clusterLabels], columns = ["HDBSCAN"], index = data.index)
        
        return self.sourceData.joinDataFrame(dataID,df)

    def fitModel(self, dataID, columnNames, timeGrouping, compGrouping = None, replicateOrder = "columnOrder", model="First Order Kinetic"):
        "Fit Model to Data"
        data = self.getData(dataID,columnNames)
        try:
            xtime = [(groupName,groupColumns) if isinstance(groupName,int) or isinstance(groupName,float) else (float(groupName.split(" ")[0]),groupColumns)\
                                                                                                            for groupName,groupColumns in timeGrouping.items()] 
        except:
            return getMessageProps("Error..","Time group could not be interpreted.")
          
        print(xtime) 
        
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
           
    def runComparison(self,dataID,grouping,test,referenceGroup=None, logPValues = True):
        """Compare groups."""

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
                results["-log10-p-1WANOVA({})".format(groupingName)] = np.log10(p) * (-1)
            else:
                results["p-1WANOVA({})".format(groupingName)] = p
            
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
                        results["-log10-p-value:({})".format(resultColumnNames[n])] = np.log10(p) * (-1)
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

    def runCluster(self,dataID,columnNames):
        ""
            
    def _nanEuclidianDistance(self,u,v):

        return np.sqrt(np.nansum((u - v) ** 2, axis=0))

    def clusterData(self, data, metric = "euclidean", method = "complete"):
        '''
        Clusters the data
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
                linkage = sch.linkage(j,method = method)
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


