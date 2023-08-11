

## permuations based statistics
from typing import Any
from numba import jit, njit, prange
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd 
from numpy.random import default_rng
import matplotlib.pyplot as plt 
import math
from itertools import combinations
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.interpolate import UnivariateSpline
from typing import List, Tuple
from abc import ABC, abstractmethod, abstractproperty
import time
from ..utils.tests import performFTest, performSAMTest
from ..utils.base import countValuesAboveThresh
    
@jit(nopython = True)
def exponentialDecay(x : np.ndarray|float, N : float, g : float) -> float:
    """
    Exponential decay fit
    """
    return N*np.exp(-x*g)

# inititate functions once to get speed up on first execution in InstnatCLue gui 
d = np.array([[2.4,2.5,2.5,2.8],[5.4,7.6,5.4,6.4]]).reshape(2,1,4)
performFTest(d)
performSAMTest(np.array([[2.3,2.5,2.6]]).reshape(1,3),np.array([[2.3,2.5,2.5]]).reshape(1,3),0.1)

class StatisticalTestABC(ABC):

    def __init__(self, 
                 dataID : str,
                 data : pd.DataFrame, 
                 groupingName : str|List[str], 
                 grouping  : dict,
                 name : str = None
                 ):
        """
        Parameter
        ===============
        :dataID ```str```The dataID from which the data are originally coming from.
        :data ```pd.DataFrame``` The data that should be used for the statistical test.
        :groupingName ```str``` or ```list``` The target groupingName(s).
        :grouping ```dict```The grouping to identify the different groups for the test (e.g. WT, KO).
        :name ```str``` The name of the statistical test.
        """
        self._statID : str = "asda"
        self._dataID = dataID
        self._result : pd.DataFrame = pd.DataFrame()
        self._groupingName = groupingName
        self._grouping = grouping
        self._data = data 
        self._dataShape = data.shape
        self._grouping = grouping
        self._name = name
        self._filteredData = pd.DataFrame()

    @abstractmethod
    def filterData(self,*args : Any, **kwargs : Any) -> pd.DataFrame:
        """
        Abstract method to filter the self._data data frame
        based on specific critera such as nan values.
        """

    @abstractmethod
    def fit(self,*args,**kwargs) -> Any:
        """
        Fits the data (e.g. calculates the statistical test)
        """

    @abstractmethod
    def getPlotNames(self) -> List[str]:
        """
        Returns the names of the test specific plots
        """
        return []
    

class StatisticalTest(StatisticalTestABC):
    """
    """
    def __init__(self, dataID: str, data: pd.DataFrame, groupingName: str | List[str], grouping: dict, name: str = None):
        super().__init__(dataID, data, groupingName, grouping, name)
    
    @property
    def ID(self) -> str:
        """
        Returns the unique id of the statistcal test (string).
        """
        return self._statID
    
    @property
    def name(self) -> str:
        """
        Returns the test name.
        """
        return self._name

    @property
    def result(self) -> pd.DataFrame:
        """
        Returns the result of the fit.
        If fit was not yet called, an empty data frame is returned.
        """
        return self._result


    

class PermutationBasedTest(StatisticalTest):
    """
    """
    def __init__(self, dataID: str, data: pd.DataFrame, groupingName: str | List[str], grouping: dict, name: str = None):
        super().__init__(dataID, data, groupingName, grouping, name)

    def generatePermutations(self, idces : np.ndarray) -> np.ndarray: 
        """
        Generate permutations without replacement based on the 
        column indices.
        """
        maxPerms = math.factorial(idces.size) - 1
        if maxPerms < self._permutations : self._permutations = maxPerms
        ll = np.empty(shape=(self._permutations,idces.size), dtype=int)
        ii = 0
        while ii < self._permutations:
            permIdces = self._rng.permutation(idces)
            if np.all(permIdces == idces):
                continue
            if np.any(np.sum(ll == permIdces, axis=1) == idces.size):
                continue
            else:
                ll[ii,:] = permIdces
                ii += 1 
            
        return ll 
    

class ANOVAStatistic(PermutationBasedTest):
    ""
    def __init__(self, dataID: str, data: pd.DataFrame, groupingName: str | List[str], grouping: dict, name: str = None, minValidInGroup : int = 3):
        super().__init__(dataID, data, groupingName, grouping, name)

        self._minValidInGroup = minValidInGroup
        self._groupColumns = [v for v in grouping[groupingName].values()]


    def filterData(self, groupColumns : List[List[str]]) -> pd.DataFrame:
        """
        Filters out data based on the min valid group (non NaN)
        """
        df = self._data
        for groupColumn in groupColumns:
            df = df.dropna(subset=groupColumn, thresh=self._minValidInGroup)

        self._filteredData = df 


    def fit(self) -> pd.DataFrame:
        """
        """
        self.filterData(self._groupColumns) #create self._filteredData object
        X = np.array([self._filteredData[groupColumn].values for groupColumn in self._groupColumns])
        #shape groups x features x replicates
        print(X)
        t1 = time.time()
        f = performFTest(X)
        print(time.time()-t1)
        print(f)
        t1 = time.time()
        f,p = stats.f_oneway(*X,axis=1)
        print(time.time()-t1)


        print(f)

    def getPlotNames(self) -> List[str]:
        return ["Heatmap"]




class SAMStatistic(PermutationBasedTest):

    def __init__(self, 
                 dataID: str, 
                 data: pd.DataFrame, 
                 groupingName: str | List[str], 
                 grouping: dict, 
                 leftGroup : str,
                 rightGroup : str,
                 name: str = "SAM-Statistics",
                 minValidInGroup : int = 3,
                 permutations : int = 500,
                 s0 : float = 0.1,
                 fdrCutoff : float = 0.05):
        super().__init__(dataID, data, groupingName, grouping, name)

        self._fdrCutoff = fdrCutoff
        self._leftGroup = leftGroup
        self._rightGroup = rightGroup
        self._minValidInGroup = minValidInGroup
        self._s0 = s0
        self._permutations = permutations
        self._rng = default_rng()
        

    def _addDataToResults(self, 
                          dataIndex : pd.Index,
                          leftGroupData : np.ndarray,
                          rightGroupData  : np.ndarray,
                          qValues  : np.ndarray,
                          testStatisticRealData : np.ndarray,
                          testStatisticPermutatedData  : np.ndarray,
                          degreeFreedomData : np.ndarray) -> None:
        ""
        self._result = pd.DataFrame(index = dataIndex)
        qValueName = self._getColumnNameSuffix("q-value")
        significantColumn = self._getColumnNameSuffix(f"Significant FDR {round(self._fdrCutoff*100)}%")
        self._result.loc[:,self._getColumnNameSuffix("log2FC")] = np.nanmean(leftGroupData,axis=1) - np.nanmean(rightGroupData,axis=1)
        self._result.loc[:,qValueName] = qValues
        samStatColumnName = self._getColumnNameSuffix("sam stat")
        self._result.loc[:,samStatColumnName] = testStatisticRealData

        #caculate regular t-test for volcano plot visualization
        self._result.loc[:,self._getColumnNameSuffix("t-test stat")] = performSAMTest(leftGroupData,rightGroupData,s0=0,sort=False)
        self._result.loc[:,self._getColumnNameSuffix("-log10 p-value")] = -np.log10(stats.t.sf(self._result[self._getColumnNameSuffix("t-test stat")].abs().values, df= degreeFreedomData)*2)

        #add expected sam statistic values
        self._result = self._result.sort_values(by=samStatColumnName)
        self._result.loc[:,self._getColumnNameSuffix("sam stat (expected)")] = np.mean(testStatisticPermutatedData,axis=1).values
        self._result.loc[:,significantColumn] = "-"
        self._result.loc[self._result[qValueName].values < self._fdrCutoff,significantColumn] = "+"

    def _getColumnNamesFromGroupings(self) -> Tuple[List[str],List[str]]:
        """
        Returns the columns names of the groupings.
        Checks if columns exists in data.
        """
        if self._groupingName not in self._grouping:
            raise ValueError(f"GroupingName {self._groupingName} not found in grouping.")
        
        if self._leftGroup not in self._grouping[self._groupingName]:
            raise ValueError(f"leftGroup {self._leftGroup} not found in grouping")
        
        if self._rightGroup not in self._grouping[self._groupingName]:
            raise ValueError(f"rightGroup {self._rightGroup} not found in grouping")
        
        leftColumnNames = self._grouping[self._groupingName][self._leftGroup]
        rightColumnNames = self._grouping[self._groupingName][self._rightGroup]
        #check if a columnNames are pandas series (+ does not work then.. transform to list)
        if isinstance(leftColumnNames,pd.Series):
            leftColumnNames = leftColumnNames.tolist()

        if isinstance(rightColumnNames,pd.Series):
            rightColumnNames = rightColumnNames.tolist()

        if not all(colName in self._data.columns for colName in leftColumnNames + rightColumnNames):
            raise ValueError("Not all columns that are defined in the grouping could be found in the data.")

        return (leftColumnNames,rightColumnNames)
    
    def _getColumnNameSuffix(self, colName : str) -> str:
        """
        Adds information to result column names including the group and s0 paramater
        """
        if colName in ["log2FC","-log10 p-value"]:
            return f"{colName} ({self._leftGroup}_{self._rightGroup})"
        
        return f"{colName} ({self._leftGroup}_{self._rightGroup})_s0({self._s0})"
    
    def _getGroupData(self, leftColNames : List[str], rightColNames : List[str]):
        """
        Filters data based on min non nan threshold and returns the group data.
        """
        return self._filteredData[leftColNames].values, self._filteredData[rightColNames].values, self._filteredData[rightColNames+leftColNames].values, self._filteredData.index

    def calculateQValue(self, 
                        testStatisticRealData : np.ndarray, 
                        absoluteSortedTestStatistic : np.ndarray, 
                        testStatisticPermutatedData : np.ndarray, 
                        absoluteStatsPermutations : np.ndarray, 
                        pi0 : float,
                        degreeFreedomData : np.ndarray,
                        degreeFreedomPermutatedData  : np.ndarray):
        """
        Caclulates QValues
        """
        Q = np.ones(shape=testStatisticRealData.size)
        deltas = [pd.Series([0,0,0,1.0,0], index=["FP","TP","d","FDR","minD"], name = -0.01)]
        #average test statistic from permutated data to calculate the test statistic delta
        averagedSorotedTestStatisticPerm = np.sort(np.mean(testStatisticPermutatedData,axis=1).abs().values)
        diff  = absoluteSortedTestStatistic - averagedSorotedTestStatisticPerm
        # get FDR for different d-statstics difference value
        for delta in np.arange(0, 10, 0.01):
            boolIdx = diff > delta
            if np.any(boolIdx):
                minD = np.min(absoluteSortedTestStatistic[diff > delta])
                TP =  countValuesAboveThresh(absoluteSortedTestStatistic,minD) 
                FP = self.countFalsePositives(absoluteStatsPermutations,minD)
                #FP = calculateFP(absoluteStatsPermutations,t_cut)
                FDR = pi0 * FP/TP
                if TP == 0:
                    print("TP equals zero, aborting FDR calculations for specific deltas")
                    break 
                r = pd.Series([FP,TP,delta,FDR,minD], index=["FP","TP","d","FDR","minD"], name = delta)
                deltas.append(r)
            else:
                break
        #combine results
        deltasDf = pd.concat(deltas,axis=1).T.drop_duplicates(subset=["minD"])
        pvaluesForFit = -np.log10(stats.t.sf(deltasDf["minD"].values, df=np.max(degreeFreedomData))*2)
       
        expoDecayFitParms, _ = curve_fit(exponentialDecay, pvaluesForFit,deltasDf["FDR"].values)
        #get param from fit
        N, g = expoDecayFitParms
        #calculate the minimum d value
        minD = -np.log(self._fdrCutoff/N)/g
        deltasDf["pvalue"] = pvaluesForFit
        deltasDf["qvalue"] = exponentialDecay(pvaluesForFit,*expoDecayFitParms)
        #estimate q values using a expontential fit
        absTestStatsRealData = np.abs(testStatisticRealData)
        pValuesForQValueFit = -np.log10(stats.t.sf(absTestStatsRealData, df=degreeFreedomData)*2)
        Q = exponentialDecay(pValuesForQValueFit,*expoDecayFitParms) 
        #correct fit errors, happens with low N and missing values

    #     pvaluesPerm = stats.t.sf(testStatisticPermutatedData.abs().values.flatten(),degreeFreedomPermutatedData.values.flatten())*2
    #     pvaluesRealData = stats.t.sf(absTestStatsRealData, df=degreeFreedomData)*2
    #     print(pvaluesPerm, pvaluesRealData)
    #     #pvaluesAvgPerm = stats.t.sf(averagedSorotedTestStatisticPerm, df = 6)*2
    #     Qs = calculateQValue(pvaluesRealData,pvaluesPerm, self._permutations)

    #    # Qs = np.array([countValuesBelowThresh(pValuesPerm,pvalue) / countValuesBelowThresh(pValuesRealData,pvalue) / self._permutations for pvalue in pValuesPerm])
     
    #     print(Qs)

        maxQValue = np.max(Q)
        if maxQValue > 1:
            Q = Q - (maxQValue - 1)
        Q[Q < 0] = 0.0
        return Q, minD
    
    def countFalsePositives(self, permutationStats : np.ndarray, minD : float) -> float:
        """
        Counts the number of False Positives. permutationsStats of shape (n_features x permutations)
        there the total number of values above minD is devided by the number of permutations.
        """
        FPs = countValuesAboveThresh(permutationStats,minD)
        numColumns = permutationStats.shape[1]
        return FPs/numColumns
    
    def estimatePi0(self, stats : np.ndarray, permutationStats : np.ndarray) -> float:
        """
        Estimates pi0, the fraction of expected features that will remain unchanged. 
        Expects absolute values of the permutated test.
        """ 
        N = stats.shape[0]
        absoluteStats = np.abs(stats)
        q25StatPerm, q75StatPerm = np.nanquantile(permutationStats, q=[0.25,0.75])

        statsInQRange = np.logical_and(absoluteStats >= q25StatPerm, absoluteStats <= q75StatPerm)
        nInRange = np.sum(statsInQRange)
        pi0 = nInRange / (0.5 * N) 
        if pi0 > 1: return 1 
        return pi0
    

    def estimateFDRLine(self, minD : float, numColumns : int) -> np.ndarray:
        """
        Estimates the FDR line for the volcano plot.
        """
        foldChangeRange =np.arange(0, 8, 0.01)
        lenFCRange = foldChangeRange.size
        standardDevRange =np.arange(0.005, 8, 0.005)
        A = np.empty(shape=(lenFCRange,4))
        A[:,0] = foldChangeRange #fold changes
        A[:,1] = np.zeros(shape=lenFCRange) ## stdev
        A[:,2] = np.zeros(shape=lenFCRange) ## t-values
        A[:,3] = np.ones(shape=lenFCRange) ## p-values
        for i,fc in enumerate(foldChangeRange):
            for standardDev in standardDevRange:
                d = fc / (standardDev + self._s0)
                if d >= minD:
                    if A[i,1] < standardDev:
                        tval = fc/standardDev
                        A[i,2] = tval
                        A[i,1] = standardDev 
        A[:,3] = stats.t.sf(A[:,2], df=numColumns - 2)*2
        boolIdx = A[:,3] < 1.0
        A = A[boolIdx,:]
        positiveFoldChanges = pd.DataFrame(A.copy(),columns=["foldChange","stdev","t-values","p-values"])
        #mirror estimated line
        A[:,0] = A[:,0] * (-1)
        negativeFoldChanges = pd.DataFrame(A,columns=["foldChange","stdev","t-values","p-values"])
        FDRLine = pd.concat([positiveFoldChanges,negativeFoldChanges], ignore_index=True)
        FDRLine["-log10 p-values"] = (-1) * np.log10(FDRLine["p-values"].values)
        self.FDRLine = FDRLine

    @property
    def fdrLine(self) -> pd.DataFrame:
        """
        """
        if not hasattr(self,"FDRLine"):
            raise AttributeError("FDRLine estimation not detected. Please call 'fit()' first.")
        
        return self.FDRLine
    
    def filterData(self, leftColNames : List[str], rightColNames : List[str]) -> pd.DataFrame:
        """
        Filters data and creae the _filteredData object which is used for the test.
        """
        df = self._data.dropna(subset=leftColNames, thresh = self._minValidInGroup)
        df = df.dropna(subset=rightColNames, thresh = self._minValidInGroup)
        self._filteredData = df

    def fit(self):
        """
        Fitting function of the SAM statistics. 
        a) calculate sam stat (d) on real data
        b) generate permutations (n = self._permutations)
        c) calculate sam statistics on permuted group values
        d) estimate pi0 (fraction of unchanged genes FDR = pi0 * FP / TP)
        e) Basae on sam stat (d) vs FDR fit an exponential decay function
        f) Estimate q-values from fit
        g) Collect results
        """
        leftColNames, rightColNames = self._getColumnNamesFromGroupings()
        self.filterData(leftColNames,rightColNames) #create self._filteredData
        numLeftColNames, numRightColNames = len(leftColNames), len(rightColNames)
        totalNumColumns = numLeftColNames + numRightColNames
        leftGroupData, rightGroupData, completeData, dataIndex = self._getGroupData(leftColNames,rightColNames)
        degreeFreedomData = np.sum(~np.isnan(leftGroupData),axis=1) + np.sum(~np.isnan(rightGroupData),axis=1) - 2

        testStatisticRealData = performSAMTest(leftGroupData,rightGroupData,s0=self._s0,sort=False)
        #sortedTestStatisticReal = np.sort(testStatisticRealData)
        absoluteSortedTestStatistic = np.sort(np.abs(testStatisticRealData))

        #generate permutated column idcs based on the total number of columns
        permutatedColumnIdcs = self.generatePermutations(np.arange(totalNumColumns))
        #calculate test statistic on permutated data
        permStats, permDf = self.performPermutationTests(
                        completeData,
                        perms=permutatedColumnIdcs,
                        s0=self._s0,
                        columnSizes=(numLeftColNames,numRightColNames))
        
        testStatisticPermutatedData = pd.DataFrame(permStats.reshape(-1,self._permutations))
        degreeFreedomPermutatedData = pd.DataFrame(permDf.reshape(-1,self._permutations))

        absoluteStatsPermutations = testStatisticPermutatedData.abs().values
        pi0 = self.estimatePi0(stats=testStatisticRealData, permutationStats=absoluteStatsPermutations)
        qValues, minD = self.calculateQValue(testStatisticRealData,
                                       absoluteSortedTestStatistic,
                                       testStatisticPermutatedData,
                                       absoluteStatsPermutations,pi0,
                                       degreeFreedomData,
                                       degreeFreedomPermutatedData
                                       )
        #add resuls to data frame (extra function?)
        self._addDataToResults(dataIndex,leftGroupData,rightGroupData,qValues,testStatisticRealData,testStatisticPermutatedData,degreeFreedomData)
        self.estimateFDRLine(minD, numColumns=totalNumColumns)

    def getPlotNames(self) -> List[str]:
        """
        Returns the list of plot names such as Volcano plot
        """
        return [self._getColumnNameSuffix("Volcano plot")]

    def performPermutationTests(self, completeData : np.ndarray, perms : np.ndarray, s0 : float, columnSizes : Tuple[int,int]) -> np.ndarray:
        """
        Performs the statistical sam test across the permutations.
        """
        permStats = np.zeros(shape=(completeData.shape[0],self._permutations))
        permDfs = np.zeros(shape=(completeData.shape[0],self._permutations))
        for n,p in enumerate(perms):
            X = completeData[:,p[:columnSizes[0]]]
            Y = completeData[:,p[columnSizes[0]:]]
            permStats[:,n] = performSAMTest(X,Y,s0,True,False)
            permDfs[:,n] = np.sum(~np.isnan(X),axis=1) + np.sum(~np.isnan(Y),axis=1) - 2
        return permStats, permDfs


# df = pd.read_csv("test-fdr.txt",sep="\t")
# df = df.set_index("Key")

# XNAMES = ["WT_01",	"WT_02",	"WT_03",	"WT_04"]
# YNAMES = ["NME6-KO_01",	"NME6-KO_02",	"NME6-KO_03",	"NME6-KO_04"]
# #YNAMES = ["NME6KO-WT_01",	"NME6KO-WT_02",	"NME6KO-WT_03",	"NME6KO-WT_04"]

# grouping = {"Genotype" : {"WT" : XNAMES,"KO":YNAMES}}
# groupingName = "Genotype"


# stat = SAMStatistic("das",df,groupingName,grouping,leftGroup="WT",rightGroup="KO",s0=0.1,fdrCutoff=0.05)
# stat.fit()

# print(stat.result)
# stat.result.to_clipboard()
# #print(ttest_ind(X,Y))
# #print(d)

