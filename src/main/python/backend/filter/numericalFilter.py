

from backend.utils.stringOperations import mergeListToString, getMessageProps


import pandas as pd 
import numpy as np
from numba import jit,prange

@jit(parallel=True)
def replaceNotDecreasingValueWithNaN(X,allowDiff = 0):
    Y = np.empty(shape=X.shape)
    for n in prange(X.shape[0]):
        lastNonNaN = 0
        x = X[n,:]
        for xi in prange(x.size):
            if xi == 0:
                Y[n,xi] = True
            elif np.isnan(X[n,xi]):
                Y[n,xi] = False
            elif X[n,lastNonNaN] - X[n,xi] > -allowDiff:
                Y[n,xi] = True
                lastNonNaN = int(xi)
            else:
                Y[n,xi] = False
    return Y



OPERATOR_OPTIONS = ["and","or"]

#stategy check if all indicies are present in filters?

funcFilter = {"Between":"findBetweenIndicies",
              "Not between":"findNotBetweenIndicies",
              "Greater than": "findGreaterThanIndicies",
              "Greater Equal than" : "findGreaterThanIndicies",
              "Smaller than": "findSmallerThanIndicies",
              "Smaller Equal than": "findSmallerThanIndicies",
              "n largest": "findNLargestIndices",
              "n smallest": "findNSmallestIndices"}

funcColumnName = {"Between":"[{},{}]",
              "Not between":"{}[]{}",
              "Greater than": ">{}",
              "Greater Equal than" : ">={}",
              "Smaller than": "<{}",
              "Smaller Equal than": "<={}",
              "n largest": "lrgst{}",
              "n smallest": "smll{}"}

minMaxForColumnName = {
              "Greater than": "max",
              "Greater Equal than" : "max",
              "Smaller than": "min",
              "Smaller Equal than": "min",
              "n largest": "max",
              "n smallest": "min"}

              
class NumericFilter(object):
    
    def __init__(self, sourceData, operator= "and"):
        ""
        self.sourceData = sourceData
        self.operator = operator
    

    def applyFilter(self, dataID, filterProps, setNonMatchNan = False, subsetData = False, selectedColumns = None):
        ""
        if setNonMatchNan:
            filterIdx = dict()
        else:
            arr = np.array([])
        for columnName, filterProp in filterProps.items():            
            filterType = filterProp["filterType"]
            if filterType in funcFilter:
                data = self.sourceData.dfs[dataID][columnName]
                idx = getattr(self,funcFilter[filterType])(data,filterProp)
                if setNonMatchNan:
                    filterIdx[columnName] = idx
                else:
                    arr = np.append(arr,idx.values)
        if setNonMatchNan:
            pass       
        elif self.operator == "or":
            filterIdx = np.unique(arr)
        elif self.operator == "and":
            filterIdx, counts = np.unique(arr, return_counts=True)
            filterIdx = filterIdx[np.where(counts == len(filterProps))]

        #get column name
        columnNameStr = []
        for columnName, filterProp in filterProps.items():
            filterType = filterProp["filterType"]
            if filterType in minMaxForColumnName:
                columnNameStr.append("{}".format(columnName) + funcColumnName[filterType].format(filterProp[minMaxForColumnName[filterType]]))
            else:
                columnNameStr.append("{}".format(columnName) + funcColumnName[filterType].format(filterProp["min"],filterProp["max"]))
        
        if subsetData:

            subsetName = "{}({})_({})".format(self.sourceData.parent.config.getParam("leading.string.numeric.filter.subset"),
                                                " ".join(columnNameStr),
                                                self.sourceData.getFileNameByID(dataID))

            return self.sourceData.subsetDataByIndex(dataID, filterIdx, subsetName)

        elif setNonMatchNan:

            return self.sourceData.setDataByIndexNaN(dataID,filterIdx,selectedColumns)

        else:
            annotationColumnName = "{}({})".format(self.sourceData.parent.config.getParam("leading.string.numeric.filter")," ".join(columnNameStr))
            return self.sourceData.addAnnotationColumnByIndex(dataID, filterIdx , annotationColumnName)


    def applyFilterForSelection(self,dataID,columnNames,metric,filterMode,filterProps,setNonMatchNan=False,subsetData=False):
        ""
        if setNonMatchNan:
            filterIdx = dict()
        else:
            arr = np.array([])

        filterType = filterProps["filterType"]
       
        data = self.sourceData.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
        if filterMode == "On individual columns":
            for columnName in columnNames.values:
                if not filterType in funcFilter:continue
                    
                idx = getattr(self,funcFilter[filterType])(data[columnName],filterProps)
                
                if setNonMatchNan:
                    filterIdx[columnName] = idx
                else:
                    arr = np.append(arr,idx.values)
            
            if setNonMatchNan:
                return self.sourceData.setDataByIndexNaN(dataID,filterIdx,columnNames)     
            elif self.operator == "or":
                filterIdx = np.unique(arr)
            elif self.operator == "and":
                filterIdx, counts = np.unique(arr, return_counts=True)
                filterIdx = filterIdx[np.where(counts == columnNames.size)]

            if subsetData:
                return self.sourceData.subsetDataByIndex(dataID, filterIdx, "subsetNumericFilter({})".format(self.sourceData.getFileNameByID(dataID)))
            else:
                return self.sourceData.addAnnotationColumnByIndex(dataID, filterIdx , "subsetNumericFilter")
        else:
            #summarize columns first
            summarizedValues = self.sourceData.transformer.summarizeTransformation(dataID,columnNames,metric=metric,justValues = True)

            filterIdx = getattr(self,funcFilter[filterType])(pd.Series(summarizedValues,index=self.sourceData.dfs[dataID].index),filterProps)
            
            self.sourceData.addColumnData(dataID,"summarizedFilterColumns({})".format(metric),summarizedValues)
            if subsetData:
                return self.sourceData.subsetDataByIndex(dataID, filterIdx, "subsetNumericFilter({}:{})".format(metric,self.sourceData.getFileNameByID(dataID)))
            elif setNonMatchNan:
                filterIdx = dict([(colName,filterIdx) for colName in columnNames.values])
                return self.sourceData.setDataByIndexNaN(dataID,filterIdx,None) 
            else:
                return self.sourceData.addAnnotationColumnByIndex(dataID, filterIdx , "subsetNumericFilter:({})".format(metric))

    def findConsecutiveValues(self,dataID,columnNames,increasing=True,annotationString=None):
        ""
        
        def diffWithNan(x,increasing,minValues):
            a = x[~np.isnan(x)]
            if a.size < minValues:
                return False
            else:
                if increasing:
                    return np.all(np.diff(a) > 0)
                else:
                    return np.all(np.diff(a) < 0)

        if columnNames.values.size < 2:
            return getMessageProps("Error..","Please select at least two numeric columns.")
        #get data and config params
        data = self.sourceData.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
        ignoreNaN = self.sourceData.parent.config.getParam("consecutive.values.ignore.nan")
        minValues = self.sourceData.parent.config.getParam("consecutive.values.min.non.nan")
        X = data.values
      
        if increasing:
            if ignoreNaN:
                filterIdx = np.apply_along_axis(diffWithNan,1,X,increasing,minValues)
            else:
                diff = np.diff(X,axis=1)
                filterIdx = np.all(diff > 0,axis=1)
            annotationColumnName = "consIncrease:({})".format(mergeListToString(columnNames.values) if annotationString is None else annotationString)
            return self.sourceData.addAnnotationColumnByIndex(dataID, filterIdx , annotationColumnName)
        else:
            if ignoreNaN:
                filterIdx = np.apply_along_axis(diffWithNan,1,X,increasing,minValues)
            else:
                diff = np.diff(X,axis=1)
                filterIdx = np.all(diff < 0,axis=1)
            annotationColumnName = "consDecrease:({})".format(mergeListToString(columnNames.values) if annotationString is None else annotationString)
            return self.sourceData.addAnnotationColumnByIndex(dataID, filterIdx , annotationColumnName)

    def filterConsecutiveValuesInGrouping(self,dataID,groupingName,increasing=True):
        ""
        grouping = self.sourceData.parent.grouping.getGrouping(groupingName)
        if grouping is not None:
            r = []
            cs = []
            for groupName, columnNames in grouping.items():
                #get data and config params
                data = self.sourceData.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
                X = data.values
                Y = replaceNotDecreasingValueWithNaN(X).astype(bool)
                X[~Y] = np.nan
                r.append(X)
                cs.extend(columnNames.values.tolist())

            rs = pd.DataFrame(np.concatenate(r,axis=1),columns=["conDecreaseFilter({}):{}".format(groupName,c) for c in cs],index=data.index)
            return self.sourceData.joinDataFrame(dataID,rs)

    def findConsecutiveValuesInGrouping(self,dataID,groupingName,increasing=True):
        ""
        grouping = self.sourceData.parent.grouping.getGrouping(groupingName)
        if grouping is not None:
            for groupName, columnNames in grouping.items():
                self.findConsecutiveValues(dataID,columnNames,increasing,annotationString=groupName)
            funcProps = getMessageProps("Done..","Consecutive values in grouping {} annotated.".format(groupingName))
            funcProps["columnNamesByType"] = self.sourceData.dfsDataTypesAndColumnNames[dataID]
            funcProps["dataID"] = dataID 
            return funcProps
        else:
            return getMessageProps("Error..","Grouping not found.")

    def findBetweenIndicies(self,data,props):
        ""
        boolIdx = data.between(props["min"],props["max"])
        return data.index[boolIdx]

    def findNotBetweenIndicies(self,data,props):
        ""
        boolIdx = data.between(props["min"],props["max"]) == False
        return data.index[boolIdx]

    def findNLargestIndices(self,data,props):
        ""
        N = int(props["max"])
        return data.nlargest(N).index

    def findNSmallestIndices(self,data,props):
        ""
        N = int(props["min"])
        return data.nsmallest(N).index

    def findGreaterThanIndicies(self,data,props):
        ""
        value = props["max"]
        if "Equal" in props["filterType"]:
            boolIdx = data >= value
        else:
            boolIdx = data > value
        return data.index[boolIdx]

    def findSmallerThanIndicies(self,data,props):
        ""
        value = props["min"]
        if "Equal" in props["filterType"]:
            boolIdx = data <= value
        else:
            boolIdx = data < value
        return data.index[boolIdx]

    def getOperator(self):
        ""
        return self.operator

    def getOperatorOptions(self):
        ""
        return OPERATOR_OPTIONS

    def setOperator(self,newOperator):
        ""
        if newOperator in OPERATOR_OPTIONS:
              setattr(self,"operator",newOperator)
    

