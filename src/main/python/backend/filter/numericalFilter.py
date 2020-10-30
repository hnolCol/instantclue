

from backend.utils.stringOperations import mergeListToString


import pandas as pd 
import numpy as np  

OPERATOR_OPTIONS = ["and","or"]

#stategy check if all indicies are present in filters?

funcFilter = {"Between":"findBetweenIndicies",
              "Not between":"findNotBetweenIndicies",
              "Greater than": "findGreaterThanIndicies",
              "Smaller than": "findSmallerThanIndicies",
              "n largest": "findNLargestIndices",
              "n smallest": "findNSmallestIndices"}


class NumericFilter(object):
    
    def __init__(self, sourceData, operator= "and"):
        ""
        self.sourceData = sourceData
        self.operator = operator
    

    def applyFilter(self, dataID, filterProps, subsetData = False):
        ""
        arr = np.array([])
        for columnName, filterProp in filterProps.items():            
            filterType = filterProp["filterType"]
            if filterType in funcFilter:
                data = self.sourceData.dfs[dataID][columnName]
                idx = getattr(self,funcFilter[filterType])(data,filterProp)
                arr = np.append(arr,idx.values)
                
        if self.operator == "or":
            filterIdx = np.unique(arr)
        elif self.operator == "and":
            filterIdx, counts = np.unique(arr, return_counts=True)
            filterIdx = filterIdx[np.where(counts == len(filterProps))]
        if subsetData:
            subsetName = "NumericSubset_({})_({})".format(mergeListToString(filterProps.keys()),self.sourceData.getFileNameByID(dataID))
            return self.sourceData.subsetDataByIndex(dataID, filterIdx, subsetName)
        else:
            annotationColumnName = "NumericFilter_({})".format(mergeListToString(filterProps.keys()))
            return self.sourceData.addAnnotationColumnByIndex(dataID, filterIdx , annotationColumnName)

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
        boolIdx = data > value
        return data.index[boolIdx]

    def findSmallerThanIndicies(self,data,props):
        ""
        value = props["min"]
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
    

