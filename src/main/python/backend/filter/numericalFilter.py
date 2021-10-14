

from backend.utils.stringOperations import mergeListToString, getMessageProps


import pandas as pd 
import numpy as np




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
    

