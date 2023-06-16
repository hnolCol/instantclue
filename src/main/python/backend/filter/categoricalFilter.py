import re
import pandas as pd 
import numpy as np 
import csv
import re
from itertools import chain

#internal imports
from .utils import buildRegex
from ..utils.stringOperations import mergeListToString, getMessageProps, buildReplaceDict, combineStrings
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

class CategoricalFilter(object):

    def __init__(self,
                sourceData,
                minStringLength = 3, 
                splitString = ";"):

        self.splitString = splitString
        self.minStringLength = minStringLength
        self.sourceData = sourceData
        self.replaceDict = {True : "+",
                            False: self.sourceData.replaceObjectNan}


    def annotateCategories(self,dataID,columnNames,searchString, operator = "and", splitString = None, inputIsRegEx=False):
        ""
        #search string is a dict (columnName -> categories)
        resultsCollection = pd.DataFrame(columns = columnNames)
        for columnName in columnNames:
            
            boolIndicator = self.searchCategory(dataID,columnName,searchString[columnName],splitString,inputIsRegEx)
            resultsCollection[columnName] = boolIndicator

        if operator == "and":
            boolIdx = np.sum(resultsCollection.values,axis=1) == len(columnNames)
        else:
            boolIdx = np.sum(resultsCollection.values,axis=1) > 0

        annotationColumn = pd.Series(boolIdx).map(self.replaceDict)
        #generate new columnName
        columnStr = mergeListToString([mergeListToString(v,",") for v in searchString.values()])
        annotationColumnName = "{}:({}):({})".format(columnStr,operator,mergeListToString(columnNames))
        return self.sourceData.addColumnData(dataID,annotationColumnName,annotationColumn)

    def annotateCategory(self,dataID,columnName,searchString, splitString = None, inputIsRegEx=False):
        ""
        
        boolIndicator = self.searchCategory(dataID,columnName,searchString,splitString,inputIsRegEx)
        
        #if searching the category gave an error, error message is a dict
        if isinstance(boolIndicator,dict):
            return boolIndicator
        else:
            #generate new columnName
            columnStr = mergeListToString(searchString)
            annotationColumnName = "{}:{}".format(columnStr,columnName)
            #replace bool with string
            annotationColumn = boolIndicator.map(self.replaceDict)
        return self.sourceData.addColumnData(dataID,annotationColumnName,annotationColumn)

    def buildRegex(self, stringList, withSeparator = True, splitString = None, matchingGroupOnly = False):
        ""
        return buildRegex(stringList, withSeparator, splitString, matchingGroupOnly)

    def columnsContainString(self,dataID, columnNames, regExp, caseSensitive):
        "Reutrns bool per row if string is is found in any column"
        collectResults = pd.DataFrame()
        #check each column if str is in row
        for columnName in columnNames:
            columnBoolIndicator = self.sourceData.dfs[dataID][columnName].astype("str").str.contains(regExp, case = caseSensitive)
            collectResults[columnName] = columnBoolIndicator
        # get bool where in at leas 1 column the string was found
        boolIndicator = collectResults.sum(axis=1) >= 1
        return boolIndicator


    def getGroupIndicator(self,dataID,columnName, regExp, flag=0):
        ""
        return self.sourceData.dfs[dataID][columnName].astype("str").str.findall(regExp, flags = flag).astype(str)	

    def getUniqueCategories(self, dataID, columnName, splitString = None):
        ""
            #if no splitString is given, take init param
        if splitString is None:
            splitString = self.splitString
        if len(columnName) == 0:
            return getMessageProps("Error..","There was an internal error. Please try again using the context menu.")
        if dataID in self.sourceData.dfs:
            if isinstance(columnName,str):
                columnName = [columnName]
            #take pandas daata for column, and split data on splitString
            splitData = self.sourceData.getDataByColumnNames(dataID,columnName)["fnKwargs"]["data"][columnName[0]].astype("str").str.split(splitString).values
            #get unique values 
            flatSplitDataList = list(set(chain.from_iterable(splitData)))
            #create data from unique categories.
            flatSplitData = pd.DataFrame(flatSplitDataList,columns=columnName)
            #return data
            return flatSplitData
        else:
            return getMessageProps("Not found","DataID was not found.")
       
    def getRegExpFromSepSearchString(self,searchString,withSeparator = False, splitString = None):
         # split search string according to "string1","string 2"
        splitString = self.splitString if splitString is None else splitString
        splitSearchString = [row for row in csv.reader([searchString], 
                                        delimiter=',', quotechar='\"')][0]
        # create reg expresion
        
        regExp = self.buildRegex(splitSearchString,withSeparator,splitString,matchingGroupOnly=True)
        return regExp, splitSearchString
    
    def getSplitString(self):
        ""
        return self.splitString


    def searchCategory(self,dataID,columnName, searchString, splitString = None, inputIsRegEx = False):
        ""

        if dataID not in self.sourceData.dfs:
            return getMessageProps("Not found","DataID was not found.")

        if splitString is None:
            splitString = self.splitString

        if isinstance(searchString, str):
            searchString = [searchString]

        if isinstance(searchString,np.ndarray):
            searchString = searchString.tolist()
     
        if len(searchString) == 0:
            
            return np.ones(shape=self.sourceData.dfs[dataID].index.size,dtype=np.bool)

        if not isinstance(searchString,list):
            raise ValueError("serach string must be list or string!")

        if columnName in self.sourceData.dfs[dataID].columns:

            #create regeluar expression
            if not inputIsRegEx:
                regExp = self.buildRegex(searchString,withSeparator=True,splitString=splitString,matchingGroupOnly=True)
            else:
                regExp = searchString
            boolIndicator = self.columnsContainString(dataID, [columnName], regExp, caseSensitive = True)
            # check if filter worked or empty df resulted
            if np.sum(boolIndicator) == 0:
                
                return getMessageProps("Empty","Data frame was empty after applying filter.")
            
            return boolIndicator

    def searchString(self, dataID, columnNames, searchString, caseSensitive = True ,inputIsRegEx = False, annotateSearchString = False, firstMatch = False, searchStringIsList = False):
        ""

        if dataID not in self.sourceData.dfs:
            return getMessageProps("Not found","DataID was not found.")

        if isinstance(columnNames,str):
            columnNames = [columnNames]
       
        if inputIsRegEx:

            regExp = searchString
            splitSearchString = ["reg"]
        else:

            regExp, splitSearchString = self.getRegExpFromSepSearchString(searchString,withSeparator=False)
        
       

        if annotateSearchString:
            #set up flags for search
            flag = 0 if caseSensitive else re.IGNORECASE
            collectResults = pd.DataFrame()
			
            if len(splitSearchString) > 1:
                #check i user entered more than one search string
                if firstMatch: 
                    #annotate only first match
                    for columnName in columnNames:
                        #using find all for regexp
                        groupIndicator  = self.getGroupIndicator(dataID,columnName,regExp,flag)					
                        #get unique values
                        uniqueValues = groupIndicator.unique()						
                        #ceate a dict containing unique values of matches and search input
                        replaceDict = buildReplaceDict(uniqueValues,splitSearchString)
                        # replace groupIndicators with input (search string)
                        annotationColumn = groupIndicator.map(replaceDict)				
                        #save result
                        collectResults[columnName] = annotationColumn
                else:
                    
                    for columnName in columnNames:
                        #extract combinations of searches
                        groupIndicator  = self.sourceData.dfs[dataID][columnName].astype("str").str.extract(regExp, flags = flag)
                        annotationColumn = groupIndicator.fillna('').astype(str).sum(axis=1)
                        collectResults[columnName] = annotationColumn

                if len(columnNames) == 1:
					# simply replaces annotation nan or empty strings with nan object string
                    # case : single search column
                    annotationColumn = \
					annotationColumn.replace('',self.sourceData.replaceObjectNan).fillna(self.sourceData.replaceObjectNan)
                else:
                    # if string search was performed in multiple columns, combine strings
                    collectResults['annotationColumn'] = \
					    collectResults.apply(lambda x: combineStrings(x, nanObjectString = self.sourceData.replaceObjectNan), axis=1)
                    annotationColumn = collectResults['annotationColumn']
            else:
                replaceDict = self.replaceDict.copy()
                replaceDict[True] = splitSearchString[0]
                for columnName in columnNames:
                 
                    columnBoolIndicator = self.sourceData.dfs[dataID][columnName].astype("str").str.contains(regExp, case = caseSensitive)
                    collectResults[columnName] = columnBoolIndicator
                boolIndicator = collectResults.sum(axis=1) >= 1
                annotationColumn = boolIndicator.map(replaceDict)	
        else:

            ## simply label rows that match by "+"
            boolIndicator = self.columnsContainString(dataID,columnNames,regExp,caseSensitive)
            #replace bool with string
            annotationColumn = boolIndicator.map(self.replaceDict)
		# generate new column name
        annotationColumnName = """{}:""".format(searchString) + mergeListToString(columnNames, joinString = "_")
        #replace " in column name
        annotationColumnName = annotationColumnName.replace('"','')
        #add column to source data and return output
       
        return self.sourceData.addColumnData(dataID,annotationColumnName,annotationColumn)

    def setSplitString(self,splitString):
        "Set SplitString."
        if isinstance(splitString,str):
            self.splitString = splitString
        else:
            self.splitString = str(splitString)

    def setupLiveStringFilter(self,dataID,columnNames,splitString = None, filterType = "category", updateData = False):
        ""
        if hasattr(self,"liveSearchData") and not updateData:
            return getMessageProps("Live filter found.","Another live filter active. Not allowed ...")

        if dataID not in self.sourceData.dfs:
            return getMessageProps("Not found","DataID was not found.")

        # check if input is str, convert to list
        if isinstance(columnNames,str):
            columnNames = [columnNames]
        elif isinstance(columnNames,pd.Series):
            columnNames = columnNames.values.tolist()
        # use class splitString if None given
        if splitString is None:
            splitString = self.splitString
        
        if filterType not in ["category","string","multiColumnCategory"]:
            return getMessageProps("Error ..","Unknown filter type selected.")

        #if more than one columnName is given, only string type filtering works
        if len(columnNames) > 1 and filterType == "category":
            filterType = "multiColumnCategory"
        
        if filterType == "string":
            
            searchData = self.sourceData.dfs[dataID][columnNames]

        elif filterType == "multiColumnCategory":
            uniqueCategoriesPerColumn = [self.getUniqueCategories(dataID, columnName, splitString).reset_index(drop=True) for columnName in columnNames]
            searchData = pd.concat(uniqueCategoriesPerColumn,ignore_index=True,axis=1)
            searchData.columns = columnNames
           
        elif filterType == "category":
            searchData = self.getUniqueCategories(dataID, columnNames, splitString)

        # if error, get unique categories returns dict
        if isinstance(searchData,dict):
        
            return searchData

        else:
            self.liveSearchData = searchData
            self.filterProps = {"type":filterType,"dataID":dataID,"columnNames":columnNames,"splitString":splitString}
            self.savedLastString = ''
            return filterType
            

    def liveStringSearch(self,searchString, updatedData = None,forceSearch = False, inputIsRegEx = False, caseSensitive = False):
        ""
        if hasattr(self,"liveSearchData"):
            dataToSearch = pd.DataFrame()
            resetDataInView = False
            nonEmptyString = searchString != ''
            #get length of search string
            lenSearchString = len(searchString)
            if not nonEmptyString and self.savedLastString != '':
                forceSearch = True
            if lenSearchString < self.minStringLength and nonEmptyString and not forceSearch:
                ## to avoid massive searching
                return
            
            if lenSearchString > 2:
                ## to start a new String search
                if searchString[-2:] == ',"':			
                    self.savedLastString = ''

            #get length of saved string
            lengthSaved = len(self.savedLastString)
            if updatedData is not None and self.savedLastString != '' and not forceSearch:
                if abs(lenSearchString - lengthSaved) == 1:
                    #if searchString is extended by user, use data from first search 
                    #or when backspace is used
                    dataToSearch = updatedData
                    # if data is for some reason of length 0, use full data
                    if len(dataToSearch.index) == 0:
                        dataToSearch = self.liveSearchData
                        resetDataInView = True
                    else:
                        resetDataInView = False
               
            else:
                dataToSearch = self.liveSearchData
                resetDataInView = True

      
            if inputIsRegEx:
                regExp = re.escape(searchString)
            else:
                regExp, _ = self.getRegExpFromSepSearchString(searchString, 
                                                            withSeparator = False)
                                                            
            collectDf = pd.DataFrame()
            for columnName in dataToSearch.columns:
               
                collectDf.loc[:,columnName] = \
					dataToSearch.loc[:,columnName].astype("str").str.contains(regExp,
													  case = caseSensitive,
                                                      regex = True)
                    
            #if only one column is searched, bool indicator = first column
            if len(collectDf.columns) == 1:
                boolInd = collectDf.iloc[:,0].values
            else:
                boolInd = collectDf.sum(axis=1) >= 1	
			
            #save search string
            self.savedLastString = searchString	
            
            return {"boolIndicator":boolInd,"resetData":resetDataInView}
        return getMessageProps("Filter not initialized","Filter not yet init.")	
			
    def applyLiveFilter(self, searchString = None, caseSensitive = True, annotateSearchString = False, inputIsRegEx = False, firstMatch = True, operator = "and"):
        ""
        try:
   
            if hasattr(self,"filterProps") and self.filterProps["type"] in ["category","string","multiColumnCategory"]:
                
                if searchString is None:
                    searchString = self.savedLastString

                if self.filterProps["type"] == "multiColumnCategory":
                    requestResponse = self.annotateCategories(
                            searchString = searchString,
                            dataID = self.filterProps["dataID"],
                            columnNames = self.filterProps["columnNames"],
                            splitString = self.filterProps["splitString"],
                            operator = operator)

                elif self.filterProps["type"] == "category":
                    requestResponse = self.annotateCategory(
                            searchString = searchString,
                            dataID = self.filterProps["dataID"],
                            columnName = self.filterProps["columnNames"][0],
                            splitString = self.filterProps["splitString"])
                else:
                    requestResponse = self.searchString(
                                    searchString = searchString,
                                    dataID = self.filterProps["dataID"],
                                    columnNames = self.filterProps["columnNames"],
                                    caseSensitive = caseSensitive,
                                    annotateSearchString = annotateSearchString,
                                    inputIsRegEx = inputIsRegEx,
                                    firstMatch = firstMatch)

                return requestResponse
        except Exception as e:
            print(e)

    def stopLiveFilter(self):
        ""
        if hasattr(self,"liveSearchData"):
            del self.liveSearchData
        if hasattr(self,"filterProps"):
            del self.filterProps
        self.savedLastString = ''

    def subsetDataOnShortcut(self,dataID,columnNames,how="keep",stringValue=""):
        ""

        config = self.sourceData.parent.config
        if dataID in self.sourceData.dfs:
            data = self.sourceData.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"]
            operator =  config.getParam("data.shortcut.subset.operator")
            #find bool matches
            boolIdx = data == stringValue
            boolSum = np.sum(boolIdx,axis=1)
            #handle multiple columns
            if data.columns.size == 1:
                #ignore operatre of column  == 1
                boolCombined = boolSum == 1
            elif operator == "and":
                boolCombined = boolSum == data.columns.size
            elif operator == "or":
                boolCombined = boolSum > 0
            #handle how parameter, if how == remove, reverse bools
            if how == "keep":
                idx = data.loc[boolCombined].index
            else:
                idx = data.loc[~boolCombined].index
            #get original file name
            fileName = self.sourceData.getFileNameByID(dataID)
            subsetName = "{}({}):({})".format(stringValue,how,fileName)
            return self.sourceData.subsetDataByIndex(dataID,idx,subsetName)
        else:
            return getMessageProps("Error..","DataID not found.")
#search string is a dict (columnName -> categories)

    def subsetDataOnMultiCategory(self,dataID = None, columnNames = None ,searchString = None, splitString = None, operator = "and"):
        
        if dataID is None:
            dataID = self.filterProps["dataID"]

        if columnNames is None:
            columnNames = self.filterProps["columnNames"]
        #find categories in respective columns
        columnNames = [colName for colName in columnNames if colName in searchString]
        if len(columnNames) == 0:
            return getMessageProps("Error..","Do not know what to do.")
        resultsCollection = pd.DataFrame(columns = columnNames)

        for columnName in columnNames:
            boolIndicator = self.searchCategory(dataID,columnName,searchString[columnName],splitString,False)
            if not isinstance(boolIndicator,dict): #would indicate an error
                resultsCollection[columnName] = boolIndicator

        if operator == "and":
            boolIdx = np.sum(resultsCollection.values,axis=1) == len(columnNames)
        else:
            boolIdx = np.sum(resultsCollection.values,axis=1) > 0

        #generate new subset name
        fileName = self.sourceData.getFileNameByID(dataID)
        columnStr = mergeListToString([mergeListToString(v,",") for v in searchString.values()])
        subsetName = "{}:({}):({})::{}".format(columnStr,operator,mergeListToString(columnNames),fileName)
        
        return self.sourceData.addDataFrame(self.sourceData.dfs[dataID][boolIdx], fileName = subsetName)

      # return self.sourceData.addColumnData(dataID,annotationColumnName,annotationColumn)

    def subsetData(self,dataID = None ,columnName = None ,searchString = None, splitString = None, inputIsRegEx=False, operator = "and", filterType = ""):
        "Splits Dataset based on category in a certain column"

        if filterType == "multiColumnCategory":
            return self.subsetDataOnMultiCategory(dataID,columnName,searchString,splitString,operator)

        if searchString is None:
            searchString = self.savedLastString 
        if dataID is None:
            dataID = self.filterProps["dataID"]

        if columnName is None:
            columnName = self.filterProps["columnNames"][0]
    
        boolIndicator = self.searchCategory(dataID,columnName,searchString,splitString,inputIsRegEx)
        #if searching the category gave an error, error message is a dict
        if isinstance(boolIndicator,dict):
            return boolIndicator
        else:
            #get original file name
            fileName = self.sourceData.getFileNameByID(dataID)
            #set up new file name
            subsetName = '{}: {} in {}'.format(searchString,columnName,fileName)
            #addDataFrame, returns dict with message
            messageProps = self.sourceData.addDataFrame(self.sourceData.dfs[dataID][boolIndicator], fileName = subsetName)
        return messageProps


    def splitDataFrame(self,dataID,columnNames):
        "Split data on each unique category"
        config = self.sourceData.parent.config
        if dataID in self.sourceData.dfs:
            if isinstance(columnNames,str):
                #input as str indicates single column, transform to lists
                columnNames = [columnNames]
            #get groupby object
            groupedData = self.sourceData.getGroupsbByColumnList(dataID,columnNames)
            if len(groupedData) == 1:
                return getMessageProps("Error..","There was only one unique value. No splitting performed.")
            if groupedData is None:
                return getMessageProps("Error..","Could not split data set. Unknown error.")
            #get file name
            fileName = self.sourceData.getFileNameByID(dataID)
            # join column names
            columnNamesJoined = mergeListToString(columnNames, joinString = " ")

            ignoreNaNGroups = config.getParam("data.quick.subset.ignore.nanString")
            fileNameAndData = [('{}({}): {}'.format(groupName,columnNamesJoined,fileName),dataFrame) for groupName, dataFrame in groupedData if not (ignoreNaNGroups and groupName == self.sourceData.replaceObjectNan)]
            nAddedDfs = self.sourceData.addDataFrames(fileNameAndData,copyTypesFromDataID = dataID)
                
            funcReturnProps = getMessageProps("Split Data Frame","Data frame {} was split on column(s): {} ".format(fileName,columnNamesJoined)+
                                           "In total {} dataframes added.".format(nAddedDfs))
            #add dataframe names
            funcReturnProps["dfs"] = self.sourceData.fileNameByID
            #do not select last df after update
            funcReturnProps["selectLastDf"] = False

        else:

            funcReturnProps = getMessageProps("Not found","DataID was not found.")

        return funcReturnProps






        









