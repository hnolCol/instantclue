



from typing import OrderedDict
from numpy.lib.function_base import diff
import pandas as pd 
import numpy as np 

from sklearn.preprocessing import scale, minmax_scale, robust_scale
from statsmodels.nonparametric.smoothers_lowess import lowess

from ..utils.stringOperations import getMessageProps

funcKeys = {
        "Standardize (Z-Score)" : "calculateZScore",
        "Scale (0 - 1)" : "minMaxNorm",
        "Quantile (25 - 75)" : "calculateQuantiles", 
        "loessRowNorm" : "fitCorrectLoess",
        "loesColNorm" : "globalLoessCorrection",
        "globalMedian" : "globalMedian",
        "normalizeMedianBySubset" : "normalizeMedianBySubset",
        "DivideByColSum" : "divideByColumnSum",
        "cumSum" : "cumulativeSum",
        "DivideByMax" : "divideByMaxValueInColumn"
                 }


class Normalizer(object):
    ""
    def __init__(self, sourceData):

        self.sourceData = sourceData

    def _addToSourceData(self,dataID,columnNames,transformedData):
        ""
        if self._transformInPlace():
            return self.sourceData.replaceColumns(dataID,columnNames,transformedData.values)
        else:
            return self.sourceData.joinDataFrame(dataID,transformedData)

    def _transformInPlace(self):
        ""
        return self.sourceData.parent.config.getParam("perform.transformation.in.place")


    def normalizeData(self, dataID, normKey, columnNames = [], **kwargs):
        ""
        if dataID in self.sourceData.dfs:
            if normKey in funcKeys and hasattr(self,funcKeys[normKey]):
                return getattr(self,funcKeys[normKey])(dataID,columnNames, **kwargs)
            elif hasattr(self,normKey):
                return getattr(self,normKey)(dataID,**kwargs)

    def calculateZScore(self,dataID, columnNames,axis=1):
        ""
        transformedColumnNames = ["Z({}):{}".format("row" if axis else "column",col) for col in columnNames.values]
        transformedValues = pd.DataFrame(
                            scale(self.sourceData.dfs[dataID][columnNames].values, axis=axis),
                            index = self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )

        return self._addToSourceData(dataID,columnNames,transformedValues)


    def calculateQuantiles(self,dataID,columnNames,axis=1):
        ""
        transformedColumnNames = ["Q({}):{}".format("row" if axis else "column",col) for col in columnNames.values]
        X = self.sourceData.dfs[dataID][columnNames].values
        
        transformedValues = pd.DataFrame(
                            robust_scale(X,
                                axis=axis, 
                                with_centering = self.sourceData.parent.config.getParam("quantile.norm.centering"),
                                with_scaling = self.sourceData.parent.config.getParam("quantile.norm.scaling")),
                            index= self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )
        return self._addToSourceData(dataID,columnNames,transformedValues)

    def cumulativeSum(self,dataID,columnNames,axis=0):
        ""
        transformedColumnNames = ["CumulativeSum:{}".format(col) for col in columnNames.values] + ["RelCumulativeSum:{}".format(col) for col in columnNames.values] + ["CumulativeSumRank:{}".format(col) for col in columnNames.values]
        X = self.sourceData.dfs[dataID][columnNames].values
        df = self.sourceData.dfs[dataID][columnNames]
        df.loc[df.index,"idx"] = df.index #add index to be used for pivoting back the data.
        #first melt to apply groupby functions
        melted = pd.melt(df,value_vars=columnNames,id_vars=["idx"]).sort_values(by="value",ascending=False)
        grouped = melted.groupby(by=["variable"])
        melted["cumsum"] = grouped["value"].transform(lambda x: x.cumsum())
        melted["rel_cumsum"] = melted["cumsum"] / melted.groupby(by=["variable"])["cumsum"].transform(lambda x: x.max())
        melted["rank"] = grouped["value"].transform(lambda x: x.rank(method="min",ascending=False))

        transformedX = melted.pivot(index="idx",columns=["variable"],values=["cumsum","rel_cumsum","rank"]).values

        transformedValues = pd.DataFrame(
                            transformedX,
                            index= self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )
        return self._addToSourceData(dataID,columnNames,transformedValues)

    def divideByColumnSum(self,dataID,columnNames,axis=0):
        ""
        transformedColumnNames = ["DivSum:{}".format(col) for col in columnNames.values]
        X = self.sourceData.dfs[dataID][columnNames].values
        transformedX = X/np.nanmax(X,axis=axis,keepdims=True)
        transformedValues = pd.DataFrame(
                            transformedX,
                            index= self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )
        return self._addToSourceData(dataID,columnNames,transformedValues)
    
    def divideByMaxValueInColumn(self,dataID,columnNames,axis=0):
        ""
        transformedColumnNames = ["DivMax:{}".format(col) for col in columnNames.values]
        X = self.sourceData.dfs[dataID][columnNames].values
        transformedX = X/np.nansum(X,axis=axis,keepdims=True)
        transformedValues = pd.DataFrame(
                            transformedX,
                            index= self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )
        return self._addToSourceData(dataID,columnNames,transformedValues)
    def minMaxNorm(self,dataID,columnNames,axis=1):
        ""
        transformedColumnNames = ["0-1({}):{}".format("row" if axis else "column",col) for col in columnNames.values]
        X = self.sourceData.dfs[dataID][columnNames].values
        Xmin = np.nanmin(X,axis=axis, keepdims=True)
        Xmax = np.nanmax(X,axis=axis,keepdims=True)
        data = (X - Xmin) / (Xmax-Xmin)
        transformedValues = pd.DataFrame(
                            data,
                            index = self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )

        return self._addToSourceData(dataID,columnNames,transformedValues)

    def normalizeGroupMedian(self,dataID,**kwargs):
        ""
        if not self.sourceData.parent.grouping.groupingExists():
            return getMessageProps("No Grouping","No Grouping set.")
        groupingName = self.sourceData.parent.grouping.getCurrentGroupingName()
        grouping = self.sourceData.parent.grouping.getCurrentGrouping() 
        columnNames = self.sourceData.parent.grouping.getColumnNames(groupingName)

        X = self.sourceData.dfs[dataID][columnNames]
        transformedValues = pd.DataFrame(index = X.index, columns = ["groupColMedian:{}".format(colName) for colName in columnNames])
        for groupName, groupColumns in grouping.items():
            columnMedians = X[groupColumns].median()
            groupMedian = np.nanmedian(columnMedians.values)
            distToGroupMedian = columnMedians.values - groupMedian
            normColumnNames = ["groupColMedian:{}".format(colName) for colName in groupColumns]
            transformedValues[normColumnNames] = X[groupColumns].subtract(distToGroupMedian, axis="columns")

        return self._addToSourceData(dataID,columnNames,transformedValues)

    def normalizeGroupQuantile(self,dataID,**kwargs):
        "Normalize Group Quantiles (e.g. adjusting within groups)"
        if not self.sourceData.parent.grouping.groupingExists():
            return getMessageProps("No Grouping","No Grouping set.")
        groupingName = self.sourceData.parent.grouping.getCurrentGroupingName()
        grouping = self.sourceData.parent.grouping.getCurrentGrouping() 
        columnNames = self.sourceData.parent.grouping.getColumnNames(groupingName)

        X = self.sourceData.dfs[dataID][columnNames]
        transformedValues = pd.DataFrame(index = X.index, columns = ["groupColQuantile:{}".format(colName) for colName in columnNames])
        for groupName, groupColumns in grouping.items():
            columnMedians = X[groupColumns].median()
            q25,q75 = np.nanquantile(X[groupColumns].values,q=[0.25,0.75])
            groupIQR = q75-q25
            groupMedian = np.nanmedian(columnMedians.values)

            scaledGroupValues = X[groupColumns].values / groupIQR
            distToGroupMedian = columnMedians.values - groupMedian
            
            normColumnNames = ["groupColQuantile:{}".format(colName) for colName in groupColumns]
            transformedValues[normColumnNames] = np.subtract(scaledGroupValues,distToGroupMedian)

        return self._addToSourceData(dataID,columnNames,transformedValues)

    def normalizeMedianBySubset(self,dataID,columnNames,subsetColumn):
        "Normalize data on a specific subset."
        if isinstance(subsetColumn,list):
            subsetColumn = subsetColumn[0]

        data = self.sourceData.dfs[dataID][columnNames.values.tolist() + [subsetColumn]]
        #uniqueValues = self.sourceData.getUniqueValues(dataID, subsetColumn)
        groupedDf = data.groupby(by=subsetColumn)
        r = []
        for groupName, groupData in groupedDf:
            medians = groupData[columnNames].median(axis=0)
            globalMedian = np.nanmedian(medians)
            diff = medians - globalMedian
            normData = data[columnNames].sub(diff,axis="columns")
            normData.columns = ["{}:{}".format(groupName,colName) for colName in columnNames.values]
            r.append(normData)
        transformedValues = pd.concat(r,axis=1)

        return self._addToSourceData(dataID,columnNames,transformedValues)


    def normalizeToGroup(self,dataID,groupingName,toGroups,withinGroupingName = None):
        ""
        grouping = self.sourceData.parent.grouping.getGrouping(groupingName)
        withinGrouping  = self.sourceData.parent.grouping.getGrouping(withinGroupingName)
        data = self.sourceData.dfs[dataID]
        r = OrderedDict()
        for specGroup in toGroups:
            if specGroup in grouping:
                specGroupColumnNames = grouping[specGroup]
                specGroupMean = data[specGroupColumnNames].mean(axis=1)
                for _, groupColumnNames in grouping.items():
                       
                        if withinGrouping is not None:
                            for withinGroup, withGroupColumnNames in withinGrouping.items():
                                withinGroupColumnNamesSubset = pd.Series(np.intersect1d(groupColumnNames,withGroupColumnNames))
                                withinGroupColumnNamesSpecGroup = pd.Series(np.intersect1d(specGroupColumnNames,withGroupColumnNames))
                               
                                
                                for colName in withinGroupColumnNamesSubset:
                                    columnName = "{} - {} ({})".format(colName,specGroup,withinGroup)
                                    r[columnName] = data[colName] - data[withinGroupColumnNamesSpecGroup].mean(axis=1)

                        else:
                            for colName in groupColumnNames:
                                transformedColumnName = "{} - {}".format(colName,specGroup)
                                r[transformedColumnName] = data[colName] - specGroupMean

                        
        df = pd.DataFrame(r, index= data.index)
        return self.sourceData.joinDataFrame(dataID,df)
            

    def fitLowess(self,y,x,**kwargs):
        "Should not be here - move function"
        fit = lowess(y,x,**kwargs, return_sorted=False)
        yMedian = np.nanmedian(y)
        
        return y - (fit - yMedian)

    def fitCorrectLoess(self,dataID,columnNames, axis=1, **kwargs):
        ""
        
        data = self.sourceData.dfs[dataID][columnNames]
        x = np.arange(columnNames.values.size)
        Y = np.apply_along_axis(self.fitLowess,axis=1,arr=data.values,x=x,frac=0.5,it=3,**kwargs)
        transformedColumnNames = ["loessFC({}):{}".format("row" if axis else "column",col) for col in columnNames.values]
        transformedValues = pd.DataFrame(Y, index= data.index, columns = transformedColumnNames)
        return self.sourceData.joinDataFrame(dataID,transformedValues)
            
    
    def globalMedian(self,dataID,columnNames, axis=1,*args,**kwargs):
        ""
        data = self.sourceData.dfs[dataID][columnNames]
        globalMedian = np.nanmedian(data.values)
        diffToColumnMedain = np.nanmedian(data.values,axis=0) - globalMedian
        
        transformedColumnNames = ["medianCorr:{}".format(colName) for colName in columnNames.values]
        transformedValues = pd.DataFrame(np.subtract(data.values,diffToColumnMedain),index=data.index,columns=transformedColumnNames)
        
        return self._addToSourceData(dataID,columnNames,transformedValues)


       # lowessLine = lowess(y,x, it=it, frac=frac)

# def transform_data(self,columnNameList,transformation):
# 		'''
# 		Calculates data transformation and adds these to the data frame.
# 		'''	
# 		newColumnNames = [self.evaluate_column_name('{}_{}'.format(transformation,columnName)) \
# 		for columnName in columnNameList]
		
# 		if transformation == 'Z-Score_row':
# 			transformation = 'Z-Score'
# 			axis_ = 1
# 		elif transformation == 'Z-Score_col':
# 			transformation = 'Z-Score' 
# 			axis_ = 0 
# 		else:
# 			axis_ = 0 
		
# 		if 'Z-Score' in transformation:
# 			transformedDataFrame = pd.DataFrame(scale(self.df[columnNameList].values, axis = axis_),
# 				columns = newColumnNames, index = self.df.index)
# 		else:
# 			transformedDataFrame = pd.DataFrame(
# 							calculations[transformation](self.df[columnNameList].values),
# 							columns = newColumnNames,
# 							index = self.df.index)
			
# 		if transformation != 'Z-Score':
# 			transformedDataFrame[~np.isfinite(transformedDataFrame)] = np.nan
		
# 		self.df[newColumnNames] = transformedDataFrame
# 		self.update_columns_of_current_data()
		
# 		return newColumnNames


    

