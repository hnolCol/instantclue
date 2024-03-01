



from collections import OrderedDict
import pandas as pd 
import numpy as np 
from ..utils.stringOperations import getMessageProps, mergeListToString, findCommonStart

from sklearn.preprocessing import scale

funcKeys = {
        "multiply" : "multiplyTransformation",
        "logarithmic" : "logarithmicTransformation",
        "rolling": "rollingWindowTransformation",
        "absolute" : "absoluteTransformation",
        "summarize": "summarizeTransformation"
         }
logarithmicBase =  {'log2': np.log2,
					'-log2':lambda x: np.log2(x)*(-1),
					'log10':np.log10,
					'-log10':lambda x: np.log10(x)*(-1),
					'ln':np.log,
					}	
summarizeMetric = OrderedDict(
                    [
                        ("min"    ,   np.nanmin),
                        ("max"    ,   np.nanmax),    
                        ("median" ,   np.nanmedian),
                        ("mean"   ,   np.nanmean),
                        ("quantile",  np.nanquantile),
                        ("std"    ,   np.nanstd),
                        ("var"    ,   np.nanvar),
                        ("coeff var", lambda x,axis: np.nanstd(x, ddof=1,axis=axis) / np.nanmean(x,axis=axis) * 100 ) 
                    ])
	
class Transformer(object):
    ""
    def __init__(self, sourceData):

        self.sourceData = sourceData
        self.config = self.sourceData.parent.config
        self.inplace = True

    def _addToSourceData(self,dataID,columnNames,transformedData,allowInPlace=True):
        ""
        if self._transformInPlace() and allowInPlace:
            return self.sourceData.replaceColumns(dataID,columnNames,transformedData.values)
        else:
            return self.sourceData.joinDataFrame(dataID,transformedData)

    def _transformInPlace(self):
        ""
        return self.config.getParam("perform.transformation.in.place")

    def transformData(self, dataID, transformKey, columnNames, **kwargs):
        ""
  
        if dataID in self.sourceData.dfs:
            if transformKey in funcKeys:
                return getattr(self,funcKeys[transformKey])(dataID,columnNames, **kwargs)
        return getMessageProps("Error..","No matching data ID?")
        
    def absoluteTransformation(self,dataID,columnNames):
        ""
        transformedColumnNames = ['abs:{}'.format(columnName) for columnName in columnNames.values] 
        transformedValues = np.abs(self.sourceData.dfs[dataID][columnNames].values)
        transformedData = pd.DataFrame(
                    transformedValues,
                    columns=transformedColumnNames,
                    index=self.sourceData.dfs[dataID].index)

        return self._addToSourceData(dataID,columnNames,transformedData)

    def multiplyTransformation(self,dataID, columnNames, value = -1):
        """Multiplies the values in a column by a constant value"""
        transformedColumnNames = ['m{}:{}'.format(value,columnName) for columnName in columnNames.values] 
        transformedValues = self.sourceData.dfs[dataID][columnNames].values * value
        transformedData = pd.DataFrame(
                    transformedValues,
                    columns=transformedColumnNames,
                    index=self.sourceData.dfs[dataID].index)

        return self._addToSourceData(dataID,columnNames,transformedData)
    

    def logarithmicTransformation(self,dataID, columnNames, base):
        ""
        if base in logarithmicBase:

            transformedColumnNames = ["t({}):{}".format(base,col) for col in columnNames.values]
            transformedData = pd.DataFrame(        
                            logarithmicBase[base](self.sourceData.dfs[dataID][columnNames].values),
                            index = self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )
            transformedData[~np.isfinite(transformedData)] = np.nan
            return self._addToSourceData(dataID,columnNames,transformedData)
        else:
            return getMessageProps("Error..","Unknown metric.")
            
    def rollingWindowTransformation(self,dataID,columnNames,windowSize,metric):
        ""
        transformedColumnNames = ['t({}:w{}):{}'.format(metric,windowSize,columnName) for columnName in columnNames.values] 
        rollingWindow = self.sourceData.dfs[dataID][columnNames].rolling(window=windowSize)
        if hasattr(rollingWindow,metric):
            transformedData = pd.DataFrame(
                    getattr(rollingWindow,metric)().values,
                    columns=transformedColumnNames,
                    index=self.sourceData.dfs[dataID].index)
            return self._addToSourceData(dataID,columnNames,transformedData)
        else:
            return getMessageProps("Error..","Unknown metric.")

    def summarizeTransformation(self,dataID,columnNames, metric, justValues = False, **kwargs):
        ""
        
        if metric in summarizeMetric:
            if justValues:
                return summarizeMetric[metric](self.sourceData.dfs[dataID][columnNames].values,axis=1, **kwargs)
                
            mergedColumns = mergeListToString(columnNames)
            if len(kwargs) > 0:
                kwargString = "".join(["-({}:{})".format(k,v) for k,v in kwargs.items()])
                transformedColumnNames = ["s({}{}):{}".format(metric,kwargString,mergedColumns)] 
            else:
                transformedColumnNames = ["s({}):{}".format(metric,mergedColumns)] 
            
            transformedData = pd.DataFrame(        
                            summarizeMetric[metric](self.sourceData.dfs[dataID][columnNames].values,axis=1, **kwargs),
                            index = self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )

            return self._addToSourceData(dataID,columnNames,transformedData,allowInPlace=False)
        else:
            return getMessageProps("Error..","Unknown metric.")
	# def calculate_rolling_metric(self,numericColumns,windowSize,metric,quantile = 0.5):
	# 	'''
	# 	Calculates rolling windows and metrices (like mean, median etc). 
	# 	Can be used for smoothing
	# 	'''
		
	# 	rollingWindow = self.df[numericColumns].rolling(window=windowSize)
		
	# 	if metric == 'mean':
	# 		self.df[newColumnNames] = rollingWindow.mean() 
	# 	elif metric == 'median':
	# 		self.df[newColumnNames] = rollingWindow.median()
	# 	elif metric == 'sum':
	# 		self.df[newColumnNames] = rollingWindow.sum() 
	# 	elif metric == 'max':
	# 		self.df[newColumnNames] = rollingWindow.max()
	# 	elif metric == 'min':
	# 		self.df[newColumnNames] = rollingWindow.min()
	# 	elif metric == 'std':
	# 		self.df[newColumnNames] = rollingWindow.std()
	# 	elif metric == 'quantile':
	# 		self.df[newColumnNames] = rollingWindow.quantile(quantile=quantile)
		
	# 	self.update_columns_of_current_data()
		
	# 	return newColumnNames
		

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


    

