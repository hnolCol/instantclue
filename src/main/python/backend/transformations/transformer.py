



import pandas as pd 
import numpy as np 
from ..utils.stringOperations import getMessageProps, mergeListToString, findCommonStart

from sklearn.preprocessing import scale


funcKeys = {
        "logarithmic" : "logarithmicTransformation",
        "rolling": "rollingWindowTransformation",
        "absolute" : "absoluteTransformation"
         }

logarithmicBase =  {'log2': np.log2,
					'-log2':lambda x: np.log2(x)*(-1),
					'log10':np.log10,
					'-log10':lambda x: np.log10(x)*(-1),
					'ln':np.log,
					}	
	
class Transformer(object):
    ""
    def __init__(self, sourceData):

        self.sourceData = sourceData

        self.inplace = True

    def transformData(self, dataID, transformKey, columnNames, **kwargs):
        ""
        if dataID in self.sourceData.dfs:
            if transformKey in funcKeys:
                return getattr(self,funcKeys[transformKey])(dataID,columnNames, **kwargs)

    def absoluteTransformation(self,dataID,columnNames):
        ""
        transformedColumnNames = ['abs:{}'.format(columnName) for columnName in columnNames.values] 
        transformedValues = np.abs(self.sourceData.dfs[dataID][columnNames].values)
        transformedData = pd.DataFrame(
                    transformedValues,
                    columns=transformedColumnNames,
                    index=self.sourceData.dfs[dataID].index)
        return self.sourceData.joinDataFrame(dataID, transformedData)


    def logarithmicTransformation(self,dataID, columnNames, base):
        ""
        if base in logarithmicBase:

            transformedColumnNames = ["t({}):{}".format(base,col) for col in columnNames.values]
            transformedValues = pd.DataFrame(        
                            logarithmicBase[base](self.sourceData.dfs[dataID][columnNames].values),
                            index = self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )
            transformedValues[~np.isfinite(transformedValues)] = np.nan
        return self.sourceData.joinDataFrame(dataID,transformedValues)

    def rollingWindowTransformation(self,dataID,columnNames,windowSize,metric):
        ""

        transformedColumnNames = ['t({}:w{}):{}'.format(metric,windowSize,columnName) for columnName in columnNames.values] 
        rollingWindow = self.sourceData.dfs[dataID][columnNames].rolling(window=windowSize)
        if hasattr(rollingWindow,metric):
            transformedData = pd.DataFrame(
                    getattr(rollingWindow,metric)().values,
                    columns=transformedColumnNames,
                    index=self.sourceData.dfs[dataID].index)
            return self.sourceData.joinDataFrame(dataID, transformedData)
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


    

