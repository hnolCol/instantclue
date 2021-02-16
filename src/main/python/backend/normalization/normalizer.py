



import pandas as pd 
import numpy as np 

from sklearn.preprocessing import scale, minmax_scale, robust_scale
from statsmodels.nonparametric.smoothers_lowess import lowess

funcKeys = {
        "Standardize (Z-Score)" : "calculateZScore",
        "Scale (0 - 1)" : "minMaxNorm",
        "Quantile (25 - 75)" : "calculateQuantiles", 
        "loessRowNorm" : "fitCorrectLoess",
        "loesColNorm" : "globalLoessCorrection"
         }


class Normalizer(object):
    ""
    def __init__(self, sourceData):

        self.sourceData = sourceData

    def normalizeData(self, dataID, normKey, columnNames, **kwargs):
        ""
        if dataID in self.sourceData.dfs:
            if normKey in funcKeys and hasattr(self,funcKeys[normKey]):
                return getattr(self,funcKeys[normKey])(dataID,columnNames, **kwargs)


    def calculateZScore(self,dataID, columnNames,axis=1):
        ""
        transformedColumnNames = ["Z({}):{}".format("row" if axis else "column",col) for col in columnNames.values]
        transformedValues = pd.DataFrame(
                            scale(self.sourceData.dfs[dataID][columnNames].values, axis=axis),
                            index = self.sourceData.dfs[dataID].index,
                            columns = transformedColumnNames
                            )

        return self.sourceData.joinDataFrame(dataID,transformedValues)

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
        return self.sourceData.joinDataFrame(dataID,transformedValues)

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

        return self.sourceData.joinDataFrame(dataID,transformedValues)

    def fitLowess(self,y,x,**kwargs):
        ""
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


    

