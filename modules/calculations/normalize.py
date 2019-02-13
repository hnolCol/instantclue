from sklearn import preprocessing as skPrep
import numpy as np
import pandas as pd
metricNames = {'Quantile (25,75)':'RobustScaler','Standardize':'StandardScaler','0->1':'MinMaxScaler'}

class dataNormalizer(object):
	'''
	'''
	def __init__(self,funcName,**kwargs):
		
		if funcName in metricNames:	
			funcName = metricNames[funcName]
			
		if hasattr(skPrep,funcName):
			self.scaler = getattr(skPrep,funcName)(**kwargs)
		
	def fit_transform(self, data):
		'''
		'''
		self.scaler.fit_transform(data)		
     	
		return self.scaler.fit_transform(data)


def quantileNormalize(df_input):
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df