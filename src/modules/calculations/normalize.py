from sklearn import preprocessing as skPrep
import numpy as np
import pandas as pd

metricNames = {'Quantile (25,75)':'RobustScaler','Standardize':'StandardScaler'}

class MinMaxScaler():

    def __init__(self, min="aroundZero", max=1):

        self.min = min
        self.max = max

    def fit_transform(self, X):

        def scale(a):
            ""
            newMin = np.random.normal(loc=0, scale = 0.00001)
            newMax = 1
            oldMin = np.nanmin(a)
            oldMax = np.nanmax(a)
            return (newMax-newMin) / (oldMax - oldMin) * (a - oldMax) + newMax

       
        maxNew = np.ones(shape=(X.shape[0],1))

        Xnew = np.apply_along_axis(scale, axis=1, arr=X)

        return Xnew

        




class dataNormalizer(object):
	'''
	'''
	def __init__(self,funcName,**kwargs):
		
		if funcName in metricNames:	
			funcName = metricNames[funcName]
			
		if hasattr(skPrep,funcName):
			self.scaler = getattr(skPrep,funcName)(**kwargs)
		elif funcName == '0->1':
        
 			self.scaler = MinMaxScaler(**kwargs)
       
		
	def fit_transform(self, data):
		'''
		'''
		self.scaler.fit_transform(data)		
     	
		return self.scaler.fit_transform(data)


def quantileNormalize(df_input):

    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df.columns:
        dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df