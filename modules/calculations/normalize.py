from sklearn import preprocessing as skPrep
import numpy as np

metricNames = {'Quantile':'RobustScaler','Standardize':'StandardScaler','0->1':'MinMaxScaler'}

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
		return self.scaler.fit_transform(data)

