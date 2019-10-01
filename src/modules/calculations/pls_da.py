
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression


class PLS_DA(object):
	'''
	'''
	def __init__(self, n_comps = 3, yIsDummyMatrix = False, scaleData = False):
		'''
		data contains n_samples, n_features
		Y - response
		'''
		self.comps = n_comps
		self.yIsDummyMatrix = yIsDummyMatrix
		self.plsr = PLSRegression(n_components=n_comps, scale=scaleData)	
		
				
	def fit(self,X,Y):
		
		if self.yIsDummyMatrix:
			self.Ym = Y
		else:
			self.Ym = self.create_dummy_y(Y)
		if self.evaluate_data(X,self.Ym):
			self.plsr.fit(X,self.Ym)
				
	def get_scores(self,block = 'x'):
		'''
		'''		
		if block == 'x':
			return self.plsr.x_scores_
		elif block == 'y':
			return self.plsr.y_scores_

	def get_weights(self,block = 'x'):
		'''
		'''
		if block == 'x':
			return self.plsr.x_weights_

	def get_loadings(self,block='x'):
		
		if block == 'x':
			return self.plsr.x_loadings_
	def get_squared_r(self,X,Y):
		'''
		'''
		return self.plsr.score(X,Y)
		
	def get_classes(self):
		'''
		'''
	def get_dummy_Y(self):
		'''
		'''
		return self.Ym
		
	@property
	def loadings_(self):
		return self.plsr.x_loadings_
		
					
	def evaluate_data(self,X,Y):
		'''
		'''
		if X.shape[0] != Y.shape[0]:
			print("Number of rows in X does not equal number of rows in Y") 
			return False
		else:
			return True
		
	def create_dummy_y(self,Y):
		'''
		'''		
		uniqueVals  = np.unique(Y)
		nClasses = uniqueVals.size
		Ydummy = np.zeros((Y.shape[0],nClasses))
		for n,target in enumerate(Y):
			col = np.where(uniqueVals==target)
			Ydummy[n,col] = 1
		self.classOrder = uniqueVals.tolist()
		return Ydummy
	