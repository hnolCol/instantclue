

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
estimators = {'random forest':RandomForestClassifier,
			  'linear SVM':LinearSVC,
			  'extra tree classifier':ExtraTreesClassifier}


defaultSettings = {'random forest':{'n_estimators':100},
				  'linear SVM':{'C':0.01,'penalty':'l1', 'dual':False},
				  'extra tree classifier':{}}

class selectFeaturesFromModel(object):
	
	def __init__(self, X, Y, model = 'random forest', max_features = 100, threshold = -np.inf):
	
		self.X = X
		self.Y = Y 
		self.model = model
		self.max_features = max_features
		self.threshold = threshold

		self.remove_nan()
		self.fit_model()
		self.select_featues()


	def fit_model(self):
		"Fits the model"
		self.est = estimators[self.model](**defaultSettings[self.model]).fit(self.X,self.Y)		
		
	
	def remove_nan(self):
		"Remove rows that contain nan"
		#colNaN = np.sum(np.isnan(self.X)) == 0
		#self.X = self.X[:,colNaN][0]
		print(self.X)
		#self.Y = self.Y[rowNaN]

	def select_featues(self,max_features = None, threshold = None):
		"Select features"
		
		if max_features is None: max_features = self.max_features
		if threshold is None: threshold = self.threshold
		
		self.model = SelectFromModel(self.est,
			max_features=max_features,
			threshold=threshold,
			prefit = True)
	@property
	def featureImportance(self):
		""
		if hasattr(self.est,'feature_importances_'):
			return self.est.feature_importances_
	@property
	def featureMask(self):
		"returns the feature mask" 
		return self.model.get_support()
		
		
## testing
		
if __name__ == "__main__":

	from sklearn.datasets import load_iris
	iris = load_iris()
	X, y = iris.data, iris.target
	selectFeaturesFromModel(X,y)

	
		
		
