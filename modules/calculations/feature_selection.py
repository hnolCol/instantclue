

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold

estimators = {'random forest':RandomForestClassifier,
			  'linear SVM':LinearSVC,
			  'extra tree classifier':ExtraTreesClassifier}


defaultSettings = {'random forest':{'n_estimators':100,'min_samples_split':2},
				  'linear SVM':{'C':0.01,'penalty':'l1', 'dual':False},
				  'extra tree classifier':{}}

class selectFeaturesFromModel(object):
	
	def __init__(self, X, Y = None, model = 'random forest', 
								max_features = 100, threshold = -np.inf, addSettings = {}):
	
		self.X = X
		self.Y = Y 
		self.model = model
		self.max_features = max_features
		self.threshold = threshold
		self.addSettings = addSettings
		#self.remove_nan()
		self.fit_model()
		self.select_featues()


	def fit_model(self):
		"Fits the model"
		if self.model in estimators:
			self.est = estimators[self.model](**defaultSettings[self.model]).fit(self.X,self.Y)		
		elif self.model == 'Variance':
			self.est = VarianceThreshold(**self.addSettings).fit(self.X)
			
	
	#def remove_nan(self):
	#	"Remove rows that contain nan"
		#colNaN = np.sum(np.isnan(self.X)) == 0
		#self.X = self.X[:,colNaN][0]
	#	print(self.X)
		#self.Y = self.Y[rowNaN]

	def select_featues(self,max_features = None, threshold = None):
		"Select features"
		if self.model in estimators:
			if max_features is None: max_features = self.max_features
			if threshold is None: threshold = self.threshold
		
			self.fitModel = SelectFromModel(self.est,
				max_features=max_features,
				threshold=threshold,
				prefit = True)
		elif self.model == 'Variance':
			self.fitModel = self.est
		
			
	@property
	def featureImportance(self):
		""
		if hasattr(self.est,'feature_importances_'):
			return self.est.feature_importances_
	@property
	def featureMask(self):
		"returns the feature mask" 
		return self.fitModel.get_support()
		
		
## testing
		
if __name__ == "__main__":

	from sklearn.datasets import load_iris
	iris = load_iris()
	X, y = iris.data, iris.target
	selectFeaturesFromModel(X,y)

	
		
		
