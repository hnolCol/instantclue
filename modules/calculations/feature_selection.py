

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from modules.dialogs.classification import randomForestWidgets, supportVectorWidgets

estimators = {'Random Forest':RandomForestClassifier,
			  'SVM':LinearSVC
			  }

# randomForestWidgets = OrderedDict([('n_estimators',['50',list(range(10,100,10)),'The number of trees in the forest.']),
# 					 ('max_features',['sqrt',['sqrt','log2'],'The number of features to consider when looking for the best split:\nIf int, then consider max_features features at each split.\nIf float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.\nIf “auto”, then max_features=sqrt(n_features).\nIf “sqrt”, then max_features=sqrt(n_features) (same as “auto”).\nIf “log2”, then max_features=log2(n_features).\nIf None, then max_features=n_features.\nNote: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.']),
# 					 ('max_depth',['None',['None','1','2','5'],'The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.']),
# 					 ('min_samples_split',['2',['2','4'],'The minimum number of samples required to split an internal node:\nIf int, then consider min_samples_split as the minimum number.\nIf float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.']),
# 					 ('min_samples_leaf',['1',['1','2','3'],'The minimum number of samples required to be at a leaf node:\nIf int, then consider min_samples_leaf as the minimum number.\nIf float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.']),
# 					 ('max_leaf_nodes',['None',['None','1','2','3'],'Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.']),
# 					 ('oob_score',['True',trueFalse,'The minimum number of samples required to split an internal node:\nIf int, then consider min_samples_split as the minimum number.\nIf float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.']),
# 					 ('bootstrap',['True',trueFalse,'Whether bootstrap samples are used when building trees.']),
# 					 ('class_weight',['balanced_subsample',['balanced_subsample','balanced','None'],'The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))\nThe “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.']),
# 					# ('n_jobs',['-2','The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.']),
# 					 ])

					 
# supportVectorWidgets = OrderedDict([('C',['1','Penalty parameter C of the error term.']),
# 					 ('kernel',['rbf',['rbf','poly','linear'],'Specifies the kernel type to be used in the algorithm.']),
# 					 ('degree',['3',['2','3',]'Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.']),
# 					 ('gamma',['auto','Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.']),
# 					 ('coef0',['0','Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.']),
# 					 ('probability',['False',trueFalse,'Whether to enable (class) probability estimates. This must be enabled prior to calling fit, and will slow down that method.']),
# 					 ('shrinking',['False',trueFalse,'Whether to use the shrinking heuristic.']),
# 					 ('tol',['1e-3','Tolerance for stopping criterion.']),
# 					 ('cache_size',['300','Specify the size of the kernel cache (in MB).']),
# 					 ('class_weight',['None',['balanced','None'],'The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))\nThe “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.']),
# 					 ])

trueFalseDict = {'True':True,'False':False}

dataTypes = {
			'n_estimators':['int',100],
			'max_features':[['sqrt','log2'],'sqrt'],
			'max_depth':['int',None],
			'min_samples_split':['int',2],
			'min_samples_leaf':['int',1],
			'max_leaf_nodes':['int',None],
			'oob_score':['bool',True],
			'class_weight':[['balanced_subsample','balanced'],None],
			'bootstrap':['bool',True],
			'C':['float',0.01],
			#'kernel':[['rbf','poly','linear','sigmoid'],'linear'],
			'gamma':['float','auto'],
			'degree':['int',3],
			'coef0':['float',0.0],
			'tol':['float',1e-3],
			'hinge':[['hinge','squared_hinge'],'squared_hinge'],
			'penalty':[['l1','l2'],'l2']
			}

def checkDataType(settings):
	newSettings = {}
	if isinstance(settings,dict):
		for k,v in settings.items():
			if k in dataTypes:
				try:
					targetType = dataTypes[k][0]
					default = dataTypes[k][1]
					if isinstance(targetType,list):
						if v in targetType:
							newSettings[k] = v
						else:
							newSettings[k] = default
					elif targetType == 'float':
						newSettings[k] = np.float(v)
					elif targetType == 'int':
						newSettings[k] = np.int(v)
					elif targetType == 'bool':
						if v in trueFalseDict:
							newSettings[k] = trueFalseDict[v]
						else:
							newSettings[k] = default
					# elif targetType == 'str_or_none':
# 						if v is not 'None':
# 							newSettings[k] = v
						
				except:
					newSettings[k] = dataTypes[k][1]
						
	
	return newSettings



estimatorSettings = {'Random Forest':randomForestWidgets,
					'SVM':supportVectorWidgets}
					
defaultSettings = {'Random Forest':{'n_estimators':100,'min_samples_split':2},
				  'SVM':{'C':0.01,'penalty':'l1', 'dual':False}}

class selectFeaturesFromModel(object):
	
	def __init__(self, X, Y = None, model = 'random forest', 
								max_features = 20, 
								threshold = -np.inf, 
								addSettings = {}):
	
		self.X = X
		self.Y = Y 
		self.model = model
		self.max_features = max_features
		self.threshold = threshold
		self.addSettings = addSettings
		self.fit_model()
		self.select_features()


	def fit_model(self):
		"Fits the model"
		if self.model in estimators:
			
			if len(self.addSettings) != 0:
				#try:
					self.est = estimators[self.model](**self.addSettings).fit(self.X,self.Y)	
				#except:
				#	self.est = estimators[self.model](**defaultSettings[self.model]).fit(self.X,self.Y)				
			else:
				self.est = estimators[self.model](**defaultSettings[self.model]).fit(self.X,self.Y)		
		
		elif self.model == 'Variance':
			
			self.est = VarianceThreshold(**self.addSettings).fit(self.X)

	def select_features(self,max_features = None, threshold = None):
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

	
		
		
