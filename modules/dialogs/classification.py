"""
	""CLASSIFICATION TASKS""
    Instant Clue - Interactive Data Visualization and Analysis.
    Copyright (C) Hendrik Nolte

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""

import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts

import matplotlib.pyplot as plt
import seaborn as sns

from modules import images
from modules.utils import *
from modules.dialogs import display_data
import webbrowser

import pandas as pd
import numpy as np

import warnings
from sklearn.model_selection import GridSearchCV 
import sklearn.ensemble as skEnsemble
import sklearn.svm as skSVM
import sklearn.naive_bayes as skNaiveBayes
import sklearn.model_selection as skModel 
import sklearn.linear_model as skLinear
import sklearn.feature_selection as skFeatSel
import sklearn.metrics as skMetrics
import sklearn.preprocessing as skPreProc

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA, NMF

from modules.dialogs.VerticalScrolledFrame import VerticalScrolledFrame

warnings.filterwarnings("ignore", category=DeprecationWarning)	
warnings.filterwarnings("ignore", category=FutureWarning)	

trueFalseDict = {'True':True,'False':False}
trueFalse = list(trueFalseDict.keys())

def tp(y_true, y_pred): 
	return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]



confusionMatrixScoring = {'tp' : skMetrics.make_scorer(tp), 'tn' : skMetrics.make_scorer(tn),
           'fp' : skMetrics.make_scorer(fp), 'fn' : skMetrics.make_scorer(fn)}


classDict = {'Random forest classifier':skEnsemble.RandomForestClassifier,
			'Extra Trees Classifier':skEnsemble.ExtraTreesClassifier,
			'Support Vector Machine': skSVM.SVC,
			'Stochastic Gradient Descent':skLinear.SGDClassifier,
			'Gaussian Naive Bayes':skNaiveBayes.GaussianNB,
			}
			
availableMethods = list(classDict.keys())

preProcessDict = {'MinMaxScaler':skPreProc.MinMaxScaler,
				'RobustScaler':skPreProc.RobustScaler,
				'UniformScaler':skPreProc.QuantileTransformer,
				'GaussianScaler':skPreProc.QuantileTransformer,
				'IdentityScaler':skPreProc.FunctionTransformer}

abbrevDict = {'Random forest classifier':'RFC',
			'Extra Trees Classifier':'ETC',
			'Support Vector Machine': 'SVM',
			'Stochastic Gradient Descent':'SGD',
			'Gaussian Naive Bayes':'GNB'}

abbrevDictRev = {item:key for key,item in abbrevDict.items()}

scorerOptions = {'chi2':skFeatSel.chi2,
				'f_classif':skFeatSel.f_classif,
				'roc_auc_score':skMetrics.roc_auc_score,
				'accuracy':skMetrics.make_scorer(skMetrics.accuracy_score),
				'average_precision':skMetrics.average_precision_score,
				'f1':skMetrics.f1_score}



crossValidation = {'Kfold':{'function':skModel.KFold,'parameters':['n_splits','shuffle']},
				   'StratifiedKFold':{'function':skModel.StratifiedKFold,'parameters':['n_splits','shuffle']},
				   'ShuffleSplit':{'function':skModel.ShuffleSplit,'parameters':['n_splits','test_size']},
				   'StratifiedShuffleSplit':{'function':skModel.StratifiedShuffleSplit,'parameters':['n_splits','test_size']},
				   'Time Series Split':{'function':skModel.TimeSeriesSplit,'parameters':['n_splits']}}

featureSelection  = {'PCA':PCA,
					'BestKFeature':skFeatSel.SelectKBest,
					'NMF':NMF,
					'IdentityFunction':skPreProc.FunctionTransformer}


sgdLossFuncs = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive']

randomForestWidgets = OrderedDict([('n_estimators',['50',list(range(10,100,10)),'The number of trees in the forest.']),
					 ('max_features',['sqrt',['sqrt','log2'],'The number of features to consider when looking for the best split:\nIf int, then consider max_features features at each split.\nIf float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.\nIf “auto”, then max_features=sqrt(n_features).\nIf “sqrt”, then max_features=sqrt(n_features) (same as “auto”).\nIf “log2”, then max_features=log2(n_features).\nIf None, then max_features=n_features.\nNote: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.']),
					 ('max_depth',['None','The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.']),
					 ('min_samples_split',['2','The minimum number of samples required to split an internal node:\nIf int, then consider min_samples_split as the minimum number.\nIf float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.']),
					 ('min_samples_leaf',['1','The minimum number of samples required to be at a leaf node:\nIf int, then consider min_samples_leaf as the minimum number.\nIf float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.']),
					 ('max_leaf_nodes',['None','Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.']),
					 ('oob_score',['True',trueFalse,'The minimum number of samples required to split an internal node:\nIf int, then consider min_samples_split as the minimum number.\nIf float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.']),
					 ('bootstrap',['True',trueFalse,'Whether bootstrap samples are used when building trees.']),
					 ('class_weight',['balanced_subsample',['balanced_subsample','balanced','None'],'The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))\nThe “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.']),
					 ('n_jobs',['-2','The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.']),
					 ])
					 
extraTreeWidgets = randomForestWidgets

					 
					 
supportVectorWidgets = OrderedDict([('C',['1','Penalty parameter C of the error term.']),
					 ('kernel',['rbf',['rbf','poly','linear'],'Specifies the kernel type to be used in the algorithm.']),
					 ('degree',['3','Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.']),
					 ('gamma',['auto','Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.']),
					 ('coef0',['0','Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.']),
					 ('probability',['False',trueFalse,'Whether to enable (class) probability estimates. This must be enabled prior to calling fit, and will slow down that method.']),
					 ('shrinking',['False',trueFalse,'Whether to use the shrinking heuristic.']),
					 ('tol',['1e-3','Tolerance for stopping criterion.']),
					 ('cache_size',['300','Specify the size of the kernel cache (in MB).']),
					 ('class_weight',['None',['balanced','None'],'The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))\nThe “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.']),
					 ])

sgdWidgets = OrderedDict([('loss',['hinge',sgdLossFuncs,'The loss function to be used. Defaults to ‘hinge’, which gives a linear SVM.The ‘log’ loss gives logistic regression, a probabilistic classifier. ‘modified_huber’ is another smooth loss that brings tolerance to outliers as well as probability estimates. ‘squared_hinge’ is like hinge but is quadratically penalized. ‘perceptron’ is the linear loss used by the perceptron algorithm. The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.']),
					 ('shuffle ',['True',trueFalse,'HIGHLY RECOMMENDED TO USE (TRUE)\nWhether or not the training data should be shuffled after each epoch. Defaults to True.']),
					 ('penalty',['l2',['none', 'l2', 'l1', 'elasticnet'],'Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.']),
					 ('learning_rate',['optimal',['optimal','constant','invscaling'],'The learning rate schedule:\n‘constant’: eta = eta0\n‘optimal’: eta = 1.0 / (alpha * (t + t0)) [default]\n‘invscaling’: eta = eta0 / pow(t, power_t)\nwhere t0 is chosen by a heuristic proposed by Leon Bottou']),
					 ('tol',['1e-3','Tolerance for stopping criterion.']),
					 ('alpha',['0.0001','Constant that multiplies the regularization term. Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.']),
					 ('class_weight',['balanced',['balanced','None'],'Preset for the class_weight fit parameter.']),
					 ('n_jobs',['-2','The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation. -1 means ‘all CPUs’. Defaults to 1.']),
					 #('class_weight',['balanced',['balanced','None'],'The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))\nThe “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.']),
					 ])

gaussianNaiveBayesWidgets  = OrderedDict()
					 
widgetCollection = {'Random forest classifier':randomForestWidgets,
					'Extra Trees Classifier': extraTreeWidgets,
					'Support Vector Machine':supportVectorWidgets,
					'Stochastic Gradient Descent':sgdWidgets,
					'Gaussian Naive Bayes':gaussianNaiveBayesWidgets}

crossValidationMethods = {'k-fold':skModel.KFold,
						 'Stratified k-fold': skModel.StratifiedKFold,
						 'Stratified Shuffle Split':skModel.StratifiedShuffleSplit,
						 'Shuffle Split':skModel.ShuffleSplit,
						 'Time Series Split':skModel.TimeSeriesSplit}




class classifierAnalysisCollection(object):
	'''
	'''
	def __init__(self):
		'''
		'''
		self.id = 0 
		self.collection = OrderedDict()
	
	def save_grid_search_results(self,results):
		'''
		'''
		self.collection[self.id] = results
		self.id += 1
		
	def get_grid_search_results(self, id):
		'''
		'''
		return self.collection[self.id]
	
	def get_general_infos(self, type = 'props'):
		'''
		'''
		collect = dict()
		for id, props in self.collection.items():
			collect[id] = props[type]
		return collect
			
		
	
	

class predictClass(object):
	
	
	def __init__(self, dataTreeview, dfClass, classifierCollection):
		
		
		self.classifierCollection = classifierCollection
		self.dfClass = dfClass
		self.dataTreeview = dataTreeview
		
		
		if self.classifier_grid_search_present():
			pass 
		else:
			tk.messagebox.showinfo('Error ...',
				'No classifer optimized yet. Use the grid search to optimize parameter.')
			return
		
		## prepare data
		self.df = self.prepare_data()
		
		if self.df is not None:
		
			self.define_variables()
			self.get_grid_searches()
			self.build_toplevel()
			self.build_widgets()
			self.define_menu()
	
		
	def define_variables(self):
		'''
		Define variables
		'''
		self.estimationType = tk.StringVar()
		
		self.get_grid_searches()
		
	
	def get_grid_searches(self):
		'''
		Get some information of the established classifiers
		'''
		self.pipelineDict = self.classifierCollection. get_general_infos(type='pipeline')
		self.generalInfo = self.classifierCollection.get_general_infos(type='props')
				
	def prepare_data(self):
		'''
		Get data
		'''
		# can only handle numeric columns
		if self.dataTreeview.onlyNumericColumnsSelected == False:
			tk.messagebox.showinfo('Error ..','Please select only numeric data.')
			return 
		     		
     	# check if selection is from one file.
		selectionIsFromSameData, selectionDataFrameId = \
		self.dataTreeview.check_if_selection_from_one_data_frame()
		
		if selectionIsFromSameData:
			# get data and dropna
			columns = self.dataTreeview.columnsSelected
			data = self.dfClass.get_current_data_by_column_list(columns)
			df = data.dropna(how='any')
			self.features = data
			return df
		
		else:
			tk.messagebox.showinfo('Error ..','Please select only data from one file.')
			return
			
			
	def close(self,event=None):
		'''
		Destroys toplevel
		'''
		self.toplevel.destroy()
	
		
	def build_toplevel(self):
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Supervised Learning - Predict Class') 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		popup.bind('<Escape>',self.close)
		if platform == 'LINUX':
			popup.bind('Button-1', self.destroy_menu)
		w = 590
		h= 540
		self.toplevel = popup
		self.center_popup((w,h))	
	
	
	def build_widgets(self):
		'''
		Builds widgets
		'''
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		self.cont.grid_columnconfigure(3,weight=1)
		self.cont.grid_rowconfigure(5,weight=1)
		
		self.contClass = tk.Frame(self.cont,background=MAC_GREY)
		self.contClass.grid(row=5,column=0,columnspan = 4, sticky= tk.NSEW)
		self.contClass.grid_columnconfigure(1,weight=1)	
				
		labelTitle = tk.Label(self.cont, text = 'Predict Class', 
                                     **titleLabelProperties)
		
		labelCombo = tk.Label(self.cont, text = 'Estimator Selection: ',
							bg = MAC_GREY)
		comboMethod = ttk.Combobox(self.cont, textvariable = self.estimationType,
						values = ['Average over outside cv (best settings may vary)',
						'Use best estimator of all outside cvs',
						'Refit best parameter settings (rank mean) on all data',
						'Refit best parameter settings (scorer mean) on all data']) 
		
		labEstimator = tk.Label(self.cont, text = 'Available grid search results',bg=MAC_GREY) 
		
		
		
		labelTitle.grid(row=0)
		labelCombo.grid(row=1,column=0)
		comboMethod.grid(row=1,column=1,columnspan=4,sticky=tk.EW)
		labEstimator.grid(row=2)
		
		self.add_collected_grid_searches()
		
		applyButton = ttk.Button(self.cont, text = 'Predict', 
								 command = self.perform_prediction)
		closeButton = ttk.Button(self.cont, text = 'Close')
		
		applyButton.grid(row=6,column=0,pady=10,padx=5)
		closeButton.grid(row=6,column=3, sticky=tk.E,pady=10,padx=5)
		
		
	def add_collected_grid_searches(self):
		'''
		Add collected grid search results
		'''
		# imitate frame that is scrollable
		vertFrame = VerticalScrolledFrame(self.contClass)
		vertFrame.pack(expand=True,fill=tk.BOTH)
		
		for id,pipeline in self.pipelineDict.items():
			var = tk.BooleanVar(value = False)
			txt = self.create_tooltip_text(id)
			cb = ttk.Checkbutton(vertFrame.interior, text = 'Grid search id - ' + str(id), 
								 variable = var) 
			CreateToolTip(cb,text = txt)
			# disabled checkbutton if number of features does not match
			# selected columns
			if self.does_input_fit_with_used_data(id) == False:
				cb.configure(state=tk.DISABLED)
			cb.grid() 
			cb.bind(right_click, self.post_menu)
		
	
	def does_input_fit_with_used_data(self, id):
		'''
		Check if number of features match, if not return False
		This leads to disabling the checkbutton
		'''
		settings = self.generalInfo[id]
		if settings['# of features'] != len(self.features):
			return False
		else:
			return True
		
			
	def create_tooltip_text(self,id):
		'''
		Assemble text to identify the correct grid search.
		'''
		s = ''
		settings = self.generalInfo[id]
		for key,value in settings.items():
			s = s + '{}: {}\n'.format(key,value)
		pipeSteps = self.pipelineDict[id]
		for step,options in pipeSteps.items():
			if len(options) == 0:
				continue
			else:
				elements = '{}: {}\n'.format(step,get_elements_from_list_as_string(options))
				s = s + elements
		return s		
				
				
	def classifier_grid_search_present(self):
		'''
		Return False if no grid search has been performed yet
		'''
		if 	len(self.classifierCollection.collection) == 0:
			return False
		else:
			return True

	def define_menu(self):
		'''
		Defines context menu
		'''
		self.menu = tk.Menu(self.toplevel,**styleDict)
		self.menu.add_command(label='Re-plot results')
		self.menu.add_command(label='Show confusion matrix')
		self.menu.add_separator()
		self.menu.add_command(label='Remove')
			
		
	
	def post_menu(self,event):
		'''
		'''
		w = event.widget
		text = w.cget('text')
		print(text)
		id = int(float(text.split('-')[-1]))
		print(id)
		self.gridSearchId = id
		x = self.toplevel.winfo_pointerx()
		y = self.toplevel.winfo_pointery()
		self.menu.post(x,y)		
     	
	def destroy_menu(self,event):
 		'''
 		'''
 		if hasattr(self,'menu') and self.menu is not None:
 			self.menu.unpost()
 			self.menu = None    	

		
	def perform_prediction(self):
		'''
		'''
		
		#self.df[]
		
		tk.messagebox.showinfo('Under Construction ..',
				'The output is currently not shown because the method being changed at the moment.',
				parent=self.toplevel)
				
		predictionByBestEstimator = results['rocCurveParams']
		for nSplit, props in predictionByBestEstimator.items():
			
			estimator = props['estimator']
			estimator.pedict(data)
			estimator.predict_proba(data)
			
	
	def center_popup(self,size):

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 				
		
		
		
class classifyingDialog(object):
	
	
	def __init__(self, dfClass, plotter, dataTreeview, classificationCollection, 
						numericColumns = [], categoricalColumns = [], initialMethod = 'Random forest classifier', cmap = 'Blues'):
		'''
		'''
		self.initialMethod = initialMethod
		self.collectCurrentWidgets = dict()
		self.settingDict = dict()
		self.calculateROCCurve = tk.BooleanVar(value=True)
		self.perfromCV = tk.BooleanVar(value=True)
		
		self.trainID, self.testID = None, None
		
		if len(categoricalColumns) == 0:
			self.targetColumn = 'Please select'
		else:
			self.targetColumn = categoricalColumns[0]
			
		self.availableTargetColumns = dfClass.dfsDataTypesAndColumnNames[dfClass.currentDataFile]['object'] + \
									  dfClass.dfsDataTypesAndColumnNames[dfClass.currentDataFile]['int64']
		self.availableDataFrames = dfClass.get_file_names()
		self.dataID = dfClass.currentDataFile
		self.data = dfClass.get_current_data_by_column_list(numericColumns).dropna()
		self.selectedFeatures = numericColumns 
		self.dfClass = dfClass
		
		self.dataTreeview = dataTreeview 
		
		print(self.availableTargetColumns)
		self.build_toplevel()
		self.add_widgets_to_toplevel()
	
	def close(self):
		'''
		Closes the toplevel.
		'''
		
		self.toplevel.destroy()
               
			
	def build_toplevel(self):
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Supervised Learning - Classification') 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w = 590
		h= 540
		self.toplevel = popup
		self.center_popup((w,h))	

	def add_widgets_to_toplevel(self):
		'''
		'''
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		self.cont.grid_columnconfigure(2,weight=1)
		
		self.contClassMethod = tk.Frame(self.cont,background=MAC_GREY)
		self.contClassMethod.grid(row=5,column=0,columnspan = 4, sticky= tk.NSEW)
		self.contClassMethod.grid_columnconfigure(1,weight=1)
		
		labelTitle = tk.Label(self.cont, text = 'Supervised Learning - Classification algorithms', 
                                     **titleLabelProperties)
		labelHelp = tk.Label(self.cont, text ='We are using the skilearn modules and therefore names \nof parameters and description'+
												'are similiar. The sklearn webpage has\na brilliant overview of the available classification methods, tips for usage, advantages and disadvantages.',
												justify=tk.LEFT, bg=MAC_GREY)
									
		clusterMethod = tk.Label(self.cont, text = 'Classification algorithm: ', bg=MAC_GREY)
		comboBoxClassif = ttk.Combobox(self.cont, values = availableMethods)
		comboBoxClassif .insert(tk.END, self.initialMethod)
		comboBoxClassif .bind('<<ComboboxSelected>>', self.new_algorithm_selected)
		
		labelSklearnWebsite = tk.Label(self.cont, text = 'sklearn.cluster webpage', **titleLabelProperties)
		labelSklearnWebsite.bind('<Button-1>', self.openWebsite)
		
		labelSettings = tk.Label(self.cont, text = 'Cluster analysis initial settings', bg=MAC_GREY)
				
		buttonPerformClassification = ttk.Button(self.cont, text = 'Done', command = self.perform_analysis)
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		
		labelTitle.grid(row=0,padx=5,sticky=tk.W,pady=5, columnspan=3)
		labelSklearnWebsite.grid(row=0, column= 3, sticky=tk.E,pady=5)
		labelHelp.grid(row=1, column= 0, columnspan=4, sticky=tk.W,pady=5,padx=5)
		
		clusterMethod.grid(row=2,column=0, sticky=tk.E,padx=5,pady=3)
		comboBoxClassif .grid(row=2,column=1, columnspan= 4, sticky=tk.EW,padx=(20,40),pady=3)
		labelSettings.grid(row=3,column=0,padx=5,pady=2,columnspan=2, stick=tk.W)
		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(row=4,columnspan=4,sticky=tk.EW,padx=2)
		
		self.create_method_specific_widgets(self.initialMethod)
		
		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(row=6,columnspan=4,sticky=tk.EW,padx=2)
		
		labelInfo = tk.Label(self.cont, text = 'Data set up ...\nRows are considered as samples, columns as features.') 
		
		labelClass = tk.Label(self.cont, text = 'Target column (classes): ', bg=MAC_GREY)
		self.comboBoxTargetColumn = ttk.Combobox(self.cont, values = self.availableTargetColumns) 
		self.comboBoxTargetColumn.insert(tk.END, str(self.targetColumn))
		self.comboBoxTargetColumn.bind('<<ComboboxSelected>>',self.add_target_column_to_data)
		
		
		labelTrainTest = tk.Label(self.cont, text = 'Define test and train data', bg=MAC_GREY)
		CreateToolTip(labelTrainTest , title_ = 'Important',text= 'Learning the parameters of a prediction function and'+
												' testing it on the same data is a methodological mistake: a model that'+
												' would just repeat the labels of the samples that it has just seen would'+
												' have a perfect score but would fail to predict anything useful on yet-unseen data.'+
												' This situation is called overfitting. To avoid it, it is common practice when'+
												' performing a (supervised) machine learning experiment to hold out part of the available'+
												' data as a test set X_test, y_test.')
		
		labelTrainData = tk.Label(self.cont, text =  'Train Data: ', bg=MAC_GREY)
		labelTestData = tk.Label(self.cont, text = 'Test Data: ', bg=MAC_GREY)
		labelSplitData = tk.Label(self.cont, text = 'Split current data: ')
		
		self.comboboxTrain = ttk.Combobox(self.cont, values = self.availableDataFrames, 
															exportselection = 0)
		self.comboboxTest = ttk.Combobox(self.cont, values = self.availableDataFrames, 
															exportselection = 0)
		self.update_combobox_texts([self.comboboxTrain,self.comboboxTest],
								   ['Please select','Please select'])
		
		
		buttonSplitData = ttk.Button(self.cont, text = 'Split data', command = self.split_data)
		
		buttonGridSearch = ttk.Button(self.cont, text = 'Find best settings', command = lambda : gridSearchClassifierOptimization(self.data, self.selectedFeatures ,self.targetColumn))
		

		crossValidationCB = ttk.Checkbutton(self.cont, 
											text = 'Perform k - fold Cross Validation (CV)', 
											variable = self.perfromCV)
											
		labelDataForCV = tk.Label(self.cont, text = 'Data used in CV: ', bg=MAC_GREY)
		
		labelKFold = tk.Label(self.cont, text = 'k :', bg=MAC_GREY)
		
		rocCurveCB = ttk.Checkbutton(self.cont, text = 'Show ROC curve', variable = self.calculateROCCurve)
		CreateToolTip(rocCurveCB, text = 'Plots roc curve from k-fold CV. The ROC curve analysis for classifier evaluation'+
										 ' on test data is always performed, unless no test data given.')
		
		## grid widgets 
		

		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(row=6,columnspan=4,sticky=tk.EW,padx=2)
		
		labelClass.grid(row=7,column=0,padx=3,pady=5,columnspan=2,sticky=tk.W)
		self.comboBoxTargetColumn.grid(row=7,column=1,pady=5,columnspan=4,sticky=tk.EW,padx=(20,40))
		
		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(row=8,columnspan=4,sticky=tk.EW,padx=2)
		
		buttonGridSearch.grid()		
				
				
		labelTrainTest.grid(row=10)
		labelSplitData.grid()
		buttonSplitData.grid()
		labelTrainData.grid()
		labelTestData.grid()
		self.comboboxTrain.grid() 
		self.comboboxTest.grid()


		crossValidationCB.grid(columnspan=4, sticky=tk.W,padx=3,pady=5)
		labelDataForCV.grid()
		labelKFold.grid()
		
		rocCurveCB.grid(columnspan=4, sticky=tk.W,padx=3,pady=5)
		
		
		buttonPerformClassification.grid()#row=12,column=0,padx=3,pady=5,sticky=tk.W)
		closeButton.grid(row=12,column=3,padx=3,pady=5,sticky=tk.E)
		

	def create_method_specific_widgets(self, method = ''):
		'''
		'''
		widgetInfoDict = widgetCollection[method]
		self.collectCurrentWidgets.clear()
		
		n = 0
		for label, widgetInfo in widgetInfoDict.items():
			labelWidget = tk.Label(self.contClassMethod, text = '{} :'.format(label), bg=MAC_GREY)
			labelWidget.grid(sticky=tk.E, padx=5,column= 0, row = n)
			
			if isinstance(widgetInfo[1],str):
				entry = ttk.Entry(self.contClassMethod) 
				entry.insert(tk.END,widgetInfo[0])
				entry.grid(sticky=tk.EW,column= 1, row = n, padx= (20,40))
				self.collectCurrentWidgets[label] = entry
			else:
				combobox = ttk.Combobox(self.contClassMethod, values = widgetInfo[1], exportselection=0)
				combobox.insert(tk.END,widgetInfo[0])
				combobox.grid(sticky=tk.EW,column= 1, row = n, padx= (20,40))
				self.collectCurrentWidgets[label] = combobox
				
			CreateToolTip(labelWidget,text = widgetInfo[-1])
			n += 1
			
	def new_algorithm_selected(self, event):
		'''
		'''
		combo = event.widget
		newAlgorithm = combo.get()
		if newAlgorithm == self.initialMethod:
			return
			
		if newAlgorithm in availableMethods:
		
			for widget in self.contClassMethod.winfo_children():
				widget.destroy()
			self.create_method_specific_widgets(newAlgorithm)
			self.initialMethod = newAlgorithm
			
	def perform_analysis(self):
		'''
		'''
		self.extract_current_settings() 
		X = self.data[self.selectedFeatures].values
		Y = self.data[self.targetColumn].values
		classifier = classDict[self.initialMethod](**self.settingDict)
		print(classifier)
		self.k_fold_cross_validation(classifier,X,Y)

		
		classifier.fit(X,Y)
		



	def k_fold_cross_validation(self,classifier,X,Y,k = 6, method = 'Stratified k-fold'):
		'''
		'''
		crossVal = crossValidationMethods[method](n_splits = k)
		
		
		for train, test in crossVal.split(X,Y):
			probsTest = classifier.fit(X[train],Y[train]).predict_proba(X[test])
			
			print(probsTest)
			print(Y[test])
			print(classifier.classes_)
			fpr, tpr, thresholds = roc_curve(Y[test],probsTest[:, 0], pos_label='+')
			print(auc(fpr,tpr))
			
	def add_target_column_to_data(self,event=None):
		'''
		'''
		self.targetColumn = self.comboBoxTargetColumn.get()
		if self.targetColumn in self.data.columns:
			return
			
		elif self.targetColumn in self.dfClass.get_columns_of_df_by_id(id=self.dataID):
			
			self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
																  definedColumnsList=[self.targetColumn])
			return True	
		else:
			tk.messagebox.showinfo('Error ..','Selected column not in data.')
			self.comboBoxTargetColumn.insert(tk.END,'Please select')
			return
			
			
			
	def split_data(self, fraction = 0.2):
		'''
		Splits data
		'''
		proceed = self.add_target_column_to_data()
		
		if proceed:
		
			X = self.data[self.selectedFeatures]
			Y = self.data[self.targetColumn]
			X_train, X_test, Y_train, Y_test = skModel.train_test_split(X,Y)
			self.add_data_to_dfClass_and_treeview(X_train, Y_train, X_test, Y_test)


	def add_data_to_dfClass_and_treeview(self,X_train, Y_train, X_test, Y_test):
		'''
		'''
		baseFileName = self.dfClass.fileNameByID[self.dataID]
		dfIds = []
		dfFileNames = []
		for n,XY in enumerate([(X_train, Y_train),(X_test, Y_test)]):
			if n == 0:
				fileName = 'TrainData {} [{}]'.format(abbrevDict[self.initialMethod],
													  baseFileName)
			else:
				fileName = 'TestData {} [{}]'.format(abbrevDict[self.initialMethod],
													  baseFileName)
													  
			# get unique file Name 
			fileName = self.dfClass.evaluate_column_name(fileName,self.availableDataFrames) ## counter intuitiv but this function also accepts a list as a template
			dfmerged = XY[0].join(XY[1])
			id = self.dfClass.get_next_available_id()
			dfIds.append(id)
			dfFileNames.append(fileName)
			## add df to dfClass
			self.dfClass.add_data_frame(dataFrame = dfmerged,
										id = id,
										fileName = fileName)
			## add to treeview to let the user explore
			self.dataTreeview.add_new_data_frame(id,
												 fileName,
												 self.dfClass.dfsDataTypesAndColumnNames[id])
		self.trainID, self.testID = dfIds
		self.update_combobox_texts([self.comboboxTrain,self.comboboxTest],
								   [dfFileNames[0],dfFileNames[1]])

		
	def update_combobox_texts(self, comboBoxList, textList):
		'''
		'''
		for n,comboBox in enumerate(comboBoxList):
			comboBox.delete(0,tk.END)
			comboBox.insert(tk.END,textList[n])		
			
			
	def extract_current_settings(self):
		'''
		'''
		if len(self.settingDict) > 0:
			self.settingDict.clear() 
		
		for key, widget in self.collectCurrentWidgets.items():
			
			stringEntry = widget.get()
			if stringEntry in trueFalse:
				value = trueFalseDict[stringEntry]
			elif stringEntry == 'None':
				value = None
			elif key == 'gamma':
				if stringEntry == 'auto':
					value = stringEntry
				else:
					value = float(stringEntry)
					
			elif key in ['C','coef0','tol','cache_size']:
				value = float(stringEntry)
			elif key in ['degree','n_estimators','max_depth','min_samples_split',
						'min_samples_leaf',
						'n_jobs']:
				value = int(float(stringEntry))
			
			else:
				value = stringEntry
				
			self.settingDict[key] = value

	def center_popup(self,size):

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))             
	
	
	def openWebsite(self,event):
		'''
		'''
		webbrowser.open_new(r"http://scikit-learn.org/stable/supervised_learning.html#supervised-learning")	

				


class gridSearchClassifierOptimization(object):
	
	
	def __init__(self, classificationCollection, dfClass, features, targetColumn = None,
					Treeview = None, plotter = None):
		
		
		self.classificationCollection = classificationCollection
		
		self.get_items_associations()
		self.define_variables()
		
		self.data = dfClass.get_current_data_by_column_list(features+targetColumn)
		
		self.targetColumn = targetColumn
		self.features = features
		self.plotter = plotter
		
		self.build_popup()
		self.add_widgets_to_toplevel()
	
	def define_variables(self):
		'''
		'''
		self.optimizeGrid = dict()
		self.functionSettings = dict()
		
		self._dragProps = {'x':0,'y':0,'item':None,'receiverBox':None}
		self.gridSearchCV = {'n_splits':3, 'KFold Procedure':'Stratified k-fold'}
		self.nestedCV = {'n_splits':5, 'KFold Procedure':'StratifiedShuffleSplit'}
		
	def close(self,event=None):
		'''
		Closes the toplevel.
		'''
		self.toplevel.destroy()
         
         			
	def build_popup(self):
		'''
		Builds the toplevel to put widgets in 
		'''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Classifier Parameter Optimization') 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		popup.bind('<Escape>',self.close)
		w = 1050
		h= 930
		self.toplevel = popup
		self.center_popup((w,h))	
		#top = self.toplevel.winfo_toplevel()
		#self.menuBar = tk.Menu(top)
		#top['menu'] = self.menuBar
		#self.subMenu = tk.Menu(self.menuBar)
		#self.menuBar.add_cascade(label='Help', menu=self.subMenu)
		#self.subMenu.add_command(label='About', command=lambda : print("hi"))
    
    

    
    
    
		# create a toplevel menu
		#menubar = tk.Menu(self.toplevel)
		#menubar.add_command(label="Hello!", command=lambda : print("hi"))
		#menubar.add_command(label="Quit!", command=lambda : print("hi"))
		# display the menu
		#self.toplevel.config(menu=menubar)	
	

	def add_widgets_to_toplevel(self):
		'''
		'''

			
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		self.cont.grid_columnconfigure(2,weight=1)
		self.cont.grid_rowconfigure(2,weight=1)
		
		self.contCanvas = tk.Frame(self.cont,background=MAC_GREY) 
		self.contCanvas.grid(row=2,column=0, sticky=tk.NSEW, columnspan=3)
			
		self.canvas = tk.Canvas(self.contCanvas, background = MAC_GREY)#, width=200, height=100)
		self.canvas.pack(fill=tk.BOTH, expand=True)
		
		self.canvas.bind('<1>',self.on_item_click)
		self.canvas.bind('<B1-Motion>',self.on_item_move)
		self.canvas.bind('<ButtonRelease-1>',self.on_item_release)
		self.canvas.bind('<Double-Button-1>', self.edit_settings)
		self.canvas.bind(right_click, self.remove_item)
						
		self.create_basic_layout()
		
		runGridButton = ttk.Button(self.cont, text = 'Perform Grid Search', command = self.perform_grid_search) 
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		
		
		runGridButton.grid(row=3,column= 0, sticky=tk.W, pady = 4, padx = 3)
		closeButton.grid(row=3, column = 2, sticky = tk.E, pady = 4, padx = 3)
		
		
		
	def edit_settings(self,event):
		'''
		Handles Double-Click events. Allows you to edit settings. 
		'''
		tagsOfSelection = self.canvas.gettags(tk.CURRENT)
		if len(tagsOfSelection) > 0:
			firstTag = tagsOfSelection[0]
		else:
			return
		if firstTag == 'current':
			return
		else:
			if firstTag == 'gridSearchCV':
				boom = defineGridSearchDialog('gridSearchCV', self.features,self)
				if len(boom.collectParamGrid) != 0 or len(boom.settingDict) != 0:
					self.gridSearchCV = merge_two_dicts(boom.collectParamGrid,boom.settingDict)
			
			elif firstTag == 'nestedCV':
				boom = defineGridSearchDialog('nestedCV', self.features,self)
				if len(boom.collectParamGrid) != 0 or len(boom.settingDict) != 0:
					self.nestedCV = merge_two_dicts(boom.collectParamGrid,boom.settingDict)
			
			else:
				boom = defineGridSearchDialog(tagsOfSelection[1].split('_')[1], self.features,self)
				self.optimizeGrid[tagsOfSelection[1]] = boom.collectParamGrid
				self.functionSettings[tagsOfSelection[1]] = boom.settingDict
			
	def remove_item(self,event):
		'''
		Handles right click action (Button-2 or Button-3). Will remove certain items.
		'''
		tagsOfSelection = self.canvas.gettags(tk.CURRENT)
		if len(tagsOfSelection) > 0:
			firstTag = tagsOfSelection[0]
		else:
			return
		if firstTag == 'current':
			return
		if firstTag in ['gridSearchCV','nestedCV']:
			tk.messagebox.showinfo('Error ..',
				'You cannot delelete this item.',
				parent=self.toplevel)
			return
		else:
			self.canvas.delete(firstTag)
			if firstTag in self.optimizeGrid:
				del self.optimizeGrid[firstTag]
			if firstTag in self.functionSettings:
				del self.functionSettings[firstTag]		
				
		receiverBox = tagsOfSelection[0].split('_')[0]	
		
		coords = self.canvas.coords(receiverBox)
		entries  = self.recieverRectangleEntries[receiverBox]
		idx = entries.index(tagsOfSelection[1])
		# delete entry and re-create items
		del self.recieverRectangleEntries[receiverBox][idx]
		del self.reciverRectangleImages[receiverBox][idx]
		if receiverBox != 'Scorer':
			self.build_images_in_receiverbox(receiverBox,coords)
		else:
			self.build_scorer_images_in_receiverbox(receiverBox,coords)
			
			
	def create_basic_layout(self):	
		'''
		'''
		self.build_nested_cv()
		self.canvas.create_text(30, 20, anchor = tk.W, text = 'Drag & Drop transform operations from here\nonto '+
															  'the right panel to build a pipeline that should be optimized. ', 
									font = LARGE_FONT, fill = '#4C626F')
		n = 0
		for key, subWidgets in 	self.itemsToDrag.items():
			y = 50+120*n
			self.canvas.create_text(30, y, anchor = tk.W, text = key, 
									font = LARGE_FONT, fill = '#4C626F')
										
			self.canvas.create_line(5,y+12,320,y+12)
			k = 0
			for procedure, image in subWidgets.items():
				if image != '':
					if k == 3 and key == 'Scorer':
						y += 30 
						k = 0
					xPos = 10 + 80*k
					yPos = y + 60
					
					self.canvas.create_image(xPos, yPos, image =  image, anchor = tk.W, tag = procedure)
					self.imagePositions[procedure] = (xPos,yPos)
					k+=1
			n += 1	
		
		self.canvas.create_image(10,yPos + 220, image = self.resetIcon, anchor = tk.W, tag = 'reset')
					
		
	def on_item_release(self,event):
		'''
		Handles button-1 (left click) release. (e.g. drag and drop)
		'''
		if self._dragProps['item'] is not None:
		
			self.canvas.itemconfigure(self._dragProps['receiverBox'],width=1,outline='black')
		
			coords = self.canvas.coords(self._dragProps['receiverBox'])
			image = self.canvas.itemcget(self._dragProps['item'],'image')
			
			if (coords[0] <= event.x <= coords[2]) and (coords[1] <= event.y <= coords[3]):
				id = len(self.recieverRectangleEntries[self._dragProps['receiverBoxTag']])
				self.recieverRectangleEntries[self._dragProps['receiverBoxTag']].append('{}_{}'.format(id,self._dragProps['itemTag']))
				self.reciverRectangleImages[self._dragProps['receiverBoxTag']].append(image)
				if self._dragProps['receiverBoxTag'] != 'Scorer':
					self.build_images_in_receiverbox(self._dragProps['receiverBoxTag'],
												 	coords)
												 
				elif self._dragProps['receiverBoxTag'] == 'Scorer':
					self.build_scorer_images_in_receiverbox(self._dragProps['receiverBoxTag'],
															coords)
															
				
			self.canvas.delete(self._dragProps['item'])	
			defaultPositionImage = self.imagePositions[self._dragProps['itemTag']]
			x,y = defaultPositionImage
			self.canvas.create_image(x,y, image=image, tag = self._dragProps['itemTag'], anchor=tk.W)
			

			self._dragProps['item'] = None
			self._dragProps['receiverBox'] = None
			self._dragProps['x'] = 0
			self._dragProps['y'] = 0
		
		else:
			tagsOfSelection = self.canvas.gettags(tk.CURRENT)
			if len(tagsOfSelection) == 0:
				return
			elif tagsOfSelection[0] == 'current':
				return
			elif tagsOfSelection[0] == 'reset':
				self.reset()
				
			elif tagsOfSelection[0] == 'performGridSearch':	
				self.perform_grid_search()
			
		
	def on_item_move(self,event):
		'''
		Change position of dragged item on canvas.
		'''
		if self._dragProps['item'] is not None:
		
			deltaX =  event.x - self._dragProps['x']
			deltaY =  event.y - self._dragProps['y']
			self.canvas.move(self._dragProps['item'], deltaX, deltaY)
			self._dragProps['x'] = event.x
			self._dragProps['y'] = event.y
										
	def on_item_click(self,event):
		'''
		Handles item click events (Button-1 left-click)
		'''
		tagsOfSelection = self.canvas.gettags(tk.CURRENT)
		if len(tagsOfSelection) > 0:
			firstTag = tagsOfSelection[0]
			if firstTag == 'current':
				return
			elif firstTag in self.checkDragAndDropTags:
				receiverBoxTag = self.get_appropiate_receiver_box(firstTag)
				self._dragProps['item'] = firstTag
				self._dragProps['x'] = event.x
				self._dragProps['y'] = event.y
				if receiverBoxTag is not None:
					
					receiverBox = self.canvas.find_withtag(receiverBoxTag)[0]
					self._dragProps['receiverBox'] = receiverBox
					self._dragProps['receiverBoxTag'] = receiverBoxTag
					self._dragProps['itemTag'] = firstTag
					self.canvas.itemconfigure(receiverBox,width=2,outline='red')
	def reset(self):
 		'''
 		Resets all made actions.
 		'''
 		#print(self.recieverRectangleEntries)
 		for receiverBox in self.recieverRectangleEntries.keys():
 			self.reciverRectangleImages[receiverBox] = []
 			self.recieverRectangleEntries[receiverBox] = []
 			if receiverBox != 'Scorer':
 				self.build_images_in_receiverbox(receiverBox,coords = None)
 			else:
 				self.build_scorer_images_in_receiverbox(receiverBox,coords = None) 

 		self.define_variables()
 		
 
 
 					
	def build_scorer_images_in_receiverbox(self, receiverBoxTag, coords):
		'''
		Displays scorer image in receiver box.
		'''		
		tagForItems = '{}_branchItems'.format(receiverBoxTag)
		imageList = self.reciverRectangleImages[receiverBoxTag]
		if len(imageList) == 0:
			self.canvas.delete(tagForItems)
			return
		
		centerWidth = coords[2] - coords[0]
		yPos = coords[1]
		branchEndPosition = self.create_branch_split(coords[0]+ centerWidth/2,yPos+25, 
													portionLeft = 0.8,length = 0.4 * centerWidth,
													numberBranch = 1, connectionLength = [330],
													tagList = [tagForItems],
													addStartLine  = True,
													yPosStartLine = yPos)
		
		self.add_scorer_images(imageList, centerXPosition = coords[2] - centerWidth/2, 
									portionLeft = 0.8,
									length = 0.4 * centerWidth,
									yStart = coords[1] + 30, lineTags = tagForItems)											
		
												
		
		self.create_branch_bottom(centerXPosition = coords[0]+ centerWidth/2, 
											yPositionBranch = branchEndPosition[-1][1], 
											yPositionEnd = coords[3],portionLeft = 0.8, 
											length = 0.4 * centerWidth, numberBranch = 1, 
											connectionLength = 20, tagLines = tagForItems)		
		 
		
	def add_scorer_images(self,imageList, centerXPosition, portionLeft, length,
											yStart, lineTags, distance = 30):
		'''
		adding the scorer image
		'''			
		startXPos = int(centerXPosition - portionLeft * length)
		extraTagList = self.recieverRectangleEntries[self._dragProps['receiverBoxTag']]
		for n,image in enumerate(imageList):
			distance = 30 + 30 * n
			self.canvas.create_line(startXPos, yStart + distance, startXPos+10, 
												yStart + distance, tag = lineTags)	
			self.canvas.create_image(startXPos+10, yStart + distance, anchor = tk.W, 
												image = image, tag= lineTags+' '+extraTagList[n])	
			
# 
# 
# 		    	
# 			for n,xPosBranch in enumerate(positions):
# 				self.canvas.create_line(xPosBranch, yPosition, xPosBranch, 
# 									yPosition + connectionLength[n],tag = tagList[0])
# 				if len(imageList) == numberBranch:
# 					if addImageTags:
# 						tagImg = tagList[n]+' '+extraTagList[n]
# 					else:	
# 						tagImg = tagList[n]
# 					imageId = self.canvas.create_image(xPosBranch, 
# 	
	def build_images_in_receiverbox(self,receiverBoxTag,coords):
		'''
		'''
		branches = len(self.recieverRectangleEntries[receiverBoxTag])
		imageList = self.reciverRectangleImages[receiverBoxTag]
		tagForItems = '{}_branchItems'.format(receiverBoxTag)
		if len(imageList) == 0:
			self.canvas.delete(tagForItems)
			return	
		centerWidth = coords[2] - coords[0]
		yPos = coords[1]		
		self.canvas.delete(tagForItems)
		branchEndPosition = self.create_branch_split(coords[0]+ centerWidth/2,yPos+25, 
												portionLeft = 0.5, length = 0.7 * centerWidth, 
												numberBranch = branches, connectionLength = [5],
												tagList = [tagForItems],
												imageList = imageList,
												addImageTags = True, addStartLine  =True,
												yPosStartLine = yPos)
		
		self.create_branch_bottom(centerXPosition = coords[0]+ centerWidth/2, yPositionBranch = branchEndPosition[-1][1], 
											yPositionEnd = coords[3],portionLeft = 0.5, 
											length = 0.7 * centerWidth, numberBranch = branches, 
											connectionLength = 5, tagLines = tagForItems)
		
	def get_appropiate_receiver_box(self, tag):
		'''
		'''
		for key, items in self.receiverTags.items():
			if tag in items:
				return key
		
				
	def center_popup(self,size):

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))    
	
	
	def build_nested_cv(self):
		'''
		Creates items for nested cross validation
		'''
		endPoint = self.build_chain_of_images([self.dataIcon,self.crossValIcon])
		branchEndPosition =  self.create_branch_split(centerXPosition = 600,yPosition = endPoint, portionLeft = 0.7,
												length = 320, numberBranch = 2,connectionLength = [380,20])
		## create reciever boxes (remove scorer [:-1])
		yPosLineEnd, yPosEndRectangle = self.create_receiver_rectangles(list(self.itemsToDrag.keys())[:-1], 
										xPosCenter = branchEndPosition[-1][0], 
										yPosStart = branchEndPosition[-1][1])
										
		self.create_grid_search_rectangle(yPosStart = branchEndPosition[-1][1] ,
										 xPosCenter = branchEndPosition[-1][0], 
										 yPosEndRectangle = yPosEndRectangle,
										 offsetRight = 250)
		
		self.build_final_estimator(yPosLineEnd, branchEndPosition[-1][0])
		
		    	
	def build_chain_of_images(self, imageList, centerXPosition = 600, imageWidth = 64, 
											   connectionLength = 10, yPositionStart = 40,
											   tagList = ['','nestedCV']):
    	    
		imageHeight = imageWidth
		yPosition  = yPositionStart
		for n, image in enumerate(imageList):
			yPosition = yPosition + n * (imageHeight + connectionLength)

			imageId = self.canvas.create_image(centerXPosition,yPosition, 
											   image = image, tag = tagList[n])
			lineId = self.canvas.create_line(centerXPosition,yPosition+imageHeight/2, 
									centerXPosition,yPosition+imageHeight/2 + connectionLength)	
		else:
		
			return yPosition+imageHeight/2 + connectionLength
			
	def build_final_estimator(self, bottomYPosition, bottomXPosition):
		'''
		'''
		xLeftPosition = bottomXPosition - 320
		self.canvas.create_line(bottomXPosition, bottomYPosition, xLeftPosition, bottomYPosition)
		
		self.canvas.create_line(xLeftPosition,bottomYPosition ,xLeftPosition, bottomYPosition-10) 
		
		self.canvas.create_image(xLeftPosition,bottomYPosition-10,
								 anchor=tk.S, image = self.evaluationIcon)
								 
		self.canvas.create_line(xLeftPosition,bottomYPosition ,xLeftPosition, bottomYPosition+20) 
				 		
		self.canvas.create_image(xLeftPosition,bottomYPosition+20,
								 anchor=tk.N, image = self.finalEstimatorIcon,
								 tag = 'performGridSearch')
		
		
		
		   	 		
	def create_branch_bottom(self, centerXPosition, yPositionBranch, yPositionEnd, 
												portionLeft, length, numberBranch, 
												connectionLength = 5, tagLines = None):
		'''
		'''
		startXPos = int(centerXPosition - portionLeft * length)
		endXPos = length + startXPos
		
		if numberBranch == 1:
			positions = [startXPos]
			endXPos = centerXPosition
		elif numberBranch == 2:
			positions = [startXPos, endXPos]
		else:
			positions = np.linspace(startXPos,endXPos,endpoint=True,num=numberBranch)
		
		for n, xPosBranch in enumerate(positions):
			self.canvas.create_line(xPosBranch,yPositionBranch,xPosBranch,
									yPositionBranch+connectionLength, tag = tagLines)
			
		self.canvas.create_line(startXPos,yPositionBranch+connectionLength, 
								endXPos ,yPositionBranch+connectionLength, tag = tagLines)
		self.canvas.create_line(centerXPosition,yPositionBranch+connectionLength,
								centerXPosition,yPositionEnd, tag = tagLines)
		
		 
    		
	def create_branch_split(self, centerXPosition, yPosition, portionLeft, length, numberBranch, 
														connectionLength = [30], branchPos = None,
														imageList = [], tagList = [None], imageWidth = 64,
														addImageTags = False, addStartLine = False,
														yPosStartLine = None):
		'''
		'''
		if addStartLine :
			self.canvas.create_line(centerXPosition, yPosStartLine,
									centerXPosition, yPosition,
									tag = tagList[0])
			
		startXPos = int(centerXPosition - portionLeft * length)
		endXPos = length + startXPos
		
		
		collectBranchYPosition = []
		
		if branchPos is None:
			if numberBranch == 1:
				positions = [startXPos] 
				endXPos = centerXPosition
			elif numberBranch == 2:
				positions = [startXPos, endXPos]
			else:
				positions = np.linspace(startXPos,endXPos,endpoint=True, num = numberBranch)
			if len(connectionLength) == 1:
				connectionLength = connectionLength * numberBranch
				
			self.canvas.create_line(startXPos,yPosition,endXPos,yPosition,tag = tagList[0])
				
				
			if len(tagList) == 1:
				tagList = tagList * numberBranch
			if addImageTags:

				extraTagList = self.recieverRectangleEntries[self._dragProps['receiverBoxTag']]
		    	
			for n,xPosBranch in enumerate(positions):
				self.canvas.create_line(xPosBranch, yPosition, xPosBranch, 
									yPosition + connectionLength[n],tag = tagList[0])
				if len(imageList) == numberBranch:
					if addImageTags:
						tagImg = tagList[n]+' '+extraTagList[n]
					else:	
						tagImg = tagList[n]
					imageId = self.canvas.create_image(xPosBranch, 
		    								 yPosition + connectionLength[n] + imageWidth/2,
		    								 image = imageList[n],
		    								 tag = tagImg)
						
					collectBranchYPosition.append((xPosBranch,yPosition + connectionLength[n] + imageWidth))
		    								 
				else:
					collectBranchYPosition.append((xPosBranch,yPosition + connectionLength[n]))
					
		return collectBranchYPosition
    		
    	
	def create_grid_search_rectangle(self,yPosStart,xPosCenter, yPosEndRectangle, offsetRight, width = 100):
		'''
		'''
		
		self.canvas.create_line(xPosCenter, yPosStart - 10, xPosCenter + offsetRight, yPosStart - 10)
		self.canvas.create_line(xPosCenter + offsetRight, yPosStart-10 ,xPosCenter + offsetRight, yPosStart)
		
		self.canvas.create_image(xPosCenter+offsetRight/2, yPosStart-15, 
								image = self.gridSearchIcon , anchor = tk.S,
								tag = 'gridSearchCV')
								
		self.canvas.create_rectangle(xPosCenter + offsetRight-width/2, yPosStart, 
									 xPosCenter+ offsetRight +width/2, yPosEndRectangle,
									 fill = '#F4F4F4', tag = 'Scorer') 
		self.canvas.create_text(xPosCenter + offsetRight-width/2 + 5, yPosStart + 5, 
								text = 'Scorer',
								anchor = tk.NW)
		self.canvas.create_line(xPosCenter + offsetRight, yPosEndRectangle, 
								xPosCenter + offsetRight, yPosEndRectangle+10) 
								
		self.canvas.create_line(xPosCenter, yPosEndRectangle+10, 
								xPosCenter + offsetRight, yPosEndRectangle+10)     	
 		
    		    
	def create_receiver_rectangles(self, nameList ,xPosCenter, yPosStart, 	
								   width  = 300, height = 120, connectionLength = 20):
		'''
		'''
		xPosStart, xPosEnd = xPosCenter - width/ 2 , xPosCenter + width/ 2
		
		
		for name in nameList:
			 
			yPosEnd = yPosStart + height
			rectangle = self.canvas.create_rectangle(xPosStart, yPosStart,
													 xPosEnd, yPosEnd, tag=name, fill = '#F4F4F4')
			text = self.canvas.create_text(xPosStart+5, yPosStart+5,anchor=tk.NW, text = name)			
			yPosLineEnd = yPosEnd + connectionLength 
			line = self.canvas.create_line(xPosCenter, yPosEnd, xPosCenter,yPosLineEnd)
			yPosStart = yPosLineEnd
		
		return yPosLineEnd, yPosEnd
		
	def get_function(self, tag, funcName,settingDict):
		'''
		'''
		
		if 'FeatureSelection' in tag:
			if funcName == 'BestKFeature':
				
				func = featureSelection[funcName](**settingDict)
			else:
				func = featureSelection[funcName](**settingDict)
		elif 'Classifier' in tag:
			func = classDict[abbrevDictRev[funcName]](**settingDict)
		elif 'Pre-Processing' in tag:
			if funcName == 'GaussianScaler':
				settingDict['output_distribution'] = 'normal'
			elif funcName == 'IdentityScaler':
				settingDict['func'] = None
			func = preProcessDict[funcName](**settingDict)
		
		return func
			
	def build_classifier_settings(self):
		'''
		Gets the classifier and its parameter to be optimized. 
		'''
		classiDict = dict()
		if 'Classifier' in self.recieverRectangleEntries:
			estimatorList = self.recieverRectangleEntries['Classifier']
			estimator = estimatorList[0]
			if estimator in self.functionSettings:
				settingDict = self.functionSettings[estimator]
			else:
				settingDict = dict()
			classiDict['Classifier'] = [self.get_function('Classifier',
										estimator.split('_')[-1],settingDict)]
			if estimator in self.optimizeGrid:
				optimizeParam = self.optimizeGrid[estimator]							
				for param, values in optimizeParam.items():
					keyParamGrid = '{}__{}'.format('Classifier',param)
					#values = [int(float(x)) for x in values]
					classiDict[keyParamGrid] = values
				
			return classiDict
	
	
	
	def build_optimize_step_settings(self,pipeLineStep):
		'''
		'''
		collectOptimGrid = []
		if pipeLineStep in self.recieverRectangleEntries:
			processes = self.recieverRectangleEntries[pipeLineStep]
			
			
			for proc in processes:
				savedGridParams = dict()
				if proc in self.functionSettings:
					settingDict = self.functionSettings[proc]
				else:
					settingDict = dict()
				func = self.get_function(pipeLineStep,
										proc.split('_')[-1],settingDict )
				savedGridParams[pipeLineStep] = [func]	
				if proc in self.optimizeGrid:					
					optimizeParam = self.optimizeGrid[proc]
				
					for param, values in optimizeParam.items():
						keyParamGrid = '{}__{}'.format(pipeLineStep,param)
						#values = [int(float(x)) for x in values]
						savedGridParams[keyParamGrid] = values
						
				collectOptimGrid.append(savedGridParams)
				
		return collectOptimGrid
		
	def combine_grid_settings(self,preProcess, featureSel, classifier):
		'''
		'''
		paramGrid = []
		pipeLine = []
		if len(preProcess) == 0:
			preProcess = [dict()]
		if len(featureSel) == 0:
			featureSel = [dict()]
			
		for preStep in preProcess:
			for featSel in featureSel:
				steps = merge_two_dicts(preStep,featSel)
				final = merge_two_dicts(steps,classifier)
				paramGrid.append(final)
				
		steps = ['Pre-Processing','FeatureSelection','Classifier']
		if len(paramGrid) > 0:
			for step in steps:
				if step in paramGrid[0]:
					functionList = paramGrid[0][step]
					pipeLine.append((step,functionList[0]))
		else:	
			tk.messagebox.showinfo('Error ..','There was in error building pipeline.')	
			return None, None
			
		return paramGrid, pipeLine

	def make_scorer(self):
		'''
		'''
		processes = self.recieverRectangleEntries['Scorer']
		scorer = {}
		if len(processes) == 0:
			scorer, refit  =  {'f1_score':'f1_micro'}, 'f1_score'
		
		for n,proc in enumerate(processes):
			tag = proc.split('_')[-1]
			if proc in self.functionSettings:
				settingDict = self.functionSettings[proc]
				print(settingDict)
				add = settingDict['average']
				scorer[tag] = '{}_{}'.format(tag,add)
			else:
				if tag == 'auc':
					scorer[tag] = 'roc_auc'
				else:
					scorer[tag] = tag
			
			if n == 0:
				refit = tag
		
		return scorer, refit	
			
		
		
		
	def perform_grid_search(self):
		'''
		'''
		estimatorList = self.recieverRectangleEntries['Classifier']
		## only one classifier possible
		if len(estimatorList) == 0:
				tk.messagebox.showinfo('No Classifier ..','Select a classifier.',parent=self.toplevel)
				return
		elif len(estimatorList) > 1:
			tk.messagebox.showinfo('Please note ..',
				'At the moment only one estimator can be evaluated at one.',
				parent=self.toplevel)
			
		progressBar = Progressbar('Grid search started ...')
		paramGrid = []
		pipelineSteps = []
		resultDF = pd.DataFrame() 
		predictionByBestEstimator = OrderedDict()
		
		
		# get data
		
		X,Y = self.data[self.features].values, np.ravel(self.data[self.targetColumn].values)
			
		## define k fold validation for nested cv
		
		n_split_nested  = self.nestedCV['n_splits']
		cvName = self.gridSearchCV['KFold Procedure']
		if cvName not in crossValidation:
			cvName = 'StratifiedShuffleSplit'
			
		if cvName in ['StratifiedShuffleSplit','ShuffleSplit']:
			testDataFraction = ts.askfloat(title = 'Define test size ..',
										   prompt = 'Please provide the fraction of data to use for testing [0,0.95]',
										   initialvalue = 0.2, minvalue = 0, maxvalue = 0.95,
										   parent = progressBar.toplevel)
			if testDataFraction is None:
				return
			else:
				cvNested = crossValidation[cvName]['function'](n_splits = n_split_nested,
															   test_size = testDataFraction) 
				cvNestedIndices = cvNested.split(X,Y)

		else:
			cvNestedIndices = crossValidation[cvName]['function'](n_splits = n_split_nested).split(X,Y) 
		
		## define k fold validation for gridsearch
		n_split_grid_search = self.gridSearchCV['n_splits']
		cvName = self.gridSearchCV['KFold Procedure']
		if cvName not in crossValidation:
			cvName = 'StratifiedKFold'
		cvInner = crossValidation[cvName]['function'](n_splits = n_split_grid_search) 
		
		progressBar.update_progressbar_and_label(2,'Extract parameter grid ..')
		classiDict = self.build_classifier_settings()
		## extract scorer
		
		scoring, refit_ = self.make_scorer()
		if classiDict is None:
			return 
			
		preprocessDict = self.build_optimize_step_settings('Pre-Processing')
		featureSelDict = self.build_optimize_step_settings('FeatureSelection')
		progressBar.update_progressbar_and_label(5,'Done ..')
		paramGrid, pipeLine = self.combine_grid_settings(preprocessDict,featureSelDict,classiDict)
		
		if paramGrid is None:
			progressBar.close()
			return 
			
		pipe = Pipeline(pipeLine)
		progressBar.update_progressbar_and_label(10,'Pipeline extracted ..')

		nSplit = 1
		for train_index, test_index in cvNestedIndices:
		
			progressBar.update_progressbar_and_label(10+80/n_split_nested*nSplit,'Nested cross validation {}/{}..'.format(nSplit,n_split_nested))
			grid = skModel.GridSearchCV(pipe, cv=cvInner , n_jobs=1, scoring = scoring,
										param_grid=paramGrid, return_train_score=True,
										refit = refit_)
			try:
				grid.fit(X[train_index], Y[train_index])
			except Exception as e:
				tk.messagebox.showinfo('Error ..','An error occured:\n{}'.format(e))
				progressBar.close()
			
			#print(grid.scorer_)
			## save results for inspection
			resultGrid = self.shorten_params(grid.cv_results_)
			cvResults = pd.DataFrame(resultGrid)
			cvNestedIndex = [nSplit] * len(cvResults.index)
			cvResults.loc[:,'#CV'] = cvNestedIndex
			cvResults = self.clean_up_results(cvResults)
			
			resultDF = resultDF.append(cvResults, ignore_index = True)	

			Y_test_pred = grid.predict(X[test_index])
			classReport = skMetrics.classification_report(Y[test_index],Y_test_pred)

			if hasattr(grid, 'predict_proba'): #'"decision_function"):
				probsTest = grid.predict_proba(X[test_index])
			else:
				probsTest = grid.decision_function(X[test_index]) 
			
			collectRocCurveParam = dict() 
			for n,class_ in enumerate(grid.classes_):
				if len(grid.classes_) == 2 and class_ in ['0','','-']:
					continue
				elif len(grid.classes_) == 2 and class_ in ['+','1']:
					probs = probsTest
				else: 
					probs = probsTest[:,n]
				fpr, tpr, _ = roc_curve(Y[test_index],probs, pos_label=class_)
				collectRocCurveParam['fpr_{}'.format(class_)] = fpr 
				collectRocCurveParam['tpr_{}'.format(class_)] = tpr 
				collectRocCurveParam['AUC_{}'.format(class_)] = round(auc(fpr,tpr),2)
			
			#print(grid.best_estimator_)
			#print(collectRocCurveParam)
			print(classReport)
			predictionByBestEstimator[nSplit] = {'Y_test_pred':Y_test_pred,
											  'best_params':grid.best_params_,
											  'ClassificationReport':classReport,
											  'roc_curve':collectRocCurveParam,
											  'classes':grid.classes_,
											  'estimator': grid.best_estimator_,
											  'scorerPrefix':refit_,
											  }
			
			nSplit += 1
			
		progressBar.update_progressbar_and_label(100,'Done. Initiate visualization ..')
		display_data.dataDisplayDialog(resultDF, waitWindow = False)
		results = dict() 
		results['nestedCVResults'] = resultDF
		results['rocCurveParams'] = predictionByBestEstimator
		results['pipeline'] = self.recieverRectangleEntries
		results['data'] = {'x':X,'y':Y}
		results['props'] = OrderedDict([('# of features',len(self.features)),
						    ('# of classes',len(grid.classes_)),('Scorer',refit_),
						    ('Features',get_elements_from_list_as_string(self.features)),
						    ('Target(Class) Column',self.targetColumn)])
		self.plotter.set_current_grid_search_results(results)
		
		self.classificationCollection.save_grid_search_results(results)
		self.plotter.initiate_chart(self.features,self.targetColumn,'grid_search_results',
			self.plotter.get_active_helper().colorMap)
		self.plotter.redraw()
		progressBar.close()
		return
		
		
	
	def shorten_params(self,gridResult): 
		'''
		Due to the way we build the pipeline, the params contains unuseful information
		that would confuse the user. To delete this we remove the "main" processes 
		'''
		newParamList = []
		toDelete = ['FeatureSelection','Classifier','Pre-Processing']
		params = gridResult['params'] 
		for param in params:
			for key in toDelete:
				if key in param:
					del param[key]
			paramString = str(param).replace('{','').replace("'",'').replace('}','')
			newParamList.append(paramString)
			
		gridResult['params']  = newParamList
		return gridResult
	
	def clean_up_results(self,df):
		'''
		'''
		for column in ['param_Classifier','param_FeatureSelection','param_Pre-Processing']:
			if column in df.columns:
				df[column] = df[column].astype(str).apply(lambda x: x.split('(')[0])#str.split('(', n = 1)
		return df
				
	def get_items_associations(self):
		'''
		Load icons and the associate with keys - same keys to get the function.
		'''
		self.recieverRectangleEntries = OrderedDict()
		self.reciverRectangleImages = OrderedDict()
		self.imagePositions = dict()
		self.dataIcon , self.naiveBayesIcon, self.svmIcon, self.crossValIcon, \
					self.treeEnsembleIcon, self.pcaIcon, \
					self.bestKFeatureIcon , self.uniformScalerIcon, \
					self.gaussianScalerIcon,self.robustScalerIcon, \
					self.minMaxScalerIcon, self.sgdIcon, \
					self.gridSearchIcon, self.finalEstimatorIcon,\
					self.nmfIcon, self.evaluationIcon, \
					self.identityFunctionIcon, self.resetIcon = images.get_workflow_builder_images()
		
		
		self.rocAucIcon, self.accuracyIcon, self.precisionIcon,\
		self.recallIcon, self.f1ScoreIcon =  images.get_scorer_images()
		
		self.itemsToDrag = OrderedDict([
				('Pre-Processing',
				{'UniformScaler':self.uniformScalerIcon,
				'GaussianScaler': self.gaussianScalerIcon,
				'RobustScaler':self.robustScalerIcon,
				'MinMaxScaler':self.minMaxScalerIcon,
				'IdentityScaler':self.identityFunctionIcon}),
				('FeatureSelection',
				{'PCA':self.pcaIcon,
				'NMF':self.nmfIcon,
				'BestKFeature':self.bestKFeatureIcon,
				'IdentityFunction':self.identityFunctionIcon,
				'RecursiveFeatureDetection':'',
				'F-statistic':'',
				}),
				
				('Classifier',
				{'RFC':self.treeEnsembleIcon,
				'SVM':self.svmIcon,
				'GNB':self.naiveBayesIcon,
				'SGD':self.sgdIcon
				}),
				
				('Scorer',
				{'roc_auc':self.rocAucIcon,
				'accuracy':self.accuracyIcon,
				'precision':self.precisionIcon,
				'recall':self.recallIcon,
				'f1':self.f1ScoreIcon})
				])
		
		self.checkDragAndDropTags = []
		self.receiverTags = dict()
		
		for key, canvasItemSettings in self.itemsToDrag.items():
			items = list(canvasItemSettings.keys())
			self.checkDragAndDropTags.extend(items)
			self.receiverTags[key] = items
			self.recieverRectangleEntries[key] = []
			self.reciverRectangleImages[key] = []


defaultValues = {'score_func':'f_classif',
			 'k':'10',
			 'n_components':'5'}


tooltipTextDict = {'KFold Procedure':'StratifiedKFold - This cross-validation object is a variation of KFold that returns '+
'stratified folds. The folds are made by preserving the percentage of samples for each class.\n'+
'StratifiedShuffleSplit - This cross-validation object is a merge of StratifiedKFold and'+
' ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving'+
' the percentage of samples for each class.\n'+
'Time Series Split - Provides train/test indices to split time series data samples that are'+
' observed at fixed time intervals, in train/test sets. In each split, test indices must be '+
'higher than before, and thus shuffling in cross validator is inappropriate.',
'n_splis':'Number of splits.',
'score_func':'Scoring function to evaluate importance of features. Note that for chi2 statistic all values have to be bigger than 0'
}


averageOptions = [('average',['binary','micro','macro','weighted','samples'])]

parameterToOptimize = {'SVM': [('C',['1,10,100','1','0.1,1,10,100,1000']),
								('kernel',['rbf','linear','poly','rbf,linear,poly']),('gamma',['auto']),('coef0',['0'])],
					  'RFC':[('n_estimators',['10,20,50','20,100,500']),
					  ('max_features',['auto','log2,sqrt','3,5,log2']),
					  ('max_depth',['None','2','3,4,5,9']),
					  ('min_samples_split',['2','7,4,2']),
					  ('min_samples_leaf',['1','2','5,3,1'])],
					  'GNB':[],
					  'SGD':[('loss',sgdLossFuncs),
					  ('learning_rate',['constant','optimal','invscaling']),
					  ('alpha',['0.0001'])],
					  'PCA':[('n_components',['4,3,2'])],
					  'NMF':[('n_components',['4,3,2'])],
					  'BestKFeature':[('score_func',['f_classif','chi2']),('k',['9,5,2','5,3','3','all,4,3,1'])],
					  'nestedCV':[('KFold Procedure',['StratifiedShuffleSplit','StratifiedKFold','Time Series Split']),
					  ('n_splits',['3','5','10'])],
					  'gridSearchCV':[('KFold Procedure',list(crossValidation.keys())),
					  ('n_splits',['3','5','10'])],
					  'precision':averageOptions,
					  'recall':averageOptions,
					  'f1':averageOptions,
					  }
					

		
class defineGridSearchDialog(object):
	
	def __init__(self, procedure = 'SVM', features = [], gridClass = None):
	
		if procedure not in parameterToOptimize:
			return
		self.procedure = procedure
		self.features = features
		self.numbFeatures = len(self.features)
		self.gridClass = gridClass 
		## dicts to save stuff
		self.paramWidgets = dict()
		self.settingDict = dict()
		self.collectParamGrid = dict()
		
		self.build_popup()
		self.build_widgets_on_toplevel()
		
		self.toplevel.wait_window() 
	
		
	def close(self):
		'''
		Closes the toplevel.
		'''
		
		self.toplevel.destroy()	
	
	def build_popup(self):
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Define Parameter for Grid Search') 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w = 410
		h= 350
		self.toplevel = popup
		self.center_popup((w,h))		

	def build_widgets_on_toplevel(self):
		'''
		'''
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY)
		self.cont.pack(expand =True, fill = tk.BOTH) 
		self.cont.grid_columnconfigure(1,weight=1)
		
		parameters = parameterToOptimize[self.procedure]
		if self.procedure == 'gridSearchCV':
			infoText = 'Choose number of cross validations performed for parameter optimization by grid search'
		else:
			infoText = ('Define parameter for GridSearch with Cross Validation\n'+
					'To define a grid search: \n'+
					'Provide grid parameters: Value1,Value2,Value3  and check the checkbutton.\n'+
					'If you provide a single parameter: Value1. This parameter will be used as'+
					' a constant parameter for the selected step in your pipeline.')
		labelInfo = tk.Label(self.cont, text = infoText, bg = MAC_GREY, wraplength = 400, justify = tk.LEFT)
		labelInfo.grid(padx=3,pady=5, columnspan=3)
		
		for n,param in enumerate(parameters):
			varCb = tk.BooleanVar()
			cb = ttk.Checkbutton(self.cont, text = param[0], variable = varCb)
			combo = ttk.Combobox(self.cont, values = param[1])
			if self.procedure in abbrevDictRev:
				description = widgetCollection[abbrevDictRev[self.procedure]][param[0]][-1]
				defaultValue = widgetCollection[abbrevDictRev[self.procedure]][param[0]][0]
				CreateToolTip(cb,title_=param[0],text= description)
				combo.set(defaultValue)
				
			elif self.procedure == 'nestedCV':
				value = self.gridClass.nestedCV[param[0]]
				combo.set(value)
			elif self.procedure == 'gridSearchCV':
				value = self.gridClass.gridSearchCV[param[0]]
				combo.set(value)
			elif self.procedure in ['recall','precision','f1']:
				combo.set('micro')
				
			elif param[0] in defaultValues:
				combo.set(defaultValues[param[0]])
			
			if param[0] in tooltipTextDict:
				CreateToolTip(cb,text= tooltipTextDict[param[0]])
					
			self.paramWidgets[param[0]] = {'CbVar':varCb,'Entry':combo} 
			cb.grid(row=n+1, column = 0, sticky=tk.W, pady=3,padx=3)
			combo.grid(row=n+1, column = 1, sticky = tk.EW, pady=3,padx=3)
			
		applyButton = ttk.Button(self.cont, text = 'Done', command = self.get_params)
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		
		applyButton.grid(row = n+2, column = 0)
		closeButton.grid(row = n+2, column = 1, sticky = tk.E)
			
	
	def get_params(self):
		'''
		'''
		for key, paramDict in self.paramWidgets.items():
			if paramDict['CbVar'].get():
				gridList = paramDict['Entry'].get().split(',')
				if len(gridList) == 1:
					tk.messagebox.showinfo('Error ..',
										   'Could not extract grid values for {}'.format(key), 
										   parent = self.toplevel)
				else:
			
					self.collectParamGrid[key] = gridList
					
			else:
				entryString = paramDict['Entry'].get()
				if entryString != '':
					self.settingDict[key] = entryString
					
		cleanUp = self.evaluate_and_transform_input()

		if cleanUp:	
			self.close() 
		
	def evaluate_and_transform_input(self):
		'''
		Evaluate input for grid search. 
		'''		
		updatedDictGrid = dict()
		updatedDictSettings = dict()
		oldDictList = [self.collectParamGrid,self.settingDict]
		newDictList = [updatedDictGrid,updatedDictSettings]
		if self.procedure == 'GNB':
			return True
		elif self.procedure in ['nestedCV','gridSearchCV']:
			
			for oldDict, updateDict in zip(oldDictList,newDictList):
				for key,values in oldDict.items():
					if isinstance(values,list):
						if key == 'n_splits':
							updateDict[key] = int(float(values[0]))
						else:
							updateDict[key]  = values[0]
					else:
						if key == 'n_splits':
							updateDict[key]  = int(float(values))
						else:
							updateDict[key]  = values
				
			self.collectParamGrid,self.settingDict = updatedDictGrid,updatedDictSettings
			return True		
							
		elif self.procedure == 'SVM':
			
			for oldDict, updateDict in zip(oldDictList,newDictList):
				for key,values in oldDict.items():
					if isinstance(values,list):
						if key == 'kernel':
							updateDict[key] = [x for x in values if x in ['linear', 'poly', 'rbf', 'sigmoid']]
						elif key == 'degree':
							updateDict[key] = [int(float(x)) for x in values]
						else:
							updateDict[key]  = [float(x) for x in values]
					else:
						if key == 'kernel':
							updateDict[key]  = values
						elif key == 'degree':
							updateDict[key] = int(float(values))
						elif key == 'gamma' and values == 'auto':
							updateDict[key] = values
						else:
							updateDict[key]  = float(values)
				
			self.collectParamGrid,self.settingDict = updatedDictGrid,updatedDictSettings
			return True			
					
		elif self.procedure in ['PCA','BestKFeature','NMF']:
			
			possibleStrings = ['all',"'all'"]
			
			for oldDict, updateDict in zip(oldDictList,newDictList):
				
				for key, values in oldDict.items():
					if key == 'score_func' and self.procedure == 'BestKFeature':
						updateDict[key] = scorerOptions[values]
						continue						
						
					if isinstance(values,list):
					
						intValues = [int(float(x)) for x in values if \
						int(float(x)) <= self.numbFeatures and x not in possibleStrings]
						
						if len(intValues) != 0:
							if len(intValues) == len(values)+1:
								intValues.append(possibleStrings)
							updateDict[key] = intValues
							
						else:
							tk.messagebox.showinfo('Error ..',
								'Invalid input in feature selection. Probably you entered a value higher than the number of features.',
								parent = self.toplevel)
							return False
					else:
						if values in possibleStrings:
							intValue = possibleStrings[0]
						else:
							intValue = int(float(values))
						if isinstance(intValue,str) == False and intValue > self.numbFeatures:
							tk.messagebox.showinfo('Error ..',
								'Number of reduced features is higher than original feature number.',
								parent = self.toplevel)
							return False
						else:
							updateDict[key] = intValue
							
			self.collectParamGrid,self.settingDict = updatedDictGrid,updatedDictSettings
			return True
			
		elif self.procedure == 'RFC':
			
			for oldDict, updateDict in zip(oldDictList,newDictList):
				
				for key, values in oldDict.items():
				
					if key in ['n_estimators','max_depth','min_sample_split','min_samples_leaf']:
						if isinstance(values,list):
							intValues = [int(float(x)) for x in values]
							updateDict[key] = intValues
						else:
							intValue = int(float(values))
							updateDict[key] = intValue
							
					elif key == 'max_features':
						stringOptions = ['auto','sqrt','log2']
						if isinstance(values,list):							
							intValues = [int(float(x)) for x in values if x not in stringOptions]
							stringVals = [x for x in values if x in stringOptions]
							resultValues = intValues + stringVals
							updateDict[key] = resultValues
						else:
							if values in stringOptions:
								updateDict[key] = values
							else:
								updateDict[key] = int(float(values))
						
			self.collectParamGrid,self.settingDict = newDictList
			return True	
									
		elif self.procedure == 'SGD':
			
			for oldDict, updateDict in zip(oldDictList,newDictList):
				
				for key, values in oldDict.items():
				
					if key in ['loss','learning_rate']:
						if isinstance(values,list):
							updateDict[key] = values
						else:
							updateDict[key] = values
							
					else:
						if isinstance(values,list):							
							floatValues = [float(x) for x in values if x not in stringOptions]
							updateDict[key] = floatValues
						else:
							updateDict[key] = float(values)
						
			self.collectParamGrid,self.settingDict = newDictList
			return True					
			
		elif self.procedure in ['recall','precision','f1']:
			for oldDict, updateDict in zip(oldDictList,newDictList):
				for key, values in oldDict.items():
						if values not in averageOptions[-1][-1]:
							tk.messagebox.showinfo('Error ..',
								'Cannot interpret input', 
								parent = self.toplevel)
							return False
						
						else:
							updateDict[key] = values
			self.collectParamGrid,self.settingDict = newDictList
			return True										
		
	def center_popup(self,size):

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))    
         	
 
 

 		
  		
  		
  		
  		

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	



