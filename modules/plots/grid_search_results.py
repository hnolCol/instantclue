import seaborn as sns
import numpy as np

from scipy import interp
from sklearn.metrics import roc_curve, auc

class gridSearchVisualization(object):
	'''
	'''
	def __init__(self,plotter,data):
	
		self.plotter = plotter
		self.figure = plotter.figure
		
		self.nCVResults = data['nestedCVResults']
		self.rocCurves = data['rocCurveParams']
		self.define_variables()	
		self.prepare_data()
		
		self.create_axes()
		self.fill_axes()
		
	def define_variables(self):
		'''
		Define variables used.
		'''
		self.axisDict = dict() 	
		
		
	def create_axes(self):
		'''
		Adds axes to figure.
		'''
		self.figure.subplots_adjust(right=.95,left=0.15,hspace=0.2,wspace=.2)
		self.axisDict[0] = self.figure.add_subplot(331)
		self.axisDict[1] = self.figure.add_subplot(332)
		self.axisDict[2] = self.figure.add_subplot(333)
		self.axisDict[3] = self.figure.add_subplot(334)
	def fill_axes(self):
		'''
		'''
		collectMacroRoc = []
		for nSplit, predData in self.rocCurves.items():	
			rocData = predData['roc_curve']
			param = predData['best_params']
			for class_ in self.rocCurves[1]['classes']:
				if 'tpr_'+class_ not in rocData:
					continue
				tpr = rocData['tpr_'+class_]
				fpr = rocData['fpr_'+class_]
				aucVal = rocData['AUC_'+class_]
				
			if len(self.rocCurves[1]['classes']) > 2:
				all_fpr = np.unique(np.concatenate([rocData['fpr_'+class_] for class_ in self.rocCurves[1]['classes']]))
				print(all_fpr)
				mean_tpr = np.zeros_like(all_fpr)
				for class_ in self.rocCurves[1]['classes']:
					mean_tpr += interp(all_fpr, rocData['fpr_'+class_], rocData['tpr_'+class_])
			
				mean_tpr /= len(self.rocCurves[1]['classes'])
				AUC = auc(all_fpr, mean_tpr)	
			else:
				all_fpr = fpr
				mean_tpr = tpr
				AUC = aucVal
   			 			
			self.axisDict[0].plot(all_fpr,mean_tpr,
							lw = 0.5,
							label='Split: {} {}\n (AUC: {})'.format(nSplit,str(param),AUC))
				
		# First aggregate all false positive rates
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points



# Finally average it and compute AUC



		leg = self.axisDict[0].legend()
		leg.draggable(state=True, use_blit=True)
		leg._legend_box.align = 'left'
		
		self.axisDict[1].bar(range(self.testScoreMeans.size),self.testScoreMeans.values)
		
		
		sns.pointplot(y='mean_test_score',hue='#CV',x='params',ax=self.axisDict[2], data = self.nCVResults)
		
		sns.pointplot(y='mean_fit_time',hue='#CV',x='params',ax=self.axisDict[3], data = self.nCVResults)
		
	def prepare_data(self):
  		'''
  		'''
  		## get parameter columns
  		paramColumns = [column for column in self.nCVResults.columns if 'param_' in column]
  		print(paramColumns) 
  		## group data 
  		groupedResults = self.nCVResults.groupby(paramColumns) 
  		# get combinations in nested cv
  		groupNames = groupedResults.groups.keys() 
  		print(groupNames)
  		self.testScoreMeans = groupedResults['mean_test_score'].mean()
  		print(self.testScoreMeans)
  	
  		
  		
  		
  		
 