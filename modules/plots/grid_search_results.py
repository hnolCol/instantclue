import seaborn
import numpy as np
from scipy import interp

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
	
	def fill_axes(self):
		'''
		'''
		collectMacroRoc = []
		for nSplit, predData in self.rocCurves.items():	
			rocData = predData['roc_curve']
			param = predData['best_params']
			for class_ in self.rocCurves[1]['classes']:
				tpr = rocData['tpr_'+class_]
				fpr = rocData['fpr_'+class_]
				auc = rocData['AUC_'+class_]
				
			
			all_fpr = np.unique(np.concatenate([rocData['fpr_'+class_] for class_ in self.rocCurves[1]['classes']]))
			print(all_fpr)
			mean_tpr = np.zeros_like(all_fpr)
			for class_ in self.rocCurves[1]['classes']:
				mean_tpr += interp(all_fpr, rocData['fpr_'+class_], rocData['tpr_'+class_])
			
			mean_tpr /= len(self.rocCurves[1]['classes'])
			
				#mean_tpr += interp(all_fpr, fpr[i], tpr[i])
   				 
			
			
			self.axisDict[0].plot(all_fpr,mean_tpr,label=class_+str(param))
				
		# First aggregate all false positive rates
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points



# Finally average it and compute AUC



		leg = self.axisDict[0].legend()
		leg.draggable(state=True, use_blit=True)
		leg._legend_box.align = 'left'
		
		self.axisDict[1].bar(range(self.testScoreMeans.size),self.testScoreMeans.values)
		
		
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
  	
  		
  		
  		
  		
 