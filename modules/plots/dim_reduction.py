




from modules.plots.scatter_annotations import annotateScatterPoints
from modules.plots.axis_styler import axisStyler
from collections import OrderedDict

import numpy as np
from modules.utils import *
from collections import OrderedDict
from modules.plots.scatter_plotter import scatterPlot

class dimensionalReductionPlot(object):
	
	def __init__(self,dfClass, plotter, colorMap):
		'''
		'''
		self.axisDict = dict()
		self.annotationClass = [None,None] 
		self.colorMap = colorMap
		self.categoricalColorDefinedByUser = OrderedDict()
		self.scatterPlots = OrderedDict() 
	
		self.dfClass = dfClass
		self.plotter = plotter
		self.dataID = plotter.get_dataID_used_for_last_chart()
		self.get_scatter_props()
		self.get_data()
		self.add_axis()
		self.fill_axis()

	def disconnect_bindings(self):
		'''
		'''
		self.plotter.figure.canvas.mpl_disconnect(self.onHover)
		
	
	def replot(self):
		'''
		'''

	def get_scatter_props(self):
		'''
		'''
		self.scatStyle = dict()
		self.scatStyle['s'], self.scatStyle['alpha'],self.scatStyle['color'] = \
		self.plotter.get_scatter_point_properties()			
		self.scatStyle['picker'] = True
		self.scatStyle['label'] = None
		self.scatStyle['linewidth'] = 0.3
		self.scatStyle['edgecolor'] ='black'							
	
	
	def add_axis(self):
		'''
		'''
		for n in range(4):
			self.axisDict[n] = self.plotter.figure.add_subplot(2,2,n+1)
		self.plotter.figure.subplots_adjust(wspace=0.3,hspace=0.3,right=0.9)

	def fill_axis(self, specificAxis = None):
		'''
		'''
		if specificAxis is None:

			ax = self.axisDict[0] if specificAxis is None else specificAxis
			self.scatterPlots[0] = scatterPlot(
									self.data,
									['Comp_1','Comp_2'],
									self.plotter,
									self.colorMap,
									self.dfClass,									
									ax,
									self.dataID,								
									self.scatStyle)
			
			
		if specificAxis is None:

			ax = self.axisDict[2] if specificAxis is None else specificAxis		
			
			self.scatterPlots[2] = scatterPlot(
									self.data,
									['Comp_2','Comp_3'],
									self.plotter,
									self.colorMap,
									self.dfClass,									
									ax,
									self.dataID,								
									self.scatStyle,
									showLegend = False)
											
		if (specificAxis  is None or specificAxis  == 1) and 'Components' in self.plotter.dimRedResults['data']:
				
				ax = self.axisDict[1] if specificAxis is None else specificAxis	
				components = self.plotter.dimRedResults['data']['Components']
				
				ax.scatter(components.iloc[0,:], components.iloc[1,:],
									**self.scatStyle)
				data = components.T
				data['experiments'] = data.index
				
				if specificAxis is None:					
					self.pcaProjectionAnnotations = annotateScatterPoints(Plotter = self.plotter,
										ax = ax, data = data,
										numericColumns = data.columns.values.tolist()[:2],
										labelColumns = ['experiments'], madeAnnotations = OrderedDict(),
										selectionLabels = OrderedDict()) 
					self.pcaProjectionAnnotations.annotate_all_row_in_data()
								
		if specificAxis is None or specificAxis == 3:
				
				if 'ExplainedVariance' in self.plotter.dimRedResults['data']:
					data = self.plotter.dimRedResults['data']['ExplainedVariance'].loc[:,0]
					xTicks = [x+1 for x in data.index.tolist()]
				elif 'klDivergence' in self.plotter.dimRedResults['data']:
					data = self.plotter.dimRedResults['data']['klDivergence']
					xTicks = 1
				elif 'noiseVariance' in self.plotter.dimRedResults['data']:
					data = self.plotter.dimRedResults['data']['noiseVariance']
					xTicks = list(range(1,data.shape[0]+1))
				else:
					data = self.plotter.dimRedResults['data']['ReconstructionError']
					xTicks = 1 
				ax = self.axisDict[3] if specificAxis is None else specificAxis
				ax.bar(xTicks, data)
				ax.set_xticks(xTicks)
				if specificAxis is None:
					ax.callbacks.connect('xlim_changed', \
						lambda event, axis = ax: axis.set_xlim(0.5,len(xTicks)+0.5))
	
	def get_data(self):
		'''
		'''
		self.data = self.plotter.dimRedResults['data']['Drivers']
		self.numericColumns = self.data.columns.values.tolist()
		
	# def change_color_by_categorical_columns(self,columnNames, updateColor=False):
# 		'''
# 		'''
# 		
# 		self.plotter.nonCategoricalPlotter.change_color_by_categorical_columns(categoricalColumn = columnNames,
# 						updateColor=False, 
# 						annotationClass = self.annotationClass[0],
# 						specificAxis = self.axisDict[0], 
# 						numericColumns = ['Comp_1','Comp_2'])
# 		
# 		self.plotter.nonCategoricalPlotter.change_color_by_categorical_columns(categoricalColumn = columnNames,
# 						updateColor=False, 
# 						annotationClass = self.annotationClass[1],
# 						specificAxis = self.axisDict[2], 
# 						numericColumns = ['Comp_2','Comp_3'])					
# 
# 		self.data = self.plotter.nonCategoricalPlotter
# 		
		
		
	def export_selection(self, specificAxis, id):	
		'''
		Export chart to main figure
		'''
		if id in [0,2]:
			self.scatterPlots[id].export_selection(specificAxis)
		
		
		
		
	def hide_show_feature_names(self):
	
		'''
		Handles feature names in projection plot of 
		a dimensional reduction procedure
		'''	
		if self.pcaProjectionAnnotations is not None:
			if len(self.pcaProjectionAnnotations.selectionLabels) != 0:
				self.pcaProjectionAnnotations.remove_all_annotations()
			else:
				self.pcaProjectionAnnotations.annotate_all_row_in_data()
			
			self.plotter.redraw(backgroundUpdate = False)		
	
	def update_color_in_projection(self,resultDict = None, ax = None):
		'''
		Projection plot in dimensional reduction show feature names and can
		be colored occording to groups. (For example if feature == name of experiment
		where expresison values of several proteins/genes have been measured (e.g Tutorial Data 2)
		'''
		if resultDict is None:
			resultDict =  self.plotter.dimRedResults
		
		if 'color' in resultDict:
			colorDict = resultDict['color']
			groupDict = resultDict['group']	 
			featureOrder = self.plotter.dimRedResults['data']['Components'].T.index
			colors = [colorDict[feature] for feature in featureOrder]
			groups = [groupDict[feature] for feature in featureOrder]
			uniqueValuesDict = OrderedDict((comb,True) for comb in list(zip(colors,groups)))
			colorsUnique,groupsUnique = zip(*list(uniqueValuesDict.keys()))
			if ax is None:
				ax = self.axisDict[1]
			ax.collections[0].set_facecolor(colors)
			axisStyler(ax,forceLegend = True ,kwsLegend = {'addPatches':True,
 									   		'legendItems' : list(groupsUnique),
 									   		'colorMap' : list(colorsUnique),
 									   		'leg_title':'Grouping',
 									   		 'patchKws':{'alpha':self.scatStyle['alpha'],
 									   		 'lw':0.5}}) 	# elif plotType == 'PCA':
# 			if onlySelectedAxis is not None:
# 				print(onlySelectedAxis)
# 				if onlySelectedAxis < 2:
# 					axisStyler(ax, xlabel='Component 1', ylabel = 'Component 2')
# 				elif onlySelectedAxis == 2:
# 					axisStyler(ax, xlabel='Component 2', ylabel = 'Component 3')
# 				else:
# 					if self.plotter.dimRedResults['method'] != 'Non-Negative Matrix Factorization':
# 						axisStyler(ax, ylabel='Explained Variance Ratio', 
# 										 xlabel = 'Components')
# 					else:
# 						axisStyler(ax, ylabel='Reconstruction Error', 
# 										 xlabel = 'Analysis ID', newXLim = (-1,3))
# 			else:
# 				for n in [0,1]:
# 					axisStyler(self.axisDict[n], xlabel='Component 1', ylabel = 'Component 2',
# 									nTicksOnYAxis = 4, nTicksOnXAxis = 4)
# 				
# 				axisStyler(self.axisDict[2], xlabel='Component 2', ylabel = 'Component 3',
# 									nTicksOnYAxis = 4, nTicksOnXAxis = 4)				
# 				if self.plotter.dimRedResults['method'] != 'Non-Negative Matrix Factorization':
# 					axisStyler(self.axisDict[3], ylabel='Explained Variance Ratio', 
# 										 xlabel = 'Components',nTicksOnYAxis = 4)
# 				elif self.plotter.dimRedResults['method'] != 'Factor Analysis':
# 					axisStyler(self.axisDict[3], ylabel='Estimated noise variance', 
# 										 xlabel = 'Feature',nTicksOnYAxis = 4)
# 				
# 				else:
# 					axisStyler(self.axisDict[3], ylabel='Reconstruction Error', 
# 										 xlabel = 'Analysis ID',newXLim = (-1,3))