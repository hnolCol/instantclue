




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
		self.pcaProjectionAnnotations = OrderedDict()

		self.scatterPlots = OrderedDict() 
	
		self.dfClass = dfClass
		self.plotter = plotter
		self.dataID = plotter.get_dataID_used_for_last_chart()
		self.get_scatter_props()
		self.get_data()
		self.replot()

	def disconnect_bindings(self):
		'''
		'''
		self.plotter.figure.canvas.mpl_disconnect(self.onHover)
		
	
	def replot(self):
		'''
		Replot the complete plot compilation.
		'''
		self.add_axis()
		self.fill_axis()
		self.style_axis()

	def get_scatter_props(self):
		'''
		Save scatter plot settings.
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

	def fill_axis(self, specificAxis = None, id = None):
		'''
		'''
		if specificAxis is None or id == 0:

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
			
			
		if specificAxis is None or id == 2:

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
				
									
		if (specificAxis is None or id == 1) and 'Components' in self.plotter.dimRedResults['data']:
				
				
				ax = self.axisDict[1] if specificAxis is None else specificAxis	
				
				components = self.plotter.dimRedResults['data']['Components']
				scatStyle = self.scatStyle.copy()
				if specificAxis is not None:
					scatStyle['color'] = self.axisDict[1].collections[0].get_facecolor()
				ax.scatter(components.iloc[0,:], components.iloc[1,:],
									**scatStyle)
				data = components.T
				data['experiments'] = data.index
				
				
				
				if specificAxis is None:					
					self.pcaProjectionAnnotations[0] = annotateScatterPoints(Plotter = self.plotter,
										ax = ax, data = data,
										numericColumns = data.columns.values.tolist()[:2],
										labelColumns = ['experiments'], madeAnnotations = OrderedDict(),
										selectionLabels = OrderedDict()) 
					self.pcaProjectionAnnotations[0].annotate_all_row_in_data()
								
		if (specificAxis is None or id == 3) and 'Components' in self.plotter.dimRedResults['data']:
				
				
				ax = self.axisDict[3] if specificAxis is None else specificAxis	
				
				components = self.plotter.dimRedResults['data']['Components']
				scatStyle = self.scatStyle.copy()
				if specificAxis is not None:
					scatStyle['color'] = self.axisDict[1].collections[0].get_facecolor()				
				ax.scatter(components.iloc[1,:], components.iloc[2,:],
									**scatStyle)
				data = components.T
				data['experiments'] = data.index
				
				if specificAxis is None:					
					self.pcaProjectionAnnotations[1] = annotateScatterPoints(Plotter = self.plotter,
										ax = ax, data = data,
										numericColumns = data.columns.values.tolist()[1:3],
										labelColumns = ['experiments'], madeAnnotations = OrderedDict(),
										selectionLabels = OrderedDict()) 
					self.pcaProjectionAnnotations[1].annotate_all_row_in_data()# 				

	
	def get_data(self):
		'''
		Retrieve data.
		'''
		self.data = self.plotter.dimRedResults['data']['Drivers']
		self.numericColumns = self.data.columns.values.tolist()
		
		
	def export_selection(self, specificAxis, id):	
		'''
		Export chart to main figure
		'''
		if id in [0,2]:
			self.scatterPlots[id].export_selection(specificAxis)
		else:
			self.fill_axis(specificAxis,id)
		
			if id == 1:
				if 0 in self.pcaProjectionAnnotations:
					for key,props in self.pcaProjectionAnnotations[0].selectionLabels.items():
						specificAxis.annotate(ha='left', arrowprops=arrow_args,**props)	
			if id == 3:
				if 1 in self.pcaProjectionAnnotations:
					for key,props in self.pcaProjectionAnnotations[1].selectionLabels.items():
						specificAxis.annotate(ha='left', arrowprops=arrow_args,**props)			
			self.update_color_in_projection(ax = specificAxis, export = True)
		
		self.style_axis(specificAxis,id)
		
	def style_axis(self, specificAxis = None, id = None):
		'''
		Style axis.
		'''
		if specificAxis is None:
			for n, ax in self.axisDict.items():
				xLabel, yLabel = self.get_axisLabels(n)
				axisStyler(ax,ylabel = yLabel, xlabel = xLabel)
		else:
			xLabel, yLabel = self.get_axisLabels(id)
			axisStyler(specificAxis,ylabel = yLabel, xlabel = xLabel)
			
	def get_axisLabels(self, n ):
		'''
		'''
		if 'ExplainedVariance' in self.plotter.dimRedResults['data']:
					data = self.plotter.dimRedResults['data']['ExplainedVariance'].loc[:,0].tolist()
					
		elif 'noiseVariance' in self.plotter.dimRedResults['data']:
					data = self.plotter.dimRedResults['data']['noiseVariance'].tolist()
		else:
			if n < 2:
				return 'Component 1', 'Component 2'
			else:
				return 'Component 2', 'Component 3'
				
		xLabel = 'Component 1 ({}%)'.format(round(data[0] * 100,0)) if n < 2 \
			else 'Component 2 ({}%)'.format(round(data[1] * 100,0))
		yLabel = 'Component 2 ({}%)'.format(round(data[1] * 100,0)) if n < 2 \
			else 'Component 3 ({}%)'.format(round(data[2] * 100,0))
		
		return xLabel, yLabel
					
	
	def hide_show_feature_names(self):
	
		'''
		Handles feature names in projection plot of 
		a dimensional reduction procedure
		'''	
		if len(self.pcaProjectionAnnotations) != 0:
		
			for id,pcaProjAnnot in self.pcaProjectionAnnotations.items():
				if len(pcaProjAnnot.selectionLabels) != 0:
					pcaProjAnnot.remove_all_annotations()
				else:
					pcaProjAnnot.annotate_all_row_in_data()
			
			self.plotter.redraw(backgroundUpdate = False)		
	
	def update_color_in_projection(self,resultDict = None, ax = None, export = False):
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
 									   		 'lw':0.5}})
			if export == False:
				self.axisDict[3].collections[0].set_facecolor(colors)	
