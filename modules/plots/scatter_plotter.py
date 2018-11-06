

import pandas as pd
import numpy as np
from modules.utils import * 
from modules.plots.scatter_annotations import annotateScatterPoints
from modules import stats
class scatterPlot(object):


	def __init__(self, data, numericColumns, plotter, colorMap, 
						dfClass, ax, dataID, scatterKwargs,
						showLegend = True, ignoreYlimChange = False):
		'''
		'''
		self.ax = ax
		self.plotter = plotter
		self.dfClass = dfClass 
		self.dataID = dataID
		self.numericColumns = numericColumns
		self.scatterKwargs = scatterKwargs
		self.colorMap = colorMap
		self.numericColumns = numericColumns
		self.data = data
		self.showLegend = showLegend
		self.nanScatterColor = scatterKwargs['color']
		self.ignoreYlimChange = ignoreYlimChange
		
		self.define_variables()
		self.get_size_interval()
		self.replot(ax=self.ax,**self.scatterKwargs)
		self.extract_axis_props()
		self.add_hover_point()
		self.add_bindings()
		
	def define_variables(self):
		'''
		Define needed variables.
		'''
		self.categoricalColorDefinedByUser = dict()
		self.sizeStatsAndColorChanges = OrderedDict()
		self.annotationClass = None
		self.idxData = None
		self.lowessData = None
		self.toolTipsActive = False
		self.argMax = -1
		
	def add_bindings(self):
		'''
		'''		
		self.onHover = self.plotter.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)
	
	def set_invisible(self,event = None, update = True):
		'''
		'''
		if self.hoverScatter[0].get_visible():
			self.hoverScatter[0].set_visible(False)
			if self.toolTipsActive:
				self.tooltip.set_visible(False)				
			if update:
				self.update_axis()
	
	def add_hover_point(self):
		'''
		'''
		styleHoverScat['markersize'] =  np.sqrt(self.scatterKwargs['s'])
		self.hoverScatter = self.ax.plot([],[],**styleHoverScat)
		self.ax.callbacks.connect('ylim_changed', \
					lambda event:self.update_background(redraw = True,updateProps=True))

	def indicate_hover_point(self):
		''
		if self.idxData is not None:
			self.plotter.annotate_in_scatter_points(self,self.idxData)

	def set_hover_data(self, dataIdx, size = None):
		'''
		'''
		boolInd = self.data.index == dataIdx
		if np.sum(boolInd) == 0:
			return
		if 'size' in self.data.columns:
			size = np.sqrt(self.data.loc[boolInd,'size'])
		dataCoords = (self.data.loc[boolInd,self.numericColumns[0]],
					  self.data.loc[boolInd,self.numericColumns[1]])
					  
		self.update_hover_data(dataCoords,size)
		

	def on_hover(self,event, size = None):
		
		if event.inaxes is None or event.button == 1:
			self.set_invisible()
			return
		if event.inaxes != self.ax:
			return
		if hasattr(self,'background') == False:
			self.update_background(redraw=False)
		pointCoords, argMax = self.check_for_close_point(event)
		if pointCoords is None:
				self.set_invisible()
				if self.plotter.nonCategoricalPlotter is not None:
					self.plotter.set_hover_points_invisible(self)
				return
		else:
			if argMax == self.argMax:
				return
			if 'size' in self.data.columns:
				size = np.sqrt(self.data['size'].values[argMax])
			self.update_hover_data(pointCoords,size)
			self.argMax = argMax

		
	def update_hover_data(self,pointCoords,size=None):
		'''
		'''		
		self.hoverScatter[0].set_visible(True)
		self.hoverScatter[0].set_data(pointCoords)
		if size is not None:
			self.hoverScatter[0].set_markersize(size)
		self.update_axis()
	
	def update_background(self, redraw=True,updateProps=True):
		'''
		'''
		if self.ignoreYlimChange:
			return
		if self.plotter.updatingAxis:
			return
		if updateProps:
			self.extract_axis_props()
		if redraw:
			if hasattr(self,'background'):
				self.set_invisible(update=False)
			self.plotter.redraw(backgroundUpdate = False)
		self.background =  self.plotter.figure.canvas.copy_from_bbox(self.ax.bbox)		
		
	def extract_axis_props(self):
		''''''
		#self.axProps = dict()
		
		self.axProps = dict(xDiff = self.ax.get_xlim()[1] - self.ax.get_xlim()[0],
							yDiff = self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
							xlim = self.ax.get_xlim(),
							ylim = self.ax.get_ylim())
			
	def check_for_close_point(self, event):
		'''
		'''
		self.idxData = None
		colIdx1 = self.numericColumns[0]
		colIdx2 = self.numericColumns[1]
		
		xValue = np.asarray(event.xdata)
		yValue = np.asarray(event.ydata)	
		boolXData = np.isclose(self.data[colIdx1].values,xValue,atol = 0.01*self.axProps['xDiff'])
		boolYData = np.isclose(self.data[colIdx2].values,yValue,atol = 0.01*self.axProps['yDiff'])
		add = np.sum([boolXData,boolYData], axis=0)
		argMax = np.argmax(add == 2)
		if argMax == 0 and add[0] != 2:	
			self.argMax = -1
			return None, None
		else:
			if self.toolTipsActive:
				textData = self.data[self.annotationColumns].values[argMax]
				text = get_elements_from_list_as_string(textData).replace(', ','\n')
				self.update_position(event,text)
				
			self.idxData = self.data.index[argMax]
			self.indicate_hover_point()
			return (self.data[colIdx1].values[argMax],self.data[colIdx2].values[argMax]), argMax

	def update_axis(self):
		'''
		Update artists using blit.
		'''
		if hasattr(self,'background') == False:
			self.background =  self.plotter.figure.canvas.copy_from_bbox(self.ax.bbox)
				
		self.plotter.figure.canvas.restore_region(self.background)
		self.ax.draw_artist(self.hoverScatter[0])	
		if self.toolTipsActive:
			self.ax.draw_artist(self.tooltip)
		self.plotter.figure.canvas.blit(self.ax.bbox)
		
	def add_color_and_size_changes_to_dict(self,changeDescription,keywords):
		'''
		Adds information on how to modify the chart further
		'''
		self.sizeStatsAndColorChanges[changeDescription] = keywords
			
	def bind_label_event(self,labelColumnList):
		'''
		Add the ability to click on points for labeling.
		'''
		self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
																  definedColumnsList=labelColumnList)
		self.textAnnotationColumns = labelColumnList
		
		if self.annotationClass is not None: ## useful to keep already added annotations by another column selectable
			
			madeAnnotations = self.annotationClass.madeAnnotations
			madeAnnotations = self.annotationClass.selectionLabels
			## avoid wrong labeling
			try:
				self.annotationClass.disconnect_event_bindings()
			except:
				pass
		else:
			madeAnnotations = OrderedDict()
			selectionLabels = OrderedDict()
		
		numColumns = self.numericColumns

		self.annotationClass = annotateScatterPoints(self.plotter,self.ax,
													  self.data,labelColumnList, numColumns,
													  madeAnnotations,selectionLabels)
													  
													  			
	def change_color_by_categorical_columns(self,categoricalColumns,
													specificAxis = None,
													updateColor = False, 
													adjustLayer = True):
		'''
		Adjust colors according to the categorical level in selected columns
		'''
		ax = self.ax if specificAxis is None else specificAxis
		
		self.colorMapDict,layerMapDict, self.rawColorMapDict = get_color_category_dict(self.dfClass,
												categoricalColumns,
												self.colorMap, self.categoricalColorDefinedByUser,
												self.nanScatterColor)
		## update data if missing columns and add column 'color'
		self.data  = self.plotter.attach_color_data(categoricalColumns, self.data, 
													self.dataID, self.colorMapDict)											
		if updateColor == False:
				self.clean_up_saved_size_and_color_changes('color')														
		
		axCollection = self.ax.collections
		if updateColor == False and adjustLayer:
			self.data.loc[:,'layer'] = self.data['color'].map(layerMapDict)		
			self.data.loc[:,'size'] =  axCollection[0].get_sizes()	
			self.data = self.data.sort_values('layer', ascending = True)		
			axCollection[0].remove() 
			## we need to replot this, otherwise the layer/order cannot be changed. 
			self.plotter.add_scatter_collection(self.ax,
											x=self.data[self.numericColumns[0]],
											y = self.data[self.numericColumns[1]], 
											size=self.data['size'],
											color = self.data['color'].values, 
											picker = True)
			self.scatterKwargs['color'] = self.data['color'].values
			if self.annotationClass is not None:
				#changed order
				self.annotationClass.update_data(self.data)
			
			self.add_color_and_size_changes_to_dict('change_color_by_categorical_columns',categoricalColumns)
			if self.showLegend:
				self.plotter.nonCategoricalPlotter.add_legend_for_caetgories_in_scatter(ax,
																	self.colorMapDict,categoricalColumns)
		elif adjustLayer == False:
			axCollection[0].set_facecolor(self.data['color'].values)
			self.add_color_and_size_changes_to_dict('change_color_by_categorical_columns',categoricalColumns)
			if self.showLegend:
				self.plotter.nonCategoricalPlotter.add_legend_for_caetgories_in_scatter(ax,
																	self.colorMapDict,categoricalColumns)
		else:
			axCollection[0].set_facecolor(self.data['color'].values)
			if specificAxis is None: ##indicating that graph is not exported but only modified
				self.update_legend(ax,self.colorMapDict)				
			else:
				if self.showLegend or specificAxis is not None:
					self.plotter.nonCategoricalPlotter.add_legend_for_caetgories_in_scatter(ax,self.colorMapDict,
														  categoricalColumns, export = True)						


	def change_color_by_numerical_column(self, numericColumn, specificAxis = None, update = True):
		'''
		Accepts a numeric column from the dataCollection class. This column is added using 
		the index ensuring that correct dots get the right color. 
		'''
		cmap = get_max_colors_from_pallete(self.colorMap)
		if isinstance(numericColumn,str):
			numericColumn = [numericColumn]
		## update data if missing columns 
		self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
																  definedColumnsList=numericColumn)	
		ax = self.ax if specificAxis is None else specificAxis
		
		if update == False:
			self.clean_up_saved_size_and_color_changes('color')
		axCollection = ax.collections
		if len(numericColumn) > 1:
			# check for updated aggregation method
			if specificAxis is None:
				self.get_agg_method()
			## merge columns 
			if self.aggMethod == 'mean':
				colorData = self.data[numericColumn].mean(axis=1)
			else:
				colorData = self.data[numericColumn].sum(axis=1)
		else:
			colorData = self.data[numericColumn[0]]
			
		scaledData = scale_data_between_0_and_1(colorData) 
		scaledColorData = cmap(scaledData)
		axCollection[0].set_facecolors(scaledColorData )
		self.scatterKwargs['color'] = scaledColorData 
		self.add_color_and_size_changes_to_dict('change_color_by_numerical_column',numericColumn)



	def change_size_by_categorical_column(self, categoricalColumn, specificAxis = None, update = True, sizeMap = None):
		'''
		changes sizes of collection by a cateogrical column
		'''
		if isinstance(categoricalColumn,str):
			categoricalColumn = [categoricalColumn]
		## update data if missing columns 
		self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
																  definedColumnsList=categoricalColumn)	
		ax = self.ax if specificAxis is None else specificAxis
			## clean up saved changes
		if update == False and specificAxis is None:
				self.clean_up_saved_size_and_color_changes('size')
		
		if sizeMap is None:
			uniqueCategories = self.data[categoricalColumn].apply(tuple,axis=1).unique()			
			numberOfUuniqueCategories = uniqueCategories.size
			scaleSizes = np.linspace(0.3,1,num=numberOfUuniqueCategories,endpoint=True)
			sizeMap = dict(zip(uniqueCategories, scaleSizes))
			sizeMap = replace_key_in_dict('-',sizeMap,0.1)
			
		scaledData = self.data[categoricalColumn].apply(tuple,axis=1).map(sizeMap)
		axCollection = ax.collections
		sizeData = (scaledData)*(self.maxSize-self.minSize) + self.minSize
		axCollection[0].set_sizes(sizeData)
		self.scatterKwargs['s'] = sizeData
		self.data.loc[:,'size'] = sizeData
		self.add_color_and_size_changes_to_dict('change_size_by_categorical_column',categoricalColumn)

	def change_size_by_numerical_column(self, numericColumn, specificAxis = None, update = True, limits = None):
		'''
		change sizes of scatter points by a numerical column
		'''
		if isinstance(numericColumn,str):
			numericColumn = [numericColumn]
		## update data if missing columns is used to encode color
		self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
																  definedColumnsList=numericColumn)	
		if specificAxis is None:
			ax = self.ax
			# clean up stuff
			if update == False:
				self.clean_up_saved_size_and_color_changes('size')
		else:
			ax = specificAxis
		
		if len(numericColumn) > 1:
			# check for updated aggregation method
			if specificAxis is None:
				self.get_agg_method()
			## merge columns 
			if self.aggMethod == 'mean':
				sizeDataRaw = self.data[numericColumn].mean(axis=1)
			else:
				sizeDataRaw = self.data[numericColumn].sum(axis=1)
		else:
			sizeDataRaw = self.data[numericColumn[0]]
		
		axCollection = ax.collections
		if limits is not None:
			min, max = limits
		else:
			min, max = None, None 
			
		scaledData = scale_data_between_0_and_1(sizeDataRaw,min,max)
		
		sizeData = (scaledData)*(self.maxSize-self.minSize) + self.minSize
		axCollection[0].set_sizes(sizeData)
		self.scatterKwargs['s'] = sizeData
		self.data.loc[:,'size'] = sizeData
		self.add_color_and_size_changes_to_dict('change_size_by_numerical_column',numericColumn)


	def export_selection(self, exportAxis):
		'''
		Export the selected axis to another axis within a main figure.
		'''
		self.replot(exportAxis,**self.scatterKwargs)
		if 'change_color_by_categorical_columns' in self.sizeStatsAndColorChanges:
			self.plotter.nonCategoricalPlotter.add_legend_for_caetgories_in_scatter(exportAxis,
									self.colorMapDict,
									self.sizeStatsAndColorChanges['change_color_by_categorical_columns'], 
									export = True)
		
		## plot done statistical tests
		for stat in ['add_regression_line','add_lowess_line']:
			if stat in self.sizeStatsAndColorChanges:
				getattr(self,stat)(exportAxis)
			
									
		if self.annotationClass is not None:
				for key,props in self.annotationClass.selectionLabels.items():
					exportAxis.annotate(ha='left', arrowprops=arrow_args,**props)
																	  
	def get_size_interval(self):
		'''
		'''
		
		self.minSize, self.maxSize = self.plotter.get_size_interval()
		
	def update_size_interval(self):
		'''
		'''
		self.get_size_interval()
		for funcName, columnNames in self.sizeStatsAndColorChanges.items():
			if 'size' in funcName:
				getattr(self,funcName)(columnNames)
				break	
	
	def replot(self, ax = None, **kwargs):
		'''
		'''
		ax.scatter(self.data[self.numericColumns[0]].values,
						self.data[self.numericColumns[1]].values,
						**kwargs)	
	
	def clean_up_saved_size_and_color_changes(self,which = 'color'):
		'''
		'''
		toDelete = []
		for functionName,column in self.sizeStatsAndColorChanges.items(): 
			if which in functionName:
				toDelete.append(functionName)
		if 'change_color_by_categorical_columns' in toDelete:
			self.plotter.delete_legend(self.ax)
			
		# if which == 'color':
# 			self
			
		for func in toDelete:
			del self.sizeStatsAndColorChanges[func]	
# 	
	def update_colorMap(self,newCmap = None):
		'''
		allows changes of color map by the user. Simply updates the color code.
		It also changes the object: self.colorMap so that it will also be used when graph is exported.
		Please note that if you just call the function it will cuase an update, this is particullary useful
		when the user used the interactive widgets to customize the color settings
		'''
		
		if newCmap is not None:
			self.colorMap = newCmap 
			
		for functionName,column in self.sizeStatsAndColorChanges.items(): 
			getattr(self,functionName)(column)  


	def add_tooltip(self, annotationColumns):
		'''
		'''

		self.toolTipsActive = True
		self.define_bbox()
		self.define_text()
		self.build_tooltip()
		self.get_tooltip_data(annotationColumns)
		self.annotationColumns = annotationColumns
		
	def build_tooltip(self):
		'''
		'''
		self.tooltip = self.ax.text(s ='', bbox=self.bboxProps,**self.textProps)
		self.textProps['text'] = ''
			
	
	def determine_position(self,x,y):
		'''
		Check how to align the tooltip.
		'''
			
		xMin,xMax = self.axProps['xlim']
		yMin,yMax = self.axProps['ylim']
		
		diff = (xMin-xMax)*0.05	
		if x > xMin + (xMax-xMin)/2:
			self.textProps['ha'] = 'right'
		else:
			self.textProps['ha'] = 'left' 
			diff *= -1
			
		if y > yMin + (yMax-yMin)/2:
			
			self.textProps['va'] = 'top'
		else:
			self.textProps['va'] = 'bottom'
		
		self.textProps['x'] = x + diff	
		self.textProps['y'] = y 		
		

	def define_bbox(self):
		'''
		Define bbox
		'''
		self.bboxProps = {'facecolor':'white', 'alpha':0.85,
						 'edgecolor':'darkgrey','fill':True,
						 }
	
	def define_text(self):
		'''
		Define text properties
		'''
		self.textProps = {'x':0,'y':0,'fontname':defaultFont,
						 'linespacing': 1.5,
						 'visible':False,
						 'zorder':20}
				
	def update_position(self,event,text):
		'''
		'''
		# get event data
		x,y = event.xdata, event.ydata
		## check if new text 
		self.textProps['text'] = text	
		self.textProps['visible'] = True
		self.determine_position(x,y)
		self.tooltip.update(self.textProps)
		

	def get_tooltip_data(self, annotationColumnList):
		'''
		'''
		self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
												 definedColumnsList = annotationColumnList)	
			
	def add_regression_line(self, specificAxis=None):
		'''
		add regression line to scatter plot
		'''
																  
		xList,yList,slope,intercept,rValue,pValue, stdErrorSlope = stats.get_linear_regression(self.data[self.numericColumns])
		ax = self.ax if specificAxis is None else specificAxis
		regressionLabel = 'Slope: {}\nIntercept: {}\nr: {}\np-val: {:.2e}'.format(round(slope,2),round(intercept,2),round(rValue,2),pValue)  	
		self.plotter.add_annotationLabel_to_plot(ax,text=regressionLabel)
		ax.plot(xList,yList,linewidth = 1, linestyle= 'dashed')
		self.add_color_and_size_changes_to_dict('add_regression_line',None)
			
		
	def add_lowess_line(self,specificAxis=None):
		'''
		add lowess line to scatter plot
		'''
			
		if self.lowessData is None: ## because lowess calculations are time consuming we save this for export to main figure
			
			self.lowessData = stats.get_lowess(self.data[self.numericColumns])		
														  
		ax = self.ax if specificAxis is None else specificAxis
		ax.plot(self.lowessData[:,0],self.lowessData[:,1],linewidth = 1, linestyle= 'dashed',color="red")														  
		self.add_color_and_size_changes_to_dict('add_lowess_line',None)	
		
		
		

		
		