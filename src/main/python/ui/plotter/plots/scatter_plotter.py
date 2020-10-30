

import pandas as pd
import numpy as np
#from modules.utils import * 
from .scatter_annotations import annotateScatterPoints
from collections import OrderedDict
from .utils import styleHoverScat, defaultFont,get_elements_from_list_as_string, styleHoverScatter

#matplotlib import 
from matplotlib.patches import Circle, Ellipse, Rectangle
#from modules import stats
class scatterPlot(object):


	def __init__(self, parent, data, numericColumns, plotter, colorMap = "Blues", 
						ax = "", dataID = "", scatterKwargs = {}, hoverKwargs = {},
						showLegend = True, ignoreYlimChange = False):
		'''
		Class to handle scatter plots. Scatter plots that are generated together
		are connected and allow to hover over points to see the location of each
		data entry row in each plot. 
		Additional information of data can be added due to additional color and size levels.
		'''
		self.parent = parent
		self.ax = ax
		self.plotter = plotter
		#self.dfClass = dfClass 
		self.dataID = dataID
		self.numericColumns = numericColumns
		self.scatterKwargs = scatterKwargs
		self.hoverKwargs = hoverKwargs
		self.colorMap = colorMap
		self.numericColumns = numericColumns
		self.data = data
		self.showLegend = showLegend
		self.nanScatterColor = scatterKwargs['color']
		self.ignoreYlimChange = ignoreYlimChange
		self.defineVariables()
		#self.get_size_interval()
		self.adjust_axis_limits()
		self.replot(ax=self.ax,**self.scatterKwargs)
		self.extract_axis_props()
		self.add_hover_point()
		self.addBindings()
		
	def addBindings(self):
		'''
		'''		
		self.onHoverEvent = self.plotter.f.canvas.mpl_connect('motion_notify_event', self.onHover)
		self.onClickEvent = self.plotter.f.canvas.mpl_connect('button_press_event', self.onClick)
	
	def adjustRectangleSize(self, axisPerc = None, updateCircle = False):
		""
		if axisPerc is None:
			axisPerc = self.parent.mC.config.getParam("selectionRectangleSize")
		xMin, xMax = self.ax.get_xlim()
		yMin, yMax = self.ax.get_ylim()
		self.xDist = np.sqrt((xMax- xMin)**2) * axisPerc
		self.yDist = np.sqrt((yMax- yMin)**2) * axisPerc
		if updateCircle:
			self.selectRectangle.set_width(self.xDist)
			self.selectRectangle.set_height(self.yDist)

	def defineVariables(self):
		'''
		Define needed variables.
		'''
		self.categoricalColorDefinedByUser = dict()
		self.sizeStatsAndColorChanges = OrderedDict()
		self.annotationClass = None
		self.idxData = None
		self.lowessData = None
		self.toolTipsActive = False
		self.savedIdxData = None
		self.resized = False
		self.ignoreEvents = False
		
	
	
	def disconnect_event_bindings(self):
		""
		self.plotter.f.canvas.mpl_disconnect(self.onHoverEvent)
		self.plotter.f.canvas.mpl_disconnect(self.onClickEvent)

	def setHoverPointsInvisible(self,event = None, update = True, resetIdx = False):
		'''
		'''
		if resetIdx:
			self.idxData = None
			self.savedIdxData = None
		#self.selectRectangle[0].set_visible(False)
		if self.hoverScatter.get_visible():
			self.hoverScatter.set_visible(False)
			if self.toolTipsActive:
				self.tooltip.set_visible(False)				
			if update:
				self.updateAxis()

	
	def adjust_axis_limits(self):
		""
		#self.ax.axis('scaled')
		
		xMin, yMin = self.data[self.numericColumns].min()
		xMax, yMax = self.data[self.numericColumns].max()
		xAdd = np.sqrt(xMin**2 + xMax**2) * 0.05
		yAdd = np.sqrt(yMin**2 + yMax**2) * 0.05
		self.ax.set_xlim(xMin-xAdd,xMax+xAdd)
		self.ax.set_ylim(yMin-yAdd,yMax+yAdd)


	def addSelectRectangle(self):
		""
		self.adjustRectangleSize() 
		
		styleSelect = styleHoverScat.copy() 
		styleSelect["alpha"] = 0.5
		styleSelect["c"] = "None"
		self.selectRectangle = Rectangle((0,0),0,0,edgecolor="black",linewidth=0.5,fill=False,zorder=100)# Circle((0,0),self.distSelect,facecolor="None" ,edgecolor="black",linewidth=0.5,linestyle="-",visible=False)# self.ax.plot([25],[25],markersize= 18 ,**styleSelect)
		self.ax.add_artist(self.selectRectangle)
		

	def add_hover_point(self, addCircle = True):
		'''
		'''
		if addCircle:
			self.addSelectRectangle()
		
		#styleHoverScatter['s'] =  self.scatterKwargs['s']
		self.hoverScatter = self.ax.scatter([],[],**self.hoverKwargs)
		self.ax.callbacks.connect('ylim_changed', \
					lambda event:self.updateBackground(redraw = True,updateProps=True))

	def indicate_hover_point(self):
		''
		#if self.idxData is not None:
			
		self.parent.setHoverData(self.idxData,self)

	#def indicate_select_data(self, dataIndex):
	#	""
	#	self.plotter.annotate_selectedData()

	def set_invisible(self):
		""
		self.setHoverPointsInvisible()

	def setHoverData(self, dataIdx, sizes = None):
		'''
		'''
		if dataIdx is None:
			self.setHoverPointsInvisible(resetIdx=True)
		else:
			self.idxData = dataIdx
			#self.idxData = self.data.index[self.data.index.isin(dataIdx)]
			#if self.idxData.size == 0:
			#	return
			self.update_hover_data()
	
	def setSelecRectInvisible(self):
		""
		if self.selectRectangle.get_visible():
			self.selectRectangle.set_visible(False)
			
	def set_select_data(self, event):
		""
		
		x,y = event.xdata, event.ydata
		self.hoverScatter.set_visible(True)

		self.selectRectangle.set_xy((x-self.xDist/2,y-self.yDist/2))
		
		return self.find_points_in_rectangle(x,y)

	def setResizeTrigger(self,resized):
		""
		self.resized = resized

	def getResizeTrigger(self):
		""
		return self.resized
		

	def find_points_in_rectangle(self,xCenter,yCenter):
		""

		ll = np.array([xCenter-self.xDist/2,yCenter-self.yDist/2])  # lower-left
		ur = np.array([xCenter+self.xDist/2,yCenter+self.yDist/2])  # upper-right
		#pts = np.array(points)

		pts = self.data[self.numericColumns].values
		
		inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
		if np.any(inidx):
			return self.data.index[inidx]
		
	def setHoverObjectsInivisible(self,leftWidget=False):
		""
		self.setHoverPointsInvisible()
		self.setSelecRectInvisible()
		self.updateAxis(onWidgetLeave=leftWidget)

	def onClick(self,event):
		""
		if event.inaxes is None:
			self.setHoverPointsInvisible()
			return
		if event.inaxes != self.ax:
			return
		self.setSelecRectInvisible()
		self.plotter.sendSelectEventToQuickSelect(self.idxData)
		
	def onHover(self,event, size = None):
		""
		if self.ignoreEvents:
			return
		if event.inaxes is None or event.inaxes != self.ax:
			self.setHoverObjectsInivisible()
			return
		self.selectRectFrac = self.parent.mC.config.getParam("selectionRectangleSize")
		if event.button == 1:
			self.setHoverPointsInvisible()
			return
		if hasattr(self,'background') == False:
			self.updateBackground(redraw=False)
		
		if self.selectRectFrac == 0:
			#single points selection
			self.setSelecRectInvisible()
			self.idxData = self.check_for_close_point(event)

		elif self.selectRectFrac > 0:
			#multiple point selection
			self.selectRectangle.set_visible(True)
			self.idxData = self.set_select_data(event)
			self.adjustRectangleSize(updateCircle=True)
		else:
			return

		if self.idxData is None:
				self.setHoverPointsInvisible(update=False)
				#if self.plotter.nonCategoricalPlotter is not None:
				self.indicate_hover_point()
				self.parent.sendIndexToQuickSelectWidget(self.idxData)
				self.savedIdxData = None
				self.updateAxis()
				return
		else:
			if self.savedIdxData is not None and len(self.savedIdxData) ==  len(self.idxData) \
					and all(x in self.savedIdxData for x in self.idxData):
				# make sure to update circle
				self.updateAxis()
				return
			self.indicate_hover_point()
			self.update_hover_data()
			#if quick select is available set this 
			self.parent.sendIndexToQuickSelectWidget(self.idxData)
			self.savedIdxData = self.idxData.copy()

	def update_hover_data(self, sizes = None):
		'''
		'''		
		#print(pointCoords)
		if self.idxData is None:
			return
		if 'size' in self.data.columns:
			sizes = self.data.loc[self.idxData,'size'].values
			self.hoverScatter.set_sizes(sizes)
		else:
			self.hoverScatter.set_sizes([self.hoverKwargs['s']])
		#if isinstance(self.idxData,pd.Int64Index):
		#	self.idxData = np.array(self.idxData.data)
		if all(x in self.data.index for x in self.idxData):

			pointCoords = self.data.loc[self.idxData,self.numericColumns].values
		
			self.hoverScatter.set_offsets(pointCoords)

			if not self.hoverScatter.get_visible():
				self.hoverScatter.set_visible(True)

			self.updateAxis()
		else:
			self.idxData = None

	def updateBackground(self, redraw=True,updateProps=True):
		""
		self.setSelecRectInvisible()
		self.setHoverPointsInvisible(update=False)
		if self.ignoreYlimChange:
			return
		
		if updateProps:
			self.extract_axis_props()
		if redraw:
			if hasattr(self,'background'):
				self.setHoverPointsInvisible(update=False)
			self.plotter.redraw(backgroundUpdate = False)
		self.background = self.plotter.f.canvas.copy_from_bbox(self.ax.bbox)	
		
	def extract_axis_props(self):
		""
		#self.axProps = dict()
		
		self.axProps = dict(xDiff = self.ax.get_xlim()[1] - self.ax.get_xlim()[0],
							yDiff = self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
							xlim = self.ax.get_xlim(),
							ylim = self.ax.get_ylim())
			
	def check_for_close_point(self, event):
		'''
		'''
		idxData = None
		colIdx1 = self.numericColumns[0]
		colIdx2 = self.numericColumns[1]
		
		xValue = np.asarray(event.xdata)
		yValue = np.asarray(event.ydata)	
		boolXData = np.isclose(self.data[colIdx1].values,xValue,atol = 0.01*self.axProps['xDiff'])
		boolYData = np.isclose(self.data[colIdx2].values,yValue,atol = 0.01*self.axProps['yDiff'])
		add = np.sum([boolXData,boolYData], axis=0)
		argMax = np.argmax(add == 2)
		if argMax == 0 and add[0] != 2:	
			return None
		else:
			if self.toolTipsActive:
				textData = self.data[self.annotationColumns].values[argMax]
				text = get_elements_from_list_as_string(textData).replace(', ','\n')
				self.update_position(event,text)
				
			idxData = self.data.index[argMax]
			
			return pd.Series(idxData)

	def updateAxis(self, onlyCirle = False, onWidgetLeave = False):
		'''
		Update artists using blit.
		'''
		hoverPoints = self.hoverScatter
		if hasattr(self,'background') == False:
			self.background =  self.plotter.f.canvas.copy_from_bbox(self.ax.bbox)

		if self.getResizeTrigger():
			#self.plotter.updateBackground_in_scatter_plots(redraw=True)
			self.setResizeTrigger(False)

		self.plotter.f.canvas.restore_region(self.background)
		
		#circle is not alway visible 
		if self.selectRectangle.get_visible():
			self.ax.draw_artist(self.selectRectangle)	

		if not onlyCirle and self.idxData is not None and not onWidgetLeave:
			#hover points are always visible
			hoverPoints.set_visible(True)
			try:
				self.ax.draw_artist(hoverPoints)
			except Exception as e:
				self.add_hover_point(addCircle=False)
			if self.toolTipsActive:
				self.ax.draw_artist(self.tooltip)
		self.plotter.f.canvas.blit(self.ax.bbox)
		
		
	def get_numeric_color_data(self, numericColumn = None):
		'''
		'''
		if numericColumn is None:
			if 'change_color_by_numerical_column' in self.sizeStatsAndColorChanges:
				numericColumn = self.sizeStatsAndColorChanges['change_color_by_numerical_column']
			else:
				return
		if len(numericColumn) > 1:
			# check for updated aggregation method
			## merge columns 
			if self.plotter.aggMethod == 'mean':
				colorData = self.data[numericColumn].mean(axis=1)
			else:
				colorData = self.data[numericColumn].sum(axis=1)
		else:
			colorData = self.data[numericColumn[0]]
		return scale_data_between_0_and_1(colorData) 
		
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
		madeAnnotations = OrderedDict()
		selectionLabels = OrderedDict()
		
		if self.annotationClass is not None: ## useful to keep already added annotations by another column selectable
			
			madeAnnotations = self.annotationClass.madeAnnotations
			selectionLabels = self.annotationClass.selectionLabels
			## avoid wrong labeling
			try:
				self.annotationClass.disconnect_event_bindings()
			except:
				pass
				
		numColumns = self.numericColumns

		self.annotationClass = annotateScatterPoints(self.plotter,self.ax,
													  self.data,labelColumnList, numColumns,
													  madeAnnotations,selectionLabels)

	def updateScatterProps(self,propsData,updatableProps = ["color","size","layer"]):
		""

		self.setHoverPointsInvisible()
		self.setSelecRectInvisible()

		columnsPresent = [x for x in  updatableProps if x in propsData.columns]
		columnsInData = [x for x in updatableProps if x in self.data.columns]
		#delete columns if they exists
		self.data.drop(columnsInData,axis=1,inplace=True) 
		if len(columnsPresent) > 0:
			#join property to self.data
			self.data = self.data.join(propsData)
		if "layer" in columnsPresent:
			#if layer is updated, we need to replot
			self.replotCollection() 
		else:
			#if layer is not updated, just update.
			if "color" in columnsPresent:
				self.ax.collections[0].set_facecolor(self.data['color'].values)
			if "size" in columnsPresent:
				self.ax.collections[0].set_sizes(self.data['size'].values)

						  
	def updateDataWithProps(self):
		""
		if "color" not in self.data.columns:
			self.data.loc[:,"color"] = self.ax.collections[0].get_facecolors()
		if "size" not in self.data.columns:
			self.data.loc[:,"size"] = self.ax.collections[0].get_sizes() 

	def updateColorData(self,colorData,setColorToCollection = True):
		""
		if "color" in colorData.columns:
			if "color" in colorData.columns and "color" in self.data.columns:
				self.data.drop(["color"],axis=1, inplace=True)
			self.data = self.data.join(colorData)
			if setColorToCollection:
				self.ax.collections[0].set_facecolor(self.data['color'].values)
			
			#self.add_color_and_size_changes_to_dict('change_color_by_categorical_columns',categoricalColumns)

	def replotCollection(self, updateLayer = True):
		""
		if updateLayer and "layer" in self.data.columns:
			self.data = self.data.sort_values('layer', kind="mergesort" ,ascending = True)
		self.updateDataWithProps()
		self.ax.collections[0].remove() 
		## we need to replot this, otherwise the layer/order cannot be changed. 
		self.plotter.add_scatter_collection(self.ax,
										x=self.data[self.numericColumns[0]],
										y = self.data[self.numericColumns[1]], 
										size=self.data['size'],
										color = self.data['color'].values, 
										picker = True)


	def change_color_by_categorical_columns(self,categoricalColumns,
													colorMapDict,
													layerColorDict,
													colorMapDictRaw,
													specificAxis = None,
													updateColor = True, 
													adjustLayer = True):
		'''
		Adjust colors according to the categorical level in selected columns
		'''
		ax = self.ax if specificAxis is None else specificAxis

		self.colorMapDict = colorMapDict
		self.rawColorMapDict = colorMapDictRaw

		## update data if missing columns and add column 'color'
		self.data  = self.plotter.attach_color_data(categoricalColumns, self.data, 
													self.dataID, self.colorMapDict)	
												
		if updateColor == False:
				self.clean_up_saved_size_and_color_changes('color')														
		
		axCollection = self.ax.collections
		if updateColor == False and adjustLayer:
			self.data.loc[:,'layer'] = self.data['color'].map(layerColorDict)	
			if 'size' not in self.data.columns:	
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
				self.plotter.add_legend_for_caetgories_in_scatter(ax,
															self.colorMapDict,categoricalColumns)
		elif adjustLayer == False:
			axCollection[0].set_facecolor(self.data['color'].values)
			self.add_color_and_size_changes_to_dict('change_color_by_categorical_columns',categoricalColumns)
			
			if self.showLegend:
				self.plotter.add_legend_for_caetgories_in_scatter(ax,
															self.colorMapDict,categoricalColumns)
		else:
			axCollection[0].set_facecolor(self.data['color'].values)
			if specificAxis is None: ##indicating that graph is not exported but only modified
				self.plotter.update_legend(ax,self.colorMapDict)				
			else:
				if self.showLegend or specificAxis is not None:
					self.plotter.add_legend_for_caetgories_in_scatter(ax,self.colorMapDict,
														  categoricalColumns, export = True)						


	# def change_color_by_numerical_column(self, numericColumn, specificAxis = None, update = True):
	# 	'''
	# 	Accepts a numeric column from the dataCollection class. This column is added using 
	# 	the index ensuring that correct dots get the right color. 
	# 	'''
	# 	cmap = get_max_colors_from_pallete(self.colorMap)
	# 	if isinstance(numericColumn,str):
	# 		numericColumn = [numericColumn]
	# 	## update data if missing columns 
	# 	self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
	# 															  definedColumnsList=numericColumn)	
	# 	ax = self.ax if specificAxis is None else specificAxis
		
	# 	if update == False:
	# 		self.clean_up_saved_size_and_color_changes('color')
			
	# 	axCollection = ax.collections
	# 	scaledData = self.get_numeric_color_data(numericColumn)
			
		
	# 	scaledColorData = cmap(scaledData)
	# 	axCollection[0].set_facecolors(scaledColorData )
	# 	self.scatterKwargs['color'] = scaledColorData 
	# 	if update == False:
	# 		self.add_color_and_size_changes_to_dict('change_color_by_numerical_column',numericColumn)



	# def change_size_by_categorical_column(self, categoricalColumn, specificAxis = None, update = True, sizeMap = None):
	# 	'''
	# 	changes sizes of collection by a cateogrical column
	# 	'''
	# 	if isinstance(categoricalColumn,str):
	# 		categoricalColumn = [categoricalColumn]
	# 	## update data if missing columns 
	# 	self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
	# 															  definedColumnsList=categoricalColumn)	
	# 	ax = self.ax if specificAxis is None else specificAxis
	# 		## clean up saved changes
	# 	if update == False and specificAxis is None:
	# 			self.clean_up_saved_size_and_color_changes('size')
		
	# 	if sizeMap is None:
	# 		uniqueCategories = self.data[categoricalColumn].apply(tuple,axis=1).unique()			
	# 		numberOfUuniqueCategories = uniqueCategories.size
	# 		scaleSizes = np.linspace(0.3,1,num=numberOfUuniqueCategories,endpoint=True)
	# 		sizeMap = dict(zip(uniqueCategories, scaleSizes))
	# 		sizeMap = replace_key_in_dict('-',sizeMap,0.1)
			
	# 	scaledData = self.data[categoricalColumn].apply(tuple,axis=1).map(sizeMap)
	# 	axCollection = ax.collections
	# 	sizeData = (scaledData)*(self.maxSize-self.minSize) + self.minSize
	# 	axCollection[0].set_sizes(sizeData)
	# 	self.scatterKwargs['s'] = sizeData
	# 	self.data.loc[:,'size'] = sizeData
	# 	if update == False:
	# 		self.add_color_and_size_changes_to_dict('change_size_by_categorical_column',categoricalColumn)
			

	# def change_size_by_numerical_column(self, numericColumn, specificAxis = None, update = True, limits = None):
	# 	'''
	# 	change sizes of scatter points by a numerical column
	# 	'''
	# 	if isinstance(numericColumn,str):
	# 		numericColumn = [numericColumn]
	# 	## update data if missing columns is used to encode color
	# 	self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
	# 															  definedColumnsList=numericColumn)	
	# 	if specificAxis is None:
	# 		ax = self.ax
	# 		# clean up stuff
	# 		if update == False:
	# 			self.clean_up_saved_size_and_color_changes('size')
	# 	else:
	# 		ax = specificAxis
		
	# 	if len(numericColumn) > 1:
	# 		# check for updated aggregation method
	# 		## merge columns 
	# 		if self.plotter.aggMethod	 == 'mean':
	# 			sizeDataRaw = self.data[numericColumn].mean(axis=1)
	# 		else:
	# 			sizeDataRaw = self.data[numericColumn].sum(axis=1)
	# 	else:
	# 		sizeDataRaw = self.data[numericColumn[0]]
		
	# 	axCollection = ax.collections
	# 	if limits is not None:
	# 		min, max = limits
	# 	else:
	# 		min, max = None, None 
			
	# 	scaledData = scale_data_between_0_and_1(sizeDataRaw,min,max)
		
	# 	sizeData = (scaledData)*(self.maxSize-self.minSize) + self.minSize
	# 	axCollection[0].set_sizes(sizeData)
	# 	self.scatterKwargs['s'] = sizeData
	# 	self.data.loc[:,'size'] = sizeData
	# 	if update == False:
	# 		self.add_color_and_size_changes_to_dict('change_size_by_numerical_column',numericColumn)


	def export_selection(self, exportAxis):
		'''
		Export the selected axis to another axis within a main figure.
		'''
		self.replot(exportAxis,**self.scatterKwargs)
		if 'change_color_by_categorical_columns' in self.sizeStatsAndColorChanges:
			self.plotter.add_legend_for_caetgories_in_scatter(exportAxis,
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

	def update_transparancy(self,alpha):
		'''
		'''
		self.scatterKwargs['alpha'] = alpha
		
	def update_size_interval(self):
		'''
		'''
		self.get_size_interval()
		for funcName, columnNames in self.sizeStatsAndColorChanges.items():
			if 'size' in funcName:
				getattr(self,funcName)(columnNames)
				break	
	
	def outsideThreadPlotting(self):

		self.replot(self.ax,**self.scatterKwargs)

	def replot(self, ax, **kwargs):
		'''
		'''
	
		ax.scatter(self.data[self.numericColumns[0]].values,
					self.data[self.numericColumns[1]].values,
					**kwargs)	
			
	def clean_up_saved_size_and_color_changes(self,which = 'color'):
		'''
		'''
		toDelete = []
		for functionName,_ in self.sizeStatsAndColorChanges.items(): 
			if which in functionName:
				toDelete.append(functionName)
		if 'change_color_by_categorical_columns' in toDelete:
			self.plotter.delete_legend(self.ax)
			
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


	def add_tooltip(self, toolTipData):
		'''
		'''
		self.toolTipsActive = True
		columnsNotInData = [x for x in toolTipData.columns if x not in self.data.columns]
		self.annotationColumns = toolTipData.columns.values.tolist()
		self.define_bbox()
		self.define_text()
		self.build_tooltip()

		if len(columnsNotInData) != 0:
			self.data = self.data.join(toolTipData[columnsNotInData])
		
		
	def build_tooltip(self):
		'''
		'''
		self.tooltip = self.ax.text(s ='', bbox=self.bboxProps,**self.textProps)
		self.textProps['text'] = ''

	def disconnect_tooltip(self):
		'''
		Destroy tooltip text
		'''		
		if hasattr(self,'tooltip'):
			self.tooltip.remove()
			del self.tooltip
			self.toolTipsActive = False	
	
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
		self.statAnnotText = self.plotter.add_annotationLabel_to_plot(ax,text=regressionLabel)
		self.statLine = ax.plot(xList,yList,linewidth = 1, linestyle= 'dashed')
		self.add_color_and_size_changes_to_dict('add_regression_line',None)
			
		
	def add_lowess_line(self,specificAxis=None):
		'''
		add lowess line to scatter plot
		'''
			
		if self.lowessData is None: ## because lowess calculations are time consuming we save this for export to main figure
			
			self.lowessData = stats.get_lowess(self.data[self.numericColumns])		
														  
		ax = self.ax if specificAxis is None else specificAxis
		self.statLine = ax.plot(self.lowessData[:,0],self.lowessData[:,1],linewidth = 1, linestyle= 'dashed',color="red")														  
		self.add_color_and_size_changes_to_dict('add_lowess_line',None)	

	def remove_stat_line(self):
		'''
		'''
		if hasattr(self,'statLine'):
			self.statLine[0].remove()
		if hasattr(self,'statAnnotText'):
			self.statAnnotText.remove()

	def set_nan_color(self,newColor = None):
		'''
		'''
		self.scatterKwargs['color'] = newColor
		#self.nanScatterColor = self.scatterKwargs['color']
		self.ax.collections[0].set_facecolor(self.scatterKwargs['color'])
		
		#self.ignoreYlimChange = ignoreYlimChange
		
		#self.defineVariables()
		#self.get_size_interval()
		#self.replot(ax=self.ax,**self.scatterKwargs)


	def update_size(self, size):
		'''
		'''
		nIdx = len(self.data.index)
		self.data.loc[:,'size'] = [size] * nIdx 
		self.scatterKwargs['s'] = [size] * nIdx 

		
	def __getstate__(self):
	
		state = self.__dict__.copy()
		for attr in ['figure', 'axisDict','annotationClass']:
			if attr in state: 
				del state[attr]
		self.annotationClass = None
		return state		
		

		
		