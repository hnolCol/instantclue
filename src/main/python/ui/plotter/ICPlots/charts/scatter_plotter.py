

import pandas as pd
import numpy as np
#from modules.utils import * 
#from .scatter_annotations import annotateScatterPoints
from collections import OrderedDict
from matplotlib.font_manager import FontProperties
 


#matplotlib import 
from matplotlib.patches import Rectangle
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#define base style for hover
styleHoverScat = dict(
				visible=False, 
				c = 'red', marker = 'o',
				markeredgecolor = 'black',
				markeredgewidth = 0.3
				)
				
class scatterPlot(object):


	def __init__(self, parent, data, numericColumns, plotter, colorMap = "Blues", 
						ax = "", dataID = "", scatterKwargs = {}, hoverKwargs = {},
						showLegend = True, ignoreYlimChange = False, multiScatter = False,interactive = True, multiScatterKwargs = dict(), adjustLimits = True):
		'''
		Class to handle scatter plots. Scatter plots that are generated together
		are connected and allow to hover over points to see the location of each
		data entry row in each plot. 
		Additional information of data can be added due to additional color and size levels.
		'''
		self.parent = parent
		self.ax = ax
		self.plotter = plotter
		self.dataID = dataID
		self.numericColumns = numericColumns
		self.scatterKwargs = scatterKwargs
		self.hoverKwargs = hoverKwargs
		self.colorMap = colorMap
		self.multiScatter = multiScatter
		self.numericColumns = numericColumns
		self.data = data
		self.showLegend = showLegend
		self.nanScatterColor = parent.mC.config.getParam("nanColor")
		self.ignoreYlimChange = ignoreYlimChange
		self.maskIndex = data.index
		self.interactive = interactive 
		self.adjustLimits = adjustLimits
		self.multiScatterKwargs = multiScatterKwargs
		self.defineVariables()
		#self.get_size_interval()
		self.adjustAxisLimits()
		self.addKwargsToData()
		self.replot(ax=self.ax,**self.scatterKwargs)
		self.extract_axis_props()
		#add potential tooltip 
		self.defineBbox()
		self.defineText()
		self.buildTooltip()
		#add bindings
		if self.interactive:
			self.add_hover_point()
			self.addBindings()

		
	def addBindings(self):
		'''
		'''	
		self.onHoverEvent = self.plotter.f.canvas.mpl_connect('motion_notify_event', self.onHover)
		self.onClickEvent = self.plotter.f.canvas.mpl_connect('button_press_event', self.onClick)
	
	def addKwargsToData(self):
		""
		if isinstance(self.scatterKwargs["s"],pd.Series):

			self.data = self.data.join(self.scatterKwargs["s"])

		if isinstance(self.scatterKwargs["color"],pd.Series):

			self.data = self.data.join(self.scatterKwargs["color"])
			self.data  = self.data.loc[self.scatterKwargs["color"].index,]
		
		if "color" in self.data.columns:
			self.scatterKwargs["color"] = self.data["color"].values
		if "size" in self.data.columns:
			self.scatterKwargs["s"] = self.data["size"].values


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
		
	
	
	def disconnectBindings(self):
		""
		if hasattr(self,"onHoverEvent"):
			self.plotter.f.canvas.mpl_disconnect(self.onHoverEvent)
		if hasattr(self,"onClickEvent"):
			self.plotter.f.canvas.mpl_disconnect(self.onClickEvent)

	def setHoverPointsInvisible(self,event = None, update = True, resetIdx = False):
		'''
		'''
		if resetIdx:
			self.idxData = None
			self.savedIdxData = None
		#self.selectRectangle[0].set_visible(False)
		if self.toolTipsActive:
			self.setTooltipInvisible()
		if self.hoverScatter.get_visible():
			self.hoverScatter.set_visible(False)
			#if self.toolTipsActive:
			#	self.tooltip.set_visible(False)				
			if update:
				self.updateAxis()

	def setTooltipInvisible(self):
		""
		self.tooltip.set_visible(False)
	
	def adjustAxisLimits(self):
		""
		if not self.adjustLimits:
			return
		#self.ax.axis('scaled')
		if self.multiScatter:
			xMin = np.nanmin(self.data[self.numericColumns[0::2]].values)
			xMax = np.nanmax(self.data[self.numericColumns[0::2]].values)
			yMin = np.nanmin(self.data[self.numericColumns[1::2]].values)
			yMax = np.nanmax(self.data[self.numericColumns[1::2]].values)
		else:
			nonNaNData = self.data[self.numericColumns].dropna()
			xMin, yMin = nonNaNData[self.numericColumns].min()
			xMax, yMax = nonNaNData[self.numericColumns].max()
		xAdd = np.sqrt(xMin**2 + xMax**2) * 0.05
		yAdd = np.sqrt(yMin**2 + yMax**2) * 0.05
		if all(not np.isnan(x) for x in [xMin,xMax,yMin,yMax,yAdd,xAdd]):
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
		
		self.hoverScatter = self.ax.scatter([],[],**self.hoverKwargs)
		self.ax.callbacks.connect('ylim_changed', \
					lambda event:self.updateBackground(redraw = True,updateProps=True))

	def indicate_hover_point(self):
		''	
		self.parent.setHoverData(self.idxData,self)

	def setInvisible(self):
		""
		self.setHoverPointsInvisible()

	def setHoverData(self, dataIdx, sizes = None):
		'''
		'''
		if self.interactive:
			if dataIdx is None:
				self.setHoverPointsInvisible(resetIdx=True)
			else:

				self.idxData = dataIdx
				self.update_hover_data()

	def setMask(self,dataIndex):
		""
		if dataIndex is None:
			self.resetMask()
		else:
			self.setHoverObjectsInvisible()
			self.maskIndex = dataIndex[dataIndex == True].index
			self.updateScatterCollection()
	
	def setScatterVisible(self,visible = False):
		""
		if isinstance(self.mainCollecion, dict):
			for scatterCollection in self.mainCollecion.values():
				scatterCollection.set_visible(visible)
		else:
			self.mainCollecion.set_visible(visible)

	def getScatterInvisibility(self,visible = False):
		""
		if isinstance(self.mainCollecion, dict):
			for scatterCollection in self.mainCollecion.values():
				return scatterCollection.get_visible()
		else:
			return self.mainCollecion.set_visible(visible)

	def toggleVisibility(self):
		""
		visible = self.getScatterInvisibility()
		if isinstance(visible,bool):
			self.setScatterVisible(not visible)

	def resetMask(self):
		""
		self.setHoverObjectsInvisible()
		self.maskIndex = self.data.index
		self.updateScatterCollection()
	
	def setSelecRectInvisible(self):
		""
		if self.selectRectangle.get_visible():
			self.selectRectangle.set_visible(False)
			
	def setSelectRectangleData(self, event):
		""
		
		x,y = event.xdata, event.ydata
		if not self.hoverScatter.get_visible():
			self.hoverScatter.set_visible(True)

		self.selectRectangle.set_xy((x-self.xDist/2,y-self.yDist/2))
		
		return self.findPointsInRectangle(x,y,event)

	def setResizeTrigger(self,resized):
		""
		self.resized = resized

	def getResizeTrigger(self):
		""
		return self.resized
		

	def setData(self,data):
		""
		self.data = data
		self.maskIndex = data.index
		self.extract_axis_props()
		self.updateScatterCollection()

	def findPointsInRectangle(self,xCenter,yCenter,event):
		""

		ll = np.array([xCenter-self.xDist/2,yCenter-self.yDist/2])  # lower-left
		ur = np.array([xCenter+self.xDist/2,yCenter+self.yDist/2])  # upper-right
		#pts = np.array(points)
		if self.multiScatter:
			numericColumnPairs = list(zip(self.numericColumns[0::2], self.numericColumns[1::2]))
			inidx = np.zeros(shape = (self.data.index.size,len(numericColumnPairs)))
			for n,numPair in enumerate(numericColumnPairs):
				pts = self.data.loc[self.maskIndex,numPair].values
				inidx[:,n] = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)

			inidx = np.any(inidx,axis=1)
		else:
			pts = self.data.loc[self.maskIndex,self.numericColumns[:2]].values
			inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
		if np.any(inidx):
			idx = self.maskIndex[inidx]
			if self.toolTipsActive:
				#reduce number of tooltips to 12
				nAnnotColumns = len(self.annotationColumns)
				idxT = idx if idx.size * nAnnotColumns <= 14 else idx.values[:int(14/nAnnotColumns)]
				self.updateTooltipPosition(event,"\n".join([str(x) if len(str(x)) < 20 else "{}..".format(str(x)[:20]) for x in self.data.loc[idxT,self.annotationColumns].values.flatten()]))
			return idx
		
	def setHoverObjectsInvisible(self,leftWidget=False, update=True):
		""
		self.setHoverPointsInvisible(update=False)
		self.setSelecRectInvisible()
		if update:
			self.updateAxis(onWidgetLeave=leftWidget)

	def onClick(self,event):
		"Handle mouse clicks on scatter. Sends the selected index to quick select if no zoom/pan is selected"
		
		if event.inaxes is None:
			self.setHoverPointsInvisible()
			return
		if event.inaxes != self.ax:
			return
		self.setSelecRectInvisible()
		if self.parent.getToolbarState() is None and not self.parent.preventQuickSelectCapture and not self.parent.eventOverAnnotation(event.inaxes,event) and event.button == 1:
			
			self.parent.sendSelectEventToQuickSelect(self.idxData)
		
	def onHover(self,event, size = None):
		""
		if self.ignoreEvents:
			return
		if event.inaxes is None or event.inaxes != self.ax:
			self.setHoverObjectsInvisible()
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
			self.idxData = self.setSelectRectangleData(event)
			self.adjustRectangleSize(updateCircle=True)
		else:
			return

		if self.idxData is None: #no hover idxData = None will reset QuickSelect and LiveGraph
				self.setHoverPointsInvisible(update=False)
				self.indicate_hover_point()
				self.parent.sendIndexToQuickSelectWidget(self.idxData)
				self.parent.sendIndexToLiveGraph(self.idxData)
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
			self.parent.sendIndexToLiveGraph(self.idxData)
			self.savedIdxData = self.idxData.copy()

	def update_hover_data(self, sizes = None):
		'''
		'''		
		if self.idxData is None:
			return
		if 'size' in self.data.columns and not self.multiScatter:
			sizes = self.data.loc[self.idxData,'size'].values
			self.hoverScatter.set_sizes(sizes)
		
		else:
			self.hoverScatter.set_sizes([self.hoverKwargs['s']])
		
		if all(idx in self.data.index for idx in self.idxData):

			pointCoords = self.data.loc[self.idxData,self.numericColumns].values
			
			if self.multiScatter:
				
				X = np.concatenate([pointCoords[:,n-2:n] for n in np.arange(2,len(self.numericColumns)+2,step=2)])
				
				if any("size({}:{})".format(*columnPair) in self.data.columns for columnPair in self.multiScatterKwargs.keys()):
					Xs = []
					for columnPair in self.multiScatterKwargs.keys():
						sizeColumnName = "size({}:{})".format(*columnPair)
						if not sizeColumnName in self.data.columns:
							Xs.append(np.array([self.scatterKwargs["s"]]*self.idxData.size))
						else:
							Xs.append(self.data.loc[self.idxData,sizeColumnName].values.flatten())
					sizes = np.concatenate(Xs)
					self.hoverScatter.set_sizes(sizes)
				
				self.hoverScatter.set_offsets(X)
			else:
				self.hoverScatter.set_offsets(pointCoords)

			if not self.hoverScatter.get_visible():
				self.hoverScatter.set_visible(True)

			self.updateAxis()
		else:
			self.idxData = None

	def updateBackground(self, redraw=True, updateProps=True):
		""
		if self.interactive:
			self.setSelecRectInvisible()
			self.setHoverPointsInvisible(update=False) #this will also set tooltip invisible
			if self.ignoreYlimChange:
				return
			
			if updateProps:
				self.extract_axis_props()
			
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
			#	text = get_elements_from_list_as_string(textData).replace(', ','\n')
				self.updateTooltipPosition(event,textData)
				
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
			self.parent.updateBackgrounds(redraw=True)
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

	def updateScatterKwargs(self,kwargs):
		""
		if isinstance(self.mainCollecion,dict):
			for scatterCollection in self.mainCollecion.values():
				scatterCollection.update(kwargs)
		else:
			self.mainCollection.update(kwargs)

	def updateScatterProps(self,propsData,updatableProps = ["color","size","layer","marker"]):
		""
		self.setHoverPointsInvisible()
		self.setSelecRectInvisible()
		columnsPresent = [x for x in  updatableProps if x in propsData.columns]
		columnsInData = [x for x in updatableProps if x in self.data.columns and x in columnsPresent]

		#delete columns if they exists
		self.data.drop(columnsInData,axis=1,inplace=True) 
		if len(columnsPresent) > 0:
			#join property to self.data
			self.data = self.data.join(propsData)
		if "marker" in columnsPresent or "layer" in columnsPresent:
			#if layer or marker is updated, we need to replot
			self.replotCollection() 
		else:
			#if layer is not updated, just update.
			self.updateScatterCollection()
			
	def updateScatterCollection(self):
		
		if isinstance(self.mainCollecion,dict) and "marker" in self.data.columns:
			for markerName, markerData in self.data.groupby("marker"):
				if markerName in self.mainCollecion:
					maskIndex = markerData.index.intersection(self.maskIndex)
					self.mainCollecion[markerName].set_offsets(markerData.loc[maskIndex,self.numericColumns].values)
					if "color" in markerData.columns:
						self.mainCollecion[markerName].set_facecolor(markerData.loc[maskIndex,"color"].values)
					if "size" in markerData.columns:
						self.mainCollecion[markerName].set_sizes(markerData.loc[maskIndex,"size"].values)
		else:
			self.mainCollecion.set_offsets(self.data.loc[self.maskIndex,self.numericColumns].values)
			if "color" in self.data.columns:
				self.mainCollecion.set_facecolor(self.data.loc[self.maskIndex,'color'].values)
			if "size" in self.data.columns:
				self.mainCollecion.set_sizes(self.data.loc[self.maskIndex,'size'].values)


	def updateScatterPropSection(self,idx, value, propName = "color"):
		"""
		#Warning
		No checking performed if idx in self.data.index
		"""
		try:
			if propName in self.data.columns:
				self.data.loc[idx,propName] = value
				self.updateScatterCollection()
		except Exception as e:
			
			print(e)


	def updateDataWithProps(self):
		""
		if "color" not in self.data.columns:
			self.data.loc[:,"color"] = pd.Series([self.mainCollecion.get_facecolors()[0]] * self.data.index.size, index=self.data.index)
		if "size" not in self.data.columns:
			self.data.loc[:,"size"] = pd.Series([self.mainCollecion.get_sizes()[0]] * self.data.index.size, index=self.data.index)
	
	def updateColorData(self,colorData,setColorToCollection = True):
		""
		if isinstance(colorData,str):
			
			self.data.loc[self.data.index,"color"] = colorData
			if setColorToCollection:
				if isinstance(self.mainCollecion,dict) and "marker" in self.data.columns:
					for markerName, markerData in self.data.groupby("marker"):
						if markerName in self.mainCollecion:
							self.mainCollecion[markerName].set_facecolor(markerData["color"].values)
				
				else:
					self.mainCollecion.set_facecolor(self.data["color"].values)

	def setNaNColorToCollection(self):
		""
		self.updateColorData(self.nanScatterColor)

	def updateSizeData(self,sizeData,setSizeToCollection = True):
		""
		self.data.loc[self.data.index,"size"] = sizeData
		if setSizeToCollection:
			if isinstance(self.mainCollecion,dict) and "marker" in self.data.columns:
					for markerName, markerData in self.data.groupby("marker"):
						if markerName in self.mainCollecion:
							self.mainCollecion[markerName].set_sizes(markerData["size"].values)
			else:
				self.mainCollecion.set_sizes(self.data["size"].values)

	def replotCollection(self, updateLayer = True):
		""
		if updateLayer and "layer" in self.data.columns:
			self.data = self.data.sort_values('layer', kind="mergesort" ,ascending = True)
			#update maskIndex
			self.maskIndex = self.data.index
		self.updateDataWithProps()#why was th
		#self.updateScatterCollection()
		
		
		#self.replot(ax = self.ax, **self.scatterKwargs)
		#self.updateScatterCollection()
		## we need to replot this, otherwise the layer/order cannot be changed. 
		
		self.removeMainCollection()
		if "marker" in self.data.columns:
			self._plotMarkerSpecScatter()

		else:
			defaultKwargs = self.scatterKwargs.copy()
			kwargs = defaultKwargs.copy()
			kwargs["s"] = self.data['size'].values
			kwargs["color"] = self.data['color'].values
			self.mainCollecion = self.ax.scatter(
						x = self.data[self.numericColumns[0]],
						y = self.data[self.numericColumns[1]], 
						**kwargs)

	def _plotMarkerSpecScatter(self):
		""
		defaultKwargs = self.scatterKwargs.copy()
		self.mainCollecion = {}
		for groupName, groupData in self.data.groupby("marker"):
			#print(data)
			kwargs = defaultKwargs.copy()
			kwargs["marker"] = groupName
			kwargs["s"] = groupData['size'].values
			kwargs["color"] = groupData['color'].values
			if groupName == "+":
				kwargs["linewidth"] = 2
				#kwargs["edgecolor"] = "black"
			scatterCollection = self.ax.scatter(
						x = groupData[self.numericColumns[0]],
						y = groupData[self.numericColumns[1]], 
						**kwargs)
			self.mainCollecion[groupName] = scatterCollection #save  marker

	def removeMainCollection(self):
		""
		if isinstance(self.mainCollecion,dict):
			for pathCollection in list(self.mainCollecion.values()):
				pathCollection.remove() 
			self.mainCollecion.clear() 
		else:
			self.mainCollecion.remove()
		
																	  
	def get_size_interval(self):
		'''
		'''
		self.minSize, self.maxSize = self.plotter.get_size_interval()
	
	def getXYData(self):
		""
		if not self.multiScatter:
			return self.data[self.numericColumns].values

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

	def replot(self, ax, **kwargs):
		'''
		'''
		if self.multiScatter and len(self.numericColumns) > 2:
			
			self.mainCollecion = {}
			numericColumnPairs = list(zip(self.numericColumns[0::2], self.numericColumns[1::2]))
			
			for xName, yName in numericColumnPairs:
				if len(self.multiScatterKwargs) != 0 and (xName,yName) in self.multiScatterKwargs:
					for k,v in self.multiScatterKwargs[(xName,yName)].items():
						kwargs[k] = v 
				self.mainCollecion[(xName,yName)] = ax.scatter(self.data[xName], self.data[yName], **kwargs) #save collection
		elif "marker" in self.data.columns:
			self._plotMarkerSpecScatter()
		else:
			self.mainCollecion = ax.scatter(self.data[self.numericColumns[0]].values,
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


	def addTooltip(self, toolTipData):
		'''
		'''
		self.toolTipsActive = True
		columnsNotInData = [x for x in toolTipData.columns if x not in self.data.columns]
		self.annotationColumns = toolTipData.columns.values.tolist()
	
		if len(columnsNotInData) != 0:
			self.data = self.data.join(toolTipData[columnsNotInData])
		
		
	def buildTooltip(self):
		'''
		'''
		self.tooltip = self.ax.text(s ='', bbox=self.bboxProps,**self.textProps)
		self.textProps['text'] = ''

	def removeTooltip(self):
		'''
		Destroy tooltip text
		'''		
		if hasattr(self,'tooltip'):
			self.toolTipsActive = False	
	
	def determinePosition(self,x,y):
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
		

	def defineBbox(self):
		'''
		Define bbox
		'''
		self.bboxProps = {'facecolor':'white', 'alpha':0.85,
						 'edgecolor':'darkgrey','fill':True,
						 }
	
	def defineText(self):
		'''
		Define text properties
		'''
		self.textProps = self.parent.getStdTextProps()
				
	def updateTooltipPosition(self,event,text):
		'''
		'''
		# get event data
		x,y = event.xdata, event.ydata
		## check if new text 
		self.textProps['text'] = text	
		self.textProps['visible'] = True
		self.determinePosition(x,y)
		self.tooltip.update(self.textProps)

	def set_nan_color(self,newColor = None):
		'''
		'''
		self.scatterKwargs['color'] = newColor
		#self.nanScatterColor = self.scatterKwargs['color']
		self.mainCollecion.set_facecolor(self.scatterKwargs['color'])
		
		#self.ignoreYlimChange = ignoreYlimChange
		
		#self.defineVariables()
		#self.get_size_interval()
		

	def setColorForMultiScatter(self, columnPair, idx, color):
		""
		colorColumnName = "color({}:{})".format(*columnPair)
		if colorColumnName in self.data.columns:
			self.data.drop(colorColumnName,inplace=True,axis=1)
		if isinstance(color,str):
			colorValues = pd.Series([color] * idx.size, index=idx, name = colorColumnName)
		elif isinstance(color,pd.Series):
			idxInters = self.data.intersection(color.index)
			if idxInters.size > 0:
				colorValues = color.loc[idxInters]
		else:
			print("Incorrect data type of color")
			return

		self.data = self.data.join(colorValues)
		self.mainCollecion[columnPair].set_facecolor(self.data.loc[self.maskIndex,colorColumnName].dropna().values)
	

	def setSizesForMultiScatter(self, columnPair, idx, size):
		""
	
		sizeColumnName = "size({}:{})".format(*columnPair)
		if sizeColumnName in self.data.columns:
			self.data.drop(sizeColumnName,inplace=True,axis=1)
		
		if isinstance(size,pd.Series):
			idxInters = self.data.intersection(size.index)
			if idxInters.size > 0:
				sizeValues = size.loc[idxInters]
		elif isinstance(size,np.int64) or isinstance(size,np.float64):
			sizeValues = pd.Series([float(size)] * idx.size, index=idx, name = sizeColumnName)
		else:
			print("Incorrect data type of size")
			return

		self.data = self.data.join(sizeValues)
		self.mainCollecion[columnPair].set_sizes(self.data.loc[self.maskIndex,sizeColumnName].dropna().values)

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
		

		
		