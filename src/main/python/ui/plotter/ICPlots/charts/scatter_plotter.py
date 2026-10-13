

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
						ax = "", dataID = "", scatterKwargs = None, hoverKwargs = None,
						showLegend = True, ignoreYlimChange = False, multiScatter = False,interactive = True, multiScatterKwargs = dict(), adjustLimits = True):
		'''
		Class to handle scatter plots. Scatter plots that are generated together
		are connected and allow to hover over points to see the location of each
		data entry row in each plot. 
		Additional information of data can be added due to additional color and size levels.
		'''
		
		if scatterKwargs is None:
		    scatterKwargs = {}
		if hoverKwargs is None:
		    hoverKwargs = {}
		self.parent = parent
		self.ax = ax
		self.plotter = plotter
		self.dataID = dataID
		self.numericColumns = numericColumns
		#Item 52: take a COPY of the caller's scatterKwargs instead of holding a
		#reference to it. addKwargsToData() below writes into self.scatterKwargs
		#("s" and "color"), and every caller passes the SAME dict object to each
		#scatterPlot it creates (e.g. ICScatter.initScatterPlots passes
		#self.scatterKwargs both when building the on-screen plot and when
		#exporting to a main-figure panel). Sharing it meant the export - whose
		#data carries sizes already multiplied by scaleFactor - overwrote the
		#SOURCE plot's stored size, so each export shrank the points again from an
		#already-shrunk baseline. That is the item 20 defect returning by a second
		#route: item 20 copied the DataFrame, but the kwargs dict was still shared.
		self.scatterKwargs = dict(scatterKwargs)
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
			self.addQuickSelectScatter()

		
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

	def addQuickSelectScatter(self):
		""
		qsScatterKwargs = self.hoverKwargs.copy()
		qsScatterKwargs["marker"] = self.scatterKwargs["marker"]
		self.quickSelectScatter = self.ax.scatter([],[], **qsScatterKwargs)
		

	def setQuickSelectScatterData(self,dataIndex,propsData):
		"Update Quick Slect Data - called if new item was selected in QS"
		if self.multiScatter:			
		
			selectedData = self.data.loc[self.data.index.intersection(dataIndex)]
			coords = np.concatenate([selectedData[list(columnPair)].values for columnPair in self.multiScatterKwargs.keys()])
			scatterColors = np.tile(propsData.loc[dataIndex,"color"].values,len(self.multiScatterKwargs))
			scatterSizes = np.tile(propsData.loc[dataIndex,"size"].values,len(self.multiScatterKwargs))

		else:
			coords = self.data[self.numericColumns].loc[dataIndex].values
			scatterColors = propsData["color"]
			scatterSizes = propsData["size"]

		self.quickSelectScatter.set_offsets(coords)
		self.quickSelectScatter.set_zorder(1e10)
		self.quickSelectScatter.set_sizes(scatterSizes)
		self.quickSelectScatter.set_facecolor(scatterColors)
		self.quickSelectScatter.set_visible(True)


	def updateQuickSelectScatter(self,scatterColors,scatterSizes):
		""
		if hasattr(self,"quickSelectScatter"):
			if self.multiScatter:
				scatterColors = np.tile(scatterColors,len(self.multiScatterKwargs))
				scatterSizes = np.tile(scatterSizes,len(self.multiScatterKwargs))
			
			self.quickSelectScatter.set_sizes(scatterSizes)
			self.quickSelectScatter.set_facecolor(scatterColors)
			self.quickSelectScatter.set_visible(True)

	def setQuickSelectScatterInivisible(self):
		""
		if hasattr(self,"quickSelectScatter"):
			self.quickSelectScatter.set_visible(False)
	
	def getQuickSelectScatterPropsForExport(self):
		""
		if hasattr(self,"quickSelectScatter"):

			coords = self.quickSelectScatter.get_offsets()
			scatterColors = self.quickSelectScatter.get_facecolors()
			scatterSizes = self.quickSelectScatter.get_sizes()
			return coords,scatterColors,scatterSizes

		return [],[],[]

	

			
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
		# if hasattr(self,"quickSelectScatter"):
		# 	if self.quickSelectScatter.get_visible():
		# 		self.quickSelectScatter.set_visible(False)

		if self.interactive and hasattr(self,"hoverScatter") and self.hoverScatter.get_visible():
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
		
		##ugly code restylye
		if self.multiScatter:
			xaxisData = self.data[self.numericColumns[0::2]].values 
			yaxisData = self.data[self.numericColumns[1::2]].values
			xMin, xMax = np.nanquantile(xaxisData, q=[0,1])
			yMin, yMax = np.nanquantile(yaxisData, q=[0,1])
		else:
			with pd.option_context('mode.use_inf_as_null', True):
				nonNaNData = self.data[self.numericColumns].dropna()
			xMin, yMin = nonNaNData[self.numericColumns].min()
			xMax, yMax = nonNaNData[self.numericColumns].max()
		
		
		xAdd = np.sqrt(xMax**2 - xMin**2) * 0.05
		yAdd = np.sqrt(yMax**2 - yMin**2) * 0.05
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
		

	def add_hover_point(self, addRectangle = True):
		'''
		'''

		if addRectangle:
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

	def getScatterInvisibility(self):
		""
		if isinstance(self.mainCollecion, dict):
			for scatterCollection in self.mainCollecion.values():
				return scatterCollection.get_visible()
		else:
			return self.mainCollecion.get_visible()

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
		if self.interactive and self.selectRectangle.get_visible():
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
				maxAnnotationLength = self.parent.mC.config.getParam("annotate.max.length")
				idxT = idx if idx.size * nAnnotColumns <= 14 else idx.values[:int(14/nAnnotColumns)]
				self.updateTooltipPosition(event,"\n".join([str(x) if len(str(x)) < maxAnnotationLength else "{}..".format(str(x)[:maxAnnotationLength]) for x in self.data.loc[idxT,self.annotationColumns].values.flatten()]))
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
		if not self.interactive:
			return
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
			self.hoverScatter.set_visible(True)
			try:
				self.ax.draw_artist(self.hoverScatter)
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

	def updateScatterProps(self,propsData,updatableProps = None):
		""
		if updatableProps is None:
		    updatableProps = ["color","size","layer","marker"]
		self.setHoverPointsInvisible()
		self.setSelecRectInvisible()
		columnsPresent = [x for x in  updatableProps if x in propsData.columns]
		columnsInData = [x for x in updatableProps if x in self.data.columns and x in columnsPresent]
		#delete columns if they exists
		self.data = self.data.drop(columnsInData,axis=1) 
		
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
	
		if isinstance(self.mainCollecion,dict):
			if "marker" in self.data.columns:
				for markerName, markerData in self.data.groupby("marker"):
					if markerName in self.mainCollecion:
						maskIndex = markerData.index.intersection(self.maskIndex)
						#item 63: stacked over all column pairs (multi-collection scatters)
						offsets, colors, sizes = self._getMarkerGroupArrays(markerData.loc[maskIndex])
						self.mainCollecion[markerName].set_offsets(offsets)
						if "color" in markerData.columns:
							self.mainCollecion[markerName].set_facecolor(colors)
						if "size" in markerData.columns:
							self.mainCollecion[markerName].set_sizes(sizes)
			else:
				maskIndex = self.data.index.intersection(self.maskIndex)
				for scatterPathCollection in self.mainCollecion.values():
					if "color" in self.data.columns:
						scatterPathCollection.set_facecolor(self.data.loc[maskIndex,"color"].values)
					
					if "size" in self.data.columns:
						scatterPathCollection.set_sizes(self.data.loc[maskIndex,"size"].values)
					
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


	def _getRowPropsFromPairCollections(self, propName):
		"""Item 63: per-row color/size of a multi-collection scatter keyed by x/y
		column pairs. Each row takes the value of the collection of the pair it
		belongs to (the pair whose columns are not NaN in that row), instead of
		every row taking the value of the first collection. Returns None if the
		collections are not keyed by column pairs."""
		if not isinstance(self.mainCollecion,dict) or len(self.mainCollecion) == 0:
			return None
		if not all(isinstance(k,tuple) and len(k) == 2 and all(c in self.data.columns for c in k) for k in self.mainCollecion.keys()):
			return None
		values = pd.Series([None] * self.data.index.size, index=self.data.index, dtype=object)
		for (xName, yName), collection in self.mainCollecion.items():
			collectionValues = collection.get_facecolors() if propName == "color" else collection.get_sizes()
			if len(collectionValues) == 0:
				continue
			rowMask = self.data[[xName, yName]].notna().all(axis=1).values
			if len(collectionValues) == self.data.index.size:
				for n in np.flatnonzero(rowMask):
					values.iloc[n] = collectionValues[n]
			else:
				for n in np.flatnonzero(rowMask):
					values.iloc[n] = collectionValues[0]
		#rows outside every pair are not drawn; give them the first value
		firstValue = next((v for v in values.values if v is not None), None)
		if firstValue is None:
			return None
		return values.apply(lambda v: firstValue if v is None else v)

	def updateDataWithProps(self):
		""
		
		if isinstance(self.mainCollecion,dict):
			firstKey = list(self.mainCollecion.keys())[0]
			scatterPathCollection = self.mainCollecion[firstKey]
		else:
			scatterPathCollection = self.mainCollecion
		
		if "color" not in self.data.columns:
			pairColors = self._getRowPropsFromPairCollections("color")
			if pairColors is not None:
				self.data.loc[:,"color"] = pairColors
		if "size" not in self.data.columns:
			pairSizes = self._getRowPropsFromPairCollections("size")
			if pairSizes is not None:
				self.data.loc[:,"size"] = pairSizes
		if "color" not in self.data.columns:
			self.data.loc[:,"color"] = pd.Series([scatterPathCollection.get_facecolors()[0]] * self.data.index.size, index=self.data.index)
		if "size" not in self.data.columns:
			self.data.loc[:,"size"] = pd.Series([scatterPathCollection.get_sizes()[0]] * self.data.index.size, index=self.data.index)

	def updateColorData(self,colorData,setColorToCollection = True):
		""
		if isinstance(colorData,str):
			
			self.data.loc[self.data.index,"color"] = colorData
			if setColorToCollection:
				self.addColorToCollection()

	def addColorToCollection(self):
		""
		if isinstance(self.mainCollecion,dict) and "marker" in self.data.columns:
			for markerName, markerData in self.data.groupby("marker"):
				if markerName in self.mainCollecion:
					self.mainCollecion[markerName].set_facecolor(self._getMarkerGroupArrays(markerData)[1]) #item 63
		elif isinstance(self.mainCollecion,dict):
			for scatterPathCollection in self.mainCollecion.values():
				scatterPathCollection.set_facecolor(self.data.loc[:,"color"].values)
		else:
			self.mainCollecion.set_facecolor(self.data["color"].values)

	def updateColorDataByIndex(self,idx,colorValue):
		""
		idxIntersection = self.data.index.intersection(idx)
		if idxIntersection.size > 0:
			self.data.loc[idxIntersection,"color"] = colorValue
			self.addColorToCollection()

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
							self.mainCollecion[markerName].set_sizes(self._getMarkerGroupArrays(markerData)[2]) #item 63

			elif isinstance(self.mainCollecion,dict):
					for scatterPathCollection in self.mainCollecion.values():
						scatterPathCollection.set_facecolor(self.data.loc[:,"size"].values)
			else:
				self.mainCollecion.set_sizes(self.data["size"].values)

	def replotCollection(self, updateLayer = True):
		""
		if updateLayer and "layer" in self.data.columns:
			self.data = self.data.sort_values('layer', kind="mergesort" ,ascending = True)
			#update maskIndex
			self.maskIndex = self.data.index
		self.updateDataWithProps()#why was th

		## we need to replot this, otherwise the layer/order cannot be changed. 
		self.removeMainCollection()
		if "marker" in self.data.columns:
			self._plotMarkerSpecScatter()
			return

		if isinstance(self.mainCollecion,dict):

				numericColumnPairs = list(zip(self.numericColumns[0::2], self.numericColumns[1::2]))
				defaultKwargs = self.scatterKwargs.copy()
			
				for xName, yName in numericColumnPairs:
					kwargs = defaultKwargs.copy()
					# if len(self.multiScatterKwargs) != 0 and (xName,yName) in self.multiScatterKwargs:
					# 	for k,v in self.multiScatterKwargs[(xName,yName)].items():
					# 		kwargs[k] = v 
					kwargs["s"] = self.data['size'].values
					kwargs["color"] = self.data['color'].values
					self.mainCollecion[(xName,yName)] = self.ax.scatter(self.data[xName], self.data[yName], **kwargs) #save collection

		else:
			defaultKwargs = self.scatterKwargs.copy()
			kwargs = defaultKwargs.copy()
			kwargs["s"] = self.data['size'].values
			kwargs["color"] = self.data['color'].values
			self.mainCollecion = self.ax.scatter(
						x = self.data[self.numericColumns[0]],
						y = self.data[self.numericColumns[1]], 
						**kwargs)

	#Item 63: marker shapes for multi-collection scatters (swarm on box/violin/bar
	#plots and swarm plots). Marker drawing used numericColumns[0] and [1] only,
	#i.e. the FIRST x/y column pair. A multi-collection scatter holds one x/y
	#pair per category, so all other categories would have lost their points.
	#The helpers below stack every pair, so each marker group is still ONE
	#collection (keyed by marker name, as before) but contains the points of
	#all pairs. For an ordinary scatter (exactly one pair) the result is
	#identical to the previous code.
	def _getMarkerColumnPairs(self):
		"x/y column pairs drawn by this scatterPlot."
		if len(self.numericColumns) > 2:
			return list(zip(self.numericColumns[0::2], self.numericColumns[1::2]))
		return [(self.numericColumns[0], self.numericColumns[1])]

	def _getPairPropArray(self, markerData, columnPair, propName, kwargsName, baseKwargs):
		"Per-row values of color/size for one marker group and one column pair."
		n = markerData.index.size
		if propName in markerData.columns:
			return list(markerData[propName].values)
		value = None
		if isinstance(self.multiScatterKwargs,dict) and columnPair in self.multiScatterKwargs:
			value = self.multiScatterKwargs[columnPair].get(kwargsName)
		if value is None and isinstance(baseKwargs,dict):
			value = baseKwargs.get(kwargsName)
		if isinstance(value,(list,tuple,np.ndarray,pd.Series)) and not isinstance(value,str):
			if isinstance(value,tuple) and len(value) in (3,4) and all(isinstance(x,(int,float,np.floating)) for x in value):
				return [value] * n #a single RGB(A) tuple
			if len(value) == self.data.index.size:
				return list(pd.Series(list(value), index=self.data.index).loc[markerData.index].values)
			if len(value) > 0:
				return [value[0]] * n
			value = None
		return [value] * n

	def _getMarkerGroupArrays(self, markerData, baseKwargs = None):
		"Offsets, colors and sizes of one marker group, stacked over all column pairs."
		if baseKwargs is None:
			baseKwargs = self.scatterKwargs
		columnPairs = self._getMarkerColumnPairs()
		offsets = np.vstack([markerData[[xName,yName]].values.astype(float) for xName, yName in columnPairs]) \
					if markerData.index.size > 0 else np.empty((0,2))
		colors, sizes = [], []
		for columnPair in columnPairs:
			colors.extend(self._getPairPropArray(markerData, columnPair, "color", "color", baseKwargs))
			sizes.extend(self._getPairPropArray(markerData, columnPair, "size", "s", baseKwargs))
		return offsets, colors, sizes

	def _plotMarkerSpecScatter(self, ax = None, baseKwargs = None):
		""
		if ax is None:
			ax = self.ax
		defaultKwargs = (self.scatterKwargs if baseKwargs is None else baseKwargs).copy()
		self.mainCollecion = {}
		for groupName, groupData in self.data.groupby("marker"):
			kwargs = defaultKwargs.copy()
			offsets, colors, sizes = self._getMarkerGroupArrays(groupData, defaultKwargs)
			kwargs["marker"] = groupName
			kwargs["s"] = sizes
			kwargs["color"] = colors
			if groupName == "+":
				kwargs["linewidth"] = 2
				#kwargs["edgecolor"] = "black"
			scatterCollection = ax.scatter(
						x = offsets[:,0],
						y = offsets[:,1], 
						**kwargs)
			self.mainCollecion[groupName] = scatterCollection #save  marker

	def removeMainCollection(self):
		""
		if isinstance(self.mainCollecion,dict):
			for pathCollection in self.mainCollecion.values():
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
		if "marker" in self.data.columns:
			#item 63: marker shapes must also be kept when a multi-collection
			#scatter is (re)built, e.g. on export to a main-figure panel; the
			#multi-collection branch below draws every point with one marker.
			self._plotMarkerSpecScatter(ax = ax, baseKwargs = kwargs)
		elif self.multiScatter and len(self.numericColumns) > 2:
			
			self.mainCollecion = {}
			numericColumnPairs = list(zip(self.numericColumns[0::2], self.numericColumns[1::2]))
			
			#Item 54: a live per-row "color" column must win over the per-columnPair
			#default in multiScatterKwargs. That dict is the ONE-TIME map the backend
			#builds at chart construction (one color per category group), whereas
			#dropping a column on color marks writes a per-ROW "color" column onto
			#this scatterPlot's own data. replotCollection() - the on-screen redraw
			#after a color change - already prefers the live column (its
			#multiScatterKwargs override is commented out and it assigns
			#kwargs["color"] = self.data['color'].values). replot() did not, and
			#replot() is what runs when a scatterPlot is CONSTRUCTED - i.e. on every
			#export to a main-figure panel. So the exported panel was recolored by
			#category group, discarding the color-mark encoding, while the on-screen
			#plot and the legend stayed correct.
			liveColor = self.data["color"].values if "color" in self.data.columns else None
			for xName, yName in numericColumnPairs:
				#copy per pair: the original assigned into the shared kwargs dict, so a
				#pair with no entry of its own silently inherited the previous pair's color.
				pairKwargs = dict(kwargs)
				if len(self.multiScatterKwargs) != 0 and (xName,yName) in self.multiScatterKwargs:
					for k,v in self.multiScatterKwargs[(xName,yName)].items():
						pairKwargs[k] = v 
				#a per-pair color that is already an ARRAY comes from a live per-group
				#recoloring (ICSwarmplot.getLiveMultiScatterKwargs) and is kept; only a
				#single stale color is replaced by the live per-row column.
				if liveColor is not None and not isinstance(pairKwargs.get("color"),(list,tuple,np.ndarray,pd.Series)):
					pairKwargs["color"] = liveColor
				self.mainCollecion[(xName,yName)] = ax.scatter(self.data[xName], self.data[yName], **pairKwargs) #save collection
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
		
	def hasToolTip(self):
		""
		return self.toolTipsActive
	
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
		

		
		