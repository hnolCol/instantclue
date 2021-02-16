from PyQt5.QtCore import QObject, pyqtSignal, QPoint
from PyQt5.QtGui import QCursor

from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from matplotlib.collections import PathCollection
from matplotlib.pyplot import scatter
from matplotlib.text import Text
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

from collections import OrderedDict
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties

#external imports
from backend.utils.stringOperations import getRandomString
from ...utils import INSTANT_CLUE_BLUE, createSubMenu
from .ICScatterAnnotations import find_nearest, find_nearest_index, xLim_and_yLim_delta
from ...custom.warnMessage import WarningMessage

from .charts.scatter_plotter import scatterPlot

import requests


class ICChart(QObject):

	#data changed signal
	dataLoaded = pyqtSignal(dict)
	updateFigure = pyqtSignal()
	clearFigure = pyqtSignal()

	def __init__(self, mainController, icPlotter, plotType, interactive = True):
		""
		QObject.__init__(self)

		self.mC = mainController
		self.p = icPlotter
		self.plotType = plotType

		self.interactive = interactive

		self.axisDict = dict() 
		self.tooltips = dict()
		self.extraArtists = OrderedDict()
		self.hoverChanged = False
		self.colorCategoryIndexMatch = None
		self.sizeCategoryIndexMatch = None
		self.quickSelectCategoryIndexMatch = None
		self.tooltipActive = False
		self.statTestEnabled = False
		self.statData = dict()
		self.quickSelectScatterDataIdx = dict()
		self.requiredKwargs = []
		
		self.saveStatTests = OrderedDict() 
		self.preventQuickSelectCapture = False

		self.dataLoaded.connect(self.onDataLoad)
		self.updateFigure.connect(self.update)
		self.clearFigure.connect(self.clear)

		if interactive:
			self.addPressBinding()

        
	def addAnnotations(self,columnNames, dataID):
		""
		self.annotations = None

	def addAxisWithTitle(self,
						ax,
						appendWhere = "top", 
						title = "Axis title", 
						axisSize = 0.25, 
						axisPadding=0, 
						textSize = 9,  
						textRotation = 90):
		"""
		Source:
		https://stackoverflow.com/questions/40796117/how-do-i-make-the-width-of-the-title-box-span-the-entire-plot
		"""
		divider = make_axes_locatable(ax)
		cax = divider.append_axes(appendWhere, size=axisSize, pad=axisPadding)
		cax.get_xaxis().set_visible(False)
		cax.get_yaxis().set_visible(False)
		cax.set_facecolor(self.getParam("axis.title.box.background"))
		at = AnchoredText(
							title, 
							loc=10,
							frameon = False,
                  			prop=dict(
									backgroundcolor=(0,0,0,0), #transparent backgroundcolor
									size=textSize, 
									color="black", 
									rotation = textRotation,
									multialignment = "center"
									)
						)

		cax.add_artist(at)

	def addHoverBinding(self):
		""
		if self.interactive:
			self.onHoverEvent = self.p.f.canvas.mpl_connect('motion_notify_event', self.onHover)

	def addPressBinding(self):
		""
		if self.interactive:
			self.onPressEvent = self.p.f.canvas.mpl_connect('button_release_event', self.onPress)

	def addTitles(self, fancyTitle = True, onlyForID = None, targetAx = None, *args, **kwargs):
		""
		if "axisTitles" in self.data and len(self.data["axisTitles"]) > 0:
			for n,ax in self.axisDict.items():
				if onlyForID is not None and targetAx is not None:
					if n == onlyForID:
						ax = targetAx
					else:
						continue
				if n in self.data["axisTitles"]:
					if isinstance(self.data["axisTitles"][n],dict):

						self.addAxisWithTitle(ax,**self.data["axisTitles"][n])
					
					elif isinstance(self.data["axisTitles"][n],list):
						for titleProps in self.data["axisTitles"][n]:
							self.addAxisWithTitle(ax,**titleProps)

					elif fancyTitle:
						self.addAxisWithTitle(ax,title=self.data["axisTitles"][n],textRotation=0, *args, **kwargs)
					else:
						self.setAxisTitle(ax,self.data["axisTitles"][n], *args, **kwargs)


	def annotateDataByIndex(self,dataIndex,annotationColumn):
		""
		if isinstance(annotationColumn,str):
			annotationColumn = pd.Series([annotationColumn])
		self.addAnnotations(annotationColumn, self.mC.getDataID())
		if hasattr(self,"annotations") and self.annotations is not None and isinstance(self.annotations,dict):
			for annotationClass in self.annotations.values():
				annotationClass.addAnnotations(dataIndex)
			self.updateFigure.emit()
	
	def restoreBackgrounds(self):
		""
		if hasattr(self,"backgrounds"):
			for ax, background in self.backgrounds.items():
				self.p.f.canvas.restore_region(background)
			#self.p.f.canvas.blit(ax.bbox)


	def checkForQuickSelectDataAndUpdateFigure(self):
		""
	
		qsData = self.getQuickSelectData()
		if qsData is not None:
			self.mC.quickSelectTrigger.emit()
		else:
			self.updateFigure.emit()

	def onPress(self,event):
		""
		
		if self.mC.mainFrames["middle"].getToolbarState() is not None:
			#if zoom or pan is activated. Reset stat data
			self.statData.clear() 
			return
		if not hasattr(self,"annotations") or self.annotations is None: #menu handling is done in annotation class, if active
			if event.inaxes is not None and event.button in [2,3]:#mac windows
				self.menuClickedInAxis = event.inaxes
				self.createAndShowMenu()
			elif event.inaxes in list(self.axisDict.values()) and event.button == 1 and self.statTestEnabled and self.isBoxplotViolinBar():
				axisID = self.getAxisID(event.inaxes)	
				
				#data are provided differently for boxplot, violin and barplot due to the matplotlib functions			
				if self.isBoxplot():
					nearestIdx = find_nearest_index(self.data["plotData"][axisID]["positions"],event.xdata)
					xdata = self.data["plotData"][axisID]["positions"][nearestIdx]
					data = self.data["plotData"][axisID]["x"][nearestIdx]
				elif self.isBarplot():
					nearestIdx = find_nearest_index(self.data["plotData"][axisID]["x"],event.xdata)
					xdata = self.data["plotData"][axisID]["x"][nearestIdx]
					data = self.data["hoverData"][axisID]["x"][nearestIdx]
				elif self.isViolinplot():
					nearestIdx = find_nearest_index(self.data["plotData"][axisID]["positions"],event.xdata)
					xdata = self.data["plotData"][axisID]["positions"][nearestIdx]
					data = self.data["hoverData"][axisID]["x"][nearestIdx]
				groupName = self.data["groupNames"][axisID][nearestIdx]
				self.saveStatData(axisID,event.ydata, xdata, data, nearestIdx, groupName)
	
	
	def eventOverAnnotation(self,ax,event):
		""
		if not hasattr(self,"annotations") or self.annotations is None:
			return False
		elif isinstance(self.annotations,dict) and hasattr(self,"addedAnnotations") and ax in self.addedAnnotations:
			#for annotationClass in self.annotations.values():
				annotations = self.addedAnnotations[ax].values()
				return any(anno.contains(event)[0] for anno in annotations)
		else:
			return False

	def createAndShowMenu(self):
		"""
		Create Menu. Modidification of the function getGraphSpecMenu allows different menus for each graph.
		Absolutely no checking is done.
		"""
		graphIndendentSubMenus = self.getSubMenus()
		graphDepSubMenus = self.getGraphSpecMenus()
		
		menus = createSubMenu(subMenus=graphIndendentSubMenus + graphDepSubMenus)
		
		self.addMainFigActions(menus)
		self.addAppActions(menus)
		self.addMenuActions(menus)
		self.addGraphSpecActions(menus)
		pos = QCursor.pos()
		pos += QPoint(3,3)
		menus["main"].exec_(pos)

	def getToolbarState(self):
		"Returns tooolbar state (ZOOM;PAN)"

		return self.mC.mainFrames["middle"].getToolbarState() 

	def saveStatData(self,axisID,ydata,xdata, data,dataIdx, groupName):
		"Saves data for statistical tests"

		if len(self.statData) == 0:
			self.statData["axisID"] = axisID
			self.statData["data"] = [data]
			self.statData["idx"] = [dataIdx]
			self.statData["y"] = [ydata]
			self.statData["x"] = [xdata]
			self.statData["groupName"] = [groupName]
			self.drawStatIndicator(self.axisDict[axisID], xdata,ydata)
		
		elif "axisID" in self.statData and self.statData["axisID"] != axisID:
			self.statData.clear() 
			self.setStatIndicatorIvisible(self.axisDict[axisID])
			
		elif "idx" in self.statData and self.statData["idx"] == dataIdx:
			self.statData.clear() 
			self.setStatIndicatorIvisible(self.axisDict[axisID])
			self.updateFigure.emit()
			w = WarningMessage(infoText="Same data selected. Selection reset.",iconDir = self.mC.mainPath)
			w.exec_()
			return
			
		else:
			self.statData["data"].append(data)
			self.statData["idx"].append(dataIdx)
			self.statData["y"].append(ydata)
			self.statData["x"].append(xdata)
			self.statData["groupName"].append(groupName)
			if self.checkDublicateTests():
				stat, p, testType = self.performStatTest()
				if p is not None:
					testType, internalID = self.saveStatResults(stat,p,testType)
					self.drawStats(self.axisDict[axisID],p,internalID,axisID) #draw test 
					#show data in statitic table
					self.setDataInStatisticTable(self.statCollection,title = "Test : {}".format(testType))
			self.statData.clear() 
		
		self.updateFigure.emit()

	def drawStatIndicator(self,ax,xdata,ydata):
		""
		if not hasattr(self,"statIndicatorLine") or ax not in self.statIndicatorLine:
			self.statIndicatorLine = {}
			self.statIndicatorLine[ax] = ax.plot([xdata],[ydata],marker="x",markeredgewidth=2, markeredgecolor="red")[0]
		else:
			self.statIndicatorLine[ax].set_data([xdata],[ydata])
			self.statIndicatorLine[ax].set_visible(True)

	def setStatIndicatorIvisible(self,ax):
		""
		if hasattr(self,"statIndicatorLine"):
			self.statIndicatorLine[ax].set_visible(False)


	def checkDublicateTests(self):
		""
		if not hasattr(self,"statCollection"):
			return True
		group1, group2 = self.statData["groupName"]
		statGroupByGroups = self.statCollection.groupby(["Group1","Group2"]).groups
		if (group1,group2) in statGroupByGroups or (group2,group1) in statGroupByGroups:
			w = WarningMessage(infoText = "Comparision ({} vs {}) exists already.".format(group1,group2))
			w.exec_()
			return False
		return True
	
	def saveStatResults(self,statValue, pValue, testType):
		"saves data to statCollection"
		if not hasattr(self,"statCollection"):
			self.statCollection = pd.DataFrame() 
		internalID = getRandomString()
		dataToAppend = {
						"Test":testType,
						"Group1":self.statData["groupName"][0], 
						"Group2": self.statData["groupName"][1],
						"p-value":pValue,
						"test-statistic":statValue,
						"internalID":internalID,
						"visible" : True
						}
		self.statCollection = self.statCollection.append(dataToAppend, ignore_index = True)
		return testType, internalID

	def performStatTest(self):
		""
		try:
			testType = self.mC.mainFrames["data"].analysisSelection.getDragTask().values[0]
			r = self.mC.statCenter.performPairwiseTest(self.statData["data"], kind = testType)
			if isinstance(r,str):
				w = WarningMessage(infoText=r)
				w.exec_()
				return None, None, None
			else:
				s,p = r #s = test statistic 
			return s,p, testType
			#self.saveStatData()
		except Exception as e:
			print(e)

	def drawStats(self, ax, p , internalID, axisID):
		""
		self.setStatIndicatorIvisible(ax)
		lineCoords, midXPoint, maxYPoint = self.getLineCoordsForStats(ax)
		if lineCoords is not None:
			#setup line
			lineProps = {"xdata": lineCoords[:,0], "ydata": lineCoords[:,1],"lw":0.75, "color":"darkgrey"}
			lineArtist = Line2D(**lineProps)
			#setup text props for p value
			tProps = self.getStdTextProps()
			tProps["x"] = midXPoint
			tProps["y"] = maxYPoint 
			tProps["text"] = "{:.4f}".format(p) if p > 0.001 else "{:.2e}".format(p) #p value formatting
			tProps["ha"] = "center"
			tProps["va"] = "bottom"
			tProps["visible"] = True
			
			#setup text
			t = Text(**tProps)

			#add artists
			ax.add_artist(lineArtist)
			ax.add_artist(t)
			if not hasattr(self,"statArtists"):

				self.statArtists = OrderedDict() 

			self.statArtists[internalID] = {
											"line" : lineArtist , 
											"text" : t,
											"axisID" : axisID,
											"lineProps" : lineProps,
											"textProps" : tProps
											}

	def toggleStatVisibilityByInternalID(self,internalID):
		"Toggles visibility of statistcs (line and p value)"
		if self.setStatArtistsVisibility(internalID):
			self.updateFigure.emit() 
		

	def removeStatsArtistsByInternalID(self,internalID):
		"Removes artits by making them invisible and removes them from statCollection - updates table in slice and marks frame"
		if self.setStatArtistsVisibility(internalID,False):
			toRemoveBoolIdx = self.statCollection["internalID"] == internalID
			self.statCollection = self.statCollection.loc[~toRemoveBoolIdx,:]
			self.setDataInStatisticTable(self.statCollection,"Test : {}".format(self.mC.mainFrames["data"].analysisSelection.getDragTask().values[0]))
			self.updateFigure.emit()

	def setStatArtistsVisibility(self,internalID,visible=None):
		"Sets artitsti visibility and returns true/false if interalID was found."
		if hasattr(self,"statArtists") and internalID in self.statArtists:
			artists = list(self.statArtists[internalID].values()) #get artists by internalID - text and line
			if visible is None or not isinstance(visible,bool):
				visible = not artists[0].get_visible() #toggle for visibilty
			for artist in artists:
				if hasattr(artist,"set_visible"):
					artist.set_visible(visible)
			try:
				boolIdx = self.statCollection["internalID"] == internalID
			
				self.statCollection.loc[boolIdx,"visible"] = visible
		
				self.setDataInStatisticTable(self.statCollection,"Test : {}".format(self.mC.mainFrames["data"].analysisSelection.getDragTask().values[0]))

			except Exception as e:
				print(e)
			return True
		return False 

	def getLineCoordsForStats(self,ax):
		""
		if hasattr(self,"statData"):
			x0,x1 = self.statData["x"]
			y0,y1 = self.statData["y"]
			yDelta = xLim_and_yLim_delta(ax)[0]
			yMax = max(y0,y1) + 0.05 * yDelta
			axisLimitWithBorder = yMax + 0.15 * yDelta
			if ax.get_ylim()[1] < axisLimitWithBorder:
				ax.set_ylim(ax.get_ylim()[0],axisLimitWithBorder)
			
			return np.array([(x0,y0),(x0,yMax),(x1,yMax),(x1,y1)]) , sum([x0,x1]) / 2, yMax

		return None, np.nan, np.nan


	def getAxisID(self,targetAxis):
		""
		for ID,ax in self.axisDict.items():
			if ax == targetAxis:
				return ID 

	def getGraphSpecMenus(self):
		""
		return []
	
	def getSubMenus(self):
		""
		return ["To main figure","To WebApp"]

	def addMainFigActions(self,menu):
		""
		try:
			mainFigMenus = self.mC.mainFrames["right"].mainFigureRegistry.getMenu(menu["To main figure"],self.mirrorAxis)
			 
		except Exception as e:
			print(e)

	def addAppActions(self,menus):
		""
		loggedIn, userProjects = self.mC.getUserLoginInfo()
		if loggedIn and "To WebApp" in menus:
			for project in userProjects:
				projectName = project["name"]
				action = menus["To WebApp"].addAction(projectName)
				action.triggered.connect(lambda chk, projectParams = project: self.sendToWebApp(projectParams))
		else:
			menus["To WebApp"].addAction("Login")

	def sendToWebApp(self, projectParams):
		""
		
		self.mC.sendTextEntryToWebApp(projectParams["ID"],"# Start","## Q: I would like to answer the question of how mitoch. are regulated.")
		# d = self.data.copy() 

		
		# URL = "http://127.0.0.1:5000/api/v1/projects"
		# r = requests.post(URL,json=projectParams)
		# print(r) 

	def addMenuActions(self, menus):
		""
		
		if hasattr(self,"colorLegend"):
			menus["main"].addAction("Remove color legend", self.removeColorLegend)
		if hasattr(self,"sizeLegend"):
			menus["main"].addAction("Remove size legend", self.removeSizeLegend)
		if hasattr(self,"markerLegend"):
			menus["main"].addAction("Remove marker legend", self.removeMarkerLegend)

	def addQuickSelectHoverScatter(self):
		""
		self.quickSelectScatter = {}
		for ax in self.axisDict.values():
			self.quickSelectScatter[ax] = ax.scatter(x=[],y=[],**self.getScatterKwargs(),zorder = 1e9)

	def removeSizeLegend(self):
		""
		self.removeLegend("sizeLegend")
		del self.sizeLegend
	
	def removeMarkerLegend(self):
		""
		self.removeLegend("markerLegend")
		del self.markerLegend

	def removeColorLegend(self):
		""
		self.removeLegend()
		del self.colorLegend
	
	def removeLegend(self,attrName = "colorLegend"):
		if hasattr(self,attrName):
			leg = getattr(self,attrName)
			leg.remove() #remove from graph
			self.updateFigure.emit()

	def addGraphSpecActions(self,menus):
		""

	def addHoverScatter(self, ax):
		""
		if not hasattr(self,"hoverScatter"):
			self.hoverScatter = dict() 

		self.hoverScatter[ax] = ax.scatter(x=[],y=[],**self.getHoverKwargs())

	def setHoverScatterData(self,coords, ax, sizes = None):
		""
		if hasattr(self,"hoverScatter") and ax in self.hoverScatter:
			hoverScatter = self.hoverScatter[ax]
			hoverScatter.set_offsets(coords)
			hoverScatter.set_visible(True)
			#add sizes
			if sizes is not None and coords.shape[0] == sizes.shape[0]:
				hoverScatter.set_sizes(sizes)
			#draw scatter
			ax.draw_artist(hoverScatter)
			#blit canvas
			self.p.f.canvas.blit(ax.bbox)
    

	def addDiagonal(self):

		for n,ax in self.axisDict.items():
			xlim = np.array(list(self.getAxisLimit(ax)))
			yData = np.array(list(self.getAxisLimit(ax,"y")))
			lc = INSTANT_CLUE_BLUE#self.mC.config.
			l = Line2D(xdata = xlim, ydata = yData, lw=0.75, color = lc)
			ax.add_artist(l)

			#save line
			lineID = getRandomString()
			self.extraArtists[lineID] = {"artist":l,"color":lc,"name":"Axis diagonal"}

	def addLinearLine(self,m=1,b=0):
		"Adds a line to each axis"
		
		for n,ax in self.axisDict.items():
			xlim = np.array(list(self.getAxisLimit(ax)))
			yData = xlim * m + b
			lc = INSTANT_CLUE_BLUE#self.mC.config.
			l = Line2D(xdata = xlim, ydata = yData, lw=0.75, color = lc)

			ax.add_artist(l)

			#save line
			lineID = getRandomString()
			self.extraArtists[lineID] = {"artist":l,"color":lc,"name":"linear line"}

	def addColorLegendToGraph(self, colorData, ignoreNaN = False,title =  None ,update=True, ax = None, export = False, legendKwargs = {}):
		""
		try:
			if colorData.index.size > 200:
				w = WarningMessage(title="To many items for legend.",
								infoText = "More than 200 items for legend which is not supported.\nYou can export the color mapping to excel instead.")
				w.exec_()	
				return			
			if ignoreNaN:
				idx = colorData["color"] != self.getParam("nanColor")
				colorData = colorData.loc[idx]
			if self.plotType in ["scatter","swarmplot","hclust"]:
				scatterKwargs = self.getScatterKwargForLegend()

				legendItems = [scatter([],[],
						color=colorData.loc[idx,"color"],
						label = colorData.loc[idx,"group"], **scatterKwargs) for idx in colorData.index]
			else:
				legendItems = [Patch(
						facecolor=colorData.loc[idx,"color"],
						edgecolor="black",
						label = colorData.loc[idx,"group"]) for idx in colorData.index]

			if title is None and "colorCategoricalColumn" in self.data:
				title = self.data["colorCategoricalColumn"]
			if ax is None:
				if self.plotType and "axLabelColor" in self.axisDict:
					ax = self.axisDict["axLabelColor"]
				else:
					ax = self.axisDict[0]
			if not export and hasattr(self,"colorLegend"):
				self.colorLegend.remove() 
			legend = ax.legend(handles=legendItems, 
								title = title, 
								title_fontsize = self.getParam("legend.title.mainfigure.fontsize") if export else self.getParam("legend.title.fontsize"), 
								fontsize = self.getParam("legend.mainfigure.fontsize") if export else self.getParam("legend.fontsize"),
								labelspacing = self.getParam("legend.label.spacing"),
								borderpad  = self.getParam("legend.label.borderpad"),
								**legendKwargs)
			if not export:
				self.colorLegend = legend
				self.checkLegend(ax,attrName="sizeLegend")
				self.checkLegend(ax,attrName="markerLegend")
				self.checkLegend(ax,attrName="quickSelectLegend")
				self.lastAddedLegend = legend
				self.colorLegendKwargs = {"colorData":colorData,"ignoreNaN":ignoreNaN,"title":title,"update":False,"export":True,"legendKwargs":legendKwargs}
				
			if update:
				self.updateFigure.emit()
			return legend
		except Exception as e:
			print(e)

	def addQuickSelectLegendToGraph(self, colorSizeData, ignoreNaN = False,title =  None ,update=True, ax = None, export = False, legendKwargs = {}):
		""
		try:
			if colorSizeData.index.size > 200:
				w = WarningMessage(title="To many items for legend.",
								infoText = "More than 200 items for legend which is not supported.\nYou can export the color mapping to excel instead.")
				w.exec_()	
				return			
			if ignoreNaN:
				idx = colorSizeData["color"] != self.getParam("nanColor")
				colorSizeData = colorSizeData.loc[idx]
			if self.plotType in ["scatter","swarmplot","hclust"]:
				scatterKwargs = self.getScatterKwargForLegend(legendType="QuickSelect")
				
				legendItems = [scatter([],[],
						color= colorSizeData.loc[idx,"color"],
						s = colorSizeData.loc[idx,"size"],
						label = colorSizeData.loc[idx,"group"], **scatterKwargs) for idx in colorSizeData.index]
			else:
				legendItems = [Patch(
						facecolor=colorSizeData.loc[idx,"color"],
						edgecolor="black",
						label = colorSizeData.loc[idx,"group"]) for idx in colorSizeData.index]

			
			title = "QuickSelect Legend"

			if ax is None:
				if self.plotType == "hclust" and "axLabelColor" in self.axisDict:
					ax = self.axisDict["axLabelColor"]
				else:
					ax = self.axisDict[0]
			if not export and hasattr(self,"quickSelectLegend"):
				self.quickSelectLegend.remove() 
			legend = ax.legend(handles=legendItems, 
								title = title, 
								title_fontsize = self.getParam("legend.title.mainfigure.fontsize") if export else self.getParam("legend.title.fontsize"), 
								fontsize = self.getParam("legend.mainfigure.fontsize") if export else self.getParam("legend.fontsize"),
								labelspacing = self.getParam("legend.label.spacing"),
								borderpad  = self.getParam("legend.label.borderpad"),
								**legendKwargs)
			if not export:
				self.quickSelectLegend = legend
				self.checkLegend(ax,attrName="sizeLegend")
				self.checkLegend(ax,attrName="markerLegend")
				self.checkLegend(ax,attrName="colorLegend")
				self.lastAddedLegend = legend
				self.quickSelectLegendKwargs = {"colorSizeData":colorSizeData,"ignoreNaN":ignoreNaN,"title":title,"update":False,"export":True,"legendKwargs":legendKwargs}
			if update:
				self.updateFigure.emit()
			return legend
		except Exception as e:
			print(e)
	
	def addMarkerLegend(self,markerData, title = "", update=True, ax = None, export = False, legendKwargs = {}):
		""

		if self.plotType in ["scatter","swarmplot"]:
			scatterKwargs = self.getScatterKwargForLegend()

			if markerData.index.size > 200:
				w = WarningMessage(title="To many items for legend.",
								infoText = "More than 200 items for legend which is not supported.")
				w.exec_()	
				return			
			if "marker" in scatterKwargs:
				del scatterKwargs["marker"]
				scatterKwargs["color"] = self.getParam("nanColor")

			legendItems = [scatter([],[],
						label = markerData.loc[idx,"group"], marker= markerData.loc[idx,"marker"], **scatterKwargs) for idx in markerData.index]
			if ax is None:
				ax = self.axisDict[0]
			if not export and hasattr(self,"markerLegend"):
				self.markerLegend.remove() 
			legend = ax.legend(
								handles=legendItems, 
								title = title, 
								title_fontsize = self.getParam("legend.title.mainfigure.fontsize") if export else self.getParam("legend.title.fontsize"), 
								fontsize = self.getParam("legend.mainfigure.fontsize") if export else self.getParam("legend.fontsize"),
								labelspacing = self.getParam("legend.label.spacing"),
								borderpad  = self.getParam("legend.label.borderpad"),
								**legendKwargs)
			if not export:
				self.markerLegend = legend
				self.checkLegend(ax,attrName="sizeLegend")
				self.checkLegend(ax,attrName="colorLegend")
				self.checkLegend(ax,attrName="quickSelectLegend")
				self.lastAddedLegend = legend
				self.markerLegendKwargs = {"markerData":markerData,"title":title,"update":False,"export":True,"legendKwargs":legendKwargs}
				
			if update:
				self.updateFigure.emit() 
			return legend

	def addVerticalLines(self, onlyForID = None, targetAx = None):
		""
		if "verticalLines" in self.data:
			for n,linesProps in self.data["verticalLines"].items():
				if n in self.axisDict and onlyForID is None:
					ax = self.axisDict[n]
				elif onlyForID is not None and targetAx is not None:
					if onlyForID == n:
						ax = targetAx
					else:
						continue
				inv = self.axisDict[n].transLimits.inverted() #using the original axis for
				xOffset,yText = inv.transform((0.01, 0.99))
				
				for lineProps in linesProps:
					ax.axvline(**lineProps)
					ax.text(x=lineProps["x"] + xOffset, 
							y=yText, 
							s=lineProps["label"], 
							fontproperties = self.getStdFontProps(),
							horizontalalignment = "left" , 
							verticalalignment = "top", 
							rotation = 90, 
							linespacing=1.1)

	def addLineByArray(self,x,y):
		""
		for _,ax in self.axisDict.items():
			l = Line2D(xdata = x, ydata = y, lw=0.75, color = INSTANT_CLUE_BLUE)
			ax.add_artist(l)
			lineID = getRandomString()
			self.extraArtists[lineID] = {"artist":l,"color":INSTANT_CLUE_BLUE,"name":"Line({})".format(lineID)}
		self.updateFigure.emit()

	def addQuadrantLines(self, quadrantCoords):
		""
		#quadrant coords x_min, x_max, y_min, y_max
		for _,ax in self.axisDict.items():
			xlim = np.array(list(self.getAxisLimit(ax)))
			ylim = np.array(list(self.getAxisLimit(ax, which="y")))
			q1 = np.array([(xlim[0],quadrantCoords[3]),(quadrantCoords[0],quadrantCoords[3]), (quadrantCoords[0],ylim[1])])
			q2 = np.array([(quadrantCoords[1],ylim[1]),(quadrantCoords[1],quadrantCoords[3]), (xlim[1],quadrantCoords[3])])
			
			q3 = np.array([(quadrantCoords[1],ylim[0]),(quadrantCoords[1],quadrantCoords[2]), (xlim[1],quadrantCoords[2])])
			q4 = np.array([(xlim[0],quadrantCoords[2]),(quadrantCoords[0],quadrantCoords[2]), (quadrantCoords[0],ylim[0])])
			for n,q in enumerate([q1,q2,q3,q4]):
				l = Line2D(xdata = q[:,0], ydata =  q[:,1], lw=0.75, color = INSTANT_CLUE_BLUE)

				ax.add_artist(l)
				#save line
				lineID = getRandomString()
				self.extraArtists[lineID] = {"artist":l,"color":INSTANT_CLUE_BLUE,"name":"Qaudrant {}".format(n)}
		self.updateFigure.emit()


	def addSizeLegendToGraph(self, sizeData, ignoreNaN = False,title =  None ,update=True, ax = None, export = False, legendKwargs = {}):
		""
		try:
			#if ignoreNaN:
			#	idx = colorData["size"] != self.getParam("minScatterSize")
			#	colorData = colorData.loc[idx]

			#if title is None and "colorCategoricalColumn" in self.data:
			#	title = self.data["colorCategoricalColumn"]
			if sizeData.index.size > 200:
				w = WarningMessage(title="To many items for legend.",
								infoText = "More than 200 items for legend which is not supported.")
				w.exec_()	
				return			
			scatterKwargs = self.getScatterKwargForLegend(legendType="size")
			legendItems = [scatter(
							[], [],
							s = sizeData.loc[idx,"size"],
							label = sizeData.loc[idx,"group"], **scatterKwargs) for idx in sizeData.index]
			if ax is None:
				ax = self.axisDict[0]
			if not export and hasattr(self,"sizeLegend"):
				self.sizeLegend.remove() 

			legend = ax.legend(handles=legendItems, title = title, 
								title_fontsize = self.getParam("legend.title.mainfigure.fontsize") if export else self.getParam("legend.title.fontsize"), 
								fontsize = self.getParam("legend.mainfigure.fontsize") if export else self.getParam("legend.fontsize"),
								labelspacing = self.getParam("legend.label.spacing"),
								borderpad  = self.getParam("legend.label.borderpad"),
								**legendKwargs)
			if not export:
				self.sizeLegend = legend
				self.checkLegend(ax,attrName="colorLegend")
				self.checkLegend(ax,attrName="markerLegend")
				self.checkLegend(ax,attrName="quickSelectLegend")
				self.lastAddedLegend = legend
				self.sizeLegendKwargs = {"sizeData":sizeData,"ignoreNaN":ignoreNaN,"title":title,"update":False,"export":True,"legendKwargs":legendKwargs}
				
					
			if update:
				self.updateFigure.emit()
			return legend
		except Exception as e:
			print(e)

	def checkLegend(self,ax,attrName):
		""
		if hasattr(self,attrName):
			legend = getattr(self,attrName)
			if legend == self.lastAddedLegend:
				newLegend = ax.add_artist(legend)


	def addLines(self, lineData):
		""
		for n,lineKwargs in lineData.items():
			if n in self.axisDict:
				l = Line2D(**lineKwargs)
				self.axisDict[n].add_artist(l)
		
	def addTooltip(self):
		""
		for ax in self.axisDict.values():
			if ax in self.hoverGroupItems:
				self.tooltips[ax] = ICChartToolTip(self.p,ax,self.hoverGroupItems[ax])
	
	def addYLimChangeEvent(self,ax,callbackFn):
		""
		self.onYLimChange = \
			ax.callbacks.connect('ylim_changed', callbackFn)


	def addSwarm(self,dataID,numericColumns,categoricalColumns, onlyForID = None, targetAx = None):
		"Add Swarm Scatter"
		try:
			if not self.isBoxplotViolinBar():
				w = WarningMessage(infoText = "Swarms can only be added to box, bar and violinplots.")
				w.exec_()
				return
			
			
			if hasattr(self,"swarmData") and onlyForID is not None and hasattr(self,"swarmScatterKwargs"):
				for n, scatter in self.swarmScatter.items():
					if not scatter.getScatterInvisibility():
						#if visibility is Flase -> dont plot anything
						return
				columnPair = self.swarmData["columnPairs"][onlyForID]
				scatterPlot(
							self,
							data = self.swarmData["plotData"],
							plotter = self.p,
							ax = targetAx,
							numericColumns = list(columnPair),
							dataID = self.swarmData["dataID"],
							scatterKwargs = self.swarmScatterKwargs,
							multiScatter = True,
							multiScatterKwargs = {},
							interactive = False,
							adjustLimits = False
							)
			elif hasattr(self,"swarmData"):
				#if swarm data are present -> just toggle visibility of swarm scatters#
				#setting swarm invisisble
				for n, scatter in self.swarmScatter.items():
					scatter.toggleVisibility()
					
				#update figure
				self.updateFigure.emit()

			elif onlyForID is None:

				self.swarmData = self.mC.plotterBrain.getSwarmplotProps(dataID,numericColumns,categoricalColumns)["data"]

				self.swarmScatter = OrderedDict() 
				#init scatter plots
				self.swarmScatterKwargs = self.getScatterKwargs().copy()
				self.swarmScatterKwargs["marker"] = self.mC.config.getParam("swarm.scatter.marker")
				self.swarmScatterKwargs["s"] = self.mC.config.getParam("swarm.scatterSize")
				self.swarmScatterKwargs["color"] = self.mC.config.getParam("swarm.facecolor")
				self.swarmScatterKwargs["zorder"] = 10000
				self.swarmScatterKwargs["alpha"] = self.mC.config.getParam("swarm.alpha")

				for n,ax in self.axisDict.items():
					columnPair = self.swarmData["columnPairs"][n]
					self.swarmScatter[n] = scatterPlot(
												self,
												data = self.swarmData["plotData"],
												plotter = self.p,
												ax = ax,
												numericColumns = list(columnPair),
												dataID = self.swarmData["dataID"],
												scatterKwargs = self.swarmScatterKwargs,
												multiScatter = True,
												multiScatterKwargs = {},
												interactive = False,
												adjustLimits = False
												)
				self.updateFigure.emit()
		except Exception as e:
			print(e)

	def buildTooltip(self):
		'''
		'''
		self.tooltip = self.ax.text(s ='', bbox=self.bboxProps,**self.textProps)
		self.textProps['text'] = ''

	def centerXToZero(self):
		""
		axes = list(self.axisDict.values())
		xLims = np.array([ax.get_xlim() for ax in axes])
		maxXLim = np.max(np.abs(xLims))
		for ax in axes:
			self.setAxisLimits(ax, xLimit=(-maxXLim,maxXLim))
		self.updateFigure.emit() 


	def rawAxesLimits(self):
		""
		if self.hasScatters():
			for scatterPlot in self.scatterPlots.values():
				scatterPlot.adjustAxisLimits()
			self.updateFigure.emit()

	def getAxesWithLimits(self):

		axes = list(self.axisDict.values())
		xLims = np.array([ax.get_xlim() for ax in axes])
		yLims = np.array([ax.get_ylim() for ax in axes])
		return axes, xLims, yLims 

	def alignLimitsOfAllAxes(self, updateFigure = True):
		""
		axes, xLims, yLims = self.getAxesWithLimits()
		xMin, xMax = np.min(xLims[:,0]), np.max(xLims[:,1])
		yMin, yMax = np.min(yLims[:,0]), np.max(yLims[:,1])
		
		for ax in axes:
			self.setAxisLimits(ax, xLimit=(xMin,xMax), yLimit=(yMin,yMax))
		if updateFigure:
			self.updateFigure.emit() 

	def alignLimitsOfXY(self):
		""
		axes, xLims, yLims = self.getAxesWithLimits()
		for n, ax in enumerate(axes):
			axMin = np.min([xLims[n,0],yLims[n,0]])
			axMax = np.max([xLims[n,1],yLims[n,1]])
			axLim = (axMin,axMax)
			self.setAxisLimits(ax, xLimit = axLim, yLimit = axLim)

		self.updateFigure.emit()

	def removeTooltip(self):
		'''
		Destroy tooltip text
		'''		
		if hasattr(self,'tooltip'):
			self.tooltip.set_visible(False)
			self.toolTipsActive = False	
	
	def removeLabels(self):
		""
		self.disconnectAnnotations()
	
	def removeAnnotationsFromGraph(self):
		""
		
		
	
	def determinePosition(self,x,y):
		'''
		Check how to align the tooltip.
		'''
		if not hasattr(self,"axProps"):
			return
			
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
		self.textProps = self.getStdTextProps()

	

	def extractAxisProps(self):
		'''
		'''
	
		self.axProps = dict()
		self.axProps['xlim'] = self.ax.get_xlim()
		self.axProps['ylim'] = self.ax.get_ylim()
		self.axProps['xDiff'] = self.axProps['xlim'][1] - self.axProps['xlim'][0]
		self.axProps['yDiff'] = self.axProps['ylim'][1] - self.axProps['ylim'][0]
				
	def updateTooltipPosition(self,event,text):
		'''
		'''
		if event.inaxes != self.ax:
			return
		# get event data
		x,y = event.xdata, event.ydata
		## check if new text 
		self.textProps['text'] = text	
		self.textProps['visible'] = True
		self.determinePosition(x,y)
		self.tooltip.update(self.textProps)
	
	
	def drawTooltip(self,background):
		""
		self.p.f.canvas.restore_region(background)
		
		self.ax.draw_artist(self.tooltip)
		# self.axisDict["axLabelColor"].set_yticklabels([self.getCurrentQuickSelectLabel()])
		self.p.f.canvas.blit(self.ax.bbox)
		#self.updateFigure.emit()

		
	def clear(self, event=None):
		""
		self.p.clearFigure()

	def clearAxes(self):
		""
		if len(self.axisDict) > 0:
			for ax in self.axisDict.values():
				ax.clear()

	def disconnectBindings(self):
		""
		if hasattr(self,"onHoverEvent"):
			self.p.f.canvas.mpl_disconnect(self.onHoverEvent)
		if hasattr(self,"onPressEvent"):
			self.p.f.canvas.mpl_disconnect(self.onPressEvent)
		if hasattr(self,"annotations") and self.annotations is not None:
			if isinstance(self.annotations,dict):
				for annotation in self.annotations.values():
					annotation.disconnectEventBindings()

	def exportAxis(self,exportAixs):
		""

	def initAbsAxes(self,axisPositions):
		""
		axisDict = dict()
		if not isinstance(axisPositions,dict):
			return axisDict

		for k, axisProps in axisPositions.items():
			if isinstance(axisProps,list):
				ax = self.p.f.add_axes(axisProps)
				axisDict[k] = ax

		return axisDict

	def initAxes(self,axisPositions):
		""
		for k,axisProps in axisPositions.items():
			ax = self.p.f.add_subplot(*axisProps)
			self.axisDict[k] = ax
		
		if "subplotBorders" in self.data:
			self.setFigureBorders()

	def isLiveGraphActive(self):
		""
		return self.mC.mainFrames["data"].liveGraph.hasData()

	def isQuickSelectActive(self):
		""
		return self.mC.mainFrames["data"].qS.hasData()

	def mirrorAxis(self,targetAx, figID, sourceAx = None):
		""
		try:
			markerLegend, sizeLegend, colorLegend, quickSelectLegend = None, None, None, None
			if sourceAx is None and hasattr(self,"menuClickedInAxis"):

				sourceAx = self.menuClickedInAxis
			axisID = self.getAxisID(sourceAx)
			targetAx.clear() 

			areaTarget = targetAx.bbox.bounds[2] * targetAx.bbox.bounds[3]
			areaSource = sourceAx.bbox.bounds[2] * sourceAx.bbox.bounds[3]
			scaleFactor = (areaSource - areaTarget) / areaSource
			self.mirrorAxisContent(axisID,targetAx, scaleFactor = scaleFactor)
			
			if hasattr(self,"colorLegendKwargs"):
				#color legend presentd
				colorLegend = self.addColorLegendToGraph(**self.colorLegendKwargs,ax=targetAx)
			if hasattr(self,"sizeLegendKwargs"):
				sizeLegend = self.addSizeLegendToGraph(**self.sizeLegendKwargs,ax=targetAx)
				if colorLegend is not None and not hasattr(self,"markerLegendKwargs"):
					targetAx.add_artist(colorLegend)
			if hasattr(self,"markerLegendKwargs"):
				markerLegend = self.addMarkerLegend(**self.markerLegendKwargs, ax=targetAx)
				if colorLegend is not None:
					targetAx.add_artist(colorLegend)
				if sizeLegend is not None:
					targetAx.add_artist(sizeLegend)

			if hasattr(self,"quickSelectLegendKwargs"):
				quickSelectLegend  = self.addQuickSelectLegendToGraph(**self.quickSelectLegendKwargs, ax = targetAx)
				if colorLegend is not None:
					targetAx.add_artist(colorLegend)
				if sizeLegend is not None:
					targetAx.add_artist(sizeLegend)
				if markerLegend is not None:
					targetAx.add_artist(markerLegend)

			self.mirrorQuickSelectArtists(axisID,targetAx)
			self.mirrorLimits(sourceAx,targetAx) #deal with user zoom
			self.mC.mainFrames["right"].mainFigureRegistry.updateFigure(figID)

		except Exception as e:
			print(e)

	def mirrorStats(self,targetAx, onlyForID):
		""
		if hasattr(self ,"statArtists"):
			for _, statDict in self.statArtists.items():
				if statDict["axisID"] == onlyForID:
					targetAx.add_artist(Line2D(**statDict["lineProps"])) 
					targetAx.add_artist(Text(**statDict["textProps"]))

	def mirrorAxisContent(self,axisID,targetAx,*args,**kwargs):
		""

	def mirrorLimits(self,sourceAx,targetAx):
		""
		
		targetAx.set_ylim(sourceAx.get_ylim())
		targetAx.set_xlim(sourceAx.get_xlim())

		
	def onDataLoad(self,data):
		"Function that should be overwritten."

	def onHover(self,event=None):
		""
		if event.inaxes is not None:
			if event.inaxes in self.tooltips:
				self.tooltips[event.inaxes].evaluateEvent(event)

	def getAxisLimit(self,ax,which="x"):
		""
		if which == "x":
			return ax.get_xlim()
		elif which == "y":
			return ax.get_ylim()


	def getCurrentQuickSelectLabel(self):
		""
		return self.mC.mainFrames["data"].qS.getCurrentLabel()

	def getHoverKwargs(self):
		""
		hoverKwargs = self.getScatterKwargs(checkQuickSelect=False)
		hoverKwargs["color"] = self.getParam("scatter.hover.color")
		return hoverKwargs

	def getParam(self,paramName):
		""
		return self.mC.config.getParam(paramName)
	
	def getScatterKwargs(self, checkQuickSelect = True):
		""
		scatterKwargs = {'s':self.getParam("scatterSize"),
							'alpha':self.getParam("alpha"),
							'picker':True,
							'label':None,
							'color':self.getParam("nanColor"),
							'edgecolor':self.getParam("scatter.edgecolor"),
							'linewidth':self.getParam("scatter.edgelinewidth"),
							"marker":self.getParam("scatter.marker")}
		
		return scatterKwargs

	def getScatterKwargForLegend(self, legendType = "color"):
		""
		scatterKwargs = self.getScatterKwargs() 
		del scatterKwargs["label"]
		if legendType == "color":
			del scatterKwargs["color"]
		elif legendType == "size":
			del scatterKwargs["s"]
		elif legendType == "QuickSelect":
			del scatterKwargs["s"]
			del scatterKwargs["color"]
		return scatterKwargs

	def getPlotData(self):
		""
		if hasattr(self,"data") and "plotData" in self.data:
			return self.data["plotData"]


	def getQuickSelectData(self):
		""
		if self.isQuickSelectActive():
			data = self.mC.mainFrames["data"].qS.getSizeAndColorData()
			return data

	def getQuickSelectMode(self):
		""
		if self.isQuickSelectActive():
			return self.mC.mainFrames["data"].qS.selectionMode

	def isQuickSelectModeUnique(self):
		""
		return self.mC.mainFrames["data"].qS.quickSelectProps["filterProps"]["mode"] == "unique"


	def getDataIndexOfQuickSelectSelection(self):
		""
		if self.isQuickSelectActive():
			return self.mC.mainFrames["data"].qS.getDataIndexOfCurrentSelection()

	def getColorArray(self):
		"Hclust overload function"
        
	def getClusterIDsByDataIndex(self):
		"Hclust overload function"
		return None, None 

	def getClusteredData(self):
		"Hclust overload function"

	def getStdTextProps(self):
		'''
		Define text properties
		'''
		textProps = {'x':0,'y':0,
					"fontproperties":FontProperties(family=self.getParam("annotationFontFamily"),
										size = self.getParam("annotationFontSize")),
					'linespacing': 1.2,
					'visible':False,
					'zorder':1e9}
		return textProps
	
	def getStdFontProps(self):
		"Returns standard font props"
		return FontProperties(
					family=self.getParam("annotationFontFamily"),
					size = self.getParam("annotationFontSize"))

	def getYlim(self,ax,offset = 0, mult = 1):
		""
		minLim, maxLim = ax.get_ylim() 
		return (minLim * mult - offset, maxLim * mult + offset)

	def isHclust(self):
		""
		return self.plotType == "hclust"
	
	def isBoxplotViolinBar(self):
		""
		return self.plotType in ["boxplot","violinplot","barplot"]
	
	def isBoxplot(self):
		""
		return self.plotType == "boxplot"

	def isBarplot(self):
		""
		return self.plotType == "barplot"

	def isLinePlot(self):
		""
		return self.plotType == "lineplot"
	
	def isViolinplot(self):
		""
		return self.plotType == "violinplot"

	def hasScatters(self):
		""
		return hasattr(self,"scatterPlots") and len(self.scatterPlots) > 0

	def enableInteractiveStats(self):
		""
		setattr(self,"statTestEnabled",True)
		self.mC.sendMessageRequest({"title":"Enabled.","message":"Interactive statistical testing enabled."})

	def resetMask(self):
		"Reset any mask::overload function"

	def setHoverItemGroups(self,itemGroups = []):
		"Each group is a group of matplotlib items"
		self.hoverGroupItems = itemGroups
		self.addTooltips()
		self.addHoverBinding()

	def sendIndexToLiveGraph(self, dataIndex):
		""
		if self.isLiveGraphActive():
			self.mC.mainFrames["data"].liveGraph.updateGraphByIndex(dataIndex)

	def sendIndexToQuickSelectWidget(self, dataIndex):
		""
		
		if self.isQuickSelectActive():
			self.mC.mainFrames["data"].qS.setHighlightIndex(dataIndex)

	def sendSelectEventToQuickSelect(self, dataIndex):
		""
		if self.isQuickSelectActive():
			self.mC.mainFrames["data"].qS.setCheckStateByDataIndex(dataIndex)

	def setAxisLimits(self,ax, xLimit = None ,yLimit = None):
		""
		if xLimit is not None:
			ax.set_xlim(xLimit)
		if yLimit is not None:
			ax.set_ylim(yLimit)

	def setAxisLabels(self,axes,labels, onlyForID = None ):
		""
		for n, ax in axes.items():
			if onlyForID is not None and n != onlyForID:
				continue
			label = labels[n]
			
			if "x" in label:
				ax.set_xlabel(label["x"])
			if "y" in label:
				ax.set_ylabel(label["y"])
	
	def setAxisOff(self,ax):
		""
		ax.axis(False)

	def setAxisOn(self,ax):
		""
		ax.axis(True)

	def setAxisTitle(self,ax,title):
		""
		ax.set_title(title)		

	def setData(self,data={}):
		""
		self.dataLoaded.emit(data)
	

	def setHoverData(self,index=None):
		""
	
	def setMask(self,dataIndex):
		""

	def setColorCategoryIndexMatch(self,categoryIndexMatch):
		""
		self.colorCategoryIndexMatch = categoryIndexMatch

	def setSizeCategoryIndexMatch(self,categoryIndexMatch):
		""
		self.sizeCategoryIndexMatch = categoryIndexMatch

	def setQuickSelectCategoryIndexMatch(self,categoryIndexMatch):
		""
		self.quickSelectCategoryIndexMatch = categoryIndexMatch
	
	def getDataInColorTable(self):
		""
		colorTable = self.mC.mainFrames["sliceMarks"].colorTable
		if hasattr(colorTable,"model") and colorTable.model._labels.index.size > 0:
			return colorTable.model._labels

	def getTitleOfColorTable(self):
		""
		colorTable = self.mC.mainFrames["sliceMarks"].colorTable
		if hasattr(colorTable,"title"):
			return colorTable.title
		return ""

	def getDataInSizeTable(self):
		""
		sizeTable = self.mC.mainFrames["sliceMarks"].sizeTable
		if hasattr(sizeTable,"model") and sizeTable.model._labels.index.size > 0:
			return sizeTable.model._labels

	def setDataInColorTable(self,data = pd.DataFrame(), title = "Colors"):
		""
		self.mC.mainFrames["sliceMarks"].colorTable.setData(data, title)

	def setDataInSizeTable(self,data = pd.DataFrame(), title = "Sizes"):
		""
		self.mC.mainFrames["sliceMarks"].sizeTable.setData(data, title)
	
	def setDataInStatisticTable(self,data = pd.DataFrame(), title = "Statistics"):
		""
		self.mC.mainFrames["sliceMarks"].statisticTable.setData(data, title)

	def setDataInLabelTable(self, data = pd.DataFrame(), title = ""):
		""
		if hasattr(self.mC.mainFrames["sliceMarks"],"labelTable"):
			self.mC.mainFrames["sliceMarks"].labelTable.setData(data, title)
	
	def setDataInTooltipTable(self, data = pd.DataFrame(), title = ""):
		""
		if hasattr(self.mC.mainFrames["sliceMarks"],"tooltipTable"):
			self.mC.mainFrames["sliceMarks"].tooltipTable.setData(data, title)

	def setHoverObjectsInvisible(self):
		"Sets hover objects invisible"
		if hasattr(self,"hoverScatter") and isinstance(self.hoverScatter,dict):
			for scatter in self.hoverScatter.values():
				scatter.set_visible(False)

	def setQuickSelectScatterInvisible(self):
		""
		

		if self.hasScatters():
			for scatterPlot in self.scatterPlots.values():
				scatterPlot.setQuickSelectScatterInivisible()
				

		if hasattr(self,"quickSelectScatter"):
			if isinstance(self.quickSelectScatter,dict):
				for _,qSScatter in self.quickSelectScatter.items():
					if hasattr(qSScatter,"set_visible"):
						qSScatter.set_visible(False)


	def setTicksOff(self,ax):
		""
		ax.tick_params(axis='both',          
    					which='both',      
						bottom=False, 
						left = False, 
						labelbottom = False, 
						labelleft = False)

	def setXTicksForAxes(self,axes,xTicks,xLabels,onlyForID = None,**kwargs):
		""
		for n, ax in axes.items():
			if onlyForID is not None and n != onlyForID:
				continue
			self.setXTicks(ax,xTicks[n],xLabels[n],**kwargs)
	
	def setYTicksForAxes(self,axes,yTicks,xLabels,onlyForID = None,**kwargs):
		""
		for n, ax in axes.items():
			if onlyForID is not None and n != onlyForID:
				continue
			self.setYTicks(ax,yTicks[n],xLabels[n],**kwargs)

	def setXTicks(self,ax,ticks,labels,**kwargs):
		""
		ax.set_xticks(ticks)
		ax.set_xticklabels(labels,**kwargs)

	def setYTicks(self,ax,ticks,labels,tickwargs = {},**kwargs,):
		""
		ax.set_yticks(ticks,**tickwargs)
		ax.set_yticklabels(labels,**kwargs)

	def setYTicksToRight(self,ax):
		""
		ax.yaxis.tick_right()		

	def setResizeTrigger(self, resized = True):
		""
		self.resized = True
	
	def update(self, event=None):
		""
		self.p.redraw()

	def updateBackgrounds(self, redraw=False):
		""
		for tooltip in self.tooltips.values():
			tooltip.update_background(redraw=False)

	def updateClim(self):
		""

	def updateData(self, data= None):
		""
	
	def updateGroupColors(self,*args,**kwargs):
		""

	def updateGroupSizes(self,*args,**kwargs):
		""

	def updateScatterPropSection(self,idx,value,propName = "color"):
		""
		if self.hasScatters():
			for scatterPlot in self.scatterPlots.values():
				scatterPlot.updateScatterPropSection(idx,value,propName)

	def updateScatterProps(self,propsData):
		""
		if self.hasScatters():
			if hasattr(self,"colorLegend"):
				self.addColorLegendToGraph(self.getDataInColorTable(),title=self.getTitleOfColorTable(),update=False)
			
			for scatterPlot in self.scatterPlots.values():
				scatterPlot.updateScatterProps(propsData)	

	def updateQuickSelectData(self,quickSelectGroup,changedCategory=None):
		""
		for ax in self.axisDict.values():
			if self.isQuickSelectModeUnique():

				scatterSizes, scatterColors, _ = self.getQuickSelectScatterProps(ax,quickSelectGroup)
				print(scatterSizes)

			elif ax in self.quickSelectScatterDataIdx: #mode == "raw"

				dataIdx = self.quickSelectScatterDataIdx[ax]["idx"]
				scatterSizes = [quickSelectGroup["size"].loc[idx] for idx in dataIdx]	
				scatterColors = [quickSelectGroup["color"].loc[idx] for idx in dataIdx]

			else:
				
				continue

			self.updateQuickSelectScatter(ax, scatterColors = scatterColors, scatterSizes = scatterSizes)
			   
                
	def getQuickSelectScatterProps(self,ax,quickSelectGroup):
		""
		#internalIDs = quickSelectGroup["internalID"]
		scatterSizes = []
		scatterColors = []
		dataIndicies = []
		#get index
		#dataIndex = np.concatenate([idx for idx in self.quickSelectCategoryIndexMatch.values()])
		intIDs = self.quickSelectScatterDataIdx[ax]["coords"]["intID"] 
		colorMapper = dict([(intID,colorValue) for intID, colorValue in quickSelectGroup[["internalID","color"]].values])
		sizeMapper = dict([(intID,sizeValue) for intID, sizeValue in quickSelectGroup[["internalID","size"]].values])
		scatterColors = intIDs.map(colorMapper)
		scatterSizes = intIDs.map(sizeMapper)
		
		return scatterSizes, scatterColors, self.quickSelectScatterDataIdx[ax]["idx"]

		# print(intIDs)
		# for intID, colorValue, sizeValue in quickSelectGroup[["internalID","color","size"]].values:
		# #for intID, indics in self.quickSelectCategoryIndexMatch.items():
		# 	indics = self.quickSelectCategoryIndexMatch[intID]
		# 	#boolIdx = internalIDs == intID
		# 	#colorValue, sizeValue = quickSelectGroup.loc[boolIdx,["color","size"]].values[0]
		# 	scatterColors.extend([colorValue] * indics.size)
		# 	scatterSizes.extend([sizeValue] * indics.size)
		# 	dataIndicies.extend(indics.tolist())

		# return scatterSizes,scatterColors,dataIndicies

		
	def updateQuickSelectScatter(self,ax,coords = None,scatterColors = None, scatterSizes = None):
		"Testing if ax in backgrounds and quickSelectScatter should be performed before."
		if not hasattr(self,"backgrounds") or not hasattr(self,"quickSelectScatter"):
			return
		self.p.f.canvas.restore_region(self.backgrounds[ax])
		if coords is not None:
			if isinstance(coords,pd.DataFrame):
				coords = coords[["x","y"]].values
			self.quickSelectScatter[ax].set_offsets(coords[:,0:2])
		self.quickSelectScatter[ax].set_visible(True)
		if scatterColors is not None:
			self.quickSelectScatter[ax].set_facecolor(scatterColors)
		if scatterSizes is not None:
			self.quickSelectScatter[ax].set_sizes(scatterSizes)

		ax.draw_artist(self.quickSelectScatter[ax])
		self.p.f.canvas.blit(ax.bbox)


	def mirrorQuickSelectArtists(self,axisID,targetAx):
		""
		if axisID in self.axisDict and hasattr(self,"quickSelectScatter"):
			sourceAx = self.axisDict[axisID]
			if sourceAx in self.quickSelectScatter:
				coords = self.quickSelectScatter[sourceAx].get_offsets()
				scatterColors = self.quickSelectScatter[sourceAx].get_facecolor()
				scatterSizes = self.quickSelectScatter[sourceAx].get_sizes()
				kwargs = self.getScatterKwargs()
				
				kwargs["zorder"] = 1e9
				kwargs["s"] = scatterSizes
				kwargs["color"] = scatterColors
				targetAx.scatter(x = coords[:,0], y = coords[:,1], **kwargs)


	def setNaNColor(self):
		""

	def setFigureBorders(self):
		""
		self.p.f.subplots_adjust(**self.data["subplotBorders"])

	
	def setLegendInvisible(self):
		try:
			if hasattr(self,"colorLegend"):
				self.colorLegend.remove() 
		except Exception as e:
			print(e)

	def subsetDataOnInternalID(self,internalID, groupName):
		""
		idx = None
		if self.colorCategoryIndexMatch is not None and internalID in self.colorCategoryIndexMatch:
			idx = self.colorCategoryIndexMatch[internalID]
		if self.sizeCategoryIndexMatch is not None and internalID in self.sizeCategoryIndexMatch:
			idx = self.sizeCategoryIndexMatch[internalID]
		if idx is not None:
			dataID = self.data["dataID"]
			subsetName = "chartSubset:({})_({})".format(groupName,self.mC.data.getFileNameByID(dataID))
			funcProps = {"key":"data::subsetDataByIndex","kwargs":{"dataID":dataID,"filterIdx":idx,"subsetName":subsetName}}
			self.mC.sendRequestToThread(funcProps)

	def resetQuickSelectArtists(self):
		""
		if hasattr(self,"quickSelectScatter"):
			for ax, qSScatter in self.quickSelectScatter.items():
				qSScatter.set_visible(False)
			
			self.updateFigure.emit()

class ICChartToolTip(object):

	def __init__(self,plotter,ax,artistProps):
		'''
		artistProp - dict. 
			Must have keys : 'artists','colors','texts'. 
			Value must be dicts in form of {key1 : color1, key2 : color2}
		'''

		self.plotter = plotter
		self.r = self.plotter.f.canvas.get_renderer()
		self.ax = ax
		self.width = self.height = 0
		self.update = True
		self.inactiveColor = self.plotter.mC.config.getParam("hover.inactive.facecolor")
		self.currentArtist = None
		self.artistProps = artistProps

		self.defineBbox()
		self.defineText()
		self.buildTooltip()
		self.extractAxisProps()
		#self.update_background(redraw=False)
		
	def adjustColor(self):
		""
		inactiveColor = to_rgba(self.inactiveColor)
		backgroundUpdate = False
		for artistID, artist in self.artistProps["artists"].items():
			currentColor = self.getColor(artist)
			#print(currentColor)
			targetColor = to_rgba(self.artistProps["colors"][artistID])
			if self.currentArtist is None and currentColor != targetColor:
				self.changeColor(artist,targetColor)
				backgroundUpdate = True
			elif self.currentArtist is None:
				continue
			elif artistID != self.currentArtist and currentColor != inactiveColor:
				self.changeColor(artist,self.inactiveColor)#
				backgroundUpdate = True
			elif artistID == self.currentArtist and currentColor != targetColor:
				self.changeColor(artist, targetColor)
				backgroundUpdate = True
		if backgroundUpdate:
			self.update_background()

			
	def buildTooltip(self):
		'''
		'''
		self.tooltip = self.ax.text(s ='', bbox=self.bboxProps,**self.textProps)
		self.textProps['text'] = ''

	def changeColor(self,artist,color):
		"Set color of artis"
		if hasattr(artist,'set_facecolor'):
				artist.set_facecolor(color)
		elif hasattr(artist,'set_color'):
					artist.set_color(color)			

	

	def evaluateEvent(self,event):
		'''
		'''
		
		artistContEvent = [artistID for artistID, artist in self.artistProps['artists'].items() if artist.contains(event)[0]]
		if len(artistContEvent) == 0:
			#mouse is not over any artist
			self.setInvisible(update=False)
			self.currentArtist = None
			#update colors to default colors
			self.adjustColor()
			return

		else:
			#an artist was found
			artistID = artistContEvent[0]
			self.mouseOverArtist = True
			self.currentArtist = artistID 
			self.adjustColor()
			#update position of tooltip
			self.updatePosition(event,self.artistProps['texts'][artistID])
			self.updateAxis()

	
	
	def getColor(self,artist):
		"Set color of artis"
		if hasattr(artist,'get_facecolor'):
				return artist.get_facecolor()
		elif hasattr(artist,'get_color'):
				return artist.get_color()		
	
	def updatePosition(self,event,text):
		'''
		'''
		# get mouse data
		x,y = event.xdata, event.ydata
		## check if new text 
		if self.textProps['text'] != text:
			self.update = True
			
		self.textProps['text'] = text	
		self.textProps['visible'] = True
		self.determinePosition(x,y)
		self.tooltip.update(self.textProps)
		self.updateAxis()
		
	def resetSize(self):
		""



	def resetOriginalColors(self):
		'''
		'''
		for artistID,artist in self.artistProps['artists'].items():
			self.changeColor(artist,self.artistProps['colors'][artistID])
						
		self.plotter.redraw()		

					
	def setInvisible(self,visble = False, update = True):
		'''
		'''
		new = {'visible':visble}
		self.tooltip.update(new)
		if update:
			self.updateAxis()

	
	def update_background(self, redraw=True):
		'''
		'''
		#redraw = False
		if redraw:
			if hasattr(self,'background'):
				self.setInvisible()
			self.plotter.redraw()
		self.background =  self.plotter.f.canvas.copy_from_bbox(self.ax.bbox)
		
	def updateAxis(self):
		'''
		'''
		if hasattr(self,'background'):
			self.plotter.f.canvas.restore_region(self.background)
			self.tooltip.draw(self.r)
			self.plotter.f.canvas.blit(self.ax.bbox)
			
	def extractAxisProps(self):
		'''
		'''
	
		self.axProps = dict()
		self.axProps['xlim'] = self.ax.get_xlim()
		self.axProps['ylim'] = self.ax.get_ylim()
		self.axProps['xDiff'] = self.axProps['xlim'][1] - self.axProps['xlim'][0]
		self.axProps['yDiff'] = self.axProps['ylim'][1] - self.axProps['ylim'][0]


	def determinePosition(self,x,y):
		'''
		Check how to align the tooltip.
		'''
			
		if self.update:
			self.extractTextDim()
			
		xMin,xMax = self.axProps['xlim']
		yMin,_ = self.axProps['ylim']
		
		width  = self.width
		height = self.height
		
		diff = (xMin-xMax)*0.05	
		
		if x + width > xMax - 0.1*xMax and x > sum(self.axProps['xlim'])/2:
			 self.textProps['ha'] = 'right'
		else:
			self.textProps['ha'] = 'left'
			diff *= -1
		
		if y - height - yMin*0.1 < yMin :
			self.textProps['va'] = 'bottom'
		else:
			self.textProps['va'] = 'top'
		
		self.textProps['x'] = x + diff	
		self.textProps['y'] = y 
		
	def extractTextDim(self):
		'''
		Extract width and height of a fake text element.
		'''
		fakeText = self.ax.text(0,0,s=self.textProps['text'],bbox=self.bboxProps)
		patch = fakeText.get_window_extent(self.r)
		inv = self.ax.transData.inverted()
		xy0 = list(inv.transform((patch.x0,patch.y0)))
		xy1 = list(inv.transform((patch.x1,patch.y1)))
		self.width = xy1[0]-xy0[0]
		self.height = xy1[1]-xy0[1]	
		fakeText.remove()
		self.update = False
	
	def defineBbox(self):
		'''
		Define bbox
		'''
		self.bboxProps = {'facecolor':'white', 'alpha':0.8,
						 'edgecolor':'darkgrey','fill':True,
						 }
	
	def defineText(self):
		'''
		Define text properties
		'''
		self.textProps = {'x':0,'y':0,
						 'fontname':'Verdana',
						 'linespacing': 1.5,
						 'visible':False,
						 'zorder':1e9}
		
	def isAnnotationInAllPlotsEnabled(self):
		""
		return False