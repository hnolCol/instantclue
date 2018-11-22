"""
	""LINE PLOTTING""
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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import itertools
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from modules.utils import *
from modules.plots.axis_styler import axisStyler


class linePlotHelper(object):

	def __init__(self,plotter,dfClass,numericColumns,colorMap):
		'''
		'''
		self.colorMap = colorMap
		self.plotter = plotter
		self.dfClass = dfClass	
		self.colors = 'lightgrey'
		self.numericColumns = numericColumns
		self.savedLegendLines = []
		self.savedMeanLines = []
		self.dataID = self.dfClass.currentDataFile
		self.categoricalColorDefinedByUser = dict()
		self.sizeStatsAndColorChanges = dict() 
		self.meanData = OrderedDict()
		self.prepare_data()
		self.replot(False)
		
	
	def replot(self,updateData):
		'''
		Performs the plotting. Easy access if loaded from a saved session.
		'''
		if updateData:
			self.prepare_data()
		self.axisDict = dict()
		self.add_axis()
		self.fill_axis()
		self.style_axis()
		self.add_line_for_pot_hover()
	
	
	def prepare_data(self):
		'''
		Prepare data.
		'''
		rawData = \
		self.dfClass.get_current_data_by_column_list(columnList = self.numericColumns)
		
		self.data = rawData.dropna(thresh=2)
		self.xData = np.arange(len(self.numericColumns))
		
		self.determine_y_axis_limits()
		self.extract_line_data()


	def determine_y_axis_limits(self):
		'''
		Scales y-axes limits
		'''		
		self.yMin = self.data.min().min()
		self.yMin -= 0.05*self.yMin
		self.yMax = self.data.max().max()
		self.yMax += 0.05*self.yMax
		
			
	def extract_line_data(self):
		'''
		Extract line coordinates.
		'''
		self.lines  = []
		
		self.data[self.numericColumns].apply(lambda y: self.add_line_data(y), axis = 1)
		
	
	
	def add_line_data(self,y):
		'''
		extract line Data to a list self.lines. Containing 
		line array data like:
		[line1,line2,..] where line1 = [(x0,y0),(x1,y1),...]
		'''
		lineData = np.array((self.xData,y)).T
		self.lines.append(lineData)
		
	
	def add_axis(self):
		'''
		Add axis to figure.
		'''
		self.axisDict[0] = self.plotter.figure.add_subplot(111)
	
	
	def fill_axis(self, specificAxis = None):
		'''
		Fill craeted axis with LineCollection.
		Parameter 
		=========
		specificAxis - axis to be used. If None the created 
			axis stored in axisDict will be used. This happens if
			the chart is NOT exported to a main figure.
		'''
		#for line in self.lines:
		#	self.axisDict[0].plot(line)
		if specificAxis is None:
			ax = self.axisDict[0]
		else:
			ax = specificAxis
		line_segments = LineCollection(self.lines, linewidths=0.95,
                               colors=self.colors, linestyle='solid',alpha=0.5)
		ax.add_collection(line_segments)
		
        
        
	
	def style_axis(self, specificAxis = None):
		'''
		Change style of axis. 
		'''
		if specificAxis is None:
			ax = self.axisDict[0]
		else:
			ax = specificAxis
		
		ax.set_xticks(range(len(self.numericColumns)))
		ax.set_xticklabels(self.numericColumns)
		axisStyler(ax=ax, ylabel = 'Value',nTicksOnYAxis = 4,rotationXTicks = 90,
			newXLim = (-0.2,len(self.numericColumns)-0.8), newYLim=(self.yMin ,self.yMax))

	def add_annotation_data(self,columnList):
		'''
		Add a column for color encoding of numerical/categorical
		column
		Parameter 
		==========
		columnList - list. Columns that are used for color encoding.
		'''
		self.data = self.dfClass.join_missing_columns_to_other_df(self.data,
												id=self.dataID,
												definedColumnsList=columnList)			
	def add_line_for_pot_hover(self):
		'''
		Add line that can be used to hover over the chart and see lines.
		'''
		self.hoverLine = self.axisDict[0].plot([],[],
							color='darkred',linewidth=1.5,label=None,
							zorder=1)
	
	def indicate_hover(self,arg):
		'''
		Indicate hovering.
		'''			
		yValues = self.data[self.numericColumns].values[arg,:]
		self.hoverLine[0].set_data(self.xData,yValues)
		self.plotter.figure.draw_artist(self.hoverLine[0])
		self.plotter.figure.canvas.blit(self.axisDict[0].bbox)	
			
	def get_data(self):
		'''
		Returns used data
		'''
		
		return self.data
		
	def get_current_colorMapDict(self):
		'''
		Returns current color map dict.
		key - categorical values (combinations) 
		value - color in rgba
		'''
		return self.colorMapDict
					
	def update_colorMap(self, newColorMap = None):
		'''
		Updates color map and chart.
		'''
		if newColorMap is not None:
			self.colorMap = newColorMap
		
		for functionName,column in self.sizeStatsAndColorChanges.items(): 
			getattr(self,functionName)(column)  
			
	
	def clean_up_old_legend_lines(self):
		
		for line in self.savedLegendLines:
			line[0].remove()
		self.savedLegendLines = []
		
		
	def add_legend(self, ax ,export = False):
		'''
		Adds a legend to a plot displaying the different categorical values. 
		'''		
		if export == False:
			self.clean_up_old_legend_lines() 
		
		leg = ax.get_legend()
		if leg is not None:
			leg.remove()
		categoricalColumn =  self.sizeStatsAndColorChanges['change_color_by_categorical_columns']
		for level,color in self.colorMapDict.items():
		 	if str(level)  not in ['nan',' ']:
		 		# generate and save line that is used for legend
		 		collectionLegend = ax.plot([],[],color = color, linewidth=0.8, label = level)
		 		if export == False:
		 			self.savedLegendLines.append(collectionLegend)
		
		 		 
		legendTitle = get_elements_from_list_as_string(categoricalColumn, addString = 'Categorical Levels: ', newLine=True)	

		axisStyler(ax,forceLegend=True,kwsLegend={'leg_title':legendTitle})
	
	
	def change_color_by_numerical_column(self,numericColumn, updateColor = True):
		'''
		Change color encoding by a numerical column. 
		'''
		self.add_annotation_data(numericColumn)
		cmap = get_max_colors_from_pallete(self.colorMap)
		
		scaledData = scale_data_between_0_and_1(self.data[numericColumn[0]])
		self.colors = cmap(scaledData)
		self.axisDict[0].collections[0].set_color(self.colors)
		self.sizeStatsAndColorChanges['change_color_by_numerical_column'] = numericColumn
				
		
	def change_color_by_categorical_columns(self,categoricalColumn,updateColor=True):
		'''
		Change color encoding by a categorical column.
		'''
		self.colorMapDict,layerMapDict, self.rawColorMapDict = get_color_category_dict(self.dfClass,categoricalColumn,
												self.colorMap,self.categoricalColorDefinedByUser
												)
		self.add_annotation_data(categoricalColumn)	
		if len(categoricalColumn) == 1:
			self.data.loc[:,'color'] = self.data[categoricalColumn[0]].map(self.colorMapDict)
		else:
			self.data.loc[:,'color'] = self.data[categoricalColumn].apply(tuple,axis=1).map(self.colorMapDict)
		if updateColor == False:
			# need to replot this to change the layer.
			self.data.loc[:,'layer'] = self.data['color'].map(layerMapDict)			
			self.data.sort_values('layer', ascending = True, inplace=True)
		
			self.colors = self.data['color'].values
			self.extract_line_data()
			self.axisDict[0].collections[0].remove()
			# have to replot to change order of lines
			self.fill_axis()
			self.sizeStatsAndColorChanges['change_color_by_categorical_columns'] = categoricalColumn
		else:
			self.axisDict[0].collections[0].set_color(self.data['color'].values)
		
		if len(self.colorMapDict) < 20:
			self.add_means()
			self.add_legend(self.axisDict[0])	
			
	def remove_color(self):
		'''
		Removes color. Triggered by user. 
		'''
		self.colors = 'lightgrey'
		self.axisDict[0].collections[0].set_color(self.colors)
		self.remove_mean_lines()
		self.sizeStatsAndColorChanges.clear()
		
		leg = self.axisDict[0].get_legend()
		if leg is not None:
			leg.remove()	
			
	def remove_mean_lines(self):
		'''
		Removes mean lines.
		'''	
		for line in self.savedMeanLines:
			line.remove()
		self.savedMeanLines = []
	
	def add_means(self,ax = None, export = False):
		'''
		Add lines for mean of categories. Done if colors were encoded by a categorical
		column. 
		Parameter 
		===========
		ax - axis to be used
		export - chart should be exported to a main figure.
		
		'''
		if ax is None:
			ax = self.axisDict[0]
		if export == False:
			self.remove_mean_lines()
		else:
			self.add_mean_lines(ax)
			return
		categoricalColumn =  self.sizeStatsAndColorChanges['change_color_by_categorical_columns']
		self.meanData.clear()
		
		groupedData = self.data.groupby(categoricalColumn)
		for level,color in self.colorMapDict.items():
		 	if str(level)  not in ['nan',' ','-']:	
		 		if level in groupedData.groups.keys():
		 			data = groupedData.get_group(level)
		 		else:
		 			continue
		 		
		 		yData = data[self.numericColumns].mean().values
		 		self.meanData[str(level)] = {'color':'black','marker':'o','markerfacecolor':color,
		 									'markeredgecolor':'darkgrey','markeredgewidth':0.4,
		 									'linestyle':'--','linewidth':0.75,
		 									'xdata':self.xData,'ydata':yData,
		 									'label':'Avarege ({})'.format(level)}
		 		if len(data.index) == 1:
		 			# if data has only one row remove dotted line since it would cover 
		 			# the actual data since mean = data
		 			self.meanData[str(level)]['linestyle'] = ''
		self.add_mean_lines(ax)
		 		
	def add_mean_lines(self,ax):
		'''
		Plot the calculated means. 
		'''
		for lineProps in self.meanData.values():
			line = Line2D(**lineProps)
			ax.add_artist(line)
			self.savedMeanLines.append(line)
	
	def get_size_color_categorical_column(self,which = 'change_color_by_categorical_columns'):
		'''
		'''
		if which in self.sizeStatsAndColorChanges:
			return self.sizeStatsAndColorChanges[which]
	
	def set_user_def_colors(self,categoricalColorDefinedByUser):
		'''
		'''
		self.categoricalColorDefinedByUser = categoricalColorDefinedByUser
								
	def export_selection(self, specificAxis):
		'''
		Export chart to a main figure.
		'''		
		self.fill_axis(specificAxis)
		self.style_axis(specificAxis)
		
		if 'change_color_by_categorical_columns' in self.sizeStatsAndColorChanges:
			self.add_legend(specificAxis,export=True)
			self.add_means(specificAxis,export=True)
		
	def __getstate__(self):
		'''
		Remove axes dict to sterilize object.
		'''
		state = self.__dict__.copy()
		#for obj in ['axisDict']:
		#	if obj in state:
		#		del state[obj]
		return state
			
			
		