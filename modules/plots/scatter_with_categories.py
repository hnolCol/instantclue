"""
	""SCATTER PLOT VARIANTS""
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
import matplotlib.gridspec as gridspec
import numpy as np

from collections import OrderedDict
import itertools
import matplotlib.patches as patches

from modules.utils import *
from modules.plots.axis_styler import axisStyler
from modules.plots.scatter_plotter import scatterPlot    			

class binnedScatter(object):
	'''
	Binned scatter is a version of a scatter plot. 
	Data are histogrammed/binnd into a given number of bins. 
	Then the points are scaled by the number of counts.
	'''
	def __init__(self,plotter,dfClass,numericColumns, categoricalColumns):
		'''
		'''
		self.plotter = plotter
		self.dfClass = dfClass
		self.numBins = self.plotter.numbBins
		self.scaleCounts = bool(self.plotter.scaleBinsInScatter)
				
		self.axisDict = dict()
		self.numericColumns = numericColumns
		self.categoricalColumns = categoricalColumns
		
		self.get_scatter_plot_props()
		self.prepare_data()
		if hasattr(self,'sizeData'):
			self.replot()
	
	
	def replot(self):
		'''
		Replot chart. (Needed when session is loaded)
		'''
		self.create_axis()
		self.fill_axis()
		self.style_axis()
	
	def get_scatter_plot_props(self):
		'''
		'''
		self.sizeScatterPoints, self.alphaScatterPoints,\
		self.colorScatterPoints = self.plotter.get_scatter_point_properties()
		
		self.scatProps = {'alpha':self.alphaScatterPoints,
						  'facecolor':self.colorScatterPoints,
						  'linewidth':0.3,
						  'edgecolor':'black'}
		
	
	def prepare_data(self):
		'''
		Prepare data. Histogram data and find counts
		'''
		rawData = self.dfClass.get_current_data_by_column_list(columnList = self.numericColumns + self.categoricalColumns)
		
		data = rawData[self.numericColumns].dropna().values
		
		twoDimHist = np.histogram2d(data[:,0], data[:,1], bins = self.numBins)		
		xs = twoDimHist[1]
		ys = twoDimHist[2]
		binnedData = []
		for (i, j),v in np.ndenumerate(twoDimHist[0]):
			if v != 0:
				binnedData.append((xs[i], ys[j], v))
		binnedData = np.array(binnedData)
		
		
		self.rawCounts = binnedData[:,2]
		if self.scaleCounts:
			sizeData = scale_data_between_0_and_1(binnedData[:,2])
			# emperically
			self.sizeData = (sizeData+0.01)*250 
		else:
			self.sizeData = binnedData[:,2]+15
			
		xyData = list(twoDimHist[-2:])
		self.xyData = binnedData[:,:2]
				
		
	def create_axis(self):
		'''
		Add axis to figure.
		'''
		self.axisDict[0] = self.plotter.figure.add_subplot(111)
		
	def fill_axis(self, specificAxis = None):
		'''
		Plot the actual data.
		'''
		if specificAxis is None:
			ax = self.axisDict[0]
		else:
			ax = specificAxis
		ax.scatter(x = self.xyData[:,0], 
								y = self.xyData[:,1],
								sizes=self.sizeData,
								label = None,**self.scatProps)
		self.add_legend(ax)
		
	def style_axis(self,ax=None):
		'''
		Style axis.
		'''
		if ax is None:
			ax = self.axisDict[0]
			
		axisStyler(ax,forceLegend=True,
						kwsLegend = dict(leg_title='Counts',ncols=3),
						nTicksOnYAxis = 5, nTicksOnXAxis = 5)
		
		 	
	def add_legend(self, ax):
		'''
		Add legend to figure.
		'''
		summary = dict() 	
		#maybe sorting first would be better and then take indexes	
		summary['min'], summary['max'], summary['mean'] = \
		np.argmin(self.sizeData), np.argmax(self.sizeData), arg_mean_median(self.sizeData,'mean')
		
		for metric in ['min','mean','max']:
			ax.scatter([],[],label= '{} ({})'.format(int(self.rawCounts[summary[metric]]),metric), 
								sizes = [self.sizeData[summary[metric]]],
								**self.scatProps)
	
		
	def export_selection(self, specificAxis):
		'''
		export selection
		'''
		self.fill_axis(specificAxis)
		self.style_axis(specificAxis)
		





	
class scatterWithCategories(object):
	'''
	'''
	def __init__(self,plotter,dfClass,figure,categoricalColumns=[],
								numericalColumns=[], colorMap = 'Blues'):#data,n_cols,n_categories,colnames,catnames,figure,size,color):
	

		self.colorMap = colorMap
		self.dataID = dfClass.currentDataFile 
		self.data = dfClass.get_current_data_by_column_list(categoricalColumns+numericalColumns)
		self.dfClass = dfClass
		self.data.dropna(inplace=True)
		self.plotter = plotter
		
		self.define_variables()
		
		self.numericalColumns = numericalColumns
		self.categoricalColumns = categoricalColumns
		self.numbNumericalColumns = len(numericalColumns)
		self.numbCaetgoricalColumns = len(categoricalColumns)
		
		self.size = self.plotter.sizeScatterPoints 
		self.color = self.plotter.colorScatterPoints
		self.alpha = self.plotter.alphaScatterPoints
		
		self.figure = figure
		
		plt.figure(self.figure.number)
		
		self.get_size_interval()
		self.get_unique_values() 
		self.group_data()
		
		n_rows,n_cols = self.calculate_grid_subplot()
		self.prepare_plotting(n_rows,n_cols) 


	def define_variables(self):
		'''
		'''
		self.grouped_data = None
		self.grouped_keys = None
		
		self.unique_values = OrderedDict() 	
		self.axes = OrderedDict() 
		self.scatterPlots = OrderedDict()
		self.label_axes = OrderedDict() 
		self.axes_combs = OrderedDict()
		self.subsets_and_scatter = OrderedDict() 
		self.sizeStatsAndColorChanges = OrderedDict()
		self.annotationClasses = OrderedDict() 
		
		self.categoricalColorDefinedByUser = dict()
		self.colorMapDict = dict()		
				
	
	def replot(self):
		'''
		'''
		self.grouped_keys = self.grouped_data.groups.keys()
		plt.figure(self.figure.number)
		n_rows,n_cols = self.calculate_grid_subplot()
		self.prepare_plotting(n_rows,n_cols) 
		
		
	def prepare_plotting(self,n_rows,n_cols):
		'''
		Function to plot different groups ...
		'''	
		self.figure.subplots_adjust(wspace=0, hspace=0, right=0.96)

		titles = list(self.unique_values[self.categoricalColumns[0]][0])
		if self.numbNumericalColumns == 1:
			# Plots the numeric column against index 
			min_y = self.data[self.numericalColumns[0]].min()
			max_y = self.data[self.numericalColumns[0]].max()
			
			
		else:
			# Plot numeric column against numeric column if there are two columns
			min_x, max_x = self.data[self.numericalColumns[0]].min(), self.data[self.numericalColumns[0]].max()
			min_y, max_y = self.data[self.numericalColumns[1]].min(), self.data[self.numericalColumns[1]].max()
			xlim = 	(min_x - 0.1*min_x, max_x + 0.1*max_x )	
		
		ylim = (min_y - 0.1*min_y, max_y + 0.1*max_y )			
			
			
		if self.numbCaetgoricalColumns  > 1:
		
			y_labels = list(self.unique_values[self.categoricalColumns[1]][0])
			
		if self.numbCaetgoricalColumns == 3:
		
			levels_3,n_levels_3 = self.unique_values[self.categoricalColumns[2]]
			outer = gridspec.GridSpec(n_levels_3, 1, hspace=0.01)
			gs_saved = dict() 
			for n in range(n_levels_3):			
				gs_ = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec = outer[n], hspace=0.0)
				gs_saved[n] = gs_
			
		for i,comb in enumerate(self.all_combinations):
			
			if comb in self.grouped_keys:
				group = self.grouped_data.get_group(comb)

				if self.numbNumericalColumns == 1:
					n_data_group = len(group.index)
					x_ = list(range(0,n_data_group))
					y_ = group[self.numericalColumns[0]]
					#self.numericalColumns.append('idx')
					group.loc[:,'idx'] = x_
				
				else:			
					x_ = group[self.numericalColumns[0]]
					y_ = group[self.numericalColumns[1]]
				

			else:
				### '''This is to plot nothing if category is not in data'''
				group = None
				x_ = []
				y_ = []

			
			pos = self.get_position_of_subplot(comb) 
			if self.numbCaetgoricalColumns < 3:
				n = 0
				ax_ = plt.subplot2grid((n_rows, n_cols), pos)
				
			else:
				n = levels_3.index(comb[2])
				ax_ = self.create_ax_from_grid_spec(comb,pos,n, gs_saved)	
			
			numColumns = self.numericalColumns
			if len(numColumns) == 1:
				numColumns = ['idx'] + self.numericalColumns
			self.scatterPlots[i] = scatterPlot(
									data=group,
									numericColumns = numColumns,
									plotter = self.plotter,
									colorMap = self.colorMap,
									dfClass = self.dfClass,									
									ax = ax_,
									dataID = self.dataID,								
									scatterKwargs  = {'s':self.size,'alpha':self.alpha,
									'picker':True,'label':None,'color':self.color,
									'edgecolor':'black','linewidth':0.3},
									showLegend = True if i == 0 else False,
									ignoreYlimChange = True)
			
			
			
			#scat = self.plotter.add_scatter_collection(ax_,x_,y_, color = self.color, 
			#					size=self.size, alpha=self.alpha,picker=True)
			self.annotate_axes(ax_)
														
			#ax_.plot(x_,y_,'o',color = self.color,ms = np.sqrt(self.size),
					#markeredgecolor ='black',markeredgewidth =0.3)#,linestyle=None)					
			if ax_.is_last_row() == False and self.numbCaetgoricalColumns < 3 and self.numbNumericalColumns > 1:
			
				ax_.set_xticklabels([])
				 
			elif self.numbCaetgoricalColumns == 3 and self.numbNumericalColumns > 1:
				if n != n_levels_3-1:
					ax_.set_xticklabels([])
				else:
					if ax_.is_last_row() == False:
						ax_.set_xticklabels([])
			else:
				ax_.set_xticklabels([])
			if ax_.is_last_col():			
				ax_.yaxis.tick_right()
			else:
				ax_.set_yticklabels([])	
			ax_.set_ylim(ylim)
			if self.numbNumericalColumns == 2:
				ax_.set_xlim(xlim)
			self.axes[i] = ax_
			#self.axes_combs[comb] = [ax_,scat]
		self.add_labels_to_figure()	
		
		for scatPlot in self.scatterPlots.values():
			setattr(scatPlot,'ignoreYlimChange',False)

	def annotate_axes(self,ax_):
		'''
		'''
		if ax_.is_first_col():
			if self.numbNumericalColumns == 2:
					text_ = self.numericalColumns[1]
			else:
					text_ = self.numericalColumns[0]
			self.plotter.add_annotationLabel_to_plot(ax_,
													text=text_,
													rotation = 90)
		if ax_.is_last_col():		
			if self.numbNumericalColumns == 2:
					text_ = self.numericalColumns[0]
			else:
					text_ = 'Index' 
			self.plotter.add_annotationLabel_to_plot(ax_,
														text=text_,
														position = 'bottomright')

	def save_color_and_size_changes(self,funcName,column):
		'''
		'''
		self.sizeStatsAndColorChanges[funcName] = column
	

	def remove_color_and_size_changes(self,which='color'):
		'''
		Removes color and size changes from all subplots.
		'''
		for ax,_ in self.axes_combs.values():
			axCollections = ax.collections 
			for coll in axCollections:
				if which == 'size' and hasattr(coll,'set_sizes'):
					coll.set_sizes([self.size])
				elif which == 'color' and hasattr(coll,'set_color'):
					coll.set_facecolor(self.color)

	def get_size_interval(self):
		'''
		'''
		self.minSize, self.maxSize = self.plotter.get_size_interval()
	
	def update_size_interval_in_chart(self):
		'''
		'''
		for key, categoricalColumns in self.sizeStatsAndColorChanges.items():
			if 'size' in key:
				self.get_size_interval()
				getattr(self,key)(categoricalColumns)
				break 
				
									
	def change_size_by_numerical_column(self,numericColumn):
		'''
		Change size of scatter points by numerical values.
		We need to calculate the limits before otherwise all
		scatter plots will have the same range of data.
		'''	
		values = self.dfClass.get_data_by_id(self.dataID)[numericColumn].values.flatten()
		limits = [np.min(values),np.max(values)]
		for scatterPlot in self.scatterPlots.values():
			scatterPlot.change_size_by_numerical_column(numericColumn,limits = limits)			
	
	
	def change_size_by_categorical_columns(self,categoricalColumn):
		'''
		
		'''
		self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
																  definedColumnsList=[categoricalColumn])
		uniqueCategories = self.data[categoricalColumn].unique()
		numberOfUuniqueCategories = uniqueCategories.size
		scaleSizes = np.linspace(0.3,1,num=numberOfUuniqueCategories,endpoint=True)
		sizeMap = dict(zip(uniqueCategories, scaleSizes))
		sizeMap = replace_key_in_dict('-',sizeMap,0.1)
				
		for scatterPlot in self.scatterPlots.values():
			scatterPlot.change_size_by_categorical_column(categoricalColumn, sizeMap = sizeMap)
		return		
		
		
		self.data.loc[:,'size'] = self.data[categoricalColumn].map(sizeMap)
		self.data.loc[:,'size'] = (self.data['size'])*(self.maxSize-self.minSize) + self.minSize
		self.adjust_size()		
		self.save_color_and_size_changes('change_size_by_categorical_columns',categoricalColumn)
	
	
	def adjust_size(self):
		'''
		'''
		self.group_data()	
		for comb in self.all_combinations:
			if comb in self.grouped_keys:
				subset = self.grouped_data.get_group(comb)
				ax,_ = self.axes_combs[comb]
				axCollection = ax.collections
				axCollection[0].set_sizes(subset['size'])


	def adjust_color(self):
		'''
		'''
		self.group_data()	
		for comb in self.all_combinations:
			if comb in self.grouped_keys:
				subset = self.grouped_data.get_group(comb)
				ax,_ = self.axes_combs[comb]
				axCollection = ax.collections
				axCollection[0].set_facecolor(subset['size'])


	def bind_label_event(self, labelColumnList):
		'''
		'''
		for scatterPlot in self.scatterPlots.values():
			scatterPlot.bind_label_event(labelColumnList)

	def add_annotation_from_df(self,df):
		'''
		'''		
		df_grouped = df.groupby(self.categoricalColumns,sort=False)
		grouped_df = df_grouped.groups.keys()
		for comb in self.all_combinations:
			if comb in grouped_df:
				annotationData = df_grouped.get_group(comb)
				self.annotationClasses[comb].addAnnotationFromDf(annotationData)		
				
				
			
	def change_color_by_numerical_column(self,numericColumn, updateColor = True):
		'''
		accepts a numeric column from the dataCollection class. numeric is added using 
		the index ensuring that correct dots get the right color. 
		'''
		for scatterPlot in self.scatterPlots.values():
			scatterPlot.change_color_by_numerical_column(numericColumn, update = updateColor)
		
		#self.save_color_and_size_changes('change_color_by_numerical_column',numericColumn)
		
			
	def change_color_by_categorical_columns(self,categoricalColumn, updateColor = True):
		'''
		'''
		for scatterPlot in self.scatterPlots.values():
			scatterPlot.change_color_by_categorical_columns(categoricalColumn,updateColor = updateColor)

		
	def get_current_colorMapDict(self):
		'''
		'''
		for scatterPlot in self.scatterPlots.values():
				return scatterPlot.colorMapDict
		
		
	def set_user_def_colors(self,categoricalColorDefinedByUser):
		'''
		'''
		for scatterPlot in self.scatterPlots.values():
			scatterPlot.categoricalColorDefinedByUser = categoricalColorDefinedByUser	
		
			
	def update_colorMap(self,newColorMap=None):
		'''
		'''
		if newColorMap is not None:
			self.colorMap = newColorMap
		for scatterPlot in self.scatterPlots.values():
			scatterPlot.update_colorMap(newColorMap)
			
	def change_nan_color(self,newColor):
		'''
		'''
		for ax in self.plotter.get_axes_of_figure():
			for coll in ax.collections:
				coll.set_facecolor(newColor)
		self.color = newColor

	def change_size(self,size):
		'''
		'''
		for ax in self.plotter.get_axes_of_figure:
			for coll in ax.collections:
				coll.set_sizes([size])
		self.color = size		
	
	def create_ax_from_grid_spec(self,comb,pos, n, gs_saved):
		'''
		Gets the appropiate axis from selected gridspec- In principle we have gridspecs in one big gridspec, 
		this allows hspacing between certain categories on y axis.
		'''
		
		gs_ = gs_saved[n]	
		ax = plt.subplot(gs_[pos])
		return ax
				
	def get_size_color_categorical_column(self, which='change_color_by_categorical_columns'):
		'''
		'''
		for scatterPlot in self.scatterPlots.values():
			if which  in scatterPlot.sizeStatsAndColorChanges:
				return scatterPlot.sizeStatsAndColorChanges[which]
		
			
	def add_labels_to_figure(self):
		'''
		Adds labels to figure  - still ugly 
		To DO. make this function cleaner. 
		'''
		
		levels_1,n_levels_1 = self.unique_values[self.categoricalColumns[0]]
		
		if self.numbCaetgoricalColumns == 1:
			bottomSpace = 0.5
			
		else:
			bottomSpace = 0.14
		
		self.figure.subplots_adjust(left=0.15, bottom= bottomSpace) 
		
		ax_top = self.figure.add_axes([0.15,0.89,0.81,0.15])
		ax_top.set_ylim((0,4))
		ax_top.axis('off') 
		width_for_rect = 1/n_levels_1
		kwargs_rectangle_main = dict(edgecolor='black',clip_on=False,linewidth=0.1,fill=True)
		kwargs_rectangle = dict(edgecolor='black',clip_on=False,linewidth=0.1,fill=False)
		ax_top.add_patch(patches.Rectangle((0,1),1,1,**kwargs_rectangle_main))
		ax_top.text(0.5, 1.5 , s = self.categoricalColumns[0], horizontalalignment='center',verticalalignment = 'center',color="white")
		
		for n,level in enumerate(levels_1):
			
			x = 0 + n * width_for_rect
			y = 0
			width = width_for_rect
			height = 1 
			ax_top.add_patch(patches.Rectangle((x,y),width,height,**kwargs_rectangle))
			ax_top.text(x + width/2 , height/2, s = level, 
				horizontalalignment='center',verticalalignment = 'center')
		self.label_axes['top_limits'] = [ax_top.get_xlim(),ax_top.get_ylim()]	
		self.label_axes['top'] = ax_top
		
		if self.numbCaetgoricalColumns > 1:
		
			ax_left = self.figure.add_axes([0.02,0.14,0.1,0.74])
			ax_left.axis('off')
			ax_left.set_xlim((0,4)) 
			ax_left.add_patch(patches.Rectangle((2,0),1,1,**kwargs_rectangle_main))
			ax_left.text(2.5, 0.5 , s = self.categoricalColumns[1], verticalalignment='center', 
				rotation=90,horizontalalignment='center',color="white")
			levels_2,n_levels_2 = self.unique_values[self.categoricalColumns[1]]
			if self.numbCaetgoricalColumns == 3:
				levels_3,n_levels_3 = self.unique_values[self.categoricalColumns[2]]
				n_levels_2 = n_levels_2 * n_levels_3
				levels_2 = levels_2 * n_levels_3
				
			height_for_rect = 1/n_levels_2
			for n,level in enumerate(levels_2):
				
				y = 1 - (n+1) * height_for_rect
				x = 3
				width = 1
				height = height_for_rect 
				
				ax_left.add_patch(patches.Rectangle((x,y),width,height,**kwargs_rectangle))
				ax_left.text(x + width/2 , y + height/2, s = level, verticalalignment='center', 
					rotation=90,horizontalalignment='center')
			if self.numbCaetgoricalColumns == 3:
			
				
				ax_left.add_patch(patches.Rectangle((0,0),1,1,**kwargs_rectangle_main))
				ax_left.text(0.5, 0.5 , s = self.categoricalColumns[2], verticalalignment='center', rotation=90,
									horizontalalignment='center',color="white")
				height_for_rect = 1/n_levels_3
				for n,level in enumerate(levels_3):
					y = 1 - (n+1) * height_for_rect
					x = 1
					height = height_for_rect
					ax_left.add_patch(patches.Rectangle((x,y),width,height,**kwargs_rectangle))					
					ax_left.text(x + width/2 , y + height/2, s = level, verticalalignment='center', rotation=90,horizontalalignment='center')
			
			self.label_axes['left'] = ax_left
		if self.numbCaetgoricalColumns > 1:
			self.inmutableAxes = [self.label_axes['top'],self.label_axes['left']]
		else:
			self.inmutableAxes = [self.label_axes['top']]
			
		

		
			
	def get_position_of_subplot(self, comb):
		'''
		Returns the position of the specific combination of levels in categorical columns. 
		Seems a bit complicated but is needed if a certain combination is missing.
		'''
		levels_1, n_levels_1 = self.unique_values[self.categoricalColumns[0]]
		if self.numbCaetgoricalColumns == 1:
			row = 0
			col = levels_1.index(comb)
		else:
			levels_2,n_levels_2 = self.unique_values[self.categoricalColumns[1]]
			col = levels_1.index(comb[0])
			row = levels_2.index(comb[1])
			
			
		return (row,col)
	
	
	
	def calculate_grid_subplot(self):
		'''
		Calculates the subplots to display data
		'''
	
		## get columns of n_cat 1 
		if self.numbCaetgoricalColumns == 1:
		
			levels_1, n_levels = self.unique_values[self.categoricalColumns[0]]
			n_cols = n_levels
			n_rows = 1 
			self.all_combinations = list(levels_1)
			
		elif self.numbCaetgoricalColumns == 2:
		
			levels_1, n_levels_1 = self.unique_values[self.categoricalColumns[0]]
			n_cols = n_levels_1
			levels_2, n_levels_2 = self.unique_values[self.categoricalColumns[1]]
			n_rows = n_levels_2
			self.all_combinations = list(itertools.product(levels_1,levels_2))
				
		elif self.numbCaetgoricalColumns == 3:
		
			levels_1, n_levels_1 = self.unique_values[self.categoricalColumns[0]]
			n_cols = n_levels_1
			levels_2, n_levels_2 = self.unique_values[self.categoricalColumns[1]]
			n_rows = n_levels_2
			levels_3, n_levels_3 = self.unique_values[self.categoricalColumns[2]]
			self.all_combinations = list(itertools.product(levels_1,levels_2,levels_3))	
			
		return n_rows, n_cols	
			
	
	def get_unique_values(self):
		'''
		Determines unique vlaues in each category, that is needed to build the subplots
		'''
		for category in self.categoricalColumns:
			
			uniq_levels = self.data[category].unique()
			n_levels = uniq_levels.size
			
			self.unique_values[category] = [list(uniq_levels),n_levels]
			
	def get_raw_colorMapDict(self):
		'''
		'''
		for scatterPlot in self.scatterPlots.values():
			return scatterPlot.rawColorMapDict
		
	def group_data(self):
		'''
		Defines a pandas groupby object with grouped data on selected categories.
		'''
		
		self.grouped_data = self.data.groupby(self.categoricalColumns, sort = False) 
		self.grouped_keys = self.grouped_data.groups.keys()
		
		
		
	def __getstate__(self):
		'''
		Remove stuff that cannot be steralized by pickle
		'''
		state = self.__dict__.copy()
		for attr in ['figure','grouped_keys']:
			if attr in state: 
				del state[attr]
		return state
			#'_Plotter',		
		
		