"""
	""MODULE TO HANDLE COLOR CHANGES""
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


import matplotlib 
import seaborn as sns

import pandas as pd
import numpy as np

from modules.utils import *


class colorChanger(object):
	'''
	'''
	def __init__(self,Plotter,dfClass,newColorMap, interactiveWidgetHelper):
		
		
		self.plotter = Plotter
		self.newColorMap = newColorMap
		
		## getting settings of current plot
		self.numericColumns,self.categoricalColumns,\
							self.selectedPlotType, self.colorMap = self.plotter .current_plot_settings
							
		self.numbCategoricalColumns = len(self.categoricalColumns)	
		self.numbNumericColumns = len(self.numericColumns)	
						
		self.axes = Plotter.get_axes_of_figure()
		self.dfClass = dfClass
		
		self.make_sure_correct_df_is_selected()
		
		self.interactiveWidgetHelper = interactiveWidgetHelper
		
		self.change_color()
		
		self.plotter.redraw()
		## update cmap in helper - > to get the right color for export and session import
		helper = self.plotter.get_active_helper()
		if helper is not None:
			helper.colorMap = newColorMap
			
	def make_sure_correct_df_is_selected(self):
		'''
		'''
		dfID = self.plotter.get_dataID_used_for_last_chart()
		self.dfClass.set_current_data_by_id(dfID)
		
	def get_items_to_color(self,ax):
		'''
		Returns items 
		'''
		if self.selectedPlotType == 'boxplot':
			axArtists = ax.artists
			return axArtists
		elif self.selectedPlotType == 'density':	
			lines = ax.lines
			if self.numbCategoricalColumns == 0:
				return lines	
			else:
				return lines[0::2]			
		elif self.selectedPlotType == 'violinplot':
			axCollections = ax.collections
			if self.plotter.addSwarm:
				helper = self.plotter.get_active_helper()
				if helper is not None:
					collectionsToUse = helper.inmutableCollections
					axCollections = [coll for coll in axCollections if coll in collectionsToUse]
			axCollections = axCollections[::2]
			return axCollections
		elif self.selectedPlotType == 'barplot':
			axPatches = ax.patches
			if self.numbCategoricalColumns > 0:
				#This is needed because the rectangles are plotted like : 0,0 ,1,1 per group e.g. not in order
				axPatches = [patch for patch in axPatches if np.isnan(patch.get_height()) == False]
				x_pos = [patch.get_x() for patch in axPatches]
				axPatches = [x for (y,x) in sorted(zip(x_pos,axPatches))] 
			return axPatches
			
		elif self.selectedPlotType == 'pointplot':
			axCollections = ax.collections
			return axCollections
			
		elif self.selectedPlotType == 'swarm':
			axCollections = ax.collections
			return axCollections
			

	def change_lines_in_pointplots(self,ax,n,colorPalette,linesPerGroup = 3):
	
		for i in range(0,n):
			lineIdx = i * linesPerGroup
			lineIdxEnd = lineIdx + linesPerGroup ## line + errors
			for lines in ax.lines[lineIdx:lineIdxEnd]:
				lines.set_color(colorPalette[i])
					
                             	 
	def change_color(self):
		'''
		First check all plot types that need a helper function
		'''
		if self.selectedPlotType in ['hclust','corrmatrix']:
			self.plotter.nonCategoricalPlotter._hclustPlotter.change_cmap_of_cluster_map(self.newColorMap)
			return
			
		elif self.selectedPlotType in ['scatter','cluster_analysis','PCA'] and self.numbCategoricalColumns == 0:
			
			self.plotter.nonCategoricalPlotter.update_colorMap(self.newColorMap)
			self.interactiveWidgetHelper.update_new_colorMap()
			return
		elif self.selectedPlotType == 'line_plot':
			self.plotter.nonCategoricalPlotter.linePlotHelper.update_colorMap(self.newColorMap)
			self.interactiveWidgetHelper.update_new_colorMap()
			return
			
		elif self.selectedPlotType in ['scatter'] and self.numbCategoricalColumns > 0:
			self.plotter.categoricalPlotter.scatterWithCategories.update_colorMap(self.newColorMap)
			self.interactiveWidgetHelper.update_new_colorMap()
			return
					
		elif self.selectedPlotType == 'time_series':
			
			self.plotter.nonCategoricalPlotter.timeSeriesHelper.change_color_map(self.newColorMap)
			return
			
		elif self.selectedPlotType == 'density':
			for ax in self.axes:
				axArtists = self.get_items_to_color(ax)
				nTotal = len(axArtists)
				newColorPalette = sns.color_palette(self.newColorMap,nTotal,desat=.75)
				for n,line in enumerate(axArtists):
					line.set_color(newColorPalette[n])
				
				self.adjust_legend_color(ax,nTotal,newColorPalette, which = 'color')
			
			return
			
		
		if self.numbCategoricalColumns == 0:
				ax = self.axes[0]
				axArtists = self.get_items_to_color(ax)
				newColorPalette = sns.color_palette(self.newColorMap,self.numbNumericColumns,desat=.75)
				for m,artist in enumerate(axArtists):
					
					if self.selectedPlotType == 'pointplot':
						artist.set_facecolor(newColorPalette)
						if m == 0:
							self.change_lines_in_pointplots(ax,self.numbNumericColumns,newColorPalette)
					else:
						artist.set_facecolor(newColorPalette[m])
		
		elif self.selectedPlotType in ['boxplot','violinplot','barplot','swarm'] \
		and self.plotter.splitCategories == False:
			
			nTotal = self.numbCategoricalColumns + 1 # +1 for one Whole population
			newColorPalette = sns.color_palette(self.newColorMap,nTotal,desat=.75)
			
			for n,column in enumerate(self.numericColumns):
				ax = self.axes[n]
				axArtists = self.get_items_to_color(ax)
				
				if len(axArtists) % nTotal == 0:
					for m in range(nTotal):
						axArtists[m].set_facecolor(newColorPalette[m])
					self.adjust_legend_color(ax,nTotal,newColorPalette)
				
		
					
		elif self.numbCategoricalColumns == 1:	
			ax = self.axes[0]
			uniqueLevels = self.dfClass.get_unique_values(self.categoricalColumns[0])
			numbUniqueLevels = uniqueLevels.size
			
			newColorPalette = sns.color_palette(self.newColorMap,numbUniqueLevels,desat=.75)
			
			if self.selectedPlotType != 'pointplot':
				newColorPalette = newColorPalette * len(self.numericColumns) * 2 #2 is a bit of security as sstripplot (swarm) adds more collections than levels	
							
			axArtists = self.get_items_to_color(ax)
			if len(axArtists) % numbUniqueLevels == 0:
				for m,artist in enumerate(axArtists):
					if self.selectedPlotType == 'pointplot':
						artist.set_facecolor(newColorPalette[m])
						
						linesPerGroup = self.numbNumericColumns * 3 + 1
						self.change_lines_in_pointplots(ax,numbUniqueLevels,
														newColorPalette, linesPerGroup = linesPerGroup)
					else:
						artist.set_facecolor(newColorPalette[m])
					
			else:
				colorDictNew = match_color_to_uniqe_value(uniqueLevels,self.newColorMap)
				collectUnique = []
				for numColumn in self.numericColumns:
					data = self.dfClass.df[[numColumn]+self.categoricalColumns].dropna(subset=[numColumn])
					collectUnique = collectUnique + data[self.categoricalColumns[0]].unique().tolist()
				for n,uniqueVal in enumerate(collectUnique):
					axArtists[n].set_facecolor(colorDictNew[uniqueVal])
					
			self.adjust_legend_color(ax,numbUniqueLevels,newColorPalette)
					
		elif self.numbCategoricalColumns > 1:
			uniqueLevels = self.dfClass.get_unique_values(self.categoricalColumns[1])
			orderInPlot = list(uniqueLevels)
			numbUniqueLevels = uniqueLevels.size
			newColorPalette = sns.color_palette(self.newColorMap,numbUniqueLevels,desat=.75)
			colorDictNew = match_color_to_uniqe_value(uniqueLevels,self.newColorMap)
			
			if self.numbCategoricalColumns == 3:
				groupedData3 = self.dfClass.get_groups_by_column_list([self.categoricalColumns[2]])
				uniqSplits3 = self.dfClass.get_unique_values(self.categoricalColumns[2])
			for n,ax in enumerate(self.axes):
				axArtists = self.get_items_to_color(ax)
				if self.numbCategoricalColumns == 2:
					data = self.dfClass.df
				else:
					data = groupedData3.get_group(uniqSplits3[n])
				
				
				xAxisSplit = data.groupby(by=self.categoricalColumns[0],sort=False)
				if self.selectedPlotType != 'pointplot':
					m = 0
					for names, group0 in xAxisSplit:
						uniquePresent = group0[self.categoricalColumns[1]].unique().tolist()
						uniquePresent.sort(key=lambda x: orderInPlot.index(x))
						for value in uniquePresent:
							color = colorDictNew[value]
							axArtists[m].set_facecolor(color)
							m += 1
				else:
					splitsPresentInData = data[self.categoricalColumns[1]].unique().tolist()
					splitsPresentInData.sort(key=lambda x: orderInPlot.index(x))
					linesPerGroup = data[self.categoricalColumns[0]].unique().size * 3 + 1
					for m, category in enumerate(splitsPresentInData):
						color = colorDictNew[category]
						axArtists[m].set_facecolor(color)
						if m == 0:
							self.change_lines_in_pointplots(ax,len(splitsPresentInData),
														newColorPalette, linesPerGroup = linesPerGroup)
						
				
				self.adjust_legend_color(ax,numbUniqueLevels,newColorPalette)
			

		
	def adjust_legend_color(self,ax,n,colorPalette,which = 'facecolor'):
		leg = ax.get_legend() 
		
		if leg is not None:
			for n in range(n):
				if which == 'facecolor':
					leg.legendHandles[n].set_facecolor(colorPalette[n])
				elif which == 'color':
					leg.legendHandles[n].set_color(colorPalette[n])
		

		
		
		
		
		
		
		
	