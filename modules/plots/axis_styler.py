"""
	""AXIS STYLING HELPER""
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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick							
					
import seaborn as sns
import numpy as np	
class axisStyler(object):
	'''
	little class to efficiently style axis. 
	'''
	def __init__(self,ax,xlabel = None,ylabel= None,rotationXTicks = None, nTicksOnYAxis = None, nTicksOnXAxis = None,
					title = None,xLabelLastRow = False,removeXTicks = False, yLabelFirstCol = False, showYTicksOnlyFirst = True,
					canvasHasBeenDrawn = False, addLegendToFirstCol = False, addLegendToFirstSubplot = False, kwsLegend = dict(),
					newXLim = None, newYLim = None, forceLegend = False):
		self.ax = ax
		self.canvasHasBeenDrawn = canvasHasBeenDrawn
		
		if xLabelLastRow:
			if self.ax.is_last_row():
				self.set_axis_x_label(xlabel)
			else:
				self.set_axis_x_label('')
				#self.remove_xticks_from_axis()
		else:
			self.set_axis_x_label(xlabel) 

		if yLabelFirstCol:
		
			if self.ax.is_first_col():
				self.set_axis_y_label(ylabel)
			else:
				self.set_axis_y_label('')
				if showYTicksOnlyFirst:
					self.remove_yticks_from_axis()
				
		else:
			self.set_axis_y_label(ylabel)
			
		
		self.set_title(title)
		
		if nTicksOnYAxis is None and nTicksOnXAxis is not None:
			self.rotate_axis_tick_labels(rotationXTicks)
			
		self.reduce_tick_number(nTicksOnYAxis,nTicksOnXAxis,rotationXTicks)
		
		if newXLim is not None:
			ax.set_xlim(newXLim)
		if newYLim is not None:
			ax.set_ylim(newYLim)
		if removeXTicks:
			self.remove_xticks_from_axis()
		
				
		if ax.is_first_col() or forceLegend:
			if (addLegendToFirstSubplot == True and ax.is_first_row()) or\
			 (addLegendToFirstCol and ax.is_first_col()) or forceLegend:
				#if ax.get_legend() is not None:
				#	leg = ax.get_legend()
				#	leg.remove()

				if len(kwsLegend) == 0:
					self.add_legend_to_axis(ax)
				else:
					if 'addPatches' in kwsLegend:
						legendItems = kwsLegend['legendItems']
						colorMap = kwsLegend['colorMap']
						patches = []
						if isinstance(colorMap,str):
							colors = sns.color_palette(colorMap,len(legendItems),desat = 0.75)
						elif isinstance(colorMap, list):
							colors = colorMap
						if 'patchKws' not in kwsLegend:
							kwsLegend['patchKws'] = {}
						
						for n,name in enumerate(legendItems):
							col = colors[n]
							patches.append(mpatches.Patch(edgecolor='black',
                                                           linewidth=0.6,
                                                           facecolor=col, 
                                                           label=name,
                                                            **kwsLegend['patchKws']))
							
						kwsLegend['patches'] = patches
						del kwsLegend['legendItems'] 
						del kwsLegend['colorMap']
						del kwsLegend['addPatches']
						del kwsLegend['patchKws']
													
					self.add_legend_to_axis(ax,**kwsLegend)
	
	def add_legend_to_axis(self,ax,patches = [] , leg_title = '', leg_font_size = '8',handles= None , labels = None, ncols=2, collection_list_legend = None):
		
		if len(patches) > 0:
			leg = ax.legend(handles = patches, bbox_to_anchor=(0., 1.0), loc=3, title = leg_title,
                                   ncol=ncols, borderaxespad=0.)
		elif handles is not None and labels is not None:
			leg = x.legend(handles,labels, bbox_to_anchor=(0., 1.0), loc=3, title = leg_title, ncol=ncols, borderaxespad=0.)                                  
		else:
			leg  = ax.legend( bbox_to_anchor=(0., 1.0), loc=3, title = leg_title,
                                   ncol=ncols, borderaxespad=0.)
            
		if leg is not None:
			leg.draggable(state=True, use_blit=True)
			leg._legend_box.align = 'left'
			#title = leg.get_title()
			#title.set_ha('left')
			#leg.set_title(title)
			
	def set_axis_x_label(self,xlabel):
		'''
		'''
		
		if xlabel is not None:
			
			if isinstance(xlabel,str):
				self.ax.set_xlabel(xlabel)
				
	def set_axis_y_label(self,ylabel):
		''' 
		'''
		if ylabel is not None:
			
			if isinstance(ylabel,str):
				self.ax.set_ylabel(ylabel)			
				
	def set_title(self,title):
		'''
		'''
		if title is not None:
			self.ax.set_title(title)
			
	def rotate_axis_tick_labels(self, rotationXTicks):
		'''
		Rotates the xtick labels.
		Y-tick labeling is not supported.
		'''
		if rotationXTicks is not None:
			self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation = rotationXTicks)   			
		
	def remove_yticks_from_axis(self):
		'''
		'''
		self.ax.set_yticklabels(['' for item in self.ax.get_yticklabels()])	
		
			
	def remove_xticks_from_axis(self):
		self.ax.set_xticklabels(['' for item in self.ax.get_xticklabels()])	
		
	def reduce_tick_number(self,nTicksOnYAxis,nTicksOnXAxis,rotationXTicks):
		'''
		Reduces tick numbers to N
		'''
		def getLabelOrEmptyString(label,index,newLabelIndexes):
			if index in newLabelIndexes:
				return label
			else:
				return ''
		
		if nTicksOnYAxis is not None:
			if self.canvasHasBeenDrawn:
			
				yAxisTickLabels = self.ax.get_yticklabels() 
				numbYAxisTickLabels = len(yAxisTickLabels) 
				newYLabelsIndexes = [int(round(x,0)) for x in np.linspace(0,numbYAxisTickLabels-1, num = nTicksOnYAxis)]
				newYLabels = [getLabelOrEmptyString(label,i,newYLabelsIndexes) for i,label in enumerate(yAxisTickLabels)]
				
			else:
				#y tick labels are all '' if canvas has not been drawn yet
				self.ax.yaxis.set_major_locator(mtick.MaxNLocator(nTicksOnYAxis))    
			
		if nTicksOnXAxis is not None:
			xAxisTickLabels = self.ax.get_xticklabels() 			
			
			if len(xAxisTickLabels) <= nTicksOnXAxis:
				if rotationXTicks is not None:
					self.rotate_axis_tick_labels(rotationXTicks)
				return
				
			if self.canvasHasBeenDrawn:
				
				numbXAxisTickLabels = len(xAxisTickLabels) 
				newXLabelsIndexes = [int(round(x,0)) for x in np.linspace(0,numbXAxisTickLabels-1, num = nTicksOnXAxis)]
			
				newxLabels = [getLabelOrEmptyString(label,i,newXLabelsIndexes) for i,label in enumerate(xAxisTickLabels)]
			
				self.ax.set_xticklabels(newxLabels, rotation = rotationXTicks)
				
			else:
				
				self.ax.xaxis.set_major_locator(mtick.MaxNLocator(nTicksOnXAxis)) 
				self.rotate_axis_tick_labels(rotationXTicks)
		
		elif rotationXTicks is not None:
			
			self.rotate_axis_tick_labels(rotationXTicks)
