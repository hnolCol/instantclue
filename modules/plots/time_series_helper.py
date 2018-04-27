"""
	""TIME SERIES ANAYLSIS""
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

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

import tkinter as tk
		
from modules.utils import *	


class aucResultCollection(object):
	'''
	Used to saved auc caclulations to be accessed at a later time.
	'''
	def __init__(self):
		'''
		'''
		self.aucCalculations = OrderedDict([('Id',[]),
											('Time Column',[]),
											('Numeric Column',[]),
											('Low Bound',[]),
											('Upper Bound',[]),
											('Area Under Curve',[])])
	def save_result(self,resultDict):
		'''
		'''
		for key, value in resultDict.items():
			self.aucCalculations[key].append(value)
	
	@property
	def performedCalculations(self):
		df = pd.DataFrame.from_dict(self.aucCalculations)
		return df
				




class _timeSeriesHelper(object):
	'''
	This class handles actions on time series plot:
		a) Interactive Area under the curve calculations
		b) Base line correction
		c) Adding Error around lines
	'''
	def __init__(self,Plotter,dfClass,ax,data,numericColumns,colors,lines):
		
		self.plotter = Plotter
		self.dfClass = dfClass
		self.lines = lines
		self.ax = ax
		self.data = data
		self.props = {}
		self.numericColumns = numericColumns
		self.timeColumn = numericColumns[0]
		self.ylim = ax.get_ylim()
		self.initiate_prop_dict_and_indicator_lines()
		self.colors = {}
		self.define_colors(colors)
		self.aucCalculations = OrderedDict([('Id',[]),
											('Time Column',[]),
											('Numeric Column',[]),
											('Low Bound',[]),
											('Upper Bound',[]),
											('Area Under Curve',[])])
		self.aucPlottingProps = OrderedDict()
		self.aucItems = OrderedDict() # items - plotted instances
		self.saveAucPolys = dict()
		for column in self.numericColumns[1:]:
			self.saveAucPolys[column] = []
		self.errorPlottingProps = OrderedDict()
		self.baseLineCorrection = OrderedDict()
		self.id = 0		
		
	def disconnect_event_bindings(self):
		'''
		'''
		self.plotter.figure.canvas.mpl_disconnect(self.onClick)

	def activate_baselineCorr_or_aucCalc(self, mode = 'baselineCorrection', columns = None,
											DataTreeview = None):
		'''
		'''
		## used currently on in base line correction
		## why this difference? Since base line correction is selected
		## from the drop down menu all these columns need be to be treated
		## whereas AUC is dropped onto the graph 
		self.extraColumns = columns
		self.DataTreeview = DataTreeview
		
		self.onClick = self.plotter.figure.canvas.mpl_connect('button_press_event',
										lambda event: self.get_clickCoords(event,mode))
		self.reset_settings()
		
		
							
	def add_error_to_lines(self, errorColumnDict):	
		'''
		'''
		
		if len(errorColumnDict) != len(self.numericColumns)-1:
			return
			
		dataId = self.plotter.get_dataID_used_for_last_chart()
		self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=dataId,
									definedColumnsList=list(errorColumnDict.values()))
		self.extract_error_data(errorColumnDict)
		for key,props in self.errorPlottingProps.items():
			self.perform_filling(self.ax,props,auc = False, alpha = 1)
				
	
	def extract_error_data(self,errorColumnDict):
		'''
		'''
		for key,value in errorColumnDict.items():
			columnSelection = [key,value,self.timeColumn]
			# check if not column given
			if all(column in self.data.columns for column in columnSelection):
				data = self.data.dropna(subset = columnSelection)
				self.errorPlottingProps[key] = {'xValues':data[self.timeColumn].values,
											'yValuesHigh' : data[key]+data[value],
											'yValuesLow' : data[key]-data[value],
											'column' : key}		
			
	def get_clickCoords(self,event,mode):
		'''
		'''
		self.plotter.castMenu = True
		if event.inaxes != self.ax:
			return
		if event.button > 1:
			for key,aucItems in self.aucItems.items():
				if aucItems[0].contains(event)[0]:
					for instance in aucItems:
						instance.remove() 
					del self.aucItems[key]
					self.plotter.redraw()
					self.background = self.plotter.figure.canvas.copy_from_bbox(self.ax.bbox)
					return
					
					
					
				
		
		x,y = event.xdata, event.ydata
		
		if self.props['firstClickCoords'] is None:
			if event.button != 1:
				self.plotter.castMenu = False
				## right click removes lines (useful for example if you want to export)
				self.determine()
			self.props['firstClickCoords'] = (x,y)
			self.background = self.plotter.figure.canvas.copy_from_bbox(self.ax.bbox)
			self.movableLine = 'secondClickLine'
			
		elif self.props['secondClickCoords'] is None:
			if event.button == 1:
				self.props['secondClickCoords'] = (x,y)
			else:
				## make a straight horizontal line
				self.plotter.castMenu = False
				self.props['secondClickCoords'] = (x,self.props['firstClickCoords'][1])
			self.plotter.figure.canvas.mpl_disconnect(self.moveLines)
			self.perform_calculations(mode)
		else:
			self.reset_settings()
					
	def move_indicator_lines(self,event):
		'''
		'''
		if self.movableLine is None or self.movableLine not in self.props:
			return
		if event.inaxes != self.ax:
			return
			
		x,y = event.xdata, event.ydata
		self.plotter.figure.canvas.restore_region(self.background)
		self.props[self.movableLine].set_data([x,x],list(self.ylim))
		self.ax.draw_artist(self.props[self.movableLine])
		self.plotter.figure.canvas.blit(self.ax.bbox)

	def perform_calculations(self, mode = 'baselineCorrection'):
		'''
		'''		
		x1,x2,y1,y2 = self.evaluate_x_vals()
		
		boolIndicator = self.data[self.timeColumn].between(x1,x2)
		data = self.data[boolIndicator].dropna(subset=self.numericColumns)
		if mode == 'baselineCorrection':
			## add extra columns 
			self.dataId = self.plotter.get_dataID_used_for_last_chart()
			self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataId,
									definedColumnsList=self.extraColumns)
									
			for column in self.numericColumns + self.extraColumns:
				if column != self.timeColumn:
					self.baseLineCorrection[column] = data[column].median()
			self.correct_baseline()
			self.disconnect_event_bindings()
							
			
		elif mode == 'aucCalculation':
		
			xValues = data[self.timeColumn].values
			xmin, xmax = np.amin(xValues), np.amax(xValues)
			if x1 < xmin : x1 = xmin 
			if x2 > xmax : x2 = xmax
			self.id += 1
			self.aucItems[self.id] = []
			for n,column in enumerate(self.numericColumns):
				if column != self.timeColumn:
					yValues = data[column].values
					xValues, yValues = self.filter_and_order_x_y(xValues,yValues)
					aucData = np.trapz(yValues,xValues)
					aucClick = np.trapz([x1,x2],[y1,y2])
					self.save_auc_calculations({'Area Under Curve':aucData-aucClick,
											'Numeric Column':column,
											'Time Column':self.timeColumn,
											'Low Bound':str((return_readable_numbers(x1),
											return_readable_numbers(y1))),
											'Upper Bound':str((return_readable_numbers(x2),
											return_readable_numbers(y2))),
											'Id':str(self.id)})
					self.aucPlottingProps['{}_{}'.format(self.id,column)] = {'xValues':xValues,
													'yValuesHigh':yValues,
													'xy':[x1,x2,y1,y2],
													 'column':column}
													 
					self.calculate_line_yValues(x1,x2,y1,y2,column)
					self.fill_auc_area(column=column, step = n-1)
			self.reset_settings()
	
	
	def fill_auc_area(self,ax = None, column = None, step = 0):
		'''
		Result
		==========
		 -Filled area indicating auc calculations
		'''
		if ax is None:
			ax = self.ax
			key = '{}_{}'.format(self.id,column)
		else:
			key = None
		if key is not None:
			self.perform_filling(ax,self.aucPlottingProps[key], step = step)
		else:
			n = 0
			for key,props in self.aucPlottingProps.items():
				self.perform_filling(ax,props,step=n)
				n += 1
			
			
	def perform_filling(self,ax,props, auc = True, alpha = 0.3, step = 0, export = False):
		'''
		'''
		if auc:
			if step == 0:	
				x1,x2,y1,y2 = props['xy']
				line = ax.plot([x1,x2],[y1,y2], color = 'k', linewidth=0.3)
				text = ax.text(x=(x1+x2)/2, y = (y1+y2)/2, s=str(self.id), ha = 'center',
							va = 'bottom')
				if export == False:
					self.aucItems[self.id].append(text)
					self.aucItems[self.id].append(line[0])			
			
			color = self.colors[props['column']]
		else:
			ax.plot(props['xValues'],props['yValuesHigh'],
					props['xValues'],props['yValuesLow'],
					color = 'k', linewidth=0.1)
			color = 'lightgrey'
			
		poly = ax.fill_between(props['xValues'],
								props['yValuesHigh'],
								props['yValuesLow'],
								color = color,
								alpha = alpha)
		if auc and export == False:
			self.aucItems[self.id].append(poly)
			self.saveAucPolys[props['column']].append(poly)
					
	
	def calculate_line_yValues(self,x1,x2,y1,y2,column):
		'''
		'''
		xValues = self.aucPlottingProps['{}_{}'.format(self.id,column)]['xValues']
		slope = (y2-y1) / (x2-x1)
		intercept = y1 - slope*x1
		yValues = xValues*slope + intercept
		self.aucPlottingProps['{}_{}'.format(self.id,column)]['yValuesLow'] = yValues
	
	def filter_and_order_x_y(self,x,y):
		'''
		x and y must be a numpy array
		'''
		yFilt = np.isnan(y)
		x = x[~yFilt]
		y = y[~yFilt]
		
		return x,y
		
	def export_selection(self,ax):
		'''
		Exports to another axis (e.g. main figure)
		'''
		if len(self.aucPlottingProps) != 0:
			self.fill_auc_area(ax=ax)
		if len(self.errorPlottingProps) != 0:
			n = 0
			for key,props in self.errorPlottingProps.items():
				self.perform_filling(ax,props,auc = False, alpha = 1, step = n,
										export = True)
				n+=1
		
	def save_auc_calculations(self,param):
		'''
		'''
		for key,value in param.items():
		
			self.aucCalculations[key].append(value)
		# save for long term storage ;) 
		
		self.plotter.store_auc_result(param)
		
	def evaluate_x_vals(self):
		'''
		'''
		x1,y1 = self.props['firstClickCoords']
		x2,y2 = self.props['secondClickCoords']
		if x1 > x2:
			return x2,x1,y2,y1
		elif x2 > x1:
			return x1,x2,y1,y2
		
			
		
	def reset_settings(self):
		'''
		'''
		self.movableLine = 'firstClickLine'	
		self.initiate_prop_dict_and_indicator_lines()
		self.plotter.redraw()
		self.background = self.plotter.figure.canvas.copy_from_bbox(self.ax.bbox)
		self.moveLines = self.plotter.figure.canvas.mpl_connect('motion_notify_event',
										lambda event: self.move_indicator_lines(event))
						
				
	def initiate_prop_dict_and_indicator_lines(self):
		'''
		'''

		self.add_indicator_lines()
		self.props['firstClickCoords'] = None
		self.props['secondClickCoords'] = None

	def add_indicator_lines(self, removeOnly = False):
		'''
		'''	
		lineKeys = ['firstClickLine','secondClickLine']
		for n,key in enumerate(lineKeys):
			if key in self.props:
				self.props[key].remove()
				if removeOnly and n == 0:
					return
		
		lines = self.ax.plot([],[],[],[], color='k', linestyle='--', 
							linewidth = 0.5, label = None)
		for n,key in enumerate(lineKeys):
			self.props[key] = lines[n]

	def	determine(self):
		'''
		Determines the process and resets everything
		'''		
		self.add_indicator_lines(removeOnly = True)
		self.plotter.figure.canvas.mpl_disconnect(self.moveLines)
		self.disconnect_event_bindings()
		self.plotter.redraw()
		self.props = {}
		return		
			
	def get_auc_data(self):
		'''
		Returns the data for auc calculations
		'''
		df = pd.DataFrame.from_dict(self.aucCalculations)
		return df
		
	def define_colors(self,colors):
		'''
		'''
		colors = colors[::-1]
				
		
		for n,column in enumerate(self.numericColumns[::-1]):
			if column != self.timeColumn:
				self.colors[column] = colors[n]
	
	
	def change_color_of_auc(self):
		'''
		'''
		for column in self.numericColumns:
			if column != self.timeColumn:
				color = self.colors[column]
				aucAreas = self.saveAucPolys[column]
				for poly in aucAreas:
					poly.set_color(color)
	
	def adjust_legend(self,colorPalette):
		'''
		'''
		leg = self.ax.get_legend()
		if leg is not None:
			for n,color in enumerate(colorPalette):
				leg.legendHandles[n].set_color(colorPalette[n])								
	
	def change_color_map(self,colorMap):
		'''
		'''
		colorPalette = sns.color_palette(colorMap,len(self.lines))
		for n,line in enumerate(self.lines):
			line.set_color(colorPalette[n])
		self.define_colors(colorPalette)
		self.change_color_of_auc()
		self.adjust_legend(colorPalette)
		self.plotter.redraw()
		self.background = self.plotter.figure.canvas.copy_from_bbox(self.ax.bbox)
		
	def correct_baseline(self):
     	 '''
     	 Corrects the baseline with calculated medians
     	 '''
		
     	 if len(self.baseLineCorrection) != 0:	
     	 	baseCorrColumns = \
     	 	self.dfClass.divide_columns_by_value(self.baseLineCorrection,baseString = 'BaselineCorr')

     	 	if len(baseCorrColumns) > 0:	
     	 		self.DataTreeview.add_list_of_columns_to_treeview(id = self.dataId,
     	 												   dataType = 'float64',
     	 												   columnList = baseCorrColumns)
     	 		tk.messagebox.showinfo('Done ..','Calculations performed.')		
				

