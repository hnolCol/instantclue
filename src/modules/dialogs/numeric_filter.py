"""
	""NUMERIC FILTER""
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

import tkinter as tk
from tkinter import ttk   
          
import tkinter.simpledialog as ts

import numpy as np
import pandas as pd

from modules.utils import *

operator_options = ['> greater than',
	'>= greater equal than','== equal',
	'!= unequal','<= smaller equal than',
	'< smaller than','[] / () within']


class numericalFilterDialog(object):

	def __init__(self,dfClass,numericalColumns):
		
		if len(numericalColumns) == 0:
			return
				
		self.numericalColumns = numericalColumns
		self.numbNumericalColumns = len(numericalColumns)
		self.mainOperator = tk.StringVar()


		self.columnName = None
		self.replaceDict = {True:'+',False:'-'}
		
		self.dfClass = dfClass
		 
		self.build_toplevel() 
		self.build_widgets()
		
		self.toplevel.wait_visibility()
		self.toplevel.grab_set() 
		self.toplevel.wait_window() 
		
	def close(self, event = None):
		'''
		Close toplevel
		'''
		self.toplevel.destroy() 
			
	
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Numerical Filter') 
		popup.bind('<Escape>', self.close) 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w = 650
		h = 140 + self.numbNumericalColumns * 45
		
		self.toplevel = popup
		self.center_popup((w,h))
		
			
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont.grid_columnconfigure(2,weight=1)
 		
 		labTitle = tk.Label(self.cont, text = 'Add a categorical column with "+" matching the given criteria.', font = LARGE_FONT, fg="#4C626F", justify=tk.LEFT, bg = MAC_GREY)
 		labMainOperator = tk.Label(self.cont, text = 'Operator used for all given conditions: ', bg = MAC_GREY)
 		
 		mainOperatorMenu = ttk.OptionMenu(self.cont, self.mainOperator,'and', *['and','or'])             
                          
 		labTitle.grid(padx=5,pady=15, columnspan=6 ,sticky=tk.W)
 		labMainOperator.grid(padx=5, columnspan=6 ,sticky=tk.W,pady=8)
 		mainOperatorMenu.grid(row=1 ,sticky=tk.E, column=3,pady=8,padx=60)
 		
 		self.add_filter_widgets_for_columns()
 		
 		addButton = ttk.Button(self.cont, text='Apply', 
 			command = self.apply_numeric_filter, width = 6)
 		closeButton = ttk.Button(self.cont, text = 'Close', 
 			command = self.close, width = 6)
 		
 		buttonRow = self.numbNumericalColumns + 5
 		addButton.grid(row=buttonRow, column=3,padx=3,pady=8,sticky=tk.E)
 		closeButton.grid(row=buttonRow, column=4,padx=3,pady=8,sticky=tk.E)
             
            

	def add_filter_widgets_for_columns(self):
		'''
		'''
		self.filterSettings = OrderedDict()
		
		for n,numColumn in enumerate(self.numericalColumns):
			## row to grid widgets 
			row = n+5
			
			variableOperator = tk.StringVar()
			
			## add checkbutton for option to use absolute values
			absoluteValuesCB = ttk.Checkbutton(self.cont, text ='abs values of:  ')
			absoluteValuesCB.state(['!alternate'])
			## column label
			columnLabel = tk.Label(self.cont, text=numColumn, 
								   bg=MAC_GREY, font = LARGE_FONT, 
								   fg="#4C626F")
			
			## operator menu 
			opCombo = ttk.Combobox(self.cont, textvariable = variableOperator ,
				values = operator_options, width = 13)
			opCombo.insert('end','> greater than')
			opCombo.configure(state='readonly')
			
			## entry for user to provide criteria 
			filterEntry = ttk.Entry(self.cont, width = 4)
			
			## grid widgets
			absoluteValuesCB.grid(row=row, column = 1, sticky=tk.W, pady=8, padx=3)
			columnLabel.grid(row=row,column = 2, sticky=tk.EW, pady=8, padx=3)
			opCombo.grid(row=row,column =3 , sticky=tk.E,pady=8, padx=3)
			filterEntry.grid(row=row,column=4, sticky=tk.EW,pady=8, padx=8,columnspan=2)
			##  Lets save everything in a dict to be used later in applying filter
			self.filterSettings[numColumn] = [absoluteValuesCB, variableOperator, filterEntry]
		
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 
         	
         	
	def  apply_numeric_filter(self):
 		'''
 		Applies the filter. The general idea is to check for all conditions separately
 		and then combine depending on how the user wants to combine the the critera.
 		'''
 		self.collectDataFrame = pd.DataFrame()
 		
 		for numColumn in self.numericalColumns:
 			absValuesCB, variableOperator,filterEntry = self.filterSettings[numColumn]
 			
 			absoluteValues = absValuesCB.instate(['selected'])
 			operator = variableOperator.get() 
 			if 'within' not in operator:
 				try:
 					filterValue = float(filterEntry.get())
 				except:
 					tk.messagebox.showinfo('Error ..',
 						'Could not convert input to float. Empty?',
 						parent=self.toplevel)
 					return
 			else:
 				filterValue = filterEntry.get()
 			
 			
 			try:
 				self.collectDataFrame[numColumn]  = self.check_if_condition_is_true(filterValue,operator,
 																				numColumn,absoluteValues)
 			except:
 				tk.messagebox.showinfo('Error ..',
 					'Could not interpret interval. Examples: (4.0,4.2),[5,14],(2,100].',
 					parent = self.toplevel)
 				return
 					
 		if self.mainOperator.get() == 'and':
 			selectionColumn = self.collectDataFrame.sum(axis=1) == self.numbNumericalColumns
 		
 		else:
 			selectionColumn = self.collectDataFrame.sum(axis=1) > 0
 		
 		outputColumn = selectionColumn.map(self.replaceDict)
 		
 		self.columnName = self.build_columnName()
 		self.columnName = self.dfClass.evaluate_column_name(self.columnName, useExact = True)
 		self.dfClass.add_column_to_current_data(self.columnName,outputColumn, evaluateName = False)
 		self.close()

	def build_columnName(self):
		'''
		Builds column name to identify the filtering
		'''
		
		operator = [variableOperator.get()[:2] for _,variableOperator,_ in self.filterSettings.values()]
		columnName = 'numFilt_{}_{}_{}'.format(self.mainOperator.get(),
							get_elements_from_list_as_string(self.numericalColumns, maxStringLength = 20),
							operator).replace(" ",'').replace("'",'')
		return columnName

	def check_if_condition_is_true(self,filterValue,operator,numColumn,absoluteBool):
		'''
		Checks the condition for the particular numerical column and returns and panda series with boolean in it
		'''
		
		if absoluteBool:
			data = self.dfClass.df[numColumn].abs()
		else:
			data = self.dfClass.df[numColumn]
		
		if 'greater than' in operator:
			boolIndicator = data > filterValue
		elif 'greater equal than' in operator:
			boolIndicator = data >= filterValue			
		elif 'unequal' in operator:
			boolIndicator = data != filterValue
		elif 'smaller equal than' in operator:
			boolIndicator = data <= filterValue
		elif 'smaller than' in operator:
			boolIndicator = data < filterValue
		elif 'equal' in operator:
			boolIndicator = data == filterValue
		elif 'within' in operator:
			min,max = filterValue.split(',')
			minValue = float(min[1:])
			maxValue = float(max[:-1])
			if min[0] == '[' and max[-1] == ']':
				boolIndicator = (data >= minValue) & (data <= maxValue)
			elif max[-1] == ')' and min[0] == '(':
				boolIndicator = (data > minValue) & (data < maxValue)
			elif max[-1] == ')' and min[0] == '[':
				boolIndicator = (data >= minValue) & (data < maxValue)
			elif max[-1] == ']' and min[0] == '(': 
				boolIndicator = (data > minValue) & (data <= maxValue)

		return boolIndicator
		
		
 