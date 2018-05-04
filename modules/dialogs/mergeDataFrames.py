"""
	""MERGE DATA FRAMES""
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
import numpy as np

import tkinter as tk
from tkinter import ttk

from modules.pandastable import core 
from modules.utils import *
from modules import images
from modules.dialogs.simple_dialog import simpleListboxSelection

parameters = {'Merge':OrderedDict([('how',['left','right','outer','inner']),
						('indicator',['True','False']),
						('suffixes',['_x,_y']),
						('left data frame','all'),
						('right data frame','all')]),
					
		  	  'Concatenate':OrderedDict([('join',['outer','inner']),
		  				 ('ignore_index',['True','False']),
		  				 ('axis',['index','columns']),
		  				 ('Data frames','all')])}
descriptionMerge = ['full outer: Use union of keys from both frames\n'+
                'inner: Use intersection of keys from both frames\n'+
                'left out.: Use keys from left frame only\n'+
                'right out.: Use keys from right frame only']
                
indicatorText = ('Add a column to the output DataFrame called _merge with information on the source of each row.'+
	' _merge is Categorical-type and takes on a value of left_only for observations whose merge key '+
	'only appears in "left" DataFrame, right_only for observations whose merge key only appears in "right"'+
	' DataFrame, and both if the observation’s merge key is found in both.')


toolTipText = {'how':descriptionMerge,'join':descriptionMerge,'indicator':indicatorText,
				'suffixes':'Column suffixes if column name appears in both data frames',
				'axis': 'The axis to concatenate along.'}



class mergeDataFrames(object):
	'''
	Window Dialog to facilitate merging and concatenation using the 
	pandas package. User has to select two data frames for merging.
	
	Output
	========
	Merged data frame which is added to the source data treeview in Instant Clue
	'''
	
	def __init__(self,dfClass,dataTreeView, method = 'Merge', images = None):
		'''
		Define variables, build toplevel and grid widgets. 
		'''
		self.treeviews = OrderedDict()
		self.dfColumns = dict()
		self.dfSelection = dict()
		self.joinOptions = dict()
		self.dataTreeView = dataTreeView 
		self.dataFrameList = dataTreeView .dataFramesSelected
		if len(self.dataFrameList) < 2:
			tk.messagebox.showinfo('Error ..','You need to select at least two data frames.')
			return
		self.dfClass = dfClass
		self.fileNames = dfClass.fileNameByID	
		self.tw = None	
		self.images = images
		if method == 'Merge':
			## update the dfColumns dict
			self.extract_df_information()
					
		
		self.method = method
		
		self.build_toplevel()
		self.build_widgets()
		
		self.toplevel.wait_window()
	
	def close(self,event=None):
		'''
		Closes the dialog window.
		'''
		if self.tw is not None:
			self.tw.destroy()
		self.toplevel.destroy()
			
	def build_toplevel(self):
		'''
		Builds toplevel of specific size and title.
		'''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Combine data frames') 
		popup.bind('<Escape>', self.close)
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=620
		h=430
		self.toplevel = popup
		self.center_popup((w,h))
			
	
	def build_widgets(self):
		'''
		Defines widgets and grids them. 
		'''
		self.cont = tk.Frame(self.toplevel, bg = MAC_GREY)
		self.cont.pack(expand=True, fill=tk.BOTH)
		
		labTitle = tk.Label(self.cont, text = 'Control of dataframe merging/concatenation\nIf'+
											   ' you select multiple columns to merge on, make'+
											   ' sure they are in the correct order.',
											   **titleLabelProperties)
							
							
		labTitle.grid(columnspan=2, sticky=tk.W, padx=10,pady=3)
		
		infoButton = ttk.Button(self.cont,text='Help',command=self.open_help)
		infoButton.grid(row = 0, column=3,padx=5,pady=3)
		
		
		n = 2
		for label, options_ in parameters[self.method].items():
			var = tk.StringVar()
			if isinstance(options_, list):
				labelWidget = tk.Label(self.cont, text = label, bg = MAC_GREY)
				comboBox = ttk.Combobox(self.cont, textvariable = var, values = options_)
				labelWidget.grid(row=n,padx=5,pady=3, sticky=tk.E)
				comboBox.grid(row=n, column =1, padx=5,pady=3, sticky=tk.EW)
				self.joinOptions[label] = var
				comboBox.insert(tk.END,options_[0])
				if label in toolTipText:
					text = toolTipText[label]
					CreateToolTip(labelWidget, text = text, title_ = label) 
			else:
				self.create_treeview_frame(n)
				
				if label == 'left data frame' :
					self.create_treeview(label)
					self.enter_information_in_treeview(self.treeviews[label],self.dfColumns[label])
				elif label == 'right data frame':
					self.create_treeview(label, row=0, column = 1)
					self.enter_information_in_treeview(self.treeviews[label],self.dfColumns[label])
				elif label == 'Data frames':
					self.create_treeview(label)
					self.enter_information_in_treeview(self.treeviews[label],self.fileNames)
					
			n+=1
		if self.method == 'Merge': 
			
			defineLeftColsButton = ttk.Button(self.cont, text  = 'Columns [left]', 
											command = lambda : self.subset_dfs('left'))
			defineRightColsButton = ttk.Button(self.cont, text  = 'Columns [right]', 	
											command = lambda : self.subset_dfs('right'))
			defineLeftColsButton.grid(row= n +1, padx=15, sticky=tk.W,column=0,pady=10)
			defineRightColsButton.grid(row= n + 1, padx=15, sticky=tk.E, column=4,pady=10)		
		
							
		combineButton = ttk.Button(self.cont, text = str(self.method), 
									command = self.perform_operation)
		closeButton = ttk.Button(self.cont, text = 'Close', 
									command = self.close)
		

		combineButton.grid(row= n +2, padx=15, sticky=tk.W,column=0,pady=10)
		closeButton.grid(row= n +2, padx=15, sticky=tk.E, column=4,pady=10)
	
	def enter_information_in_treeview(self,treeview,iidAndItemDict):
		'''
		Insert items in treeview.
		
		Parameter
		==========
		treeview - Tkinter treeview widget
		iidAndItemDict - dict. Key are iids. Items in dict are text information.
		
		Output
		==========
		None
		'''
		for iid, entry in iidAndItemDict.items():
			
			treeview.insert('','end',iid,text=entry)
	
	
	def extract_df_information(self):
		'''
		Get information from df and store them in a dict (dfColumns) 
		'''					
		for n,dfID in enumerate(self.dataFrameList):
			dfColumnDict = OrderedDict()
			columnList = self.dfClass.get_columns_of_df_by_id(dfID)
			for column in columnList:
				dfColumnDict['{}_{}'.format(dfID,column)] = column
			if n == 0:
				self.dfColumns['left data frame'] = dfColumnDict
				self.dfSelection['left data frame'] = []
			elif n == 1:	
				self.dfColumns['right data frame'] = dfColumnDict
				self.dfSelection['right data frame'] = []
		
		
	def extract_merge_props(self,columnsForMerge = None):
		'''
		Extracts selected properties. 
		
		Parameter
		==========
		columnsForMerge - dict. This is only not None if merge is used inseated of
						  concatenation. This is used to extract the selected columns
						  to be used for merging for the left and right data frame. 
		Output 
		==========
		props 			- dict. Properties accepted keywords of the function. pd.concat
						  or pd.merge from the pandas package. 
		'''
		props = dict()
		for prop, variable in self.joinOptions.items():	
			if prop == 'suffixes':
				propValue = variable.get().split(',')
				if len(propValue) != 2:
					propValue = ('_x','_y')
				else:
					pass
			elif prop == 'indicator':
				propValue = variable.get()
				if propValue == 'True':
					propValue= True
				elif propValue == 'False':
					propValue = False
				else:
					## this will create a column called like the string passed..
					pass
			else:
				propValue = variable.get()
				
			props[prop] = propValue
		if columnsForMerge is not None:
			# get the column names for merge
			props['left_on'] = [column[len(self.dataFrameList[1])+1:] for column in columnsForMerge['left data frame']] #cutoff the data frame id from the iid = columnName
			props['right_on'] = [column[len(self.dataFrameList[1])+1:] for column in columnsForMerge['right data frame']]
			if self.evaluate_input(props) is None:
				return	
		return props	

	def extract_columns(self,props):
		
		collect = []
		for key,selection in self.dfSelection.items():
			allColumns = list(self.dfColumns[key].values())
			
			if len(selection) == 0:
				collect.append(allColumns)
				
			else:
				side = key.split(' ')[0]
				selectedColumns = [column for column in allColumns  if \
								   column in props['{}_on'.format(side)] or column in selection]
				collect.append(selectedColumns)		
		
		return collect
			
			
			
			
			
				
	def evaluate_input(self,props):
		'''
		Evaluates input of columns to be used for merging. We take this extra computing
		time to warn the user if a column was selected that contains less unique values
		than rows (e.g. the merging could result in unexpected results since the merging
		occurs at the first matching item)
		'''
		leftId = self.dataFrameList[0]
		rightId = self.dataFrameList[1]
		nonSuitableColumn = []
		
		if len(props['left_on']) != len(props['right_on']):
			tk.messagebox.showinfo('Error ..','Number of columns to be used for matching'+
								' has to be the same for left and right data frames.',parent =self.toplevel)
			return 
		
		self.dfClass.set_current_data_by_id(leftId)
		uniqueValues = self.dfClass.get_unique_values(props['left_on'],forceListOutput=True)
		rows = self.dfClass.get_row_number()
		for n,column in enumerate(props['left_on']):
			if uniqueValues[n].size != rows:
				nonSuitableColumn.append(column)
			
			
		# now check right df
		self.dfClass.set_current_data_by_id(rightId)
		uniqueValues = self.dfClass.get_unique_values(props['right_on'],forceListOutput=True)
		rows = self.dfClass.get_row_number()
		for n,column in enumerate(props['right_on']):
			if uniqueValues[n].size != rows:
				nonSuitableColumn.append(column)
		
		if len(nonSuitableColumn) != 0:
			quest = tk.messagebox.askquestion('Warning ..','Warning - Column(s) used for merging contain(s)'+
					' less unique values than rows. Unexpected results are possible.\n'+
					' Columns: {}\n Abort?'.format(get_elements_from_list_as_string(nonSuitableColumn)),
					parent= self.toplevel)
			if quest == 'yes':
				return
		return True
		
		
	def subset_dfs(self,which):
		'''
		'''
		columnKey = '{} data frame'.format(which)
		if columnKey not in self.dfColumns:
			return
		else:
			columns = list(self.dfColumns[columnKey].values())
			
			
		dialog = simpleListboxSelection('Selected columns will be \nused for merging.',
										columns)
		selection = dialog.selection
		print(selection)
		self.dfSelection[columnKey] = selection
		print(self.dfSelection)
		
		
		
			
	def create_treeview_frame(self, row):
		'''
		Create tkinter Treeview to display colum headers or dataframes.
		User must select from all present tree views.
		'''
		if hasattr(self, 'treeviewFrame') == False:
			self.treeviewFrame = tk.Frame(self.cont, bg=MAC_GREY)
			self.treeviewFrame.grid(row=row, padx= 20, columnspan=6, sticky=tk.NSEW + tk.W)
			self.treeviewFrame.grid_rowconfigure(1,weight=1)
			self.cont.grid_columnconfigure(5,weight=1)
			self.cont.grid_rowconfigure(row,weight=1)
			
	def create_treeview(self, caption, row = 0, column = 0):
		'''
		Create a treeview to display either column name of two 
		data frames (merge) or data frames (concatenate).
		
		Parameter
		=========
		caption - Caption of treeview
		row     - row where to grid treeview
		column	- column where to grid treeview
		=========
		Output	- None 
		'''
		
		labelTree = tk.Label(self.treeviewFrame, text = caption, bg=MAC_GREY)
		labelTree.grid(row=row, sticky=tk.W, column = column)
		treeview = ttk.Treeview(self.treeviewFrame,show='tree', 
							style='source.Treeview')
								 
		scbarVert = ttk.Scrollbar(self.treeviewFrame, orient=tk.VERTICAL)
		treeview.config(yscrollcommand = scbarVert.set)
		scbarVert.config(command = treeview.yview)
			
		scbarHor = ttk.Scrollbar(self.treeviewFrame, orient=tk.HORIZONTAL)
		treeview.config(xscrollcommand = scbarHor.set)
		scbarHor.config(command = treeview.xview)			
			
			
		#grid scrollbars and treeview
		treeview.grid(row=row+1, sticky=tk.NSEW, column = column, padx=(0,16),pady=(0,16))
		scbarVert.grid(row=row+1,column = column, sticky=tk.NS+tk.E, padx = (0,4))
		scbarHor.grid(row=row+1,column = column, sticky=tk.EW+tk.S)
		self.treeviewFrame.grid_columnconfigure(column,weight=1)
		self.treeviews[caption] = treeview
						
					
		
	def perform_operation(self):
		'''
		Either merges or concatenates selected data frames.
		'''
		columnsForMerge = dict()
		
		for label, treeview in self.treeviews.items():
			selectedItems =  list(treeview.selection())
			numItemsSelected = len(selectedItems)
			if self.method == 'Merge' and numItemsSelected == 0:
				tk.messagebox.showinfo('Please select ..',
					'Please select columns to be used for merging.',
					parent = self.toplevel)
				return
			elif self.method == 'Concatenate' and numItemsSelected == 0:
				tk.essagebox.showinfo('Please select ..',
					'Please select dataframes to be used for concatenation.',
					parent = self.toplevel)
				return
			if self.method == 'Merge':
				columnsForMerge[label] = selectedItems
			else:
				dfsToConat = [self.dfClass.dfs[iid] for iid in selectedItems]
				dfsFileNames = [self.fileNames[iid] for iid in selectedItems]
				
		if self.method == 'Concatenate':
			propsConcat = self.extract_merge_props()
			
			try:
				newDf = pd.concat(dfsToConat, **propsConcat)
			except Exception as e:
				tk.messagebox.showinfo('Error ..',
					'An error occured during merge:\n'+str(e),
					parent=self.toplevel)
				return
			nameOfNewDf = '{}: {}'.format(self.method,
										get_elements_from_list_as_string(dfsFileNames,maxStringLength=10))
		else:
			propsMerge = self.extract_merge_props(columnsForMerge)
			## check if user has selected columns to use	
			if propsMerge is None:
				return		
			leftCol, rightCol = self.extract_columns(propsMerge)			
			
			leftDf = self.dfClass.dfs[self.dataFrameList[0]][leftCol]
			rightDf = self.dfClass.dfs[self.dataFrameList[1]][rightCol]
			
			try:
				newDf = pd.merge(leftDf, rightDf, sort = False, **propsMerge)
			except Exception as e:
				tk.messagebox.showinfo('Error ..',
					'An error occured during merge:\n'+str(e),
					parent=self.toplevel)
				return
			nameOfNewDf = '{}: {} {}'.format(self.method,self.dataFrameList[0],self.dataFrameList[1])
		id = self.dfClass.get_next_available_id()
		dfName = self.dfClass.evaluate_column_name(nameOfNewDf,self.fileNames.values())
		
		self.dfClass.add_data_frame(newDf, id = id, fileName = dfName)
		columnDataTypeRel = self.dfClass.get_columns_data_type_relationship_by_id(id)
		self.dataTreeView.add_new_data_frame(id,dfName,columnDataTypeRel)
		tk.messagebox.showinfo('Done ..','Done. Merged/Concatenated data frame was added.'+
										  'The shape of the new data is {}. \n\n Please note'.format(newDf.shape)+
										  ' that due to insertion of NaN for non matching rows, '+
										  'integers are cosnidered as floats.'+
										  'This does not happen using "inner" join.', parent = self.toplevel)
		self.close()




	def open_help(self):
		'''
		Opens a tkinter toplevel and display a help picture.
		'''
		if self.tw is None:
			self.tw = tk.Toplevel()
			if 'how' in self.joinOptions:
				key = 'how'
			else:
				key = 'join'
			value = self.joinOptions[key].get()
			if platform in ['WINDOWS','LINUX']:
				self.tw.wm_overrideredirect(True)
			else:
				self.tw.tk.call("::tk::unsupported::MacWindowStyle","style",self.tw._w, "plain", "none")
			self.lab = tk.Label(self.tw , image = self.images[value])
			self.lab.grid()
			x, y = self.tw.winfo_pointerxy()
			self.tw.wm_geometry("+%d+%d" % (x, y))
		else:
			self.tw.destroy()
			self.tw = None
        	

					
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 
