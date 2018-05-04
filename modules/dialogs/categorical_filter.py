"""
	""CATEGORICAL FILTERING""
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
import tkinter.font as tkFont
           

import numpy as np
import pandas as pd

from itertools import chain
from functools import reduce

import csv
import re

from modules.utils import *
from modules import images
from modules.pandastable import core 

operationTitle = {'Find category & annotate':'Select unique categories for annotation.\nMatching rows are annotated by a "+" sign.',
				  'Search string & annotate':'Enter string to search in selected column.\nMatching rows are annotated by a "+" or the search string itself.',
				  'Subset data on unique category':'Select unique categories for subset.\nMatching rows will be kept, other will be dropped from the source file.',
				  'Annotate scatter points':'Select rows that you would like to annotate in the current plot.',
				  'Find entry in hierarch. cluster':'Select row that you would like to find in the cluster.',
				  'Find entry in line plot':'Select row that you would like to find in the line plot.'}
				
operationMessage = {'Find category & annotate':'Annotation done. Column has been added to the tree view.',
				  'Search string & annotate':'Searching and annotation done. Column has been added to the tree view.',
				  'Subset data on unique category':'Subset of data has been added.',
				  'Annotate scatter points':'',
				  'Find entry in hierarch. cluster':'',
				  'Find entry in line plot':''}
				

class categoricalFilter(object):
	'''
	Categorical filter can be used for several operations. 
	
	=================
	Operations
		- Find category & annotate
		- Search string & Annotate
		- Subset data on unique category
		- Annotate scatter points 
		- Find rows in hierarchichal clustering
		- 
	=================
	
	Annotate scatter points and Finding rows in hierarchichal clsustering may seem
	unreasonable but it is meant to be applied if the user wants to label a specific
	row. Then the search function that is also used to find strings/categories is 
	very much suitable
	
	'''
	def __init__(self,dfClass, dataTreeview, plotterClass ,
				operationType = 'Find category & annotate', 
				dataSubset = None, columnForFilter = None, addToTreeview = True):
		'''
		=====
		Parameter
			dataSubset - If you dont want to make the whole data set available that is currently
				selected but only a subset (for example for annotations). Subset must have
				the same columns as the currently selected df by self.dfClass
		=====
		'''
		self.define_annotation_command_relation()
		
		self.searchString = tk.StringVar()
		self.separatorString = tk.StringVar()
		self.separatorString.set(';')
		self.caseSensitive = tk.BooleanVar(value=True)
		self.annotateSearchString = tk.BooleanVar(value=False)
		self.onlyFirstFind = tk.BooleanVar(value=False)
		self.userRedExpression = tk.BooleanVar(value=False)
		self.annotationColumn = tk.StringVar()
		self.protectEntry = 1
		self.operationType = operationType
		self.closed = False
		self.addToTreeview = addToTreeview
		
		self.plotter = plotterClass
		self.dfClass = dfClass
		
		## make sure data of plot is selected
		if self.operationType in ['Annotate scatter points',
								  'Find entry in hierarch. cluster',
								  'Find entry in line plot']:
				self.dataID = plotterClass.get_dataID_used_for_last_chart() 
				self.dfClass.set_current_data_by_id(self.dataID)
		
		
		if self.operationType == 'Find entry in line plot':
			ax = self.plotter.axes[0]
			self.background = self.plotter.figure.canvas.copy_from_bbox(ax.bbox)
		
		if dataSubset is None:
			self.df = self.dfClass.get_current_data() 
		else:
			self.df = dataSubset
		self.dataTreeview = dataTreeview
		self.columnForFilter = columnForFilter
		
		self.replaceDict = {True : "+",
                        False: self.dfClass.replaceObjectNan}
		
		self.saveLastString = ''

		self.build_toplevel()
		self.build_widgets()
		
		self.prepare_data()
		self.display_data()
		
		self.toplevel.wait_window()
		
	def close(self,event = None):
		'''
		Close toplevel
		'''
		self.closed = True
		if hasattr(self,'pt'):
			del self.pt
		self.toplevel.destroy() 	
		

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Categorical Filter - '+self.operationType) 
		popup.grab_set()
		popup.bind('<Escape>', self.close) 
        
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=520
		h=630
		self.toplevel = popup
		self.center_popup((w,h))
		
			
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont_widgets = tk.Frame(self.cont,background=MAC_GREY) 
 		self.cont_widgets.pack(fill=tk.X, anchor = tk.W) 
 		self.cont_widgets.grid_columnconfigure(1,weight=1)
 		self.create_preview_container() 
 		
 		self.doneIcon, self.refreshIcon = images.get_done_refresh_icons()
 		
 		labelTitle = tk.Label(self.cont_widgets, text= operationTitle[self.operationType], 
                                     **titleLabelProperties)
 		labelSearch = tk.Label(self.cont_widgets, text = 'String :', bg = MAC_GREY) 
 		entrySearch = ttk.Entry(self.cont_widgets, textvariable = self.searchString)
 		entrySearch.unbind('<Command-v>')
 		
 		if self.operationType == 'Search string & annotate':
 			entrySearch.bind('<Control-v>', self.copy_from_clipboard)
 			if platform == 'MAC':
 				entrySearch.bind('<Command-v>', self.copy_from_clipboard)
 		if platform == 'MAC':
 			entrySearch.bind('<Command-z>', self.undo) 	
 			
 		entrySearch.bind('<Control-z>', self.undo) 			
 		entrySearch.bind('<Return>',lambda event: \
 		self.update_data_upon_search(event, forceUpdate = True))
 		self.searchString.trace(mode="w", callback=self.update_data_upon_search)
 		
 		if self.operationType in ['Find category & annotate','Subset data on unique category']:
 		
 			## for these types we want to provide the possibility to change the separator to get unique values
 			labelSeparator = tk.Label(self.cont_widgets, text = 'Sep. :', bg = MAC_GREY)
 			sepComboBox = ttk.Combobox(self.cont_widgets, textvariable = self.separatorString,
 									   exportselection = 0, 
 									   values = [';',',',':','-','_','/'])
 			self.separatorString.set(';')
 			
 			refreshButton = tk.Button(self.cont_widgets, image=self.refreshIcon, 
 														 command = self.refresh_separator)
 			
 			labelSeparator.grid(row=1,padx=5, pady=5, sticky=tk.W)  
 			sepComboBox.grid(row=1, column=1, sticky=tk.EW, padx=2)
 			refreshButton.grid(row=1, column=2, sticky=tk.E, padx=2)
 			
 		elif self.operationType == 'Search string & annotate':
 		
 			labelInfo = tk.Label(self.cont_widgets,text= 'For multiple search strings type: "String1","String2" ..', 
 								 bg=MAC_GREY, justify=tk.LEFT)
 		
 			annotateStringCb = ttk.Checkbutton(self.cont_widgets, 
 				variable = self.annotateSearchString, text = 'Annotate matches by search string(s)')
 			onlyFirstFind = ttk.Checkbutton(self.cont_widgets, 
 				variable = self.onlyFirstFind , text = 'Annotate combinations (String1,String2)',
 				command = self.check_cbs)
 			caseSensitive = ttk.Checkbutton(self.cont_widgets, 
 					variable = self.caseSensitive, text = 'Case sensitive',
 					command = lambda: self.update_data_upon_search(forceUpdate = True))
 				
 			useRegRexpress = ttk.Checkbutton(self.cont_widgets, 
 										variable = self.userRedExpression, 
 										text = 'Input is a regular expression')
 										
 			labelInfo.grid(row=1,column=0,columnspan=3,padx=3,sticky = tk.W,pady=4)
 			annotateStringCb.grid(row=2,column=0,columnspan=2,padx=3,sticky=tk.W) 
 			onlyFirstFind.grid(row=3,column=0,columnspan=2,padx=3,sticky=tk.W) 
 			useRegRexpress.grid(row=3,column=1,columnspan=2,padx=3,sticky=tk.E)
 			caseSensitive.grid(row=2,column=1,columnspan=2,padx=3,sticky=tk.E)
 		
 		elif self.operationType in ['Annotate scatter points',
 				'Find entry in hierarch. cluster','Find entry in line plot']:
 			
 			labelColumn = tk.Label(self.cont_widgets,text= 'Column:', bg=MAC_GREY)
 								 
 			optionMenuColumn = ttk.OptionMenu(self.cont_widgets, self.annotationColumn, 
 											  self.dfClass.df_columns[0], *self.dfClass.df_columns,
 											  command = self.update_data) 
 			caseSensitive = ttk.Checkbutton(self.cont_widgets,
 										variable = self.caseSensitive, 
 										text = 'Case sensitive',
 										command = lambda : self.update_data_upon_search(forceUpdate = True))
 			
 			caseSensitive.grid(row=2,column=1,columnspan=2,padx=3,sticky=tk.E)
 			optionMenuColumn.grid(row=3,column=1,columnspan=1,sticky=tk.EW)
 			labelColumn.grid(row=3,column=0,sticky=tk.W,padx=3)
 		
 		## creating buttons for applying 
 		
 		applyButton = tk.Button(self.cont_widgets, image=self.doneIcon, command = self.commandDict[self.operationType])

 		labelTitle.grid(row=0,column=0,padx=4, pady = 15, columnspan = 3, sticky=tk.W)
 		labelSearch.grid(row=5,padx = 5, pady = 5, sticky=tk.W)
 		entrySearch.grid(column = 1, row = 5, sticky=tk.EW, padx=2)
 		applyButton.grid(row=5,column=2,sticky=tk.E,padx=2)
 		
	def prepare_data(self):
		'''
		Prepares the data.
		'''
		if self.operationType in ['Find category & annotate','Subset data on unique category']:
			sepString = self.separatorString.get()
			splitData = self.df[self.columnForFilter].astype('str').str.split(sepString).values
			flatSplitData = list(set(chain.from_iterable(splitData)))
			self.uniqueFlatSplitData = pd.DataFrame(flatSplitData,columns=[self.columnForFilter])
			self.splitString = sepString

		elif self.operationType == 'Search string & annotate':
			
 			self.uniqueFlatSplitData = self.df[self.columnForFilter].astype('str')
		
		elif self.operationType in ['Annotate scatter points',
									'Find entry in hierarch. cluster',
									'Find entry in line plot']:
 			if self.columnForFilter is None:
 				self.columnForFilter = self.annotationColumn.get()
 			
 			if self.plotter.nonCategoricalPlotter is not None:
 				self.numericColumns = self.plotter.nonCategoricalPlotter.numericColumns
 			elif self.plotter.categoricalPlotter.scatterWithCategories is not None:
 				self.numericColumns = self.plotter.categoricalPlotter.scatterWithCategories.numericalColumns
 			
 			if self.operationType != 'Find entry in line plot': # allows NaNs 
 				self.df = self.df.dropna(subset=self.numericColumns)
 			
 			if self.columnForFilter not in self.df.columns:
 				## add columns if needed. Since the data in plotting classes do not 
 				## automatically contain all needed columns
 				self.df = self.dfClass.join_missing_columns_to_other_df(self.df, 
 																		self.dataID, 
 																		[self.columnForFilter])
 					
 			self.uniqueFlatSplitData = pd.DataFrame(self.df[self.columnForFilter].astype('str'),
 																columns=[self.columnForFilter])

	def update_data(self,columnName = None):
 		'''
 		Updates the data if users uses a new column in the option menu. 
 		'''
 		self.columnForFilter = columnName
 		self.prepare_data()
 		self.pt.model.df = self.uniqueFlatSplitData 
 		self.pt.redraw()		
			
	def update_data_upon_search(self,varname = None, elementname = None, 
										mode=None, forceUpdate = False):
		'''
		Updates data upon change of the StringVar self.searchString. Will return None if
		the search string is short. 
		'''
		if self.protectEntry < 1:
			if self.protectEntry == -1:
				self.protectEntry += 1
				pass
			elif self.protectEntry == 0:
				self.searchString.set(self.outputString)
				self.protectEntry += 1
				
				return		
		
		searchString = self.searchString.get()
		nonEmptyString = searchString != ''
		lenSearchString = len(searchString)
		if lenSearchString < 3 and nonEmptyString and forceUpdate == False:
			## to avoid massive searching when data are big
			return
			
		if self.operationType == 'Search string & annotate' and lenSearchString > 2:
			## to start a new String search
			if searchString[-2:] == ',"':			
				self.saveLastString = ''
			
		lengthSaved = len(self.saveLastString)
		
		
		if lenSearchString == lengthSaved + 1 and self.saveLastString != '':
			dataToSearch = self.pt.model.df
		elif lenSearchString + 1 == lengthSaved and self.saveLastString != '':
			## avoid research on backspace
			dataToSearch = self.pt.model.df
		else:
			dataToSearch = self.uniqueFlatSplitData
		if len(dataToSearch.index) == 0:
			dataToSearch = self.uniqueFlatSplitData
			
		if self.operationType == 'Search string & annotate':
			
			
			if self.userRedExpression.get():
				regExp = re.escape(searchString)
			else:
				splitSearchString = [row for row in csv.reader([searchString], 
											delimiter=',', quotechar='\"')][0]
				regExp = self.build_regex(splitSearchString,withSeparator=False)
				
			collectDf = pd.DataFrame()
			for n,column in enumerate(self.columnForFilter):
				try:
					collectDf.loc[:,str(n)] = \
					dataToSearch[column].str.contains(regExp,
													  case = self.caseSensitive.get())
				except:
					return
			boolIndicator = collectDf.sum(axis=1) >= 1			
			subsetData = dataToSearch[boolIndicator]
		
 			#self.uniqueFlatSplitData = pd.DataFrame(self.df[self.columnForFilter].astype('str'),columns=[self.columnForFilter])
		
		else: 	
			boolIndicator = dataToSearch[self.columnForFilter].str.contains(searchString,
																	case = self.caseSensitive.get()).values
			subsetData = dataToSearch[boolIndicator]
		
		self.pt.model.df = subsetData
		self.pt.redraw()
		
		self.saveLastString = searchString 
	
	def check_cbs(self):
		'''
		Controls checkbuttons status
		'''
		if self.userRedExpression.get():
			self.annotateSearchString.set(False)
			self.onlyFirstFind.set(False)
					
		if self.onlyFirstFind.get():
			self.annotateSearchString.set(True)
			
		
	def refresh_separator(self):
		'''
		Data are split on a spearator. Allowing extraction of unique categories.
		This function splits the data and inserts them in the pandastable.
		'''
		splitString = self.separatorString.get()
		# compare to used split string, if the same - return
		if 	splitString == self.splitString:
			return
		else:
			self.saveLastString = ''
			self.prepare_data()
			self.update_data_upon_search()
								
	def display_data(self):
		'''
		Displays data in a pandastable. The 
		'''
		self.pt = core.Table(self.cont_preview,
						dataframe = self.uniqueFlatSplitData, 
						showtoolbar=False, 
						showstatusbar=False)
						
		## unbind some events that are not needed
		if platform == 'MAC':			
			self.pt.unbind('<MouseWheel>') # for Mac it looks sometimes buggy 

		self.pt.unbind('<Double-Button-1>')
		
		self.pt.show()		
				
	def create_preview_container(self,sheet = None):
		'''
		Creates preview container for pandastable. 
		'''
		self.cont_preview  = tk.Frame(self.cont,background='white') 
		self.cont_preview.pack(expand=True,fill=tk.BOTH,padx=(1,1))		       
 		
          
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 

	def get_selected_row(self):
		'''
		Returns the rows selected in pandastable
		'''
		return self.pt.multiplerowlist	
		
	def get_selected_category(self):
		'''
		Checks the pandastable object for selected rows
		'''
		rowsSelected = self.get_selected_row()
		
		textSelected = [self.pt.model.getValueAt(row, 0) for row in rowsSelected]
		return textSelected
		
		
	def add_annotation_column(self):
		'''
		Gets user's selection and adds the new column to the data frame and tree view.
		=====
		Input:
			None
		'''

		textSelected = self.get_selected_category()
		if len(textSelected) == 0:
			tk.messagebox.showinfo('Select category ..',
				'Please select a category for annotation.',
				parent=self.toplevel)
			return
		
		regExp = self.build_regex(textSelected)
		
		boolIndicator = self.df[self.columnForFilter].astype(str).str.contains(regExp)
		annotationColumn = boolIndicator.map(self.replaceDict)
		
		if self.addToTreeview == False:
			self.boolIndicator = boolIndicator
			self.annotationColumn = annotationColumn
			self.splitString = self.separatorString.get()
			self.close()
			return
			
		textString = get_elements_from_list_as_string(textSelected, maxStringLength = 15)
		columnName = '{}:{}'.format(textString,self.columnForFilter)
		
		self.add_column_to_df_and_tree(columnName,annotationColumn)
		
		
	def add_column_to_df_and_tree(self,columnName, annotationColumn):
		'''
		=====
		Input:
			columnName - Name of the annotation column created. Needs to be a string
			
			annotationColumn - The actual data object. Can be anything that can be added 
							   to a pandas data frame.
		====
		'''
		columnName = self.dfClass.evaluate_column_name(columnName)
		self.dfClass.add_column_to_current_data(columnName,annotationColumn,evaluateName=False)
		self.dataTreeview.add_list_of_columns_to_treeview(self.dfClass.currentDataFile, 
														'object', [columnName])
														 
		tk.messagebox.showinfo('Done ..',
			operationMessage[self.operationType],
			parent=self.toplevel) 
		
	         
	def build_regex(self,categoriesList,withSeparator = True, splitString = None):
		'''
		Build regular expression that will search for the selected category. Importantly it will prevent 
		cross findings with equal substring
		=====
		Input:
			List of categories that were selected by the user
		====
		'''
		if splitString is None:
			splitString = self.separatorString.get()
		regExp = r''
		
		for category in categoriesList:
			category = re.escape(category) #escapes all special characters
			if withSeparator:
				regExp = regExp + r'({}{})|(^{}$)|({}{}$)|'.format(category,splitString,category,splitString,category)
			else:
				regExp = regExp + r'({})|'.format(category)
				
		regExp = regExp[:-1] #strip of last |
		return regExp
		
	def find_string_and_annotate(self):
		'''
		Finds strings and either annotates them by a "+" sign or by the search string itself.
		=====
		Input:
			None
				
		====
		'''
		searchStrings = self.searchString.get() 
		collectResults = pd.DataFrame()	
		boolIndicator = None
		
		if self.userRedExpression.get():
			regExp = searchStrings
		else:
			splitSearchString = [row for row in csv.reader([searchStrings], 
											delimiter=',', quotechar='\"')][0]
								
			regExp = self.build_regex(splitSearchString,withSeparator=False)
		
		
		if self.annotateSearchString.get():
		
			if self.caseSensitive.get():
				flag = 0
			else:
				flag = re.IGNORECASE 
			#self.uniqueFlatSplitData[column] = self.uniqueFlatSplitData[column].astype(str)
			if len(splitSearchString) > 1:
				if self.onlyFirstFind.get(): 
					for column in self.columnForFilter:
						groupIndicator  = self.uniqueFlatSplitData[column].str.findall(regExp, flags = flag).astype(str)
						uniqueValues = groupIndicator.unique()
						replaceDict = self.build_replace_mapDict(uniqueValues,splitSearchString)
						annotationColumn = groupIndicator.map(replaceDict)
						collectResults[column] = annotationColumn
				else:
					for column in self.columnForFilter:
						groupIndicator  = self.uniqueFlatSplitData[column].str.extract(regExp, flags = flag)
						annotationColumn = groupIndicator.fillna('').astype(str).sum(axis=1)
						collectResults[column] = annotationColumn
				
				if len(self.columnForFilter) == 1:
					# simply take replaced annotation column
					annotationColumn = \
					annotationColumn.replace('',self.dfClass.replaceObjectNan).fillna(self.dfClass.replaceObjectNan)
				else:
					collectResults['annotationColumn'] = \
					collectResults.apply(lambda x: self.combine_string(x), axis=1)
					annotationColumn = collectResults['annotationColumn']
									
			else: 
				replaceDict = self.replaceDict
				replaceDict[True] = splitSearchString[0]
				for column in self.columnForFilter:
					columnBoolIndicator = self.uniqueFlatSplitData[column].str.contains(regExp,case = self.caseSensitive.get())
					collectResults[column] = columnBoolIndicator
				boolIndicator = collectResults.sum(axis=1) >= 1
				annotationColumn = boolIndicator.map(replaceDict)
				
		else:
			## simply label rows that match by "+"
			for column in self.columnForFilter:
					columnBoolIndicator = self.uniqueFlatSplitData[column].str.contains(regExp,case = self.caseSensitive.get())
					collectResults[column] = columnBoolIndicator
			boolIndicator = collectResults.sum(axis=1) >= 1
			annotationColumn = boolIndicator.map(self.replaceDict)
		
		if self.addToTreeview == False:
	
			self.boolIndicator  = boolIndicator 
			self.annotationColumn = annotationColumn
			self.close()
			return
			
			
		textString = get_elements_from_list_as_string(splitSearchString, maxStringLength = 15)
		
		columnName = '{}:{}'.format(textString,self.columnForFilter)
		
		self.add_column_to_df_and_tree(columnName,annotationColumn)
		
		
	def combine_string(self,row):
		'''
		Might not be the nicest solution and defo the slowest. (To do..)
		But it returns the correct string right away without further
		processing/replacement.
		'''
		nanString = ''
		base = ''
		if all(s == nanString for s in row):
			return self.dfClass.replaceObjectNan
		else:
			n = 0
			for s in row:
				if s != nanString:
					if n == 0:
						base = s
						n+=1
					else:
						base = base+','+s
			return base			
		
				
	def build_replace_mapDict(self,uniqueValues,splitSearchString):
		'''
		Subsets the currently selected df from dfClass on selected categories. Categories are 
		retrieved from user's selection.
		=====
		Input:
			uniuqeValues - unique values of a findall procedure. values of this
				object present the keys in the returned dict
			splitSearchString - List of strings that were entered by the user without quotes
				
		====
		'''
		replaceDict = dict() 
		naString = ''
		
		for value in uniqueValues:
			if all(x in value for x in splitSearchString):
				replaceDict[value] = splitSearchString
			if any(x in value for x in splitSearchString):
				repString = ''
				for category in splitSearchString:
					if category in value:
						repString =repString + '{},'.format(category)
				replaceDict[value] = repString[:-1]
			else:
				replaceDict[value] = naString
		return replaceDict		
						
		
				
	def subset_data_on_category(self):
		'''
		Subsets the currently selected df from dfClass on selected categories. Categories are 
		retrieved from user's selection.
		=====
		Input:
			None
		====
		'''
		textSelected = self.get_selected_category()
		
		if len(textSelected) == 0:
			tk.messagebox.showinfo('Select category ..',
				'Please select a category for annotation.',
				parent=self.toplevel)
			return
		regExp = self.build_regex(textSelected)	
		
		boolIndicator = self.df[self.columnForFilter].astype(str).str.contains(regExp)
		
		fileName = self.dfClass.get_file_name_of_current_data()
		textString = get_elements_from_list_as_string(textSelected,maxStringLength = 15)
		
		nameOfNewSubset = '{}: {} in {}'.format(textString,self.columnForFilter,fileName)
		
		subsetDf = self.df[boolIndicator]
		
		## adds data to dfClass and to the treeview 
		subsetId = self.dfClass.get_next_available_id()
		self.dfClass.add_data_frame(subsetDf, id = subsetId, fileName = nameOfNewSubset)
		columnDataTypeRelation = self.dfClass.get_columns_data_type_relationship_by_id(subsetId)
		
		self.dataTreeview.add_new_data_frame(subsetId,nameOfNewSubset,columnDataTypeRelation)
		
		tk.messagebox.showinfo('Done ..',
			operationMessage[self.operationType],
			parent=self.toplevel) 
	
	def annotate_scatter_points(self, event = None):
		'''
		Annotates selected rows in the current plot (scatter)
		Procedure: get_selected_rows - get_index since it the same in pandastable.model.df
		- then combine the columns with the ones that were used in the chart (numericColumns)
		- adds a annotation event in categorical plotter class - add annotations by this class
		
		=====
		Input:
			None
		====
		'''
		rowsSelected = self.get_selected_row()
		selection = self.pt.model.df.iloc[rowsSelected]
		
		labelColumn = selection.columns.values.tolist()
		if self.plotter.categoricalPlotter is not None:
			if self.plotter.categoricalPlotter.scatterWithCategories is not None:
				labelColumn = labelColumn + self.plotter.categoricalPlotter.categoricalColumns
		columnsNeededForAnnotation = list(set(self.numericColumns+labelColumn))
		annotationData = self.df.loc[selection.index.tolist(),columnsNeededForAnnotation]
		
		if self.plotter.categoricalPlotter is not None:
			if self.plotter.categoricalPlotter.scatterWithCategories is not None:
				self.plotter.categoricalPlotter.scatterWithCategories.bind_label_event(labelColumn)
				self.plotter.categoricalPlotter.scatterWithCategories.add_annotation_from_df(annotationData)
		else:
			self.plotter.nonCategoricalPlotter.bind_label_event(labelColumn)
			self.plotter.nonCategoricalPlotter.annotationClass.addAnnotationFromDf(annotationData)
	
	def search_for_entry_in_hclust(self, event = None):
		'''
		Find entry in hierarcichal clustering and center.
		'''
		rowsSelected = self.get_selected_row()
		selection = self.pt.model.df.iloc[rowsSelected[:1]]
		labelColumn = selection.columns.values.tolist()
		
		index = selection.index.tolist()[0]		
		
		## add this column as label 
		self.plotter.nonCategoricalPlotter._hclustPlotter.add_label_column(labelColumn)
		self.plotter.nonCategoricalPlotter._hclustPlotter.find_index_and_zoom(index)
		self.plotter.redraw()

	def search_entry_in_line(self, event = None):
		'''
		Find entry in line plot.
		'''
		rowsSelected = self.get_selected_row()
		selection = self.pt.model.df.iloc[rowsSelected[:1]]
		labelColumn = selection.columns.values.tolist()
		
		idx = selection.index.tolist()[0]	
		loc = self.df.index.tolist().index(idx)
		
		if len(self.plotter.tooltips) != 0:
			self.plotter.tooltips[0].set_invisible()
		else:
			self.plotter.figure.canvas.restore_region(self.background)
		
		self.plotter.nonCategoricalPlotter.linePlotHelper.indicate_hover(loc)
		
		
		
	def copy_from_clipboard(self,event):
		'''
		Try to infer paste as a valid input in the search entry.
		E.g separating each entry by comma and put string in ""
		'''
		data = pd.read_clipboard('\t',header=None).values.ravel()
		output = r''
		for value in data:
			output = output + r',"{}"'.format(value)
		self.outputString = output[1:]
		self.searchString.set(self.outputString)
		self.protectEntry = 0
		
	def undo(self,event):
		'''
		Undo/Delete last entry
		'''
		currentInput =  self.searchString.get()
		lastSeparation = currentInput.split('","')
		
		if len(lastSeparation) == 1:
			
			self.searchString.set('')
		else:
			lenLastString = len(lastSeparation[-1])
			totalLen = len(currentInput)
			# -2 for ,"
			idx = totalLen-2-lenLastString
			truncString = currentInput[:idx]
			self.searchString.set(truncString)
			
	def define_annotation_command_relation(self):
		'''
		Defines a dictionary describing the function to be used by applyButton.
		
		=====
		Input:
			None
		====
		'''		
		self.commandDict = {'Find category & annotate':self.add_annotation_column,
				  'Search string & annotate':self.find_string_and_annotate,
				  'Subset data on unique category':self.subset_data_on_category,
				  'Annotate scatter points':self.annotate_scatter_points,
				  'Find entry in hierarch. cluster': self.search_for_entry_in_hclust,
				  'Find entry in line plot': self.search_entry_in_line}
				







