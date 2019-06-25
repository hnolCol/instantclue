"""
	""FIND AND REPLACE DATA""
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

import csv
import re

from modules import images
from modules.pandastable import core 
from modules.utils import *



operationTitle = {'ReplaceColumns': 'Find & Replace column names',
				  'ReplaceRowEntries': 'Find & Replace values in selected column.'
				  }
             
infoText = {'object':'You can use "String1","String2" .. to find multiple strings.'+
		 			'\nThen you have to provide one or exactly the same number of '+
		 			'values to be used for replacement.',
		 	'float64':'You can use Value1,Value2, .. to replace multiple values at once.'+
		 			'\nThen you have to provide one or exactly the same number of '+
		 			'values to be used for replacement.',
		 	'int64':'You can use Value1,Value2, .. to replace multiple values at once.'+
		 			'\nThen you have to provide one or exactly the same number of '+
		 			'values to be used for replacement.'}		 

class findAndReplaceDialog(object):
	'''
	findAndReplaceDialog can be used for several operations. 
	
	=================
	Operations
		- Find and replace column names
		- Find and replace data in a selected column

	=================
	'''

	def __init__(self, mode = 'ReplaceRowEntries', dfClass = None, dataTreeview = None):
		
		self.operationType = mode
		
		self.searchString = tk.StringVar()
		self.replaceString = tk.StringVar() 
		self.exactMatch = tk.BooleanVar() 
		self.saveLastString  = ''
		
		if self.operationType == 'ReplaceRowEntries':
		
			self.columnForReplace = self.evaluate_column_selection(dataTreeview)
			
			if self.columnForReplace is None:
				return
			
			if isinstance(self.columnForReplace,str):
				subsetColumns = [self.columnForReplace]
			else:
				subsetColumns = self.columnForReplace
			
			self.dataType = dataTreeview.dataTypesSelected[0]
			self.df = dfClass.get_current_data_by_column_list(subsetColumns)						
			
			
		else:
			self.dataType = 'object'
			self.dataTreeview = dataTreeview 
			
		self.dfClass = dfClass	
		
		self.define_commands()
		self.build_toplevel()
		self.build_widgets()
		self.prepare_data()
		self.display_data()
		
		self.toplevel.wait_visibility()
		self.toplevel.grab_set() 
		
	def close(self, event = None):
		'''
		Close toplevel
		'''
		if hasattr(self,'pt'):
			del self.pt
		self.toplevel.destroy() 	
		

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Find & replace - '+self.operationType) 
		popup.bind('<Escape>',self.close)
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=615
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
        
        
 		if self.dataType == 'object':
   
 			exactMatchCB = ttk.Checkbutton(self.cont_widgets, variable = self.exactMatch, 
 															text = 'Match entire cell content')
 			exactMatchCB.grid(row=3,column=0,columnspan=2,padx=3,sticky=tk.W)
 			
 		labelInfo = tk.Label(self.cont_widgets, text = infoText[self.dataType], 
 												bg=MAC_GREY, justify=tk.LEFT) 	
 		
 		labelSearch = tk.Label(self.cont_widgets, text = 'Replace: ', bg = MAC_GREY)
 		labelReplace = tk.Label(self.cont_widgets, text = 'with: ', bg = MAC_GREY)
 		 
 		entrySearch = ttk.Entry(self.cont_widgets, textvariable = self.searchString)
 		entryReplace = ttk.Entry(self.cont_widgets, textvariable = self.replaceString)
 		
 		entrySearch.bind('<Return>', \
 		lambda event :self.update_data_upon_search(forceUpdate = True))
 		
 		self.searchString.trace(mode="w", callback=self.update_data_upon_search) 
 		
 		applyButton = create_button(self.cont_widgets, image=self.doneIcon, 
 										command = self.commandDict[self.operationType])	
		
		
 		labelTitle.grid(row=0,column=0,padx=4, pady = 15, columnspan = 3, sticky=tk.W)
 		labelInfo.grid(row=1,column=0,padx=4, pady = 3, columnspan = 3, sticky=tk.W)
 		
 		
 		labelSearch.grid(row=4,padx = 5, pady = 5, sticky=tk.W)
 		labelReplace.grid(row=5,padx = 5, pady = 5, sticky=tk.W)
 		entrySearch.grid(column = 1, row = 4, sticky=tk.EW, padx=2)
 		entryReplace.grid(column = 1, row = 5, sticky=tk.EW, padx=2)
 		applyButton.grid(row=4,column=2,sticky=tk.E,padx=2, rowspan=2) 
 
 	
				
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
         	
	def evaluate_column_selection(self, dataTreeview):
		'''
		'''
		columnForReplace = dataTreeview.columnsSelected
		lenSelectedColumns = len(columnForReplace)
		if  lenSelectedColumns > 1:
			uniqueDataType = list(set(dataTreeview.dataTypesSelected))
			if len(uniqueDataType) > 1:
					tk.messagebox.showinfo('Error ..',
						'Please select only columns of one data type')
					return 
					
			else:
					columnForReplace = dataTreeview.columnsSelected
		else:
			columnForReplace = dataTreeview.columnsSelected		
		
		return columnForReplace
			
		
	def prepare_data(self):
		'''
		'''
		if self.operationType == 'ReplaceRowEntries':	
			self.uniqueFlatSplitData = self.df.copy() 
		else:
		
			self.uniqueFlatSplitData = pd.DataFrame(self.dfClass.df_columns , columns = ['Column Names'])
			self.columnForReplace = 'Column Names'		
			
					
	def display_data(self):
		'''
		Displays data in a pandastable.  
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
		
	def update_data_upon_search(self,varname = None, elementname = None, mode=None,
								forceUpdate = False):
		'''
		Traces user's input an changes the data shown in pandastable.
		
		Notes - 
		=============
		
		Data are only updated if entry string is > 2 characters to avoid heavy changes 
		
		Dont renew changes when a comma or a " is entered. 
		
		For numeric data - the data that are shown when they are close to
		8 %.
		'''
		searchString = self.searchString.get()
		nonEmptyString = searchString != ''
		lenSearchString = len(searchString)
		if len(self.pt.model.df.index) > 10000:
			return
		if self.dataType == 'object':
			if lenSearchString < 3 and nonEmptyString and forceUpdate == False:
				## to avoid massive searching when data are big
				return
			if lenSearchString > 2:
				## to start a new String search
				if searchString[-2:] == ',"':			
					self.saveLastString = ''		
			lengthSaved = len(self.saveLastString)
		
			if lenSearchString == lengthSaved + 1 and self.saveLastString != '':
				dataToSearch = self.pt.model.df
			elif lenSearchString + 1 == lengthSaved and self.saveLastString != '':
				## avoid re-search on backspace
				dataToSearch = self.pt.model.df
			else:
				dataToSearch = self.uniqueFlatSplitData
			if len(dataToSearch.index) == 0:
				dataToSearch = self.uniqueFlatSplitData
			
			splitSearchString = [row for row in csv.reader([searchString], delimiter=',', quotechar='\"')][0]
			regExp = self.build_regex(splitSearchString)
			
			if self.operationType == 'ReplaceRowEntries':	
				collectDf = pd.DataFrame() 
				for n,column in enumerate(self.columnForReplace):
					collectDf.loc[:,str(n)] = dataToSearch[column].str.contains(regExp,case = True)
				boolIndicator = collectDf.sum(axis=1) >= 1
			else:
				boolIndicator = dataToSearch[self.columnForReplace].str.contains(regExp,case = True)
			
			subsetData = dataToSearch[boolIndicator]
			self.saveLastString = searchString

		else:
			valueList = self.transform_string_to_float_list(searchString)
			if valueList is None:
				return
			valuesToReplace = np.asarray(valueList)			
			
			## 8 % tolerance to be shown.
			boolIndicator = self.uniqueFlatSplitData.applymap(lambda value: \
			np.isclose(value,valuesToReplace, rtol = 0.8).item(0)).sum(axis=1) > 0
			
			subsetData = self.uniqueFlatSplitData[boolIndicator]
			
				
				 
		self.pt.model.df = subsetData
		self.pt.redraw()	
			         
	def build_regex(self,stringList, saveInList = False):
		'''
		Build regular expression that will search for the selected category. Importantly it will prevent 
		cross findings with equal substring
		=====
		Input:
			List of strings that were entered by the user
		====
		'''
		regExp = r''
		if self.exactMatch.get():
			baseString = r'(^{}$)|'
		else:
			baseString = r'({})|'
		listRegEx = []
		for category in stringList:
			category = re.escape(category) #escapes all special characters
			if saveInList:
				listRegEx.append(baseString.format(category)[:-1])
			else:
				regExp = regExp + baseString.format(category)
				
		
		if saveInList:
			return listRegEx
		else:		
			regExp = regExp[:-1] #strip of last |
			return regExp
			
	def extract_regExList(self,string):
		'''
		'''
		splitString = [row for row in csv.reader([string], delimiter=',', quotechar='\"')][0]
		regExList = self.build_regex(splitString, saveInList=True) 
		
		return regExList
		
	def evaluate_input(self,searchList,replaceList):
		'''
		'''
		lenSearchList = len(searchList)
		lenReplaceList = len(replaceList) 
		
		if 	lenSearchList == lenReplaceList:
			return True, searchList, replaceList, False if lenReplaceList > 1 else True
			
		elif lenSearchList > lenReplaceList:
			if lenReplaceList == 1:
				replaceList = replaceList * lenSearchList
				return True, searchList, replaceList, True
			else:
				tk.messagebox.showinfo('Error ..','Number strings for replacement does not match'+
									   ' the number of values to be replaced. Please revisit.'+
									   ' {} versus {}'.format(lenSearchList,lenReplaceList),
									   parent=self.toplevel)
		elif lenReplaceList > lenSearchList:
			tk.messagebox.showinfo('Error ..','Number of "To replace" strings is smaller than then number'+
									' of strings that should be used to replace them with. Please revisit.'+
									' {} versus {}'.format(lenSearchList,lenReplaceList),
									parent=self.toplevel)
			return False, None, None, False
				
	def transform_string_to_float_list(self,string):
		'''
		Takes a string and convert it to floats by splitting it at ',' and returns a list 
		of vlaues
		'''
		if string[-1] == ',':
			string = string[:-1]
		try:
			valueList = [float(x) for x in string.split(',')]
		except ValueError:
			tk.messagebox.showinfo('Error ..',
				'Cannot convert string to float.', 
				parent=self.toplevel)
			return
		return valueList 
			
	def perform_replacement_of_rowValues(self):
		'''
		Actually replace values.
		'''	
		searchString = self.searchString.get()
		replaceString = self.replaceString.get()
		if searchString == '':
			tk.messagebox.showinfo('Error ..','Please insert string to be replaced.',parent=self.toplevel)
			return
			
		elif replaceString == '':
			tk.messagebox.showinfo('Error ..','Please insert string to be used for replacement.',parent=self.toplevel)
			return
	
			
		if self.dataType == 'object':
			
			toReplaceList = self.extract_regExList(searchString)
			
			valueList = [row for row in csv.reader([replaceString], delimiter=',', quotechar='\"')][0]
			proceedBool,toReplaceList,valueList, matchToOne = self.evaluate_input(toReplaceList,valueList)
			if proceedBool == False:
				return
						
			for column in self.columnForReplace:
					if matchToOne:
						self.dfClass.df[column] = self.dfClass.df[column].str.replace('|'.join(toReplaceList),
														   					valueList[0],
														   					)
					else:
						for toReplace, value in zip(toReplaceList,valueList):
							self.dfClass.df[column] = self.dfClass.df[column].str.replace(toReplace,
														   					value,
														   					)
		else:
			searchList = self.transform_string_to_float_list(searchString)
			replaceList = self.transform_string_to_float_list(replaceString)
			
			proceedBool,toReplaceList,valueList = self.evaluate_input(searchList,replaceList)
			if proceedBool == False:
				return
			
			replaceDict = dict(zip(searchList,replaceList))			
			replacedDf = self.pt.model.df.applymap(lambda value: self.replace_if_match(value, replaceDict))
			index = replacedDf.index
			self.dfClass.df.loc[index,self.columnForReplace] = replacedDf
																	   
		tk.messagebox.showinfo('Done ..', 'Replacement done.',parent=self.toplevel)
	
	def replace_if_match(self, value, replaceDict):
		'''
		'''
		if value in replaceDict:
			return replaceDict[value]
		else:
			return value	
	
	def get_unique_changes_from_column_dict(self,replaceDict):
		'''
		'''
		keyList = []
		
		for key,item in replaceDict.items():
			if key == item:
				keyList.append(key)
		
		for key in keyList:
			del replaceDict[key]
			
		return replaceDict
				
	def rename_column_names(self):
		'''
		Rename column names.
		'''		
		searchString = self.searchString.get()
		replaceString = self.replaceString.get()
		if searchString == '':
			tk.messagebox.showinfo('Error ..','Please insert string to be replaced.',parent=self.toplevel)
			return
			
		elif replaceString == '':
			tk.messagebox.showinfo('Error ..','Please insert string to be used for replacement.',parent=self.toplevel)
			return
		toReplaceList = self.extract_regExList(searchString)
		valueList = [row for row in csv.reader([replaceString], delimiter=',', quotechar='\"')][0]
		
		if len(valueList) == 1 and len(toReplaceList) != len(valueList):
			valueList = valueList*len(toReplaceList)
		elif len(valueList) > len(toReplaceList):
			tk.messagebox.showinfo('Error..',
				'Number of strings/values that should be replaced must match the number of "replaced with" strings',
				parent=self.toplevel)
			return
		print(toReplaceList)
		self.uniqueFlatSplitData['Column Names'].replace(toReplaceList,valueList,
														   regex=True,inplace=True)
		
		replaceDict = dict(zip(self.dfClass.df_columns,self.uniqueFlatSplitData['Column Names']))
		replaceDictFiltered = self.get_unique_changes_from_column_dict(replaceDict )
		
		self.dfClass.rename_columnNames_in_current_data(replaceDictFiltered)	
		
		dataFrameID = self.dfClass.currentDataFile 
		iidList = ['{}_{}'.format(dataFrameID,columnName) for columnName in replaceDictFiltered.keys()]
		self.dataTreeview.rename_itemText_by_iidList(iidList,list(replaceDictFiltered.values()))
		
		tk.messagebox.showinfo('Done ..', 'Replacement done.',parent=self.toplevel)
		
	def define_commands(self):
		'''
		'''
		self.commandDict = {'ReplaceColumns':self.rename_column_names,
							'ReplaceRowEntries':self.perform_replacement_of_rowValues}
		











