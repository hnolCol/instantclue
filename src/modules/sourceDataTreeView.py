"""
	""TREEVIEW - DATA ORGANIZATION""
	* Data are presented by their column names
	* This classs handles clicks by user, filters unwanted selection
	* and organizes the data.
	
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

from collections import OrderedDict
from modules.utils import *

columnTypeNaming = OrderedDict([('float64','Numeric Floats'),
	('int64','Integers'),('object','Categories'),('bool','Boolean')])


class sourceDataTreeview(object):
	'''
	InstantClue uses a ttk.Treeview to display columnHeaders from the Uploaded File
	We use the word Column therefore for a new entry in the Treeview 
	'''
	
	def __init__(self, tkinterTreeviewObject):
	
		self.sourceDataTree = tkinterTreeviewObject
		
		self.define_tags_for_navigation()
		self.allItemsSelected = []
		self.dataFramesSelected = []
		self.columnsSelected = []
		self.dataTypesSelected = []
		
		self.stopDragDrop = True
		self.stopSelection = False
		self.do_some_bindings()

	def add_all_data_frame_columns_from_dict(self,columnDataTypeRelationshipDict,fileNameDict = {}):
		'''
		Will delete all children and then enter all uploaded files and columns.
		fileNameList must have same length as columnDataTypeRelationshipDict
		'''
		if len(fileNameDict) != len(columnDataTypeRelationshipDict):
			return
		
		self.delete_all_entries()
		
		for id, columnDataTypeRelation in columnDataTypeRelationshipDict.items():
		
			fileName = fileNameDict[id]
			
			self.add_new_data_frame(id,fileName,columnDataTypeRelation) 
			
			
	def add_new_data_frame(self,id,fileName,columnDataTypeRelation):
		'''
		Adds a new DataFrame that can be called from outside the class 
		needing only the properties of the data frame from the data class : dataCollection .. 
		'''
		self.create_file_name_header(id,fileName)
		self.add_data_type_separator(id,fileName)
		
		for dataType, columnList in columnDataTypeRelation.items():
			self.add_list_of_columns_to_treeview(id,dataType,columnList)
		

	def add_data_type_separator(self,id,fileName):
		'''
		Adds the separtor for floats, integers, categories and boolean
		'''
	
		for dataType,dataTypeSeparator in columnTypeNaming.items():			
			parent = '{}_{}'.format(id,fileName)
			keyDataType = '{}_{}'.format(id,dataType)
			text = dataTypeSeparator
			
			
			self.insert_entry_to_treeview(parent,'end',keyDataType,text,'dataType') 
			
			
	def add_column_to_end(self,currentDataID,columnName,dataType):
		'''
		Adds a new column simply at the end of its dataType section.
		'''
		#to Do
		pass
		
	
	def add_list_of_columns_to_treeview(self,id,dataType,columnList,startIndex = None):
		'''
		Adds a list of columns to the source data. dataType can be a list that
		give per element in columnList a specific dataType
		'''

		for n,columnName in enumerate(columnList):
				if isinstance(dataType,list):
					dataTypeItem = dataType[n]

				else:
					dataTypeItem = dataType
				
				if startIndex is not None:
					if isinstance(startIndex,list) and len(startIndex) == len(columnList):
						index = startIndex[n]
					else:
						index = startIndex+n+1
				else:
					index = 'end'
						
				parent = '{}_{}'.format(id,dataTypeItem) 
				iid = '{}_{}'.format(id,columnName)
				text = columnName
			
				self.insert_entry_to_treeview(parent,index,iid,text)
				

	def change_data_type_by_iid(self,iidList, newDataType):
		'''
		Changes the data type by iid. Extracts the dataframe id and name from the iid s
		that are given by iidList.
		'''
		for n,iid in enumerate(iidList):
			columnName = self.sourceDataTree.item(iid)['text']
			if isinstance(newDataType,str) == False and len(newDataType) == len(iidList):
				newDataType = newDataType[n]
				
			id = iid.split('_'+columnName)[0]
			parent = '{}_{}'.format(id,newDataType) 
			
			self.delete_entry_by_iid(iid)	
			
			self.insert_entry_to_treeview(parent,'end',iid,columnName)
			
					
		
	def check_if_selection_from_one_data_frame(self):
		'''
		Returns True if the current selection is only from one dataframe
		'''
		uniqueDataFrames = self.get_data_frames_from_selection()
		
		if len(uniqueDataFrames) == 1:
			return True, uniqueDataFrames[0]
		elif len(uniqueDataFrames) == 0:
			return None, None
		else:
			return False, None
			
	def check_if_item_was_selected(self,event):
		'''
		Checks if treeview item was selected alreay - dont remove it for better drag & drop experience
		'''
		self.stopSelection = False
		iid = self.sourceDataTree.identify_row(event.y) 
		
		if iid in self.allItemsSelected:
		
			self.currentClickItemWasSelected = True
		else:
			self.currentClickItemWasSelected = False
			
		if iid.split('_')[-1] in ['float64','int64','object','bool']:
			self.stopSelection = True
		
		
		
	
	def check_for_shift(self,event):
		'''
		Checks if shift is pressed.
		'''
		s = event.state
		self.pressedShift =  (s & 0x1) != 0
		
		    
    
	def create_file_name_header(self,id,fileName):
		'''
		'''
		parent = ''
		iid = '{}_{}'.format(id,fileName)
		text = fileName
		self.insert_entry_to_treeview(parent,'end',iid,text,'mainFile',open=True)
	
	def clicked_item(self,event):
		'''
		'''
		
		return self.sourceDataTree.identify('item',event.x,event.y)	
		
	def define_tags_for_navigation(self):
		'''
		'''

		self.sourceDataTree.tag_configure('mainFile', background = '#E4DEB6', 
								  foreground="black", font = tkFont.Font(size=defaultFontSize))
		self.sourceDataTree.tag_configure('dataType', font = tkFont.Font(size=defaultFontSize))
		##E4DED4
		
	def delete_all_entries(self):
		'''
		Deletes every entry in the treeview
		'''
		childrenTree = self.get_children() 
		self.sourceDataTree.delete(*childrenTree)
		
	def delete_entry_by_iid(self,iid):
		'''
		'''
		if isinstance(iid,str):
			self.sourceDataTree.delete(iid)
		elif isinstance(iid,list):
			for iid_ in iid:
				self.sourceDataTree.delete(iid_)
				
	def delete_selected_entries(self):
		'''
		'''
		currentSelection = self.get_current_selection() 
		for iid in currentSelection:
			self.delete_entry_by_iid(iid)
	
	def deselect_all_items(self,event = None):
		'''
		Removes all selected items
		'''
		selection = self.get_current_selection()
		for item in selection:
			self.sourceDataTree.selection_remove(item)
	
	
	def do_some_bindings(self):
		'''
		'''
		self.sourceDataTree.bind('<<TreeviewSelect>>', self.on_treeview_selection)
		self.sourceDataTree.bind('<1>', self.check_if_item_was_selected)
	
	
	def evaluate_output_for_single_df(self, dataFramesSelected):
		'''
		'''
		
			
	def get_data_types_from_selection(self,columnsSelected = None):
		'''
		returns a list of data types that are selected from the treeview. (To avoid that categories and floats can 
		be used for drag& drop at the same time)
		'''

		if columnsSelected is None:
			columnsSelected = self.get_column_selections()
		dataTypes = [self.sourceDataTree.parent(item).split('_')[-1] for item in columnsSelected]
		return dataTypes
			
	def get_column_selections(self, itemsSelected=None):
		'''
		columns means here, the iids of any item in the treeview that represents a column in the 
		data frame. The filter criteria is that they do not have any children and their parents parent is not '' to
		exclude data type separator (floats,objects..) that do not have any entries. 
		'''
		if itemsSelected is None:
			itemsSelected = self.get_current_selection()
		columnNamesSelected = [item for item in itemsSelected if len(self.sourceDataTree.get_children(item)) == 0 and \
													self.sourceDataTree.parent(self.sourceDataTree.parent(item)) != '']
		return columnNamesSelected
			
	def get_data_frames_from_selection(self,itemsSelected = None):
		'''
		Uses the identifier that is used in the source data treeview iid creation 
		to obtain data frames. This is important when functions can only be applied to 
		one data frame at a time.
		'''
		if itemsSelected is None:
			itemsSelected = self.get_current_selection()
		dataFrames = [item.split('_')[0] for item in itemsSelected]
		uniqueDataFrames = list(set(dataFrames))
		
		return uniqueDataFrames
	
	def get_current_selection(self):
		'''
		Returns current selection iid
		'''
		selectedItems =  list(self.sourceDataTree.selection())
		return selectedItems
			
		
	def get_children(self):
		'''
		Returns all children in the tkinterTreeviewObject
		'''
		childrenTree = self.sourceDataTree.get_children() 
		
		return childrenTree
	
		
	def insert_entry_to_treeview(self, parent='',index='end',iid = None, text = '', tag  = '',open=False):
	
		'''
		Inserts entries to tkinter treeview.
		'''
		self.sourceDataTree.insert(parent,index,iid,text=text,tag=tag,open=open)
		
	def on_treeview_selection(self,event):
		'''
		handles item selection of the sourceDataTree
		'''		
		self.onlyDataFramesSlected = False
		self.onlyDataTypeSeparator = False

		
		if self.currentClickItemWasSelected:
			# avoid strange selection in treeview when double clikc + shift on separtor
			if self.stopSelection:
				for iid in self.get_current_selection():
					self.sourceDataTree.selection_remove(iid)
				self.stopSelection = False
				self.currentClickItemWasSelected = False
				return
			
			for iid in self.allItemsSelected:
				for iid in self.allItemsSelected:
					self.sourceDataTree.selection_add(iid) 
					## sadly a bit of ugly for loop, but selection_set has problems with 
					## some iids that contain spaces (even after patch)		
				self.currentClickItemWasSelected = False			
		else:
			self.allItemsSelected = self.get_current_selection() 
		
		
		self.onlyDataFramesSelected = all([self.sourceDataTree.parent(itemIID) == '' for itemIID in self.allItemsSelected])
		self.dataFramesSelected = self.get_data_frames_from_selection(self.allItemsSelected)	
		self.columnsIidSelected = self.get_column_selections(self.allItemsSelected)
		self.columnsSelected = [self.sourceDataTree.item(iid)['text'] for iid in self.columnsIidSelected]
		#data types of the selected columns, will be empty if data type separators are selected
		self.dataTypesSelected = self.get_data_types_from_selection(self.columnsIidSelected )
		
		self.onlyDataTypeSeparator =  all(items in list(columnTypeNaming.values()) \
		for items in [self.sourceDataTree.item(iid)['text'] for iid in self.allItemsSelected])
				
		
		self.currentClickItemWasSelected = False
		
		if self.onlyDataTypeSeparator:
			self.stopDragDrop = True
			return

		uniqueDataTypes = len(set(self.dataTypesSelected))
		numColumnsSelected = len(self.columnsSelected)
		if len(self.dataFramesSelected) > 1 and numColumnsSelected == 0:
			### only dfs selected - needed to cast a different menu upon right_click 
			self.onlyDataFramesSlected = True
			self.stopDragDrop = True
			return
			
		if numColumnsSelected == 0:
		
			self.stopDragDrop = True
			return
		
		self.stopDragDrop = False	
		
		floatAndInts = all([dataType in ['float64','int64'] for dataType in self.dataTypesSelected])
		intsAndObjects = all([dataType in ['object','int64'] for dataType in self.dataTypesSelected])
		
		if uniqueDataTypes == 0 or floatAndInts or intsAndObjects:		
			# Checks if the data types fit together e.g. int and floats or ints and objects 
			# and dont do anything	
			
			if floatAndInts:
				self.onlyNumericColumnsSelected = True
			elif intsAndObjects:
				self.onlyNumericColumnsSelected = False
		else:
			pass ## to do! 
			
		
		
		
	def rename_itemText_by_iidList(self,iidList,newNameList):
		'''
		'''
		newIids = []
		for n,iid in enumerate(iidList):	
			
			newIID = '{}_{}'.format(iid.split('_')[0],newNameList[n])
			idx = self.sourceDataTree.index(iid)
			parent = self.sourceDataTree.parent(iid)
			self.sourceDataTree.delete(iid)
			self.insert_entry_to_treeview(parent=parent,index=idx,iid=newIID,text=newNameList[n])
			newIids.append(newIID)
		return newIids
					
			
			
	#	def insert_entry_to_treeview(self, parent='',index='end',iid = None, text = '', tag  = '',open=False):
	
	#	'''
	#	Inserts entries to tkinter treeview.
	#	'''
	#	self.sourceDataTree.insert(parent,index,iid,text=text,tag=tag,open=open)
		
		
		
		
		
		
		
		
	
	
	
	
	
	
	
	
	
	
					
			
	
	
