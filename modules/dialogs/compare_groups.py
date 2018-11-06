"""
	""ROW WISE GROUP COMPARE""
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
import numpy as np           
import pandas as pd 
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
from modules.dialogs.simple_dialog import simpleListboxSelection
from modules import stats
from modules import images
from modules.utils import *

from tslearn.metrics import SoftDTW, SquaredEuclidean
from scipy.spatial.distance import cdist

availableTests = ['t-test','Welch-test','Whitney-Mann U [unpaired non-para]',
			'Wilcoxon [paired non-para]','1-W-ANOVA','Kruskal-Wallis',
			'Soft-TDW (time series)']


class compareGroupsDialog(object):


	def __init__(self, selectedColumns = [], dfClass = None, treeView = None):
		''
		self.dfClass = dfClass 
		self.treeView = treeView
		self.groups = OrderedDict()
		self.selectedColumns = selectedColumns
		self.testSelected = tk.StringVar()
		self.get_images()
		self.build_toplevel()
		self.build_widgets()
		self.add_column_to_group('Group_1',selectedColumns)
		self.add_column_to_group('Group_2',[])
		

	def close(self,event=None):
		''
		self.toplevel.destroy() 

	def build_toplevel(self):
		''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Compare Groups row-wise') 
		popup.bind('<Escape>', self.close) 
		self.toplevel = popup
	
	def build_widgets(self):
		''
		cont = tk.Frame(self.toplevel,background = MAC_GREY)
		cont.pack(fill='both', expand=True)
		cont.grid_columnconfigure(1,weight=1)
		cont.grid_rowconfigure(4,weight=1)
		
		labelTitle = tk.Label(cont, text = 'Compare two or multiple groups.\nDouble-click on groups to enter names.', **titleLabelProperties)
		labelTitle.grid(pady=5,padx=5,sticky=tk.W,columnspan=2)
		
		testLabel = tk.Label(cont, text='Test :', bg = MAC_GREY)
		combo = ttk.Combobox(cont, textvariable = self.testSelected, values = availableTests) 
		combo.insert(0,'t-test')
		combo.configure(state = 'readonly')
		testLabel.grid(row=2,column=0)
		combo.grid(row=2,column=1)

		self.group_tree(cont)
		
		addGroupButton = create_button(cont, command=self.add_group, image = self.addImg)
		applyButton = create_button(cont, image = self.check_icon, command = self.perform_calculation)
		closeButton = ttk.Button(cont,text='Close',command=self.close)
		
		addGroupButton.grid(row=2, column=2)
		applyButton.grid(row=5,column=0)
		closeButton.grid(row=5,column=1)
		
		
	def group_tree(self,cont):
		''
		treeFrame = tk.Frame(cont)
		treeFrame.grid(row=4,sticky=tk.NSEW, columnspan=2)
		treeFrame.grid_rowconfigure(0,weight=1)
		treeFrame.grid_columnconfigure(0,weight=1)
		self.group_treeview = ttk.Treeview(treeFrame, height = "4", 
           						show='tree', style='source.Treeview')
		self.group_treeview.bind('<Double-Button-1>', self.column_selection)
		self.group_treeview.grid(sticky=tk.NSEW)
		
		self.add_group()
		self.add_group()

	def add_column_to_group(self,groupName,columns,update=True):
		'''
		'''
		for col in columns:
			name = '{}_{}_{}'.format('addedColumn',groupName,col)
			self.group_treeview.insert(groupName,'end',iid = name, text = col)
			self.groups[groupName].append(col)
		
		if self.group_treeview.item(groupName)['open'] == 0 and update:
			self.group_treeview.item(groupName,open=True)
		

	def add_group(self):
		'''
		'''	
		groupName = 'Group_{}'.format(len(self.groups)+1)
		
		self.group_treeview.insert('','end',iid = groupName,text=groupName)
		self.groups[groupName] = []

	def perform_calculation(self):
		'''
		'''
		progBar = Progressbar('Comparing two groups ..')
		
		if any(len(k) < 2 for v,k in self.groups.items()):
			tk.messagebox.showinfo('Warning..',
				'There are groups with less than two columns selected.\nNaN will be returned.',
				parent=self.toplevel)
						
		combinations = list(itertools.combinations(self.groups.keys(),2))
		nTotal = len(combinations)
		n = 0
		#nTotal = len(list(combinations))
		df = self.dfClass.get_current_data()
		addedColumnNames = []
		for group1,group2 in combinations:
			if any(len(group) == 0 for group in [group1,group2]):
				continue
			n+=1
			s1 = common_start_string(*self.groups[group1])
			s2 = common_start_string(*self.groups[group2])
			colName = '{}_vs_{}'.format(s1,s2)
			progBar.update_progressbar_and_label(n/nTotal * 100,
				'Comparing groups - {} vs {}.\n{} out {} comparisions.\nCalculating ..'.format(s1,s2,n,nTotal))
			
			if self.testSelected.get() in ['1-W-ANOVA','Kruskal-Wallis']:
				
				groupColumns = [list(v) for v in self.groups.values()]
				data  = df.apply(self.compare_multiple_groups, axis=1, test = self.testSelected.get(),
					groupColumns = groupColumns)
				data = data.values
				result = pd.DataFrame(data,columns=['{}_{}'.format(colName,self.testSelected.get())], index = df.index)
			
			elif self.testSelected.get() == 'Soft-TDW (time series)':				
				groupColumns = [list(v) for v in self.groups.values()]
				data = df.apply(self.calcualteTDW, axis = 1, groupColumns = groupColumns) 
				result = pd.DataFrame(data,columns=['{}_softTDW'.format(colName)], index = df.index)
			else:	
				data = df.apply(self.compare_two_groups, axis=1, testSettings = {'paired':False,
									  'test':self.testSelected.get(),
									  'mode':'two-sided [default]'},
									  groupColumns = [self.groups[group1],self.groups[group2]])
				data = data.values
				result = pd.DataFrame(data,columns=['{}_{}'.format(colName,self.testSelected.get())], index = df.index)			 
			
			columnNames = self.dfClass.join_df_to_currently_selected_df(result, exportColumns = True)
			addedColumnNames.extend(columnNames)
		progBar.close()		
		tk.messagebox.showinfo('Done ..','Calculations done.')			  
		
		self.treeView.add_list_of_columns_to_treeview(self.dfClass.currentDataFile,
     													'float64',addedColumnNames)  

	def get_images(self):
		'''
		Get images for buttons.
		'''
		_,_,self.check_icon,_ = images.get_custom_filter_images()
		_,_,_,self.addImg = images.get_data_upload_and_session_images()
		
	def calcualteTDW(self,row,groupColumns):
		
		data = [row[col].values.astype(np.float) for col in groupColumns]
		#data = [x[~np.isnan(x)] for x in data]
		distMatrix = cdist(data[0].reshape(len(groupColumns[0]),1),data[1].reshape(len(groupColumns[1]),1))
		return SoftDTW(distMatrix).compute()		

	def compare_multiple_groups(self,row,test, groupColumns = []):
		'''
		'''
		data = [row[col].values.astype(np.float) for col in groupColumns]
		
		data = [x[~np.isnan(x)] for x in data]
		if any(x.size < 2 for x in data):
			return (np.nan,np.nan)
		#print(data)
		return stats.compare_multiple_groups(test,data)
		
		
	def compare_two_groups(self,row,testSettings = {}, groupColumns = []):
		
		data = [row[col].values.astype(np.float) for col in groupColumns]
		
		data = [x[~np.isnan(x)] for x in data]
		if any(x.size < 2 for x in data):
			return (np.nan,np.nan)
			
		#print(x1)
		#print(x2)
		#print(np.isnan(x1))
		#if np.sum(np.isnan(x1)) > len(col1) - 2 or np.sum(np.isnan(x2)) > len(col2) - 2:
		#	return (np.nan, np.nan)
		
		return stats.compare_two_groups(testSettings,data)
		
	
	def delete_group_member(self,groupName):
		'''
		'''
		iids = self.group_treeview.get_children(groupName)
		for iid in iids:
			self.group_treeview.delete(iid)
		self.groups[groupName] = []
		
		return iids
	
	def get_columns(self):
		'''
		'''	
		columnsAvailable = []
		numericColumns = self.dfClass.get_numeric_columns()
		columnsUsed = []
		for v in self.groups.values():
			columnsUsed.extend(v)
		columnsUsed = list(set(columnsUsed))
		
		for column in numericColumns:
			if column not in columnsUsed:
					columnsAvailable.append(column)
			
		return columnsAvailable
		

	def column_selection(self,event):
		'''
		'''
		selection = [iid for iid in list(self.group_treeview.selection()) if iid.split('_')[0] == 'Group']
		
		if len(selection) == 0:
			return
		groupName = selection[0]
		columns = self.get_columns()
		selectionDialog = simpleListboxSelection('Select columns for Group ..',
									data = columns)
		
		if len(selectionDialog.selection) != 0:
			
			iids = self.delete_group_member(groupName)
			self.add_column_to_group(groupName,selectionDialog.selection,update=True)
		
				
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		