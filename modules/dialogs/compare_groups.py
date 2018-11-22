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
from modules.dialogs.simple_dialog import simpleListboxSelection, simpleUserInputDialog
from modules import stats
from modules import images
from modules.utils import *
try:
	from tslearn.metrics import SoftDTW, SquaredEuclidean
except:
	pass
from scipy.spatial.distance import cdist

availableTests = ['t-test','Welch-test','Whitney-Mann U [unpaired non-para]',
			'Wilcoxon [paired non-para]','1-W-ANOVA',#'Kruskal-Wallis',
			'Soft-TDW (time series)']


class compareGroupsDialog(object):


	def __init__(self, selectedColumns = [], dfClass = None, treeView = None, statTesting = True):
		''
		self.dfClass = dfClass 
		self.treeView = treeView
		self.groups = OrderedDict()
		self.selectedColumns = selectedColumns
		# stat Testing == False can be used to use this dialog to define groups
		# instead of testing row wise
		self.statTesting = statTesting
		self.testSelected = tk.StringVar()
		self.pairedVar = tk.BooleanVar()
		self.sideVar = tk.StringVar()
		self.logPVal = tk.BooleanVar(value=True)
		
		self.get_images()
		self.build_toplevel()
		self.build_widgets()
		self.build_menu()
		self.add_column_to_group('Group_1',selectedColumns)
		self.add_column_to_group('Group_2',[])
		self.toplevel.wait_window()
		
		

	def close(self,event=None):
		''
		self.toplevel.destroy() 

	def build_toplevel(self):
		''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Define groups for test') 
		popup.bind('<Escape>', self.close) 
		self.toplevel = popup
		self.center_popup((520,460))
	
	def build_widgets(self):
		''
		cont = tk.Frame(self.toplevel,background = MAC_GREY)
		cont.pack(fill='both', expand=True)
		cont.grid_columnconfigure(1,weight=1)
		cont.grid_rowconfigure(4,weight=1)
		
		labelTitle = tk.Label(cont, text = 'Compare two or multiple groups.\nTo manage groups, right click on their names.', **titleLabelProperties)
		labelTitle.grid(pady=5,padx=5,sticky=tk.W,columnspan=2)
		if self.statTesting:
		
			testLabel = tk.Label(cont, text='Test :', bg = MAC_GREY)
			combo = ttk.Combobox(cont, textvariable = self.testSelected, values = availableTests) 
			combo.insert(0,'t-test')
			combo.configure(state = 'readonly')
			testLabel.grid(row=2,column=0)
			combo.grid(row=2,column=1, sticky = tk.EW, padx=(0,53))
			
			pairedCB = ttk.Checkbutton(cont, variable = self.pairedVar, text = 'Paired')
			pairedCB.grid(row=3,column=0)
			
			sideCombo = ttk.Combobox(cont, textvariable = self.sideVar, values = ['less','two-sided [default]','greater'])
			sideCombo.insert(0,'two-sided [default]')
			sideCombo.configure(state = 'readonly')
			sideCombo.grid(row=3,column=1, sticky = tk.EW, padx=(0,151))
			
			logPCB = ttk.Checkbutton(cont, variable = self.logPVal, text = '-log10 p-values')
			logPCB.grid(row=3,column=1, sticky=tk.E)
		self.group_tree(cont)
		
		addGroupButton = create_button(cont, command=self.add_group, image = self.addImg)
		applyButton = ttk.Button(cont,text='Apply',command = self.perform_calculation)
		closeButton = ttk.Button(cont,text='Close',command=self.close)
		
		addGroupButton.grid(row=2, column=1, sticky = tk.E)
		applyButton.grid(row=5,column=0)
		closeButton.grid(row=5,column=1, sticky = tk.E)
		

	def build_menu(self):
		'''
		Define menu to handle groups
		'''
		
		menu = tk.Menu(self.toplevel, **styleDict)
		menu.add_command(label='Add columns',command = self.column_selection)
		menu.add_command(label='Add group',command = self.add_group)
		menu.add_separator()
		menu.add_command(label='Rename group', command = self.rename_groups)
		menu.add_command(label='Name by longest match', command = self.name_longest_match)
		menu.add_separator()
		menu.add_command(label='Clear group',command = self.delete_clear_group)
		menu.add_command(label='Delete group',command = lambda: self.delete_clear_group(deleteGroup=True))
		menu.add_separator()
		menu.add_command(label='Reset grouping', command = self.clear_tree)
		self.menu = menu
	
	
	def post_menu(self, event = None):
		x = self.toplevel.winfo_pointerx()
		y = self.toplevel.winfo_pointery()
		self.menu.focus_set()
		self.menu.post(x,y)

	
	def group_tree(self,cont):
		''
		treeFrame = tk.Frame(cont)
		treeFrame.grid(row=4,sticky=tk.NSEW, columnspan=2)
		treeFrame.grid_rowconfigure(0,weight=1)
		treeFrame.grid_columnconfigure(0,weight=1)
		self.group_treeview = ttk.Treeview(treeFrame, height = "4", 
           						show='tree', style='source.Treeview')
		self.group_treeview.bind(right_click, self.post_menu)
		self.group_treeview.grid(pady=3,padx=2,sticky=tk.NSEW)
		
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
		if self.statTesting == False:
		
			self.close()
			return
				

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
			#s1 = common_start_string(*self.groups[group1])
			#s2 = common_start_string(*self.groups[group2])
			if self.testSelected.get() in ['1-W-ANOVA','Kruskal-Wallis']:
				colName = '{}_{}'.format(self.testSelected.get(), get_elements_from_list_as_string(list(self.groups.keys())))
				nTotal = 1.2
			else:
				colName = '{}_vs_{}'.format(group1,group2)
			
			progBar.update_progressbar_and_label(n/nTotal * 100,
				':: Calculating .. {}/{}'.format(n,nTotal))
			
			if self.testSelected.get() in ['1-W-ANOVA','Kruskal-Wallis']:
				if n > 1:
					continue
				groupColumns = [list(v) for v in self.groups.values()]
				data  = df.apply(self.compare_multiple_groups, axis=1, test = self.testSelected.get(),
					groupColumns = groupColumns)
				result = pd.DataFrame(data,columns=['results'],index=df.index)
				newColumnNames = ['test_stat_{}_{}'.format(colName,self.testSelected.get()), 
					  'p-value_{}_{}'.format(colName,self.testSelected.get())]
				result[newColumnNames] = result['results'].apply(pd.Series)
				if self.logPVal.get():
					result['-log10_{}'.format(newColumnNames[-1])] = (-1)*np.log10(result[newColumnNames[-1]].values)
					newColumnNames[-1] = '-log10_{}'.format(newColumnNames[-1])
				result = result[newColumnNames]
			
			elif self.testSelected.get() == 'Soft-TDW (time series)':
				if platform == 'WINDOWS':
					tk.messagebox.showinfo('Error..',
						'Only available on Mac and Linux at the moment.', 
						parent = self)	
					return			
				groupColumns = [list(v) for v in self.groups.values()]
				data = df.apply(self.calcualteTDW, axis = 1, groupColumns = groupColumns) 
				result = pd.DataFrame(data,columns=['{}_softTDW'.format(colName)], index = df.index)
			else:	
				data = df.apply(self.compare_two_groups, axis=1, testSettings = {'paired':self.pairedVar.get(),
									  'test':self.testSelected.get(),
									  'mode':self.sideVar.get()},
									  groupColumns = [self.groups[group1],self.groups[group2]])
				#data = data
				result = pd.DataFrame(data,columns=['results'],index=df.index)
				#data.columns = ['results']
				newColumnNames = ['test_stat_{}_{}'.format(colName,self.testSelected.get()), 
					  'p-value_{}_{}'.format(colName,self.testSelected.get())]
				result[newColumnNames] = result['results'].apply(pd.Series)
				if self.logPVal.get():
					result['-log10_{}'.format(newColumnNames[-1])] = (-1)*np.log10(result[newColumnNames[-1]].values)
					newColumnNames[-1] = '-log10_{}'.format(newColumnNames[-1])
				result = result[newColumnNames]
			
			
			
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

		statResults = stats.compare_two_groups(testSettings,data)
		
		return statResults
		
	
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
		

	def column_selection(self,event = None):
		'''
		'''
		selection = self.get_selected_groups()
		if len(selection) == 0:
			return
		groupName = selection[0]
		columns = self.get_columns()
		selectionDialog = simpleListboxSelection('Select columns for Group ..',
									data = columns)
		
		if len(selectionDialog.selection) != 0:
			
			iids = self.delete_group_member(groupName)
			self.add_column_to_group(groupName,selectionDialog.selection,update=True)
		
	def get_selected_groups(self):
		'''
		'''
		return [iid for iid in list(self.group_treeview.selection()) if self.group_treeview.parent(iid) == '']	
												

	def find_longest_item_match(self, selection, replaceIfNone = 'Insert name..'):
		'''
		'''
		commonStart = []
		for n,groupId in enumerate(selection):
			
			commonStr = common_start_string(*self.groups[groupId])
			if isinstance(commonStr,list) or commonStr == '':
				if isinstance(replaceIfNone,str):
					commonStr = replaceIfNone
				elif isinstance(replaceIfNone,list) and len(replaceIfNone) == len(selection):
					commonStr = replaceIfNone[n]
				else:
					commonStr = groupId
			
			commonStart.append(commonStr)
		return commonStart

	def rename_groups(self,event = None):
		'''
		'''
		selection = self.get_selected_groups()		
		commonStart = self.find_longest_item_match(selection)
			
		optionValues = [[commonStart[n],groupId] for n,groupId in enumerate(selection)]
		groupRenameDialog = simpleUserInputDialog(selection,commonStart,optionValues,
														title='Rename groups',infoText='')
		renameOutput = groupRenameDialog.selectionOutput
		if len(renameOutput) != 0:
			
			for groupId, newName in renameOutput.items():
				if newName == '':
					continue
				self.rename_group_in_tree(groupId,newName)

	def rename_group_in_tree(self,groupId,newName):
		'''
		'''
		columnsAdded = self.groups[groupId]
		if groupId in self.groups:
				del self.groups[groupId]
		self.groups[newName] = []
		idx = self.group_treeview.index(groupId)
		self.group_treeview.delete(groupId)
		self.group_treeview.insert('',index=idx, iid = newName, text = newName)
		self.add_column_to_group(newName,columnsAdded,update=True)
				

	def clear_tree(self,event=None):
		'''
		'''
		self.group_treeview.delete(*self.group_treeview.get_children())
		if hasattr(self,'groups'):
			self.groups.clear()		
	
	def name_longest_match(self):
		'''
		'''
		selection = self.get_selected_groups()
		alternativeNames = [self.groups[id][0] if len(self.groups[id]) > 0 else id for id in selection]
		commonStart = self.find_longest_item_match(selection,alternativeNames)
		for groupId, newName in zip(selection,commonStart):
			self.rename_group_in_tree(groupId,newName)
									
		
	def delete_clear_group(self, groupId = None, deleteGroup = False):
		'''
		'''
		if groupId is None:
			selection = self.get_selected_groups()	
			if len(selection) == 0:
				tk.messagebox.showinfo('Error..','No groups selected')
				return
		elif isinstance(groupId,str):
			selection = [str]
		elif isinstance(groupId,list):
			selection = groupId
		else:
			return

		for groupId in selection:
			if groupId in self.groups and deleteGroup:
				del self.groups[groupId]
				self.group_treeview.delete(groupId)
			elif groupId in self.groups:
				groupItems = self.group_treeview.get_children(groupId)
				self.groups[groupId] = []
				for iid in groupItems:
					self.group_treeview.delete(iid)
				
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))			
		
		
				
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		