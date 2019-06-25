"""
	""MASKING - LIVE FILTERING""
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

from itertools import chain

import modules.images as img

from modules.dialogs.categorical_filter import categoricalFilter

from modules.utils import * 
from modules.dialogs.VerticalScrolledFrame import VerticalScrolledFrame


class clippingMaskFilter(object):


	def __init__(self,plotter,dfClass,analysisClass):
		
		self.operator = tk.StringVar()
		self.takeAbsValues = tk.BooleanVar(value=False)
		self.activeNumFilter = tk.BooleanVar(value=False)
		
		self.ignoreIds = []
		
		self.operator.set('and')
		
		self.dfClass = dfClass
		self.plotter = plotter
		self.analysisClass = analysisClass
	
		self.dataID = dfClass.currentDataFile
		if self.dataID is None:
			return
			
		self.data = dfClass.get_current_data() 
		
		self.filterCollection = OrderedDict() 
		self.id = 0
		self.treeCollection = OrderedDict() 
		
		self.create_toplevel() 	
		self.build_widgets()
		self.toplevel.wait_window()
		
		
	def close(self, event = None):
		'''
		'''
		self.dfClass.reset_clipping(self.dataID)
		try:
			self.plotter.save_axis_limits()		
			self.plotter.reinitiate_chart(updateData = True)
		except:
			pass			
		self.toplevel.destroy() 
		
						
	def create_toplevel(self):
	
		'''
		'''
		self.toplevel = tk.Toplevel() 
		self.toplevel.wm_title('Live Filter ...')
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close)
		self.toplevel.bind('<Escape>', self.close)
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(expand=True, fill='both')
		cont.grid_rowconfigure(4,weight = 1) 
		
		self.cont = cont	
		self.center_popup((400,650))
	


	def build_widgets(self):
	
		'''
		'''	
		## get image
		_,_,_,self.addImg = img.get_data_upload_and_session_images()
		self.delIcon,_,_ = img.get_delete_cols_images()
		
		
		labelTitle = tk.Label(self.cont, text = 'Live masking ..', **titleLabelProperties) 
		labelTitle.grid(padx=5,pady=15, columnspan=6 ,sticky=tk.W)	
		
		
		labelInfo = tk.Label(self.cont, bg = MAC_GREY,
							 text = 'Masking allows you to subset your data on the fly.')
		labelInfo.grid(columnspan=6,sticky=tk.W)
		
		labelOperator = tk.Label(self.cont, text = 'Logical Operator: ', bg = MAC_GREY)
		labelOperator.grid(row = 2, column = 0,pady=5,padx=3) 
		
		comboOperator = ttk.Combobox(self.cont, textvariable = self.operator ,
												values = ['and','or'], width=8)
		comboOperator.bind('<<ComboboxSelected>>', self.update_clipping_mask)
		comboOperator.grid(row = 2, column = 1,pady=5,padx=3) 
		
		resetButton = ttk.Button(self.cont, text = 'Reset', width=5, command = self.reset_filters)
		resetButton.grid(row = 2, column = 2,pady=5,padx=3) 
		
		addButton = create_button(self.cont, image = self.addImg, command = self.define_new_filter)
		addButton.grid(sticky = tk.W,padx=5,pady=5) 
		
		
		self.create_filterSelection_frame() 
		
		
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		closeButton.grid(columnspan=10,column=4,sticky=tk.E,pady=5,padx=5)
		
	def reset_filters(self):
		'''
		'''
		
		for filterDict in self.treeCollection.values():
			filterDict['frame'].destroy() 
			
			
		self.dfClass.reset_clipping(self.dataID)
		self.treeCollection.clear() 
			
		
		
	def create_filterSelection_frame(self):
	
	
		self.filterSelFrame = ttk.Frame(self.cont) 
		self.filterSelFrame.grid(sticky=tk.NSEW,columnspan=8)
		
	
	def construct_numerical_filter(self,columnName):
		'''
		'''
		columnFrame = tk.Frame(self.filterSelFrame,bg=MAC_GREY,relief=tk.GROOVE,bd=2)
		columnFrame.pack(expand=True, fill=tk.BOTH, side=tk.RIGHT)
		
		labelColumn = tk.Label(columnFrame, text = columnName, **titleLabelProperties) 
		labelColumn.grid(padx=5,pady=5, sticky=tk.W, columnspan=2) 
		
		self.numLimits = dict() 
		
		for n,lim in enumerate(['Min','Max']):
			lab = tk.Label(columnFrame, text = '{}: '.format(lim), bg = MAC_GREY) 
			lab.grid(row = n+1, column = 0, padx=2,pady=2) 
			ent = ttk.Entry(columnFrame, width = 4) 
			ent.grid(row = n+1, column = 1, padx=2,pady=2) 
			ent.bind('<Return>', lambda event, columnName = columnName, id = self.id:\
			self.create_num_clipping_mask(event,columnName,id))
			
			self.numLimits[lim] = ent 
		
		cbAbs = ttk.Checkbutton(columnFrame, variable = self.takeAbsValues , text = 'Absolute Values')
		cbAbs.grid(columnspan=2,pady=3,padx=2)
		
		cbActive = ttk.Checkbutton(columnFrame, variable = self.activeNumFilter,
									 text = 'Active', 
									 command = lambda id = self.id: self.num_filter_handle(id))
		cbActive.grid(columnspan=2,pady=3,padx=2)		
		
		
		delButton = create_button(columnFrame, 
								image = self.delIcon, 
								command = lambda id = self.id: self.remove_filter(id))
		delButton.grid(row=0,column = 3, sticky=tk.NW)
		self.filterCollection[self.id] = {'Type':'Numeric',
										  'columnNames':columnName,
										  'columnFrame':columnFrame,
										  'id':self.id
										  }
		self.id  += 1 
		
	def num_filter_handle(self,id):
		'''
		'''		
		if self.activeNumFilter.get():
			
			if id in self.ignoreIds:
				idx = self.ignoreIds.index(id) 
				del self.ignoreIds[idx]
				
			self.create_num_clipping_mask(event = None,
								columnName = self.filterCollection[id]['columnNames'],
								id = id)
			
		else:	
		
			self.ignoreIds.append(id)	
			self.update_clipping_mask()
		
			
		

	def construct_categorical_filter(self, items, columnName, sep, filterInfo = None):
		'''
		'''
		columnFrame = tk.Frame(self.filterSelFrame,bg=MAC_GREY,relief=tk.GROOVE,bd=2)
		columnFrame.pack(expand=True, fill=tk.BOTH, side=tk.RIGHT)
		vertFrame = ttk.Frame(columnFrame)
		vertFrame.pack(expand=True,fill=tk.BOTH)
		
		scrVert = ttk.Scrollbar(vertFrame,orient=tk.VERTICAL)
		scrHor  = ttk.Scrollbar(vertFrame,orient=tk.HORIZONTAL)
		
		tree = ttk.Treeview(vertFrame, xscrollcommand = scrHor.set, 
							yscrollcommand = scrVert.set)
		scrVert.configure(command = tree.yview)
		scrHor.configure(command = tree.xview)
		for item in items:
			tree.insert('',tk.END,text=item)
			
		tree.heading('#0',text = columnName)		
		scrVert.grid(row=1,column=3,sticky=tk.NS+tk.W,padx=(0,10))
		scrHor.grid(row=2,column=0,columnspan=3,sticky=tk.EW+tk.N,padx=(10,0))
		tree.grid(row=1,column=0,columnspan=3,sticky=tk.NSEW,padx=(10,0))
		
		vertFrame.grid_rowconfigure(1,weight = 1)
		vertFrame.grid_columnconfigure(0,weight = 1)
		
		
		tree.bind('<Double-Button-1>',self.create_clipping_mask)
		tree.tag_configure('selected', background = '#E4DEB6')
		
		delButton = create_button(vertFrame, image = self.delIcon, command = lambda id = self.id: self.remove_filter(id))
		delButton.grid(row=1,column = 4, sticky=tk.NW)
		self.treeCollection[columnName] = {'items':items,'columnName':columnName,'tree':tree,
										   'splitString': sep, 'frame':columnFrame,'id':self.id}
		if filterInfo is not None:
			self.treeCollection[columnName]  = merge_two_dicts(self.treeCollection[columnName],filterInfo)
		
		self.filterCollection[self.id] = {'Type':filterInfo,'selCategories':[],
										  'columnFrame':columnFrame,
										  }

	def remove_filter(self,id):
		'''
		'''
		self.filterCollection[id]['columnFrame'].destroy()
		
		
	def create_clipping_mask(self,event):
		'''
		
		'''		
		for keys,value in self.treeCollection.items():
			if event.widget == value['tree']:
				
				selectedItems = value['tree'].selection()
				if len(selectedItems) == 0:
					return
					
				id = value['id']
				
				iid = selectedItems[0]
				textSelected = value['tree'].item(iid,"text")
				
				if value['splitString'] is None:
						value['splitString'] = ''
						
				if textSelected in self.filterCollection[id]['selCategories']:
						idx = self.filterCollection[id]['selCategories'].index(textSelected)
						del self.filterCollection[id]['selCategories'][idx]
						value['tree'].item(iid, tags = '')
				else:
						self.filterCollection[id]['selCategories'].append(textSelected)
						value['tree'].item(iid, tags = 'selected')
						value['tree'].selection_remove(iid)
						
				regExp = categoricalFilter.build_regex('',
												self.filterCollection[id]['selCategories'],
												splitString = value['splitString'])
				if value['0'] == 'Select from all':
					searchColumn = self.data[value['columnName']]
				
				elif value['0'] in ['Find Strings','Find Categories']:
				
					searchColumn = value['annotationColumn']
					
				boolIndicator = searchColumn.astype(str).str.contains(regExp)
				self.filterCollection[id]['boolIndic'] = boolIndicator
								 
						
		self.update_clipping_mask()
		


	def update_clipping_mask(self,event=None):
		'''
		'''
		self.dfClass.set_current_data_by_id(self.dataID)
		boolDf = pd.DataFrame() 
		
		if len(self.ignoreIds) == len(self.filterCollection):
			if self.operator.get() == 'or':
				boolDf['mask'] = [1] * len(self.dfClass.df.index)
			else:
				# this is a bit confusing but with 'and' we check 
				# if number True == len(ignored Ids) == len(constructed Filters) 
				# which is in this case zero (== 0) 
				boolDf['mask'] = [0] * len(self.dfClass.df.index)
				
		for id, filterSettings in self.filterCollection.items():
			if id not in self.ignoreIds:
				boolDf[str(id)] = filterSettings['boolIndic']
		
		if self.operator.get() == 'or':
			pooledIndic = boolDf.sum(axis=1) > 0
		else:
			
			pooledIndic = boolDf.sum(axis=1) == len(self.filterCollection) - len(self.ignoreIds)
		
		if np.sum(pooledIndic) == 0:
			tk.messagebox.showinfo('Error ..',
								   'Filtering results in an empty data frame.',
								   parent=self.toplevel)
								   
		self.dfClass.create_clipping(self.dataID, pooledIndic)
		
		self.plotter.save_axis_limits()		
		self.plotter.clean_up_figure()
		
		self.plotter.reinitiate_chart(updateData=True, fromFilter = True)			
		

	def create_num_clipping_mask(self,event,columnName,id):
		'''
		'''
		limitsFloat = dict() 
		
		for lim in ['Min','Max']: 
					
			if self.numLimits[lim].get() == '':
				if lim == 'Max':
					limitsFloat[lim] = np.inf
				else:
					limitsFloat[lim] = -np.inf
			else:
				try:
					limitsFloat[lim] = float(self.numLimits[lim].get())
				except:
					tk.messagebox.showinfo('Error ..',
										'Could not convert input to float.',
										parent = self.toplevel)
					return
					
		data = self.dfClass.get_current_data(ignoreClipping = True)
		
		if self.takeAbsValues.get():
		
			boolIndicator = (data[columnName[0]].abs() < limitsFloat['Max']) & \
			(data[columnName[0]].abs() > limitsFloat['Min'])
		
		else:
			boolIndicator = (data[columnName[0]] < limitsFloat['Max']) & \
			(data[columnName[0]] > limitsFloat['Min'])
		
		self.filterCollection[id]['boolIndic'] = boolIndicator
		self.activeNumFilter.set(True)
		self.update_clipping_mask()
						
		
		
	def define_new_filter(self):		
		'''
		'''
		dialog = defineFilter(self.dfClass,self.analysisClass)
		filterSettings = dialog.filterSettings
		
		self.dfClass.set_current_data_by_id(self.dataID)
		
		if filterSettings is not None:
			if filterSettings['0'] == 'Select from all':
			
				data = self.dfClass.get_current_data() 
				
				sep = filterSettings['pickSep']
				columns = filterSettings['pickColumn']
				
				if sep is not None:
				
					splitData = data[columns[0]].astype('str').str.split(sep).values
				
				else:
					splitData = data[columns[0]].astype('str').values
					
				flatSplitData = list(set(chain.from_iterable(splitData)))
				self.construct_categorical_filter(flatSplitData,columns[0],sep,filterSettings)
			
			
			elif filterSettings['0'] == 'Find Strings':
				
				catFilter = categoricalFilter(self.dfClass, '' ,self.plotter,
								 operationType = 'Search string & annotate',
								 columnForFilter = filterSettings['pickColumn'],
								 addToTreeview = False)
				if hasattr(catFilter,'boolIndicator') == False:
					return # user canceled filter creation
					
				filterInfo =   {'boolIndicator':catFilter.boolIndicator,
									'annotationColumn': catFilter.annotationColumn,
									'annotateSearchString':catFilter.annotateSearchString.get()}
				
				filterSettings = merge_two_dicts(filterInfo,filterSettings) 
					
				if catFilter.annotateSearchString.get() == False:
															
					self.construct_categorical_filter(['+','-'],
									str(filterSettings['pickColumn']),'', filterSettings)
				
				else:
				
					self.construct_categorical_filter(np.unique(catFilter.annotationColumn),
										str(filterSettings['pickColumn']),'',filterSettings)
					
				
				del catFilter
							 
			elif filterSettings['0'] == 'Numerical Filter':
				
				self.construct_numerical_filter(filterSettings['pickColumn'])
				
			elif filterSettings['0'] == 'Find Categories':
			
				catFilter = categoricalFilter(self.dfClass, '' ,self.plotter,
								 operationType = 'Find category & annotate',
								 columnForFilter = filterSettings['pickColumn'][0],
								 addToTreeview = False) 	
				if hasattr(catFilter,'boolIndicator') == False:
					return # user canceled filter creation
					
				filterInfo =   {'boolIndicator':catFilter.boolIndicator,
								'annotationColumn': catFilter.annotationColumn,
								'annotateSearchString':False,
								'splitString':catFilter.splitString}
							
				filterSettings = merge_two_dicts(filterInfo,filterSettings) 
				
				self.construct_categorical_filter(np.unique(filterInfo['annotationColumn']),
										str(filterSettings['pickColumn']),
										filterInfo['splitString'],filterSettings)
				del catFilter 
							
			self.id += 1			
		
		
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 		



		
titleText0 = 'Choose type of filter ..'
options0 = ['Select from all','Find Strings','Find Categories','Numerical Filter']			
tooltipInfo0 = ('Filter Type Selection\nSelect from all will display all unique categorical values'+
						  ' found using a separator (;).\nFind Strings will open a dialog after column selection'+
						  ' that allows you to search for certain strings.\n\nFind Categories will allow you to create'+
						  ' a column contain "+" and "-" indicating weather the searched category is present or not.'+
						  '\n\nNumerical Filter will allow you to filter data on numeric criteria on the fly.')

titleTextPickSep = 'Choose separator'
optionsPicksep = ['None',';',',','.','tab','space','//','-','_']
toolTipInfoPicksep = 'Pick a separator from the list. Data will be split first and then filter for unique categorical value.s.' 

tilteTextColumns = 'Select a column'
optionsColumns = None
toolTipInfoColumns = 'Select one or muliple columns for filter.'





filterSelInfo = {'0':[titleText0,options0,tooltipInfo0],
				 'pickSep':[titleTextPickSep,optionsPicksep,toolTipInfoPicksep],
				 'pickColumn':[tilteTextColumns,optionsColumns,toolTipInfoColumns]}			
				
				
			 		
 		
class defineFilter(object):
	
	
	def __init__(self,dfClass,analysisClass):
	
		'''
		'''
		self.id = '0' 
		self.saveFilter = OrderedDict()
		
		self.dfClass = dfClass
		self.analysisClass = analysisClass
		
		filterSelInfo['pickColumn'][1] = self.dfClass.get_columns_of_current_data()
		self.filterOptions = filterSelInfo
		
		
		self.build_toplevel()
		self.build_page()
		self.toplevel.grab_set()
		self.toplevel.wait_window()
	
	
	
	def close(self,event=None, reset = True):
		'''
		'''
		if reset:
			self.saveFilter = None
		self.toplevel.destroy()
		
		
	def build_toplevel(self):
		'''
		'''
		self.toplevel = tk.Toplevel() 
		self.toplevel.bind('<Escape>', self.close)
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close)        	
        	
			
	
	def	build_page(self):
		'''
		'''
		self.cont = ttk.Frame(self.toplevel) 
		self.cont.pack(expand=True,fill=tk.BOTH) 
		self.add_widgets_to_page(self.id)

		
		
	def add_widgets_to_page(self,id):
	
		if id not in self.filterOptions:
			return
			
		titleText,options,tooltipInfo = self.filterOptions[id]
			
		if hasattr(self,'labelTitle'):
			self.labelTitle.configure(text=titleText) 
			self.fill_lb(self.listbox,options)
			
			self.toolTip.text = tooltipInfo
			if self.saveFilter['0'] == 'Find Strings' and self.id == 'pickColumn':
				self.listbox.configure(selectmode = tk.EXTENDED)
			else:
				self.listbox.configure(selectmode = tk.BROWSE)
		else:
			
			self.labelTitle = tk.Label(self.cont, text = titleText,**titleLabelProperties)
			self.labelTitle.grid(padx=5,pady=5,sticky=tk.NW,columnspan=2) 
		
			self.listbox = self.add_listbox()
			self.fill_lb(self.listbox,options) 
			self.toolTip = CreateToolTip(self.listbox,text=tooltipInfo)
		
		
	def add_listbox(self):
		
		
		scrVert = ttk.Scrollbar(self.cont,orient=tk.VERTICAL)
		scrHor  = ttk.Scrollbar(self.cont,orient=tk.HORIZONTAL)
		lb = tk.Listbox(self.cont, selectmode=tk.BROWSE, setgrid = True,
			xscrollcommand = scrHor.set, yscrollcommand = scrVert.set)
		scrVert.configure(command = lb.yview)
		scrHor.configure(command = lb.xview)
		
		scrVert.grid(row=1,column=3,sticky=tk.NS)
		scrHor.grid(row=2,column=0,columnspan=3,sticky=tk.EW)
		lb.grid(row=1,column=0,columnspan=3,sticky=tk.NSEW,padx=(10,0))
		lb.bind('<Double-Button-1>', self.next_page) 
		lb.bind('<Return>', self.next_page) 
		return lb		
		
	def fill_lb(self,lb,data):
		lb.delete(0,tk.END)
		for item in data:
			lb.insert(tk.END,item)
	
	
	def next_page(self,event):
		'''
		'''
		if len(self.listbox.curselection()) == 0:
			return
		if self.id != 'pickColumn':
			sel = self.listbox.curselection()[0]
			item = filterSelInfo[self.id][1][sel]
		else:
			item = [filterSelInfo[self.id][1][idx] for idx in self.listbox.curselection()]
		
		self.saveFilter[self.id] = item
		
		if self.id == '0':
			if item == 'Select from all':
			
				self.id = 'pickSep'
				
			elif item in ['Find Strings','Find Categories','Numerical Filter']:
				
				self.id = 'pickColumn'
						
		elif self.id == 'pickSep':
		
			self.id = 'pickColumn'
		
		elif self.id == 'pickColumn':
			
			self.close(reset=False) 	
			return
		
		self.add_widgets_to_page(self.id)
		
		
	@property
	def  filterSettings(self):
		return self.saveFilter
		
		
	
	
		
		
		
		
		
			
		
		 		
 						
		
			
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		