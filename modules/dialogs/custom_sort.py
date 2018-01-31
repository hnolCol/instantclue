import tkinter as tk
from tkinter import ttk  
import tkinter.font as tkFont
           

import numpy as np
import pandas as pd

import csv
import re

from modules import images
from modules.utils import *



class customSortDialog(object):
	'''
	customSortDialog can be used to :  
	
	=================
	Operations
		- customary reorder categorical values
		- customary reorder column names

	=================
	'''

	def __init__(self, inputValues, dfClass = None, dataTreeview = None):		
		'''
		input Values  - dict like.
		'''
		
		self.inputValues = inputValues
		self.resortedValues = None
		
		self.build_toplevel()
		self.build_widgets()
		
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
		popup.wm_title('Custom Sorting') 
		popup.bind('<Escape>',self.close)
		popup.grab_set() 
		popup.protocol("WM_DELETE_WINDOW", self.discard_changes)
		w=400
		h=500
		self.toplevel = popup
		self.center_popup((w,h))
		
			
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		 		
 		labelTitle = tk.Label(self.cont, text= 'Move items in list to reorder', 
                                     **titleLabelProperties)        
 		labelTitle.grid(padx=30, pady=15, sticky=tk.W)
 		self.create_listbox()
 		
 		sortButton = ttk.Button(self.cont,text='Sort',width=6,command = self.extract_sorted_values)
 		closeButton = ttk.Button(self.cont,text='Close',width=6, command = self.discard_changes)
 		
 		sortButton.grid(row=4,column=0,sticky=tk.W,padx=3,pady=8)
 		closeButton.grid(row=4,column=2,columnspan=2,sticky=tk.E,padx=3,pady=8)
        
        
	def discard_changes(self):
		'''
		'''     
		self.resortedValues = None
		self.close()

	def create_listbox(self):
		'''
		'''
		
		self.cont.grid_columnconfigure(2,weight=1)
		self.cont.grid_rowconfigure(1,weight=1)
		scrVert = ttk.Scrollbar(self.cont,orient=tk.VERTICAL)
		scrHor  = ttk.Scrollbar(self.cont,orient=tk.HORIZONTAL)
		self.treeView = ttk.Treeview(self.cont, xscrollcommand = scrHor.set,
							yscrollcommand = scrVert.set,
							show='tree')
		scrVert.configure(command = self.treeView.yview)
		scrHor.configure(command = self.treeView.xview)
		
		
		scrVert.grid(row=1,column=3,sticky=tk.NS)
		self.treeView.grid(row=1,column=0,columnspan=3,sticky=tk.NSEW,padx=(10,0))
		scrHor.grid(row=2,column=0,columnspan=3,sticky=tk.EW)
		
		self.treeView.bind('<B1-Motion>',self.on_motion)
		self.enter_values()
		
		
	def enter_values(self):
	
		for key,valueList in self.inputValues.items():
			parent = key
			self.treeView.insert('',tk.END,iid=parent,text=parent)
			
			for n,value in enumerate(valueList):
				self.treeView.insert(parent,tk.END,iid = 'iid{}:{}_{}'.format(n,key,value),text=value)
		
	def on_motion(self,event):
		'''
		'''
		selection = list(self.treeView.selection())		
		itemUnderCurs = self.treeView.identify_row(event.y)
		

		if len(selection) == 0:
			return
		
		if itemUnderCurs in selection:
			return
		elif itemUnderCurs == '':
			return
		elif self.treeView.parent(itemUnderCurs) != self.treeView.parent(selection[0]):
			return
		else:
			idx_ = self.treeView.index(selection[0])
			idxSelection = [self.treeView.index(item) for item in selection]
			idxCurs = self.treeView.index(itemUnderCurs)
			if max(idxSelection) < idxCurs:
				pass

			elif min(idxSelection) > idxCurs:
				selection = selection[::-1]
			elif idx_ == 0:
				return			
			for n,item in enumerate(selection):

				parent = self.treeView.parent(item)
				index = self.treeView.index(item)
				self.treeView.move(item,parent,idxCurs)
				
				
	def extract_sorted_values(self):
		'''
		'''
		self.resortedValues = OrderedDict()
		columns = self.treeView.get_children()
		for column in columns:
			splitString = ':{}_'.format(column)
			sortedValues = [iid.split(splitString)[-1] for iid in self.treeView.get_children(column)]
			self.resortedValues[column] = sortedValues				
		self.close()				
		
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 











# 
# 
#              def move_items_around(event,lb):
#                  
#                  selection = lb.curselection()
#                  if len(selection) == 0:
#                      return 
#                  pos_item = selection[0]
#                  curs_item = lb.nearest(event.y)
#                  if pos_item == curs_item:
#                     return
#                  else:
#                      text = lb.get(pos_item)
#                      lb.delete(pos_item)
#                      lb.insert(curs_item,text)
#                      lb.selection_set(curs_item)
#              
#def resort_source_column_data(lb,idx_,popup):
#                  new_sort = list(lb.get(0, END))
#                  self.sourceData.df_columns = new_sort
#                  self.sourceData.df = self.sourceData.df[new_sort]
#                  
#                  index = range(0,len(new_sort))
#                  dict_for_sorting = dict(zip(new_sort,index))
#                  
#                  for col in new_sort:
#                      item = idx_+str(col)
#                      parent = self.source_treeview.parent(item)
#                      
#                      index = dict_for_sorting[col]
#                      self.source_treeview.move(item,parent,index)
#                  tk.messagebox.showinfo('Done..','Columns were resorted. This order will also appear when you export the data frame.',parent=popup)    
#              
#              def resort_dtype(lb,idx_,pop,dtype):
#                  new_sort = list(lb.get(0, END))
#                  old_cols = self.sourceData.df_columns
#                  old_cols_s = [col for col in old_cols if col not in new_sort]
#                  self.sourceData.df_columns = new_sort + old_cols_s
#                  self.sourceData.df = self.sourceData.df[self.sourceData.df_columns]
#                   
#                  for i,col in enumerate(new_sort):
#                      item = idx_+str(col)
#                      parent = self.source_treeview.parent(item)                    
#                      index = i
#                      self.source_treeview.move(item,parent,index)
#                      
# 
#                  
#                  tk.messagebox.showinfo('Done..','Columns were resorted in the provided order.Please note that the newly sorted columns are placed at the beginning of the source file. Visible upon export.',
#                                         parent = popup) 
# 
#                  
#              def resort_source_data(lb,col,popup):
#                  new_sort = lb.get(0, END)
#                  index = range(0,len(new_sort))
#                  dict_for_sorting = dict(zip(new_sort,index))
#                  self.sourceData.df['sorting_idx_instant_clue'] = self.sourceData.df[col].replace(dict_for_sorting)
#                  self.sourceData.df.sort_values(by = 'sorting_idx_instant_clue',inplace=True)
#                  self.sourceData.df.drop('sorting_idx_instant_clue',1, inplace=True)
#                  
#                  if len(last_called_plot) > 0:
#                      self.prepare_plot(*last_called_plot[-1])
#                  tk.messagebox.showinfo('Done..','Custom re-sorting performed.',parent=popup)
#                  
#              if  (len(self.DataTreeview.columnsSelected  ) > 1 and mode == 'Custom sorting') or (mode == 'Re-Sort Columns' and len(self.data_sources_selected) > 1) or (self.only_datatypes_selected and len(self.DataTreeview.columnsSelected  ) > 1 and mode == 'Re-Sort'):
#                  if mode == 'Re-Sort Columns':
#                      typ = 'Dataframe'
#                      sel = self.data_sources_selected[0]
#                  elif mode == 'Re-Sort':
#                      typ = 'Data type'
#                      sel = self.data_types_selected[0]
#                  else:
#                      typ = 'Column'
#                      sel = str(self.DataTreeview.columnsSelected  [0])
#                  tk.messagebox.showinfo('Note..','Can perform this action only on one {}.\nFirst one selected: {}\nSorting is stable, meaning that you can perform sorting on columns sequentially.'.format(typ,sel))
#         
#              if mode == 'Re-Sort Columns':
#                  sel = self.data_sources_selected[0]
#                  for key,value in self.idx_and_file_names.items():
#                          if value == sel:
#                              idx_ = key
#                  self.set_source_file_based_on_index(idx_)            
#                  uniq_vals = self.sourceData.df_columns
#                  
#              elif mode == 'Re-Sort':
#                  if self.only_datatypes_selected:
#                      sel = self.items_selected[0]
#                      idx_,dtype = sel.split('_')[0]+'_',sel.split('_')[-1]
#                      
#                      
#                  else:
#                      return
#                 
#                  self.set_source_file_based_on_index(idx_)
#                  
#                  ##set_data_to_current_selection()
#                  
#                  uniq_vals = [col for col in self.sourceData.df_columns if self.sourceData.df[col].dtype == dtype]
#              else:
#                  ##set_data_to_current_selection()
#                  col_to_sort = self.DataTreeview.columnsSelected  [0]
#                  uniq_vals = list(self.sourceData.df[col_to_sort].unique()) 
#                  
#                  
# 
#              
#              popup.attributes('-topmost', True)
#              
#              cont = self.create_frame(popup)  
#              cont.pack(fill='both', expand=True)
#              cont.grid_rowconfigure(2, weight=1)
#              cont.grid_columnconfigure(0, weight=1)
#              lab_text =  'Move items in listbox in custom order'
#              
#              lab_main = tk.Label(cont, text= lab_text, 
#                                      font = LARGE_FONT, 
#                                      fg="#4C626F", 
#                                      justify=tk.LEFT, bg = MAC_GREY)
#              lab_main.grid(padx=10, pady=15, columnspan=6, sticky=tk.W)
#              scrollbar1 = ttk.Scrollbar(cont,
#                                           orient=VERTICAL)
#              scrollbar2 = ttk.Scrollbar(cont,
#                                           orient=HORIZONTAL)
#              lb_for_sel = Listbox(cont, width=1500, height = 1500,  xscrollcommand=scrollbar2.set,
#                                       yscrollcommand=scrollbar1.set, selectmode = tk.SINGLE)
#              lb_for_sel.bind('<B1-Motion>', lambda event, lb=lb_for_sel : move_items_around(event,lb))
#              lb_for_sel.grid(row=2, column=0, columnspan=3, sticky=tk.E, padx=(20,0))
#              scrollbar1.grid(row=2,column=4,sticky = 'ns'+'e')
#              scrollbar2.grid(row=5,column =0,columnspan=3, sticky = 'ew', padx=(20,0))
#              
#              scrollbar1.config(command=lb_for_sel.yview)
#              scrollbar2.config(command=lb_for_sel.xview)
#              
#              fill_lb(lb_for_sel,uniq_vals)
#              if mode == 'Re-Sort Columns':
#                  but_okay = ttk.Button(cont, text = 'Sort', command = lambda lb=lb_for_sel,idx_=idx_, pop = popup: resort_source_column_data(lb,idx_,pop))
#              elif mode == 'Re-Sort':
#                  but_okay = ttk.Button(cont, text = 'Sort', command = lambda lb=lb_for_sel,idx_=idx_, pop = popup: resort_dtype(lb,idx_,pop,dtype))
#              
#              else:   
#                  but_okay = ttk.Button(cont, text = 'Sort', command = lambda lb=lb_for_sel, col = col_to_sort, pop = popup: resort_source_data(lb,col,pop))
#              but_close = ttk.Button(cont, text = 'Close', command = popup.destroy)
#              but_okay.grid(row = 6, column = 1, pady=5)
#              but_close.grid(row = 6, column = 2, pady=5)
#                          
       