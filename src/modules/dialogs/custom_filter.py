"""
	""CUSTOM FILTERING""
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
import numpy as np
import pandas as pd 

import os

import tkinter as tk
from tkinter import ttk

from itertools import chain 

import re

from modules import images
from modules.pandastable import core
from modules.utils import *


class customFilterDialog(object):
	
	def __init__(self,dfClass,selectedColumns):
	
		self.data = dfClass.get_current_data()
		self.selectedColumns = selectedColumns
		
		
		self.get_images()
		self.define_variables()
		self.create_toplevel() 		
		self.build_menu()
		self.build_widgets() 
		
		
		self.toplevel.wait_window()
		
	def define_variables(self):
		'''
		'''
		self.mot_button = None
		self.items_selected = None
		self.widget = None
		self.output_data_frame = None
		self.mode = None
		self.trees = dict() 
		self.operator_var = tk.StringVar(value='OR') 
		self.annotate_category = tk.BooleanVar(value=False)
				
	def create_toplevel(self):
		self.toplevel = tk.Toplevel() 
		self.toplevel.wm_title('Custom Categorical Filter...')
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close_toplevel)
		self.toplevel.bind('<Escape>', self.close_toplevel)
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(expand=True, fill='both')
		cont.grid_rowconfigure(4,weight = 1) 
		self.cont = cont
		
		
	def build_menu(self):
	
			
		self.menu = tk.Menu(self.cont,**styleDict) 
		self.menu.add_command(label='Custom filter ..',state=tk.DISABLED,foreground="darkgrey")
		self.menu.add_separator()
		self.menu.add_command(label='Keep all', command = lambda: self.select_all(mode='select'))
		self.menu.add_command(label='Trash all', command = lambda: self.select_all(mode='trash'))
		
		
	def select_all(self,mode):
		if mode == 'select':
			im_ = self.keep_icon
		else:
			im_ = self.trash_icon
			
		for iid in self.tree_menu.get_children():
			self.tree_menu.item(iid, image = im_) 
		
		
	def cast_menu(self,event):
	
		self.tree_menu = event.widget
		x = self.cont.winfo_pointerx()
		y= self.cont.winfo_pointery() 
		self.menu.post(x,y) 
		
	
		
	def build_widgets(self):
	
		lab1 = tk.Label(self.cont, 
			text = 'Custom categorical filter\n\nDrag and Drop unique categories to trash or cart.', 
			**titleLabelProperties)		
		
		lab1.grid(padx=5,pady=15, columnspan=6 ,sticky=tk.W)
		
		
		for n,column in enumerate(self.selectedColumns):
			n = n*2
			var_ = tk.StringVar() 
                         
			tree_data = ttk.Treeview(self.cont, height = "9", show="tree") 
			scroll_y = ttk.Scrollbar(self.cont ,orient = tk.VERTICAL, command = tree_data.yview)
			tree_data.configure(yscrollcommand = scroll_y.set)
			
			lab_sep = tk.Label(self.cont, text='Separator: ',bg=MAC_GREY)
			lab_column = tk.Label(self.cont, text=column[0:40], bg=MAC_GREY,fg="#4C626F")
			combo_sep = ttk.Combobox(self.cont, values = [';',':','_','-','/'], width=5) 
			combo_sep.bind('<Return>', self.update_separator)
			combo_sep.bind('<<ComboboxSelected>>', self.update_separator)
			combo_sep.set(';')		
			lab_search = tk.Label(self.cont, text= 'Search: ', bg=MAC_GREY)
			search_entry = ttk.Entry(self.cont, textvariable = var_, width=5) 
			
			lab_column.grid(row=1,column=n,padx=1,sticky=tk.W,columnspan=2) 
			lab_sep.grid(row=2, column=n,padx=1,sticky=tk.W)
			lab_search.grid(row=3, column=n,padx=1,sticky=tk.W)
			combo_sep.grid(row=2, column=n+1,padx=1,sticky=tk.EW)
			search_entry.grid(row=3, column=n+1,padx=1,sticky=tk.EW) 
			tree_data.grid(row=4, column = n,columnspan=2,padx=(2,10), sticky=tk.NSEW) 
			scroll_y.grid(row=4,column=n+1, sticky= tk.NS+tk.E) 
			self.cont.grid_columnconfigure(n+1,weight = 1, minsize=80) 
			tree_data.bind("<<TreeviewSelect>>", self.on_select) 
			tree_data.bind("<B1-Motion>", self.on_motion) 
			tree_data.bind("<ButtonRelease-1>", self.on_release) 
			tree_data.bind("<Double-Button-1>", self.keep_clicked_items)
			tree_data.bind(right_click, self.cast_menu)
			
			var_.trace(mode="w", callback= lambda varname, elementname, mode, tree=tree_data,column=column, var_ = var_: self.on_trace(varname, elementname, mode, tree,column,var_))
			self.trees[column] = [tree_data,combo_sep,search_entry]
		
		self.add_data_to_trees()
		self.keep_button = tk.Button(self.cont, image = self.car_icon) 
		self.trash_button = tk.Button(self.cont, image = self.dustbin_icon) 
		self.keep_button.grid(row=6, column = 0, padx=20, columnspan=2,pady=8, sticky=tk.W)
		self.trash_button.grid(row=6, column = 1, padx=20, columnspan=n+1,pady=8, sticky=tk.E) 
		
		
		cont_buttons = tk.Frame(self.cont, background =MAC_GREY)
		cont_buttons.grid(columnspan=40,row=8,column=0,sticky=tk.EW)
		
		for col_num in [1,3,5,7]:
			cont_buttons.grid_columnconfigure(col_num ,weight=1)
		
		
		filt_label = tk.Label(cont_buttons, text = 'Perform filtering using this ...', font = LARGE_FONT, fg="#4C626F", justify=tk.LEFT, bg = MAC_GREY)	
		operator_label = tk.Label(cont_buttons, text = 'operator for filtering: ', bg=MAC_GREY)
		operator_om = ttk.OptionMenu(cont_buttons, self.operator_var, self.operator_var.get(), *['AND','OR'])
		
		filt_button1 =  ttk.Button(cont_buttons, text='Filter', command = \
		lambda: self.filter_and_close(mode='remove')) 
		filt_button2 =  ttk.Button(cont_buttons, text='Subset', command = \
		lambda: self.filter_and_close(mode = 'subset')) 
		filt_button3 =  ttk.Button(cont_buttons, text='Annotate ', command = \
		lambda: self.filter_and_close(mode = 'annotate'))
		
		CreateToolTip(filt_button1, text = 'This activity will remove all rows that do match any given criteria')
		CreateToolTip(filt_button2, text = 'This activity creates a new data frame containing data that match any of the given criteria')
		CreateToolTip(filt_button3, text = 'Adds a column indicating rows that match given criteria by a "+" sign.'+
			' Adds the categorical value instead of a "+" sign if the check button "Annotate matching'+
			' categories" is enabled.')		
		
		close_button = ttk.Button(cont_buttons, text='Close', command = self.close_toplevel) 
		cb_matches = ttk.Checkbutton(cont_buttons,text='Annotate matching categories', variable = self.annotate_category)
		CreateToolTip(cb_matches, text = 'If checked, rows that match given criteria will be annotated by their category value (name)')
		
		filt_label.grid(row=0, column = 0 ,padx=10,pady=3,columnspan=3)
		ttk.Separator(cont_buttons, orient= tk.HORIZONTAL).grid(sticky=tk.EW, columnspan=12, in_ = cont_buttons,pady=(2,4))
		operator_label.grid(row=2,column=0, padx=5, pady=3, columnspan=2)
		operator_om.grid(row=2,column=2,padx=5,pady=3,sticky=tk.W)
		cb_matches.grid(row=2,column=4,padx=5,pady=3,sticky = tk.E, columnspan=3)
		
		filt_button1.grid(row=3, column = 0 ,padx=2,pady=3,in_ = cont_buttons,sticky=tk.EW) 
		filt_button2.grid(row=3, column = 2 ,padx=2,pady=3,in_ = cont_buttons,sticky=tk.EW)
		filt_button3.grid(row=3, column = 4 ,padx=2,pady=3,in_ = cont_buttons,sticky=tk.EW)
		close_button.grid(row=3, column = 6 ,padx=(5,2),pady=3,in_ = cont_buttons,sticky=tk.EW) 

		
	def update_separator(self,event):
		'''
		If User updates the separator: Splits data on given separator 
		and adds them to the tree view. 
		'''
		widget = event.widget
		sep_ = widget.get() 
		
		for key,values in self.trees.items():
			if widget == values[1]:
				column = key
				break
		dat_ = self.get_unique_vals(column,widget) 		
		self.add_data_to_trees(data = dat_, column = column, new_separator=True) 
		

	def add_data_to_trees(self, data = None, column= None, new_separator = False):
		'''
		Adds data in tree view.
		'''
		if new_separator:
			tree_ = self.trees[column][0]
			tree_.delete(*tree_.get_children())
			for entry_ in data:
					tree_.insert('','end',iid=column+'_##Instant__'+entry_, text = entry_, image=self.trash_icon) 
			return
		if data is None:
			for column in self.selectedColumns:
				tree_,combo_sep,search_entry = self.trees[column]
				data_ = self.get_unique_vals(column,combo_sep) 
				for entry_ in data_:
					tree_.insert('','end',iid=column+'_##Instant__'+entry_, text = entry_, image=self.trash_icon) 
					
				self.trees[column] = [tree_,combo_sep,search_entry,pd.DataFrame(data_, columns=[column])]
		else:
			tree_ = self.trees[column][0]
			for i,entry_ in enumerate(data):
				iid = column+'_##Instant__'+entry_
				tree_.move(iid,'',index=i)
				
			
	def get_unique_vals(self,column,combo_sep):
		'''
		Return unique values.
		'''
		if self.data[column].dtype in [np.float64,np.int64]:
			dat_ = self.data[column].astype(str)
		else:
			dat_ = self.data[column]
		dat_ = dat_.str.split(combo_sep.get()).dropna() 
		dat_ = list(set(chain.from_iterable(dat_)))
		return dat_
		
	def on_select(self,event): 
		w = event.widget
		self.items_selected =  list(w.selection())
		self.items_from_source = [w.item(item)['text'] for item in self.items_selected]
	
	
	def keep_clicked_items(self,event):
	
		tree_ = event.widget
		for item in self.items_selected:
			if tree_.item(item)['image'][0] == str(self.keep_icon):
			
					tree_.item(item, image = self.trash_icon)
			else:
					tree_.item(item, image = self.keep_icon)
					
	def on_trace(self, varname = None, elementname = None, mode = None, tree = None,column = None, var_ = None):
	
		
		data_ = self.trees[column][-1]
		data_ = data_[data_[column].str.contains(var_.get())]
		self.add_data_to_trees(data=data_[column], column= column)
		
		
		
	def on_motion(self,event):
		
		if self.mot_button is None:
			self.mot_button = tk.Button(self.cont, text=str(self.items_from_source),
                                     bd=0,
                                     fg="grey", bg=MAC_GREY)
                                    
		self.widget = self.cont.winfo_containing(event.x_root, event.y_root)
		if self.widget == self.keep_button:	
			self.mot_button.configure(fg="darkgreen") 
		elif self.widget == self.trash_button:
			self.mot_button.configure(fg="red")
		else:
			self.mot_button.configure(fg = "grey")
		
		
		x = self.cont.winfo_pointerx() - self.cont.winfo_rootx()
		y = self.cont.winfo_pointery() - self.cont.winfo_rooty()
		self.mot_button.place( x= x-20 ,y = y-30) 
			
	def on_release(self,event):
	
		if self.widget in [self.keep_button, self.trash_button]:
			col_ = self.items_selected[0].split('_##Instant__')[0]
			tree_ = self.trees[col_][0]
			if self.widget == self.keep_button:
				
				for item in self.items_selected:
					tree_.item(item, image = self.keep_icon)

			else:
				for item in self.items_selected:
					tree_.item(item, image = self.trash_icon)
		
		if self.mot_button is not None:
				self.mot_button.destroy() 
				self.mot_button = None
				self.widget = None
				
				
	def join_strings(self,row):
		row_ = row.dropna() 
		row_ = row_.str.cat(sep=";") 
		return row_
		
	def custom_search_annotate(self,row,keep,sep = ';'):
		
		matches_ = []
		cat_ = np.nan
		rowSplit = row.split(sep)
		for cat in keep:
			if cat in rowSplit:
				matches_.append(cat)
		number_of_matches =len(matches_)
		if number_of_matches == 0:
			return cat_
		elif number_of_matches == 1:
			return matches_[0]
		else:
			cat_  = ';'.join(matches_)
			return cat_
			
	def custom_search(self,row,keep):
			for cat in keep:
				if cat in row:
					return True
				else:
					pass
			return False
		
	def filter_and_close(self, mode = ''):
		'''
		Identifies categorical values that the user wish to keep. Identified by the icon type.
		Then it builds for each tree view a regular expression to find the rows that match.
		Parameter 
		==========
		Input  - mode Filter, Annotate, Subset .. 
		
		'''
		self.mode = mode
		data_ = self.data
		operator = self.operator_var.get()
		collect_subframes = pd.DataFrame()
		for column,tree_ in self.trees.items():
			tree = tree_[0]
			separator = tree_[1].get()			
			remove_ = [tree.item(item)['text'] for item in tree.get_children() if tree.item(item)['image'][0] == str(self.trash_icon)]
			keep_ = [tree.item(item)['text'] for item in tree.get_children() if tree.item(item)['image'][0] == str(self.keep_icon)]
			regEx = self.build_regex(keep_, separator)
			
			if len(keep_) == 0 and operator == 'AND':
				tk.messagebox.showinfo('Error..',
					'Your filter selection leads to an empty data frame. Aborting...', 
					parent=self.toplevel)
				return	
				
			if operator == 'AND':
					data_ = data_[data_[column].astype(str).str.contains(regEx)]
			else:
				if self.annotate_category.get():
					collect_subframes[column] = self.data[column].astype(str).apply(lambda row, sep = separator: self.custom_search_annotate(row,keep_,sep))
				else:
					collect_subframes[column] = self.data[column].astype(str).str.contains(regEx)
				
			if data_.empty and operator == 'AND':
				tk.messagebox.showinfo('Error..',
					'Your filter selection leads to an empty data frame. Aborting...', 
					parent=self.toplevel)
				return
		
		if operator == 'OR' and self.annotate_category.get() == False:
			data_ = self.data[collect_subframes.sum(axis=1) > 0]
			
		elif self.annotate_category.get():

			collect_subframes = collect_subframes.dropna(how='all')
			data_ = pd.DataFrame()
			if len(self.selectedColumns) == 1:
				data_ = collect_subframes
			elif len(self.selectedColumns) > 1:
				data_ = collect_subframes[self.selectedColumns].apply(lambda row: self.join_strings(row), axis=1)
			elif len(self.selectedColumns) == 2:
				data_.loc[:,'comb_'] = collect_subframes[self.selectedColumns[0]].str.cat(collect_subframes[self.selectedColumns[1]],sep=';',na_rep ='-')
			else:
				collect_cols = []	
				for col in self.selectedColumns:
					if col == self.selectedColumns[0]:
						pass
					else:
						collect_cols.append(collect_subframes[col].values.tolist())
				
				data_.loc[:,'comb_'] = collect_subframes[self.selectedColumns[0]].str.cat(collect_cols,sep=';',na_rep ='-')				
			
				
		if len(data_.index) == len(self.data.index) and self.mode != 'annotate' and self.annotate_category.get() == False:
			tk.messagebox.showinfo('Error..','You have not selected a single category to be removed.', parent=self.toplevel)
			return		
		self.output_data_frame = data_	
		self.close_toplevel()
	         
	
	def build_regex(self,categoriesList,splitString, withSeparator = True):
		'''
		Build regular expression that will search for the selected category. Importantly it will prevent 
		cross findings with equal substring
		=====
		Input:
			List of categories that were selected by the user
			Split String - String by which row-content is split to achieve unique categories.
		====
		'''
		regExp = r''
		
		for category in categoriesList:
			category = re.escape(category) #escapes all special characters
			if withSeparator:
				regExp = regExp + r'({}{})|(^{}$)|({}{}$)|'.format(category,splitString,category,splitString,category)
			else:
				regExp = regExp + r'({})|'.format(category)
		regExp = regExp[:-1] #strip of last |
		return regExp
		
		
	def get_images(self):
		'''
		Get images from base64 code
		'''
		self.dustbin_icon,self.car_icon,self.keep_icon,self.trash_icon = images.get_custom_filter_images()
		
		
		
		
	def get_data(self):
		'''
		Returns data
		'''
		return self.output_data_frame, self.mode, self.annotate_category.get()

				 
	def close_toplevel(self, event = None):
		'''
		'''
		self.toplevel.destroy()
		
