"""
	""CUSTOM SORTING""
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
from modules.utils import *



class customSortDialog(object):
	'''
	customSortDialog can be used to :  
	
	=================
	Activities using this dialog:
		- custom reorder categorical values
		- custom reorder column names
		- custom sort of columns in receiver boxes

	=================
	'''

	def __init__(self, inputValues, dfClass = None, infoText = '', enableDeleting = False,
							dataTreeview = None, parentOpen = []):		
		'''
		input Values  - dict like.
		'''
		
		self.inputValues = inputValues
		self.resortedValues = None
		self.parentOpen = parentOpen
		self.infoText = str(infoText)
		self.enableDeleting = enableDeleting
		
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
		popup.bind('<Return>',self.extract_sorted_values)
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
 		labelTitle.grid(padx=30, pady=15, sticky=tk.W,columnspan=2)
 		
 		labelInfo = tk.Label(self.cont, text = self.infoText, bg = MAC_GREY)
 		labelInfo.grid(padx=5, pady=15, sticky=tk.W,columnspan=3)
 		self.create_listbox()
 		
 		sortButton = ttk.Button(self.cont,text='Sort',width=6,command = self.extract_sorted_values)
 		closeButton = ttk.Button(self.cont,text='Close',width=6, command = self.discard_changes)
 		
 		sortButton.grid(row=5,column=0,sticky=tk.W,padx=3,pady=8)
 		closeButton.grid(row=5,column=1,columnspan=3,sticky=tk.E,padx=3,pady=8)
        
        
	def discard_changes(self):
		'''
		Discard sorting and close.
		'''     
		self.resortedValues = None
		self.close()

	def create_listbox(self):
		'''
		Grids listbox and scrollbars.
		'''
		self.cont.grid_columnconfigure(2,weight=1)
		self.cont.grid_rowconfigure(2,weight=1)
		scrVert = ttk.Scrollbar(self.cont,orient=tk.VERTICAL)
		scrHor  = ttk.Scrollbar(self.cont,orient=tk.HORIZONTAL)
		self.treeView = ttk.Treeview(self.cont, xscrollcommand = scrHor.set,
							yscrollcommand = scrVert.set,
							show='tree')
		
		scrVert.configure(command = self.treeView.yview)
		scrHor.configure(command = self.treeView.xview)
		
		if self.enableDeleting:
			self.treeView.bind('<BackSpace>', self.delete_selection)
			self.treeView.bind('<Delete>', self.delete_selection)
		
		scrVert.grid(row=2,column=3,sticky=tk.NS)
		self.treeView.grid(row=2,column=0,columnspan=3,sticky=tk.NSEW,padx=(10,0))
		scrHor.grid(row=3,column=0,columnspan=3,sticky=tk.EW)
		
		self.treeView.bind('<B1-Motion>',self.on_motion)
		self.enter_values()
		

	def delete_selection(self,event):
		'''
		'''
		selection = list(self.treeView.selection())	
		self.treeView.delete(selection[0])
		
				
	def enter_values(self):
		'''
		Fill listbox.
		'''
		for key,valueList in self.inputValues.items():
			parent = key
			if parent != '':

				if parent in self.parentOpen:
					self.treeView.insert('',tk.END,iid=parent,text=parent,open=True)
				else:
					self.treeView.insert('',tk.END,iid=parent,text=parent)
				
			
			for n,value in enumerate(valueList):
				self.treeView.insert(parent,tk.END,iid = 'iid{}:{}_{}'.format(n,key,value),text=value)
		
	def on_motion(self,event):
		'''
		Handles motion when left-click (Button-1) is pressed.
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
				
				
	def extract_sorted_values(self, event = None):
		'''
		Extract re-sorted values and save them in resortedValues dict. Key 
		is the "column". 
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
   