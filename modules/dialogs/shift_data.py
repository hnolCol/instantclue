"""
	""COLUMN NAME CHANGER""
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
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
from collections import OrderedDict
from modules.utils import *
from modules.dialogs.import_subset_of_data import importDataFromDf


class shiftTimeData(object):
	
	
	def __init__(self,columns,dfClass,sourceTreeView,analyzeClass):
						
		
		self.columns = columns 
		self.dfClass = dfClass
		self.analyzeClass = analyzeClass
		
		self.dataTreeview = sourceTreeView
		
		self.sortColumn = tk.BooleanVar(value=True)
		self.entries = {}
		
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
		popup.wm_title('Shift data') 
		popup.bind('<Escape>', self.close) 
		#popup.bind('<Return>', lambda _,entryDict = self.entryDict : self.rename_columns_in_df(entryDict))
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=480
		h=170+int(len(self.columns))*35 ##emperically
		self.toplevel = popup
		self.center_popup((w,h))
		
			
	def build_widgets(self):
		'''
		'''
		
		
		self.cont = tk.Frame(self.toplevel,background = MAC_GREY)
		self.cont.pack(fill='both', expand=True)
		self.cont.grid_columnconfigure(1,weight=1,minsize=200)
		self.cont.grid_columnconfigure(0,weight=1,minsize=200)
		self.cont.grid_rowconfigure(3,weight=1)
		labelTitle = tk.Label(self.cont, text = 'Shift data to adjust starting point.', **titleLabelProperties)
		labelTitle.grid(pady=5,padx=5,sticky=tk.W,columnspan=2)
		
		labelColumn = tk.Label(self.cont, text = 'Select time (x) column:', 
			bg = MAC_GREY)
		labelColumn.grid(row = 1, column = 0, pady = 2, padx = 2, sticky = tk.E)
		
		
		self.cbColumns = ttk.Combobox(self.cont, values = self.dfClass.get_numeric_columns())
		self.cbColumns.set(self.dfClass.get_numeric_columns()[0])
		self.cbColumns.grid(row = 1, column = 1, pady = 2, padx = 2, sticky = tk.EW)
		self.cbColumns.configure(state='readonly')
		
		cbSort = ttk.Checkbutton(self.cont,variable = self.sortColumn, 
			text = 'Sort time column')
		cbSort.grid(column = 1, sticky = tk.W)
		
		self.build_interval_widgets()
		
		applyButton = ttk.Button(self.cont, text = 'Apply', 
			command = self.shift_data)
		closeButton = ttk.Button(self.cont, text = 'Close', 
			command = self.close)
		
		importButton = ttk.Button(self.cont, text = 'Import from file', 
			command = self.import_data)
		
		importButton.grid(row = 4, column = 1, sticky = tk.E)
		applyButton.grid(row = 5, column = 0, sticky = tk.W)
		closeButton.grid(row = 5, column = 1, sticky = tk.E)
		
	def build_interval_widgets(self):
	
		win = tk.Frame(self.cont, bg = MAC_GREY)
		win.grid(row=3,sticky = tk.NSEW,columnspan=2)
		
		lab1 = tk.Label(win, text = 'Remove data\nbefore',bg = MAC_GREY)
		lab2 = tk.Label(win, text = 'Zero time\npoint',bg = MAC_GREY)
		lab1.grid(row=0,column = 1, pady = (10,0))
		lab2.grid(row=0,column = 2, pady = (10,0))
		for n,column in enumerate(self.columns):
			lab = tk.Label(win,text=column, bg = MAC_GREY)
			lab.grid(row=n+1,column=0)
			
			entStart = ttk.Entry(win)
			entStart.grid(row=n + 1,column=1)
			entStart.insert(0,'0')
			entEnd = ttk.Entry(win)
			entEnd.grid(row=n+1,column=2, sticky = tk.EW)
			self.entries[column] = [entStart,entEnd]
		
	def center_popup(self,size):
         	'''
         	Casts the popup in center of screen
         	'''

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 		
	
	
	
	def import_data(self):
		'''
		'''
		importer = importDataFromDf(self.dfClass,
 						title = 'Select data from preview as your start/zero values.'+
 						' They must match either in row- or column-number the'+
 						' selected numeric columns: {}.'.format(len(self.columns)),
 						requiredDataPoints = len(self.columns),
 						allowMultSelection = True)
		selectionData = importer.get_data_selection()
		
		del importer
		if selectionData is None:
			return		
		
		vals = selectionData.values
		for n,column in enumerate(self.columns):
			entries  = self.entries[column]
			row = vals[n,:2].tolist()
			entries[0].insert(0,row[0])
			entries[1].insert(0,row[1])				
		
	def shift_data(self):	
		'''
		'''
		intervalData = self.read_entries()
		if intervalData is None:
			return
		if True:
		
			df = self.dfClass.shift_data_by_row_matches(matchColumn = self.cbColumns.get(), 
											  adjustColumns = self.columns,
											  intervalData = intervalData,
											  sort = self.sortColumn.get(),
											  removeOtherData = True)
			self.analyzeClass.add_new_dataframe(df,
				'ShiftedData_{}'.format(self.dfClass.get_file_name_of_current_data()))
			
			tk.messagebox.showinfo('Done',
				'Data shifting applied. Data frame has been added.',
				parent = self.toplevel)
		
		else:
			
			tk.messagebox.showinfo('Error..',
				'There was an error shifting your data.',
				parent = self.toplevel)
	def read_entries(self):
		
		intervalData = {}
		self.columns  = [col for col in self.columns if col != self.cbColumns.get()]
		for column in self.columns:
			entries = self.entries[column]
			try:
				start, stop = [float(ent.get()) for ent in entries]
				if start > stop:
					tk.messagebox.showinfo('Error..',
						'Value for "remove data" cannot be greater than the "new zero" point.',
						parent=self.toplevel)
					return

				intervalData[column] = (start,stop)
			except:
				tk.messagebox.showinfo('Error..',
					'Cannot interpret as float.Erorr happened in \n{}.'.format(column), 
					parent = self.toplevel)
				return
					
		return intervalData
		
			
			
		
		
		
		
	
	
	
				
             
             
             
             
             
             