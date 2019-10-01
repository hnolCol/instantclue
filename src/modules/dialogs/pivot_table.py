"""
	""PIVOT TABLE""
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
import pandas as pd


from modules.utils import *

class pivotDialog(object):

	def __init__(self,dfClass, dataID):
		
		self.dfClass = dfClass
		self.dataID = dataID
		
		
		self.data = self.dfClass.get_current_data()
		self.pivotedDf = pd.DataFrame()
				
		self.build_toplevel() 
		self.build_widgets()
		
		self.toplevel.wait_visibility()
		self.toplevel.grab_set() 
		self.toplevel.wait_window()
		
			
	def close(self):
		'''
		Close toplevel
		'''
		self.toplevel.destroy() 	
		

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Pivot Table') 
        
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=340
		h=220
		self.toplevel = popup
		self.center_popup((w,h))
	
	def build_widgets(self):

 		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont.grid_columnconfigure(1,weight=1)
 		 
 		
 		labelTitle = tk.Label(self.cont, text= 'Transform data form long to wide format (pivot).'+
 									' Note: The index is very import, if not specified no rows will be '+
 									'aggregated (input and output will have the same number of rows).',wraplength=330,**titleLabelProperties)
                                     
                                     
                                     
 		labelTitle.grid(columnspan=2, sticky=tk.W, padx=3,pady=5)
 		
 		self.selectedColumns = {}
 		
 		for n,kw in enumerate(['index','columns','values']):
 			var = tk.StringVar()
 			lab = tk.Label(self.cont, text = kw, bg=MAC_GREY)
 			comboBox = ttk.Combobox(self.cont, textvariable = var,
 									values = self.dfClass.df_columns)
 			comboBox.grid(row=n+2, column=1, sticky=tk.EW, padx=3,pady=3)
 			if 'PriorMeltIndex' in self.data.columns and kw == 'index':
 				var.set('PriorMeltIndex')
 			else:
 				var.set('Please Select')
 				
 			lab.grid(row=n+2, column=0, sticky=tk.E, padx=3,pady=3)
 			self.selectedColumns[kw] = var
  			
  			
 			
 		applyButton = ttk.Button(self.cont, text = 'Transform', command = self.perform_transform)
 		
 		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
 		
 		applyButton.grid(row=5, column = 0, sticky=tk.W)
 		closeButton.grid(row=5, column = 1, sticky=tk.E)
 		
	def perform_transform(self):
		'''
		'''
		idx = self.selectedColumns['index'].get()
		values = self.selectedColumns['values'].get()
		columns = self.selectedColumns['columns'].get()
		for kw in [idx,values,columns]:
			if kw not in self.dfClass.df_columns:
				kw = None
		try:
			self.pivotedDf  = self.data.pivot(index = idx ,values=values,columns=columns)
		except Exception as e:
			tk.messagebox.showinfo('Error ..','An error occured:\n'+str(e))
			# to be save
			self.pivotedDf = pd.DataFrame()
			return
			
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
 
 
# to do: make this class more general                                 
class transformColumnDialog(object):

	def __init__(self,slectedColumn, allColumns):
		
		self.allColumns = allColumns
		self.newColumnName = tk.StringVar(value=slectedColumn)
		self.columnForColumns = None
		self.build_toplevel() 
		self.build_widgets()
		
		self.toplevel.wait_window()
		
			
	def close(self):
		'''
		Close toplevel
		'''
		self.toplevel.destroy() 	
		

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Pivot Table') 
		popup.grab_set() 
        
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=340
		h=160
		self.toplevel = popup
		self.center_popup((w,h))
	
	def build_widgets(self):

 		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont.grid_columnconfigure(1,weight=1)
 		 
 		
 		labelTitle = tk.Label(self.cont, text= 'Transpose Data\nUnique values in column that is used to'+
 									' assign columns should have the same length as the number of '+
 									'columns in the transposed data\nYou can also set this to "None" and '+
 									'the column names will be simply the index', wraplength=320,
                                     **titleLabelProperties)
                                     
                                     
 		labelTitle.grid(columnspan=2, sticky=tk.W, padx=3,pady=5)
 		labColumns = tk.Label(self.cont, text = 'New Column Names: ', bg=MAC_GREY)
 		labColumns.grid(row=2,column=0,padx=3,pady=3,sticky=tk.E)
 		
 		comboBox = ttk.Combobox(self.cont, textvariable = self.newColumnName,
 												values =self.allColumns)
 		comboBox.grid(row=2,column=1,padx=3,pady=3,sticky=tk.EW)
 		
 			
 		applyButton = ttk.Button(self.cont, text = 'Done', command = self.save_selection)
 		
 		closeButton = ttk.Button(self.cont, text = 'Close', command = self.discard_selection)
 		
 		applyButton.grid(row=5, column = 0, sticky=tk.W)
 		closeButton.grid(row=5, column = 1, sticky=tk.E)
	def discard_selection(self):
 		'''
 		'''
 		self.columnForColumns = None
 		self.close() 	
	def save_selection(self):
		'''
		'''
		self.columnForColumns = self.newColumnName.get()
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
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     


