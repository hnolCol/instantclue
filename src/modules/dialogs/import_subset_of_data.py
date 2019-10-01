"""
	""IMPORT SUBSET DATA""
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

"""
Explanation
================
This dialog allows the selection of rows/columns from already loaded data.
This is beneficial, when you would like to correlate data or selected 
x-axis values for curve fits. It also has a parameter claled
requiredDataPoints that will trigger the indication if requirement is met or not.
"""
import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
from modules.pandastable import core
import numpy as np
import pandas as pd
from itertools import chain
from modules.utils import *


class importDataFromDf(object):

	def __init__(self,dfClass,title,requiredDataPoints = None, allowMultSelection = False,dataFrame = None):
		
		self.data_to_export = pd.DataFrame()
		
		
		self.dfClass = dfClass
		self.title = title
		
		self.get_file_names()
		self.dfSelection = tk.StringVar()
		self.selectionIndicator = tk.StringVar()
		
		self.requiredDataPoints = requiredDataPoints 
		self.allowMultSelection = allowMultSelection
		self.build_toplevel() 
		self.build_widgets()
		self.toplevel.wait_visibility()
		self.toplevel.grab_set() 
		self.toplevel.wait_window() 
		
	def close(self,reset = False):
		'''
		Close toplevel
		'''
		if reset or self.selectionMatches == False:
			self.currentRows = []
			self.currentColumns = [] 
		self.toplevel.after_cancel(self.dataSelection)
		self.toplevel.destroy() 	
		
	
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Data Importer')  
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=880
		h=430
		self.toplevel = popup
		self.center_popup((w,h))
		
			
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont_widgets = tk.Frame(self.cont,background=MAC_GREY) 
 		self.cont_widgets.pack(fill=tk.X, anchor = tk.W) 
 		self.cont_widgets.grid_columnconfigure(1,weight=1)
 		self.create_preview_container() 
 		
 		
 		
		## define widgets 
 		labTitle = tk.Label(self.cont_widgets, text = self.title, **titleLabelProperties)
                                        
                                      
 		labelDF = tk.Label(self.cont_widgets, text = 'Data frames:', bg=MAC_GREY) 
 		
 		self.labelSelection = tk.Label(self.cont_widgets, textvariable = self.selectionIndicator, 
 																**titleLabelProperties)
                
 		self.optionmenuDataFrames = ttk.OptionMenu(self.cont_widgets, self.dfSelection, self.currFileName,
 											 *self.fileNames, command = self.refresh_preview)
 										  											 		
 		buttonClose = ttk.Button(self.cont_widgets, text = "Close", command = lambda: self.close(reset=True), width=9)
 		buttonLoad = ttk.Button(self.cont_widgets, text = "Done", width=9, command = self.close)
 		
 				
 		## grid widgets
 		labTitle.grid(padx=5,pady=5, columnspan=7, sticky=tk.W) 
 		 		
 		labelDF.grid(padx=5,pady=5, row=2, column=0, sticky = tk.E) 				
 		self.optionmenuDataFrames.grid(padx=5,pady=5, row=2, column=1,columnspan=9, sticky = tk.EW)
 		self.labelSelection.grid(row=4,column=0, pady=5, padx=10, sticky=tk.W, columnspan=3)  		
		

 		buttonClose.grid(padx=3, row=4,column=6, pady=5, sticky=tk.E) 
 		buttonLoad.grid(padx=3, row=4, column=5, pady=5, sticky=tk.E) 
 		
		
 		self.initiate_preview()
 		self.identify_selected_data()
 	

	def create_preview_container(self,sheet = None):
		'''
		Creates preview container for pandastable. Mainly to delete everything easily and fast.
		'''
		self.cont_preview  = tk.Frame(self.cont,background='white') 
		self.cont_preview.pack(expand=True,fill=tk.BOTH)

 		
	def initiate_preview(self):
		'''
		Actually displaying the data.
		'''
		dataFrameID = self.getIdFromName[self.currFileName]
	 	
		df = self.dfClass.get_data_by_id(dataFrameID)
		self.pt = core.Table(self.cont_preview,
						dataframe = df, 
						showtoolbar=False, 
						showstatusbar=False)
		self.pt.show()
		
		
		
	def identify_selected_data(self):
		
		
		self.currentRows = self.pt.multiplerowlist
		self.currentColumns = self.pt.multiplecollist
		self.selectionMatches = True
		nRows = len(self.currentRows)
		nCols = len(self.currentColumns)
		shapeString = 'Rows: {} x Columns: {}'.format(nRows,
														   nCols)
		
		
		
		if nRows == self.requiredDataPoints and (nCols == 1 or self.allowMultSelection):
			selectionString = 'Row selection matches - '+shapeString
			self.labelSelection.configure(**titleLabelProperties)
			
		elif nCols == self.requiredDataPoints  and (nRows == 1 or self.allowMultSelection):
			selectionString =  'Column selection matches - '+shapeString
			self.labelSelection.configure(**titleLabelProperties)
		else:
			selectionString = 'Selection does not match - '+shapeString
			self.labelSelection.configure(fg='red')
			self.selectionMatches = False
			
		self.selectionIndicator.set(selectionString)
			 
	
		self.dataSelection = self.toplevel.after(200, self.identify_selected_data)
		
		
		
	def refresh_preview(self, idAndName):
		'''
		Refreshing the preview of a file. 
		'''
		if idAndName in self.getIdFromName:
			dataFrameID = self.getIdFromName[idAndName]
			data = self.dfClass.get_data_by_id(dataFrameID)
			self.pt.model.df = data
			self.pt.redraw()
	
	def get_file_names(self):
		'''
		'''
		self.getIdFromName = {}
		fileNameSelected = self.dfClass.get_file_name_of_current_data()
		for id, fileName in self.dfClass.fileNameByID.items():
			self.getIdFromName['{} {}'.format(id,fileName)] = id
			if fileNameSelected == fileName:
				self.currFileName = '{} {}'.format(id,fileName)
		
		self.fileNames = list(self.getIdFromName.keys())
		
		
	def get_data_selection(self):
		if len(self.currentRows) > 0 and len(self.currentColumns) > 0:
			return self.pt.model.df.iloc[self.currentRows,self.currentColumns]
		else:
			return
         

	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 		
  