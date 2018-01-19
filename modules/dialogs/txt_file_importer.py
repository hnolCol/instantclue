import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
from pandastable import Table, TableModel
import numpy as np
import pandas as pd
from modules.pandastable import core
from modules.utils import *
from itertools import chain
from collections import OrderedDict




'''
Need: live link to data
Enocidng
separator
decemial point
Parse only selected columns -> selected ones!! Checkbutton? 

'''


encodingsCommonInPython = ['utf-8','ascii','ISO-8859-1','iso8859_15','cp037','cp1252','big5','euc_jp']
commonSepartor = ['tab',',','space',';','/','&','|','^','+','-']
decimalForFloats = ['.',','] 
compressionsForSourceFile = ['infer','gzip', 'bz2', 'zip', 'xz']
nanReplaceString = ['-','None', 'nan','  ']
comboBoxToGetInputFromUser = OrderedDict([('Encoding:',encodingsCommonInPython),
											('Column Separator:',commonSepartor),
											('Decimal Point String:',decimalForFloats),
											('Decompression:',compressionsForSourceFile),
											('Skip Rows:',list(range(0,20))),
											('Replace NaN in Object Columns:',nanReplaceString)])


class fileImporter(object):
	
	def __init__(self, pathUpload):
		
		self.headerRow = tk.StringVar()
		self.headerRow.set('1')
		
		
		self.data_to_export = None
		self.replaceObjectNan = None
		self.pt = None
		self.pathUpload = pathUpload
		self.comboboxVariables = OrderedDict()
		
		

		self.build_toplevel() 
		self.build_widgets()
		
		self.preview_df = self.load_n_rows_of_file(self.pathUpload, N = 50)
		self.initiate_preview(self.preview_df)
		
		
		self.toplevel.wait_window() 	
	
	def close(self):
		'''
		Close toplevel
		'''
		if hasattr(self,'pt'):
			self.pt.remove()
			del self.pt	
		self.toplevel.destroy() 

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Import Files') 
         
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
 		labTitle = tk.Label(self.cont_widgets, text = 'Settings for file upload',
                                      **titleLabelProperties)  
 		labPreview = tk.Label(self.cont_widgets, text = 'Preview', **titleLabelProperties)
 		labInfo = tk.Label(self.cont_widgets, text = 'If you do not want upload all columns - select columns and delete them using the drop-down menu', bg=MAC_GREY)
 		     		
 		buttonClose = ttk.Button(self.cont_widgets, text = "Close", command = self.discard_changes, width=9)
 		buttonLoad = ttk.Button(self.cont_widgets, text = "Upload", width=9, command  = self.save_changes)
 		self.toplevel.bind('<Return>', self.save_changes)
 		
 		buttonUpdate = ttk.Button(self.cont_widgets, text = "Update", width = 9, command = self.update_preview)
 				
 		## grid widgets
 		labTitle.grid(padx=5,pady=5, columnspan=7, sticky=tk.W) 
 		for comboBoxLabel,comboBoxValues in comboBoxToGetInputFromUser.items():

 			self.create_combobox_with_options(self.cont_widgets,comboboxLabel=comboBoxLabel,
 											optionsForCombobox = comboBoxValues,
 											) 	
 											
 		if '.csv' in self.pathUpload:
 			self.comboboxVariables['Column Separator:'].set(';')


 		buttonClose.grid(padx=3, row=6,column=5, pady=3, sticky=tk.E) 
 		buttonLoad.grid(padx=3, row=5, column=5, pady=3, sticky=tk.E)
 		buttonUpdate.grid(padx=3, row=4, column=5, pady=3, sticky=tk.E)
 		
 		
 		labPreview.grid(padx=5,pady=5, row=8, column=0, sticky = tk.W) 
 		labInfo.grid(row=8, column= 0, padx=(80,0), columnspan = 3, sticky = tk.W)
 		

 		#self.initiate_preview(self.sheets_available[1])	
 			
	def create_preview_container(self,sheet = None):
		'''
		Creates preview container for pandastable. Mainly to delete everything easily and fast.
		'''
		self.cont_preview  = tk.Frame(self.cont,background='white') 
		self.cont_preview.pack(expand=True,fill=tk.BOTH)
		
		
	def create_combobox_with_options(self,tkFrame,comboboxLabel, optionsForCombobox):		
		'''
		Creates Comboboxes with Labels, creates varialbes and saves to dcit (self.comboboxVariables)
		'''
		columnInGridLabel = 0
		columnInGridCombobox = 1 
		comboboxVariable = tk.StringVar() 
		comboboxVariable.set(str(optionsForCombobox[0])) 
		
		labelCombobox = tk.Label(tkFrame, text  = comboboxLabel, bg = MAC_GREY) 
		comboBox = ttk.Combobox(tkFrame, textvariable = comboboxVariable, values = optionsForCombobox)
		
		labelCombobox.grid(in_=tkFrame,padx=5,
							column = columnInGridLabel,
							pady=3,sticky=tk.E) 
		rowCombobox = labelCombobox.grid_info()['row']
							
		comboBox.grid(in_=tkFrame,padx=5,
							row = rowCombobox,
							column = columnInGridCombobox,
							pady=3,sticky=tk.EW) 
		
		self.comboboxVariables[comboboxLabel] = comboboxVariable 
		
		
					
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 		

	

	def initiate_preview(self,df):
		'''
		Actually displaying the data.
		'''
	 
		self.pt = core.Table(self.cont_preview,
						dataframe = df, 
						showtoolbar=False, 
						showstatusbar=False)	
						
		self.pt.parentframe.master.unbind_all('<Return>') #we use this as a shortcut to upload data- will give every time an error	
		self.pt.show()

	def discard_changes(self):
	
 		'''
 		No Export and close toplevel
 		'''
 		self.data_to_export = None
 		self.close()	
 		
	def do_some_bindings(self, comboBox):
		'''
		Bindings to update column headers.
		'''
		comboBox.bind('<<ComboboxSelected>>', self.update_header) 
		comboBox.bind('<Return>', self.update_header)

	def evaluate_columns(self,columnList):
		'''
		Turn columns in columnList into strings. This is useful, when this columns contains NaN
		'''
		
		columnList = [str(col) for col in columnList]
		return columnList
		
		
	def extract_combobox_variables(self):
		'''
		Returns values from the created comboboxes.
		'''
		uploadSettings = [value.get() for value in self.comboboxVariables.values()]
		
		return uploadSettings
	
	
		 
	def load_n_rows_of_file(self,path,N,usecols=None,lowMemory=False):
		'''
		Loads file given by path to display preview. If N is None -> Load all.
		'''
		encoding, separator, decimal, compression, skipRows, self.replaceObjectNan = self.extract_combobox_variables()
		if separator == 'tab':
			separator = '\t'
		elif separator == 'space':
			separator = '\s+'
		try:	
		
			dfNRowsChunks = pd.read_table(path, encoding=encoding, sep = separator,
							decimal = decimal, compression = compression, low_memory=lowMemory,
							skiprows = int(float(skipRows)), nrows = N,
							usecols = usecols, chunksize = 10000)
			chunkList = []
			for chunk in dfNRowsChunks:
				chunkList.append(chunk)
				
			dfNRows = pd.concat(chunkList)
							
		except:
			try:
				dfNRows = pd.read_table(path, encoding=encoding, sep = separator,
							decimal = decimal, compression = compression, low_memory=lowMemory,
							skiprows = int(float(skipRows)), nrows = N,
							usecols = usecols)
			except:
				tk.messagebox.showinfo('Please revise ..','There was an error parsing your file. Please revise upload settings.')
				return
		

		return dfNRows
		
	def update_preview(self):
		'''
		Update preview when user clicks update.
		'''
		self.cont_preview.destroy()
		self.create_preview_container()
		## reload data with new settings
		
		self.preview_df = self.load_n_rows_of_file(self.pathUpload, N = 50)
		if self.preview_df is not None:
			self.initiate_preview(self.preview_df)
		
		
		
	def save_changes(self, event = None):
		'''
		Defines self.data_to_export to set the data to be exported of the importer class
		'''
		columnsToUpload = self.pt.model.df.columns
		#print(columnsToUpload)
		
		self.data_to_export = self.load_n_rows_of_file(self.pathUpload, N = None, 
														usecols = columnsToUpload)
		if self.data_to_export is None:
			return
		else:
			self.pt.parentframe.destroy()												
			self.close()

	
				
         
		