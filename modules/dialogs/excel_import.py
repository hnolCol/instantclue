import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
from modules.pandastable import core
import numpy as np
import pandas as pd
from itertools import chain
from modules.utils import *


class ExcelImporter(object):

	def __init__(self,excel_sheets, excel_file):
	
		self.data = dict()
		self.excel_file = excel_file
		self.replaceObjectNan = '-'		
		self.sheets_available = ['All Excel Sheets'] + excel_sheets
		self.sheet_selected = tk.StringVar() 
		self.sheet_selected.set(excel_sheets[0])
		self.headerRow = tk.StringVar()
		self.headerRow.set('1')
		self.df_dimensions = tk.StringVar()
		self.pt = None
		
		self.parse_sheets_into_dict() ## will create self.data 
		self.build_toplevel() 
		self.build_widgets()
		
		self.toplevel.wait_window() 
		
	def close(self):
		'''
		Close toplevel
		'''
		self.data_to_export = False
		self.toplevel.destroy() 	
		
	def get_files(self):
	
		'''
		Returns loaded (and modified) files.
		'''
		
		sheet_to_export = self.sheet_selected.get()
		
		if sheet_to_export == self.sheets_available[0]: ##preventing unwanted scenario if the sheet is called by accident 'All Excel Sheets'	
			
			data_to_export = self.data		
			pass
			
		else:
			
			data_to_export = self.data[sheet_to_export]
			
		return data_to_export
			
	
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Excel Importer') 
         
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
 		labTitle = tk.Label(self.cont_widgets, text = 'Selected Excel file contains several sheets.\nYou can choose one or load all. Use the interactive mode to customize the file import ...',
                                      font = LARGE_FONT, fg="#4C626F",bg = MAC_GREY, justify=tk.LEFT)  
 		labSheets = tk.Label(self.cont_widgets, text = 'Excel Sheets:', bg=MAC_GREY) 
 		labHeaderRow = tk.Label(self.cont_widgets, text = 'Headers are in row:', bg=MAC_GREY)
 		labDataFrameShape = tk.Label(self.cont_widgets, textvariable = self.df_dimensions, bg=MAC_GREY, fg="#4C626F" )
 		comboboxHeaderRow = ttk.Combobox(self.cont_widgets, textvariable = self.headerRow ,values = list(range(0,30)))
 		
 		self.do_some_bindings(comboboxHeaderRow)
                
 		self.optionMenuSheets = ttk.OptionMenu(self.cont_widgets, self.sheet_selected, self.sheets_available[1],
 											 *self.sheets_available, command = self.refresh_preview)
 		buttonMelt = ttk.Button(self.cont_widgets, text = 'Melt', command = self.melt_df)
 											  											 		
 		buttonClose = ttk.Button(self.cont_widgets, text = "Close", command = self.close, width=9)
 		buttonLoad = ttk.Button(self.cont_widgets, text = "Load", width=9, command  = self.save_changes)
 		# bind Return to upload stuff
 		self.toplevel.bind('<Return>', self.save_changes)
 				
 		## grid widgets
 		labTitle.grid(padx=5,pady=5, columnspan=7, sticky=tk.W) 
 		 		
 		labSheets.grid(padx=5,pady=5, row=2, column=0, sticky = tk.E) 				
 		self.optionMenuSheets.grid(padx=5,pady=5, row=2, column=1,columnspan=9, sticky = tk.EW)
 		
 		labHeaderRow.grid(padx=5,pady=5, row=3, column=0, sticky = tk.E) 
 		comboboxHeaderRow.grid(padx=5,pady=5, row=3, column=1,columnspan=9, sticky = tk.EW)
 		
 		labDataFrameShape.grid(padx=3, row=4,column=0, pady=5, sticky=tk.W,columnspan=2)
 		
 		buttonMelt.grid(padx=3, row=4,column=1, pady=5, sticky=tk.E) 

 		buttonClose.grid(padx=3, row=4,column=6, pady=5, sticky=tk.E) 
 		buttonLoad.grid(padx=3, row=4, column=5, pady=5, sticky=tk.E) 

 		self.initiate_preview(self.sheets_available[1])
 		
 		
	def do_some_bindings(self, comboBox):
		'''
		Bindings to update column headers.
		'''
		comboBox.bind('<<ComboboxSelected>>', self.update_header) 
		comboBox.bind('<Return>', self.update_header) 


	def create_preview_container(self,sheet = None):
		'''
		Creates preview container for pandastable. Mainly to delete everything easily and fast.
		'''
		self.cont_preview  = tk.Frame(self.cont,background='white') 
		self.cont_preview.pack(expand=True,fill=tk.BOTH)

 		
	def initiate_preview(self,sheet = None):
		'''
		Actually displaying the data.
		'''
	 
		df_ = self.data[sheet]
		self.get_shape_and_display(df_)
		self.pt = core.Table(self.cont_preview,
						dataframe = df_, 
						showtoolbar=False, 
						showstatusbar=False)
		self.current_sheet = sheet				
		self.pt.show()
		self.pt.parentframe.master.unbind_all('<Return>')
		
	def refresh_preview(self, new_excel_sheet):
		'''
		Refreshing the preview of a file. 
		'''

		if new_excel_sheet == 'All Excel Sheets':
			return
		if self.current_sheet == new_excel_sheet:
			return
		else:
			self.save_changes(destroy = False)
			self.cont_preview.destroy()
			self.create_preview_container() 
			self.initiate_preview(sheet = new_excel_sheet) 
			
			#self.pt.model.df.columns = self.data[self.current_sheet].iloc[2,:]
			
	def update_header(self,event):
		'''
		If the user wants to skip set row as header in your file. 
		'''
		try:
			row_ = int(float(self.headerRow.get())-1)	
		except:
			tk.messagebox.showinfo('Error..','Cannot convert entry string to integer.')	
			return
		
		columnList = self.evaluate_columns(self.data[self.current_sheet].iloc[row_,:])	
		self.pt.model.df.columns =  columnList# sets new columns headers
		self.pt.tablecolheader.redraw()
		
		## basically only needed because we cannot undo this. (maybe in the future..)
		
		quest_to_delete = tk.messagebox.askquestion('Delete ?','Would you like to delete this row from your data?')
		
		if quest_to_delete == 'yes':
			self.pt.model.deleteRows(rowlist = [row_])
			self.pt.redraw()
			
		else:
			pass 
		
	def evaluate_columns(self,columnList):
		'''
		Turn columns in columnList into strings.
		'''
		
		columnList = [str(col) for col in columnList]
		return columnList
	
		
	def parse_sheets_into_dict(self,skipped_rows = 0):
		'''
		Parses Excel file into a dictionary
		
		'''
		self.data = {sheet_name: self.excel_file.parse(sheet_name) 
          	for sheet_name in self.excel_file.sheet_names}
         
         
	def save_changes(self, destroy = True):
		'''
		Saves the modified dataframe in self.data dictionary for exporting. 
		'''
		#print(self.pt)
	
		data_updated = self.pt.model.df
		self.data[self.current_sheet] = data_updated
		if destroy:
			## delete
			self.data_to_export = True
			del self.pt	
			self.toplevel.destroy()
		
		
	def get_shape_and_display(self,df):
		shape_ = df.shape
		dimensions = 'Row {} x Columns {}'.format(shape_[0],shape_[1])
		self.df_dimensions.set(dimensions)	
	
	
	def melt_df(self):
		'''
		Melt Dialog with user. User can preview results. Then the melted df is added into the Option Menu
		'''
		columns_selected = self.pt.multiplecollist	
		df_name = 'Melt_of_'+self.current_sheet
		
		if len(columns_selected) > 1:
			quest = tk.messagebox.askquestion('Perform?','''You have selected some columns. 
Would you like to melt them directly (click YES) 
or would you like to first check out a preview of 
the melting procedure (click NO).''')	
				
			if quest == 'yes':
				melted_data = self.perform_melt_in_import_window(columns_selected)
				self.add_new_data(melted_data,df_name)	
				return
			else:
				pass
		
		melt_dialog = _HelperMelt(dataframe = self.data[self.current_sheet])
		melted_data = melt_dialog.melted_df
		del melt_dialog
		if melted_data.empty:		
			return				
			
		else:		
			self.add_new_data(melted_data,df_name)			
			
			
			
	def add_new_data(self,df,df_name):
		'''
		Add new data to the option menu (from melting etc ...)
		'''
	
		## check if such a df is already there...
		
		melted_df_in_list = len([sheet for sheet in self.sheets_available if df_name in sheet])
		df_name = df_name+'_'+str(melted_df_in_list)
		self.data[df_name] = df		
		self.sheets_available.append(df_name)
		
		## ugly
		self.create_new_option_menu()
		tk.messagebox.showinfo('Performed ..','Data frame added. Well done.',parent = self.toplevel) 
			

	def create_new_option_menu(self):
		'''
		This is very ugly, but the creation of another Toplevel caused problems for
		the OptionMenu in the other Toplevel.. 
		'''
		self.optionMenuSheets.destroy()
		self.optionMenuSheets = ttk.OptionMenu(self.cont_widgets, self.sheet_selected, self.sheets_available[1],
 											 *self.sheets_available, command = self.refresh_preview)
 											
		self.optionMenuSheets.grid(padx=5,pady=5, row=2, column=1,columnspan=9, sticky = tk.EW)
		
	
	def perform_melt_in_import_window(self, column_indic):

		'''
		Does the melting. columns are provided by indices. id_vars are actually all columns,
		but the selected ones.
		'''
		df_ = self.data[self.current_sheet]
		df_column = df_.columns.values.tolist() 
		column_names = [col for n,col in enumerate(df_column) if n not in column_indic]
		for_column_name = [col for col in df_column if df_column.index(col) in column_indic]
		name_ = ','.join(tuple(for_column_name))
		variable_name = 'variable_['+name_+']' 
		value_name = 'value_['+name_+']' 
		melted_df = pd.melt(df_, 
									id_vars = column_names,
									var_name= variable_name,
							 		value_name = value_name)	
		return melted_df
		
		
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 		
         

		
class _HelperMelt(object):

	def __init__(self,dataframe):
		self.cont_preview = None
		self.id_vars_for_melt = None
		self.melted_df = pd.DataFrame()
		self.data = dataframe
	
		self.df_head = self.check_df_and_reduce(dataframe)
		self.df_columns = dataframe.columns.values.tolist()
		
		
		
		self.build_toplevel()
		self.build_widgets()
		self.initiate_preview(self.df_head) 
	
	
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
		popup.wm_title('Melt DataFrame') 
         
		popup.protocol("WM_DELETE_WINDOW", self.discard_melt)
		w=510
		h=280
		self.toplevel = popup
		self.center_popup((w,h))
		
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		cont_widgets = tk.Frame(self.cont,background=MAC_GREY) 
 		cont_widgets.pack(fill=tk.X, anchor = tk.W) 
 		cont_widgets.grid_columnconfigure(1,weight=1)
 		
 		
 		
 		
		## define widgets 
 		labTitle = tk.Label(cont_widgets, text = 'By melting a dataframe you change from wide- to long-format.\nSimply try and check out the preview ..',
                                      font = LARGE_FONT, fg="#4C626F",bg = MAC_GREY, justify=tk.LEFT)  
                                    
        #listboxColumns  = tk.Listbox(cont_widgets, selectmode = tk.EXTENDED) 
         
 		buttonClose = ttk.Button(cont_widgets, text = "Close", command = self.discard_melt)
 		buttonOkay = ttk.Button(cont_widgets, text = "Okay", command  = self.return_melted_df)
 		buttonPreview = ttk.Button(cont_widgets, text = "Preview" , command = self.display_preeview)
 		buttonUndo = ttk.Button(cont_widgets, text = "Undo", command = self.undo_everything)
         
	
 		## grid widgets
 		labTitle.grid(padx=5,pady=5, columnspan=6, sticky=tk.W) 
 		buttonClose.grid(padx=3, row=4,column=6, pady=5, sticky=tk.E) 
 		buttonOkay.grid(padx=3, row=4, column=5, pady=5, sticky=tk.E) 	
 		buttonPreview.grid(padx=3, row=4, column=2, pady=5, sticky=tk.E) 
 		buttonUndo.grid(padx=3, row=4, column=3, pady=5, sticky=tk.E)  	 	

 		 		
	def return_melted_df(self):
		'''
		Returns the melted data to be further processed.
		'''

		self.melt_the_data(self.data, id_vars = self.id_vars_for_melt)
		self.close() 
		
	def discard_melt(self):
		'''
		Set melted_df None and closes dialog.
		'''
		del self.melted_df
		self.melted_df = pd.DataFrame()
		self.close() 
		
	def check_df_and_reduce(self,dataframe,n=100):

 		'''
 		Allows only a limit of rows (n) in df
 		'''
 		n_rows = len(dataframe.index)
 		if n_rows < n:
 			return dataframe
 		else:
 			dataframe = dataframe.head(n) 
 			return dataframe
		
	def create_preview_container(self):
		'''
		Creates preview container for pandastable. Mainly to delete everything easily and fast.
		'''
		self.cont_preview  = tk.Frame(self.cont,background='white') 
		self.cont_preview.pack(expand=True,fill=tk.BOTH)
		
	def melt_the_data(self,df,id_vars = None, column_indices = None):
		'''
		Does the melting. columns are provided by indices. id_vars are actually all columns,
		but the selected ones.
		'''
		if id_vars is None:
			if column_indices is None:
				column_indices = self.pt.multiplecollist				
			column_names = [col for n,col in enumerate(self.df_columns) if n not in column_indices]
			for_column_name = [col for col in self.df_columns if self.df_columns.index(col) in column_indices]
		else:
			column_names = id_vars
			for_column_name = [col for col in self.df_columns if col not in id_vars]
		name_ = ','.join(tuple(for_column_name))
		
		variable_name = 'variable_['+name_+']' 
		value_name = 'value_['+name_+']' 
		
		self.melted_df = pd.melt(df, 
									id_vars = column_names,
									var_name= variable_name,
							 		value_name = value_name)
				 
		self.id_vars_for_melt =  column_names ## saves the ids 
		
	
	def display_preeview(self):
		'''
		Calculates the melted Dataframe on the shortened data set. 
		'''
		columns_selected_index = self.pt.multiplecollist
		
		self.melt_the_data(self.df_head,column_indices = columns_selected_index) 
		
						 					 		
		### displaying the data 
							 		
		self.cont_preview.destroy() 
		self.cont_preview = None
		self.initiate_preview(self.melted_df)
	
 	
		
	def initiate_preview(self,df_):
		'''
		Actually displaying the data.
		'''

		if self.cont_preview is None:
			self.create_preview_container()
		
		self.pt = Table(self.cont_preview,
						dataframe = df_, 
						showtoolbar=False, 
						showstatusbar=False)				
		self.pt.show()				
		
	def undo_everything(self):
		'''
		Going back to beginning.
		'''
		del self.melted_df
		self.melted_df  = pd.DataFrame()
		self.cont_preview.destroy() 
		self.cont_preview = None
		self.initiate_preview(self.df_head)
		
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 
      