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

import xml.etree.ElementTree as ET
from modules.dialogs.simple_dialog import simpleUserInputDialog, simpleListboxSelection
from modules.dialogs.VerticalScrolledFrame import VerticalScrolledFrame
import tkinter.filedialog as tf


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
thoursandsString = ['None',',','.']
comboBoxToGetInputFromUser = OrderedDict([('Encoding:',encodingsCommonInPython),
											('Column Separator:',commonSepartor),
											('Decimal Point String:',decimalForFloats),
											('Thousand Separator:',thoursandsString),
											('Decompression:',compressionsForSourceFile),
											('Skip Rows:',list(range(0,20))),
											('Replace NaN in Object Columns:',nanReplaceString)])



pandasInstantTranslate = {'Encoding:':'encoding',
						  'Column Separator:':'sep',
						  'Decimal Point String:':'decimal',
						  'Thousand Separator:':'thousands',
						  'Decompression:':'compression',
						  'Skip Rows:':'skiprows',
						  }

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
		
		self.toplevel.wait_visibility()
		self.toplevel.grab_set() 
		self.toplevel.wait_window() 	
	
	def close(self,event=None):
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
		popup.bind('<Escape>', self.close) 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=880
		h=630
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
 		labInfo = tk.Label(self.cont_widgets, text = 'If you do not want upload all columns - select'+
 						' columns and delete them using the drop-down menu.\nPlease '+
 						'note that deleting rows in the preview will not effect the data.', bg=MAC_GREY)
 		     		
 		buttonClose = ttk.Button(self.cont_widgets, text = "Close", command = self.discard_changes, width=9)
 		buttonLoad = ttk.Button(self.cont_widgets, text = "Load", width=9, command  = self.save_changes)
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
 		buttonLoad.grid(padx=3, row=4, column=5, pady=3, sticky=tk.E)
 		buttonUpdate.grid(padx=3, row=5, column=5, pady=3, sticky=tk.E)
 		
 		
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
		encoding, separator, decimal, thousands, compression, skipRows, self.replaceObjectNan = \
		self.extract_combobox_variables()
		
		if separator == 'tab':
			separator = '\t'
		elif separator == 'space':
			separator = '\s+'
		if thousands == 'None':
			thousands = None
		try:	
		
			dfNRowsChunks = pd.read_table(path, encoding=encoding, sep = separator,
							decimal = decimal, compression = compression, low_memory=lowMemory,
							thousands = thousands, skiprows = int(float(skipRows)), nrows = N,
							usecols = usecols, chunksize = 10000)
			chunkList = []
			for chunk in dfNRowsChunks:
				chunkList.append(chunk)
				
			dfNRows = pd.concat(chunkList)
							
		except:
			try:
				dfNRows = pd.read_table(path, encoding=encoding, sep = separator, thousands = thousands,
							decimal = decimal, compression = compression, low_memory=lowMemory,
							skiprows = int(float(skipRows)), nrows = N,
							usecols = usecols)
				
			except:
				tk.messagebox.showinfo('Please revise ..',
					'There was an error parsing your file. Please revise upload settings.\n'+
					'Changing the encoding (utf-8 -> ISO-8859-1) or to adjust the decimal or thousands point string might help.')
				return pd.DataFrame()
		
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
		
		if self.pt is None:
			return
			
		columnsToUpload = self.pt.model.df.columns
		
		self.data_to_export = self.load_n_rows_of_file(self.pathUpload, N = None, 
														usecols = columnsToUpload)
		if self.data_to_export is None:
			return
		else:
			self.pt.parentframe.destroy()												
			self.close()






class multipleTxtFileLoader(object):
	
	
	
	def __init__(self):
		
		self.generalLoadSettings = comboBoxToGetInputFromUser
		#self.define_variables()
		self.create_toplevel()
		self.create_widgets()
		
		self.toplevel.wait_window()
	


	def close(self,event=None):
		'''
		'''
		self.toplevel.destroy()
				
		
	def create_toplevel(self):
		'''
		'''
		
		self.toplevel = tk.Toplevel() 
		self.toplevel.wm_title('Multiple file loader...')
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close)
		self.toplevel.bind('<Escape>', self.close)
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(expand=True, fill='both')
		cont.grid_rowconfigure(4,weight = 1)
		cont.grid_columnconfigure(1,weight = 1) 
		self.cont = cont 
		
		
	def create_widgets(self):
		'''
		'''
		labelTitle = tk.Label(self.cont,text = 'Load multiple txt files.',**titleLabelProperties)
		labelTitle.grid()
		
		dirButton = ttk.Button(self.cont, text = 'Select folder', command = self.select_dir)
		settingButton = ttk.Button(self.cont, text = 'General Settings', command = self.get_upload_settings)
		
		fileFrame = tk.Frame(self.cont, bg = MAC_GREY)
		fileFrame.grid_rowconfigure(0,weight=1)
		fileFrame.grid_columnconfigure(0,weight=1)
		
		self.create_file_options(fileFrame)
		
		
		loadButton = ttk.Button(self.cont, text = 'Load', command = self.upload_files)
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		
		
		dirButton.grid(row=1,column = 0, sticky = tk.W, padx = 5, pady = 2)
		settingButton.grid(row=1,column = 1, sticky = tk.E, padx = 5, pady = 2)		
		
		
		fileFrame.grid(row = 4, columnspan = 2,sticky = tk.NSEW)
		loadButton.grid(row=5,column = 0, sticky = tk.W, padx = 5, pady = 2)
		closeButton.grid(row=5,column = 1, sticky = tk.E, padx = 5, pady = 2)
		
	
	def create_file_options(self,frame):
		'''
		'''
		self.vertFrame = VerticalScrolledFrame(frame)
		self.vertFrame.grid(sticky = tk.NSEW)
				
		self.vertFrame.grid_columnconfigure(0,weight=1,minsize=200)
		self.vertFrame.grid_columnconfigure(3,weight=1)
		
		lab = tk.Label(self.vertFrame.interior, text = 'Select a folder. Txt files will be shown here ..', bg = MAC_GREY)		
		lab.grid(sticky = tk.EW+tk.W)
		for n,textLabel in enumerate(['File Name','Columns','Columnd\nIndex','Concatenate\nIndex','Ignore','Settings']):
		
			lab = tk.Label(self.vertFrame.interior, text = textLabel, bg = MAC_GREY)
			lab.grid(row=1, column = n, padx=8, sticky = tk.EW)
		
		ttk.Separator(self.vertFrame.interior, orient = tk.HORIZONTAL).grid(row = 1, sticky = tk.EW+tk.S,columnspan=6)
		
		
	def insert_files_in_frame(self,txtFiles):
		'''
		'''		
		self.fileWidgets = dict()
		self.vars = dict()
		
		for n,file in enumerate(txtFiles):
			self.fileWidgets[file] = dict()

			
			
			lab = tk.Label(self.vertFrame.interior, text = '{} - {}'.format(n+1,file), bg = MAC_GREY)
			lab.grid(row = n+2, sticky = tk.EW,pady=2)
			
			columnButton = ttk.Button(self.vertFrame.interior, text = '..', width = 2)
			columnButton.grid(row = n+2, column = 1,pady=2, padx = 10, sticky = tk.W)
			
			
			ent = ttk.Entry(self.vertFrame.interior, width = 6)
			ent.grid(row = n+2, column = 2,pady=2, padx = 10, sticky = tk.W)
			ent.insert(0,'all')
			self.fileWidgets[file]['fileLabel'] = lab
			self.fileWidgets[file]['columnButton'] = columnButton
			self.fileWidgets[file]['columnIndex'] = ent
			
			self.fileWidgets[file]['ignore'] = tk.BooleanVar(value = False)
			cb = ttk.Checkbutton(self.vertFrame.interior, 
								variable = self.fileWidgets[file]['ignore'],
								command = lambda file = file: self.ignore_df(file))
			cb.grid(row = n+2, column = 4, pady = 2, padx = 10)
			ent = ttk.Entry(self.vertFrame.interior,width=2)
			
			ent.grid(row = n + 2, column = 3, padx = 10)
			ent.insert(0,'0')
			
			but = ttk.Button(self.vertFrame.interior, 
							 text = '..', width = 2,
							 command = lambda file=file: self.get_upload_settings(general=False, file = file))
			but.grid(row = n + 2 , column = 5, padx = 10)
			self.fileWidgets[file]['concatIndex'] = ent
			
			ttk.Separator(self.vertFrame.interior, orient = tk.HORIZONTAL).grid(row = n+2, 
				sticky = tk.EW+tk.S,columnspan=6)

			
		
				
	def define_variables(self, csvFilesOnly = False):
		'''
		'''
		self.fileColumns = OrderedDict()
		self.loadSettings = self.adjust_setting_names(
			dict([k,v[0]] for k,v in self.generalLoadSettings.items()))
			
		self.collectDfs = OrderedDict()
		self.fileSpecificSettings = dict()
		self.fileSpecificUserSettings = dict()
		
		
		
	def select_dir(self):
		'''
		'''
		self.define_variables()
		dir = tf.askdirectory()
		if dir is None or dir == '':
			return
		txtFiles = self.get_txt_files(dir)
		if txtFiles is None: return    
		self.insert_files_in_frame(txtFiles) 
		self.get_file_columns(dir,txtFiles)	
		self.dir = dir
     	
     	
     	
	def get_upload_settings(self, general = True, file = None):
		'''
		'''
		
		if file in self.fileSpecificSettings:
			prevSettings = OrderedDict([(k,[v]) for k,v in self.fileSpecificUserSettings[file].items()])
		else:
			prevSettings = self.generalLoadSettings
			
		dialog = simpleUserInputDialog(list(prevSettings.keys()),
							[x[0] for x in prevSettings.values()],
							list(self.generalLoadSettings.values()),
							'Upload Settings',
							'Settings have to be define for all files that should be uploaded.')		
		
		if len(dialog.selectionOutput) == len(comboBoxToGetInputFromUser):
			if general:
			
				self.loadSettings = self.adjust_setting_names(dialog.selectionOutput)
				if len(self.fileSpecificSettings) != 0:
					quest = tk.messagebox.askquestion('Overwrite?',
						'File specific upload paramters were given. Do you want to discard them?',
						parent=self.toplevel)
					if quest == 'yes':
						self.fileSpecificSettings = dict()
			
			elif file is not None and general == False:
				self.fileSpecificUserSettings[file] = dialog.selectionOutput.copy()
				self.fileSpecificSettings[file] = self.adjust_setting_names(dialog.selectionOutput)
			
			self.get_file_columns(self.dir,list(self.fileColumns.keys()))
					     		
	def get_txt_files(self,dir):
		'''
		'''
		files = os.listdir(dir)     
		txtFiles = [file for file in files if file[-3:] =='txt']
		csvFiles = [file for file in files if file[-3:] =='csv']
		
		if len(csvFiles) == 0 and len(txtFiles) != 0:
			pass
		elif len(csvFiles) != 0 and len(txtFiles) != 0:
			tk.messagebox.showinfo('Attention ..',
				'Found csv and txt files in selected dir. Carefully check the upload settings', 
				parent = self.toplevel)
		elif len(csvFiles) != 0 and len(txtFiles) == 0:
			tk.messagebox.showinfo('Attention ..',
				'Csv files detected. Please check the upload parameter.',
				parent=self.toplevel)
		txtFiles = txtFiles + csvFiles
			
		if len(txtFiles) == 0:
			tk.messagebox.showinfo('Error .',
     			'No text files found in directory.',
     			parent = self.toplevel) 
			return
		else:
			return txtFiles     		
 	
	def get_settings(self,file):
 		
 		if file in self.fileSpecificSettings:
				
					settings =  self.fileSpecificSettings[file]
 		else:
					settings = self.loadSettings    		
 		return settings
     	     	
	def get_file_columns(self,dir,txtFiles):
		'''
		Gets column headers of files.
		'''
		for file in txtFiles:
			try:
				settings = self.get_settings(file)
				df = pd.read_table(os.path.join(dir,file),nrows = 1,**settings)
				columns = df.columns.values.tolist()
			except:
				columns = []
			self.fileColumns[file] = columns
			
			self.fileWidgets[file]['columnButton'].configure(command = lambda file=file: self.custom_column_select(file))
		
		
	
	def custom_column_select(self,file):
		'''
		'''
		dialog = simpleListboxSelection('Select columns for upload. Please note that if you want to'+
									   ' upload always the same column index (e.g. at the same position of each file'+
									   ' you can also just enter the index in the main window.',
     								   self.fileColumns[file])
		selection = dialog.selection
		if len(selection) != 0:		
			index = [self.fileColumns[file].index(x)+1 for x in selection]
			if len(index) == 1:
				indexStr = index[0]
			else:
				indexStr = get_elements_from_list_as_string(index,
										addString = '',
										newLine = False, 
										maxStringLength = None)
			self.fileWidgets[file]['columnIndex'].delete(0,tk.END)
			self.fileWidgets[file]['columnIndex'].insert(0,indexStr)
			
			
	def ignore_df(self,file):
		'''
		'''
		for widgetID in ['columnButton','columnIndex','concatIndex']:
			if self.fileWidgets[file]['ignore'].get():
				self.fileWidgets[file][widgetID].configure(state=tk.DISABLED)
			else:
				self.fileWidgets[file][widgetID].configure(state=tk.ACTIVE)
		
		if self.fileWidgets[file]['ignore'].get():		
			
			self.fileWidgets[file]['fileLabel'].configure(fg = 'darkgrey')
		else:
			self.fileWidgets[file]['fileLabel'].configure(fg = 'black')
			


	def upload_files(self):
		'''
		'''
		if hasattr(self,'fileWidgets') == False:
			tk.messagebox.showinfo('Error..','No dir selected yet',parent=self.toplevel)
			return
		pb = Progressbar('Multiple Files')
		txtFiles = list(self.fileColumns.keys())
		dir = self.dir
		concatKeys = [self.fileWidgets[file]['concatIndex'].get() for file in txtFiles]
		concats = OrderedDict([(key,{'dfs':[],'columns':[],'files':[]}) for key in concatKeys])
		
		for n,file in enumerate(txtFiles):
			
			if self.fileWidgets[file]['ignore'].get():
				continue
			else:
				indexStr = self.fileWidgets[file]['columnIndex'].get()
				if indexStr != 'all':
					try:
						indexCols = [int(float(n))-1 for n in indexStr.split(',')]
						columns = [self.fileColumns[file][n] for n in indexCols]
					except:
						columns = self.fileColumns[file]
				else:
					columns = self.fileColumns[file]
				settings = self.get_settings(file) 

				df = pd.read_table(os.path.join(dir,file),usecols = columns, **settings)
				concatKey = self.fileWidgets[file]['concatIndex'].get()
				concats[concatKey]['dfs'].append(df)
				concats[concatKey]['files'].append(file)
				concats[concatKey]['columns'].extend(['{}_{}'.format(col,n) for col in columns])
								
			pb.update_progressbar_and_label(n/len(txtFiles) * 100, 
						'File {} of {} loaded.'.format(n,len(txtFiles)))
		
		for key, dfsAndColumns in concats.items():
			if 	len(dfsAndColumns['dfs']) == 1:
				self.collectDfs[dfsAndColumns['files'][0]] = dfsAndColumns['dfs'][0]
			else:
				dfs = pd.concat(dfsAndColumns['dfs'], ignore_index=True, axis=1)
				dfs.columns = dfsAndColumns['columns']
				self.collectDfs['ConcatFiles_({})'.format(len(dfsAndColumns['files']))] = dfs
		
		pb.close()
		self.close()
		

	def get_results(self):
		'''
		'''
		
		if hasattr(self,'collectDfs') :
			return self.collectDfs, '' ,self.naString
		else:
			return dict(),None,None
				
			
	def adjust_setting_names(self,settingDict):
		'''
		'''	
		self.naString = settingDict['Replace NaN in Object Columns:']
		settingDict = dict([(pandasInstantTranslate[k],v) 
			for k,v in settingDict.items() 
			if k in pandasInstantTranslate])
		if settingDict['sep'] in ['tab','space']:
			if settingDict['sep'] == 'tab':
				settingDict['sep'] = '\t'
			else:
				settingDict['sep'] = '\s+'
		if settingDict['thousands'] == 'None':
			settingDict['thousands'] = None
		settingDict['skiprows'] = int(float(settingDict['skiprows']))
		return settingDict
		
		
class XML2DataFrame:

    def __init__(self, xml_data):
    	#self.root = xml_data
    	
        self.root = ET.XML(xml_data)

    def parse_root(self, root):
        return [self.parse_element(child) for child in iter(root)]

    def parse_element(self, element, parsed=None):
        if parsed is None:
            parsed = dict()
        for key in element.keys():
            parsed[key] = element.attrib.get(key)
        if element.text:
            parsed[element.tag] = element.text
        for child in list(element):
            self.parse_element(child, parsed)
        return parsed

    def process_data(self):
        structure_data = self.parse_root(self.root)
        return pd.DataFrame(structure_data)


class xmlImporter(object):	
	
	def __init__(self,pathUpload):

		self.load_file(pathUpload)	
		
		
		self.replaceObjectNan = '-'	
		self.build_toplevel()
		self.build_widgets()
		self.initiate_preview(self.df)
		self.toplevel.wait_window() 


	def load_file(self, pathUpload):
	
		tree = ET.parse(pathUpload)
		root = tree.getroot()
		xml2df = XML2DataFrame(ET.tostring(root))
		self.df = xml2df.process_data()
				
		
	def close(self,event=None):
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
		popup.bind('<Escape>', self.close) 
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
 		labTitle = tk.Label(self.cont_widgets, text = 'Settings for file upload',
                                      **titleLabelProperties)  
 		labPreview = tk.Label(self.cont_widgets, text = 'Preview', **titleLabelProperties)
 		     		
 		buttonClose = ttk.Button(self.cont_widgets, text = "Close", command = self.discard_changes, width=9)
 		buttonLoad = ttk.Button(self.cont_widgets, text = "Load", width=9, command  = self.save_changes)
 		self.toplevel.bind('<Return>', self.save_changes)
 		
 		labTitle.grid(padx=5,pady=5, columnspan=7, sticky=tk.W) 
 		labPreview.grid(padx=5,pady=5, row=8, column=0, sticky = tk.W) 
 		
 		buttonClose.grid(padx=3, row=6,column=5, pady=3, sticky=tk.E) 
 		buttonLoad.grid(padx=3, row=5, column=5, pady=3, sticky=tk.E)
 		 		 		
 			
	def create_preview_container(self,sheet = None):
		'''
		Creates preview container for pandastable. Mainly to delete everything easily and fast.
		'''
		self.cont_preview  = tk.Frame(self.cont,background='white') 
		self.cont_preview.pack(expand=True,fill=tk.BOTH)
	
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
	
	def save_changes(self, event = None):
		'''
		Defines self.data_to_export to set the data to be exported of the importer class
		'''
		columnsToUpload = self.pt.model.df.columns
		#print(columnsToUpload)
		
		self.data_to_export = self.pt.model.df[columnsToUpload]
		if self.data_to_export is None:
			return
		else:
			self.pt.parentframe.destroy()												
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
         
		