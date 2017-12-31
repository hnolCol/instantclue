import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
from collections import OrderedDict
from modules.utils import *


class ColumnNameConfigurationPopup(object):
	
	
	def __init__(self,columns,dfClass,sourceTreeView):
						
		
		self.columns = columns 
		self.dfClass = dfClass
		self.renamed = True
		
		self.dataTreeview = sourceTreeView
		self.iidList = sourceTreeView.columnsIidSelected
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
		popup.wm_title('Change column name') 
         
		popup.protocol("WM_DELETE_WINDOW", self.discard_changes)
		w=600
		h=100+int(len(self.columns))*33 ##emperically
		self.toplevel = popup
		self.center_popup((w,h))
		
			
	def build_widgets(self):
			
             '''
             Building needed tkinter widgets 
             '''
             
             cont = tk.Frame(self.toplevel,background = MAC_GREY)
             cont.pack(fill='both', expand=True)
             cont.grid_columnconfigure(1,weight=1,minsize=130)
             cont.grid_columnconfigure(0,weight=1,minsize=230)
			
             labelTitle = tk.Label(cont, text = 'Change column name', font = LARGE_FONT, fg="#4C626F", justify=tk.LEFT, bg = MAC_GREY)
             labelTitle.grid(pady=5,padx=5,sticky=tk.W,columnspan=2)
             
             entryDict = OrderedDict()
             
             for n,columnName in enumerate(self.columns): 
             	columnLabel = tk.Label(cont, text = columnName, bg=MAC_GREY)
             	newColumnEntry = ttk.Entry(cont,width=400)
             	
             	newColumnEntry.insert('end',columnName)
             	
             	entryDict[columnName] = newColumnEntry
             	
             	columnLabel.grid(row=n+2,column=0,sticky=tk.E,padx=5, pady=3)
             	newColumnEntry.grid(row=n+2, column=1, padx=5, pady=3, columnspan=5, sticky=tk.EW)
             
             
             renameButton = ttk.Button(cont, text = 'Rename',
             						   command = lambda: self.rename_columns_in_df(entryDict))  
             closeButton = ttk.Button(cont, text = 'Close', command = self.discard_changes)
             
             closeButton.grid(row=n+3, column = 1, padx=5, pady=3, sticky=tk.E) 
             renameButton.grid(row = n+3, column = 0, padx=5, pady=3, sticky=tk.W, columnspan=2)
	
	
	def discard_changes(self):
		'''
		'''
		self.renamed = False 
		self.close()            	
             	
	def rename_columns_in_df(self,entryDict):
		'''
		'''
		renameDict = OrderedDict() 
		columnNamesToChange = []
		iidList = []
		n = 0
		for oldName, entry in entryDict.items():
			entryText = entry.get()
			if entryText != oldName:
				columnNamesToChange.append(oldName)
				renameDict[oldName] = entryText
				iidList.append(self.iidList[n])
				n+=1
		columnNotToChange = [col for col in self.dfClass.df_columns if col not in columnNamesToChange]
		for oldName,newName in renameDict.items():
			newNameEval = self.dfClass.evaluate_column_name(newName,columnNotToChange,useExact=True)
			columnNotToChange.append(newNameEval)
			renameDict[oldName] = newNameEval
			
		self.dfClass.rename_columnNames_in_current_data(renameDict)
		self.dataTreeview.rename_itemText_by_iidList(iidList, list(renameDict.values()))
		self.close()
		
	def center_popup(self,size):
         	'''
         	Casts the popup in center of screen
         	'''

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))  	
	
	