import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import pandas as pd
import numpy as np

from modules.utils import *
from modules.dialogs.VerticalScrolledFrame import VerticalScrolledFrame


class transformDimRedDialog(object):
	
	def __init__(self, dimRedCollection, dfClass, dataTreeview, numericColumns = []):
	
		self.dimRedModelResults = dimRedCollection.dimRedResults
		self.dfClass = dfClass 
		self.dataTreeview = dataTreeview 
		
		if len(numericColumns) == 0:
			self.numericColumns  = self.dataTreeview.columnsSelected
		else:
			self.numericColumns = numericColumns
		

		if len(self.dimRedModelResults) == 0:
			
			tk.messagebox.showinfo('Error ..','Could not find any performed dimensional reduction.')
			return
		
		self.build_popup()
		self.add_widgets_to_toplevel() 
		
		
		
	def close(self):
		'''
		Closes the toplevel.
		'''
		
		self.toplevel.destroy()
         
         			
	def build_popup(self):
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Apply Dimensional Reduction') 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w = 525
		h= 420
		self.toplevel = popup
		self.center_popup((w,h))			

	def add_widgets_to_toplevel(self):
		'''
		'''
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		self.cont.grid_columnconfigure(1,weight=1)
		self.cont.grid_rowconfigure(5,weight=1)
		
		self.contClust = tk.Frame(self.cont,background=MAC_GREY)
		self.contClust.grid(row=5,column=0,columnspan = 4, sticky= tk.NSEW)
		self.contClust.grid_columnconfigure(1,weight=1)	
		
		labelTitle = tk.Label(self.cont, text = 'Apply dimensional reduction', 
                                     **titleLabelProperties)
                                     
		labelInfo = tk.Label(self.cont, text = 'Select dimensional reduction model'+
											   '\nThe result will be additional columns in the \ndata treeview per selected model.',
											   justify=tk.LEFT, bg=MAC_GREY)
											   
		labelWarning = tk.Label(self.cont, text = 'Warning: Only if the feature/column names are\.'+
			'exactly the same as they were when the model was constructed the order does not matter.\nOtherwise'+
			' only(!) the order of features matters and names will be neglected.',wraplength = 520,**titleLabelProperties)
                                     
                                     
		self.create_widgets_for_dimRed() 
		
		applyButton = ttk.Button(self.cont, text = 'Apply', command = self.perform_transformation)
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		labelTitle.grid(row=0,padx=5,sticky=tk.W,pady=5, columnspan=3)
		labelInfo.grid(row=1,padx=5,sticky=tk.W,pady=5, columnspan=4)
		labelWarning.grid(row=2,padx=5,sticky=tk.W,pady=5, columnspan=4)
		ttk.Separator(self.cont, orient = tk.HORIZONTAL).grid(row=3,columnspan=4,sticky=tk.EW, padx=1,pady=3)
		
		ttk.Separator(self.cont, orient = tk.HORIZONTAL).grid(row=6,columnspan=4,sticky=tk.EW, padx=1,pady=3)
		applyButton.grid(row=7,column=0,padx=3,pady=5)
		closeButton.grid(row=7,column=3,padx=3,pady=5, sticky=tk.E)
        

	def create_widgets_for_dimRed(self):
		'''
		'''
		self.dimRed_cbs_var = dict() 
		vertFrame = VerticalScrolledFrame(self.contClust)
		vertFrame.pack(expand=True,fill=tk.BOTH)
		for id,props in self.dimRedModelResults.items():
		
			varCb = tk.BooleanVar(value = False) 
			textInfo = self.dimRedModelResults[id]['data']['Predictor'].get_params()
			columnsInDimRed = self.dimRedModelResults[id]['numericColumns']
			modelName = self.dimRedModelResults[id]['name']
			cb = ttk.Checkbutton(vertFrame.interior, 
				text = modelName, variable = varCb) 
			self.dimRed_cbs_var[id] = varCb			
			
			if len(columnsInDimRed) != len(self.numericColumns):
				cb.configure(state=tk.DISABLED) 
				title_ = ' == Number of selected columns/features does NOT match the\nnumber of columns/features used in model creation! == '
			else:		
				title_ =  'Dimensional Reduction Model\nMethod: {}\nColumns: {}'.format(self.dimRedModelResults[id]['method'],get_elements_from_list_as_string(columnsInDimRed))
			
			CreateToolTip(cb, text = str(textInfo).replace("'",''), title_ = title_)
			cb.grid(sticky=tk.W, column=0,padx=3,pady=3)
				
		vertFrame.grid_rowconfigure(len(self.dimRedModelResults)+1,weight=1)
    	
	def perform_transformation(self):
		'''
		'''
		dataToPredict = self.dfClass.df[self.numericColumns].dropna()
		columnsToAdd = []
		for key, variable in self.dimRed_cbs_var.items():
			if variable.get():
				resortedColumns = self.resort_columns_if_same_name(self.dimRedModelResults[key]['numericColumns'])
				
				if resortedColumns is not None:
					dataToPredict = dataToPredict[resortedColumns]
				
				transformedDf = self.dimRedModelResults[key]['data']['Predictor'].transform(dataToPredict.as_matrix())

				modelID = self.dimRedModelResults[key]['name'].split('_')[-1]
				nComps = self.dimRedModelResults[key]['data']['Predictor'].n_components_
				columnNames = ['Id_{}_Comp_{}'.format(modelID,i+1) for i in range(nComps)]
				
				columnNamesEval = [self.dfClass.evaluate_column_name(column)\
				for column in columnNames]
				
				outputDf = pd.DataFrame(transformedDf, 
					index = dataToPredict.index, 
					columns = columnNamesEval)
				columnsToAdd.extend(columnNamesEval)
							
				self.dfClass.join_df_to_currently_selected_df(outputDf)
				del outputDf
		
		self.dataTreeview.add_list_of_columns_to_treeview(self.dfClass.currentDataFile, 
														 'float64', columnsToAdd)		
				
		tk.messagebox.showinfo('Done ..','Dimensional reduction applied. Columns were added.')
		del dataToPredict
    		
	
	def resort_columns_if_same_name(self,predictorColumns):
		'''
		'''
		if all(column in predictorColumns for column in self.numericColumns):		
			indices = [predictorColumns.index(column) for column in self.numericColumns]
			
			numericColumnsSorted = [y for x,y in zip(indices,self.numericColumns)]
			return numericColumnsSorted
		else:
			return None
			
    		

    	                                 		
	def center_popup(self,size):

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))     		
		
	
