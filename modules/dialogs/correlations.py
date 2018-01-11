
import numpy as np
import pandas as pd 
from itertools import chain 

import os

import tkinter as tk
import tkinter.filedialog as tf

from tkinter import ttk
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from modules.utils import *
from modules.dialogs.import_subset_of_data import importDataFromDf
from modules.dialogs.simple_dialog import simpleUserInputDialog
coefficients = ['Spearman','Pearson']


class correlationDialog(object):


	def __init__(self,dfClass,selectedColumns):
		self.cont = None
		
		self.coefficient = tk.StringVar(value='Pearson')
		self.correlationColumns = []
		self.selectedColumns = selectedColumns
		self.n_cols = len(self.selectedColumns) 
		self.dfClass = dfClass
		self.data = self.dfClass.get_current_data_by_column_list(self.selectedColumns)
		self.results = pd.DataFrame()
		self.build_toplevel() 
		self.build_widgets() 
		
		self.toplevel.wait_window()
		
	def build_toplevel(self):
	
		self.toplevel = tk.Toplevel() 
		self.toplevel.wm_title('Correlations ...')
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close_toplevel)
		cont = tk.Frame(self.toplevel, background = MAC_GREY)
		cont.pack(expand=True, fill='both')
		#cont.grid_columnconfigure(5,weight=1)
		self.cont = cont
		
		
	def build_widgets(self):
		lab1 = tk.Label(self.cont, text = 'Calculation of correlation coefficients', **titleLabelProperties)	
		lab1.grid(padx=5,pady=15, columnspan=6 ,sticky=tk.W)
		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(column=0,columnspan=6,padx=1,sticky=tk.EW,pady=(0,4)) 
		lab2 = tk.Label(self.cont, text='Each row of the selected columns will be correlated to your input.'+
							'\nThe number of rows/columns determines on which axis the correlation will be calculated.',
							justify=tk.LEFT,bg=MAC_GREY)
		combo_coeff = ttk.Combobox(self.cont, textvariable=self.coefficient , values = coefficients)

		self.corrButton = ttk.Button(self.cont, text = 'Correlate', command = self.calculate_correlation, state=tk.DISABLED)
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close_toplevel)
		upload_but = ttk.Button(self.cont, text = 'Select data', command = self.import_data)
		lab2.grid(columnspan=5)		
		combo_coeff.grid(columnspan=5, padx=2,pady=2)
		upload_but.grid(row=5, column=0,padx=2,pady=2) 
		
		self.corrButton.grid(row=5, column=1,padx=2,pady=2)
		closeButton.grid(row=5, column=2,padx=2,pady=2) 
		
	def close_toplevel(self):
		'''
		Closes the dialog window.
		'''
		self.toplevel.destroy() 
		
		
	def determine_correlation(self,row,axis,coeff):
		collect_coeff = []
		if coeff == 'Pearson':
			for column in self.corr_data.columns:
				try:
					corrResults = pearsonr(row,self.corr_data[column])
				except:
					corrResults  = (np.nan,np.nan)
				collect_coeff.extend(corrResults )
		
		elif coeff == 'Spearman':
			for column in self.corr_data.columns:
				try:
					corrResults  = spearmanr(row,self.corr_data[column],axis=0)
				except:
					corrResults  = (np.nan, np.nan)
				collect_coeff.extend(corrResults )

		return collect_coeff
				
		
	def calculate_correlation(self, axis=0):
		
		resultData = pd.DataFrame() 
		resultData['Collect'] = self.data.apply(lambda row, axis=axis, coeff = self.coefficient.get(): self.determine_correlation(row,axis,coeff), axis=1) 
		resultDfNames = []
		roundDict = {}
		coeff_ = self.coefficient.get()
		
		for column in self.correlationColumns:
			n_ = str(column)+'_'+coeff_[0:5]+'_Coeff'
			n_1 = str(column)+'_'+coeff_[0:5]+'_pVal'
			roundDict[n_] = 2
			roundDict[n_1] = 5
			resultDfNames.extend((n_,n_1))
		
		results = pd.DataFrame(resultData['Collect'].values.tolist(), 
					columns = resultDfNames, 
					index = self.data.index)
		## round the data
		results = results.round(roundDict)
		# evaluate column names to prevent different names in source data treeview and source data df
		newColumnNames = [self.dfClass.evaluate_column_name(columnName) for columnName in results.columns]
		results.columns = newColumnNames
		self.results = results
		self.close_toplevel()
				
	
	def get_correlations(self):
	
		return self.results 

	
	def import_data(self):
		'''
		'''
		importer = importDataFromDf(self.dfClass,
 						title = 'Select data from preview as your x values.'+
 						' They must match either in row- or column-number the'+
 						' selected numeric columns: {}.'.format(self.n_cols),
 						requiredDataPoints = self.n_cols, allowMultSelection = True)
 		## get selected data				
		selectionData = importer.get_data_selection()
		## ensure that df Class has correct data selected (if user change it to select data) 
		#self.dfClass.set_current_data_by_id(self.dfID)
		#print(selectionData)
		if selectionData is None:
			del importer
			return
		else:
			shape = selectionData.shape
			if shape[0] == self.n_cols and shape[1] == self.n_cols:
				askAxis = simpleUserInputDialog(['observations are in'],
					['rows'],['rows','columns'],
					title = 'Select axis.',
					infoText = 'Data selection shape matches in rows and columns. Please select where the observations are stored.')
				results = askAxis.selectionOutput 
				
				if len(results) == 0:
					tk.messagebox.showinfo('Error..','No axis selected.',parent=self)
				elif results['observations are in'] == 'rows':
					self.corr_data = selectionData
				elif results['observations are in'] == 'columns':
					self.corr_data = selectionData.transpose()
			elif shape[1] == self.n_cols:
				self.corr_data = selectionData.transpose()
			elif shape[0] == self.n_cols:
				self.corr_data = selectionData
			self.correlationColumns = self.corr_data.columns.values.tolist()		
			self.corrButton.configure(state=tk.ACTIVE)
   
	
	