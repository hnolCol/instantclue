
####
#### Class that handles correlation of rows to data
####

import numpy as np
import pandas as pd 
from itertools import chain 

import os

import tkinter as tk
import tkinter.filedialog as tf

from tkinter import ttk
from scipy.stats import spearmanr
from scipy.stats import pearsonr

MAC_GREY = '#ededed'
LARGE_FONT = ("Helvetica", 13, "bold")

coefficients = ['Spearman','Pearson']


class correlate_rows_to(object):


	def __init__(self,platform,data,columns):
		self.platform = platform
		self.cont = None
		self.toplevel = None
		self.coefficient = tk.StringVar()
		self.coefficient.set('Pearson')
		self.column = columns
		self.n_cols = len(columns) 
		self.data = data
		
		self.build_toplevel() 
		self.build_widgets() 
		
		self.toplevel.wait_window()
		
	def build_toplevel(self):
	
		self.toplevel = tk.Toplevel() 
		self.toplevel.wm_title('Correlations ...')
		if self.platform != 'MAC':
			self.toplevel.attributes('-topmost', True)
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close_toplevel)
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(expand=True, fill='both')
		#cont.grid_columnconfigure(5,weight=1)
		self.cont = cont
		
		
	def build_widgets(self):
		lab1 = tk.Label(self.cont, text = 'Calculation of correlation coefficients', font = LARGE_FONT, fg="#4C626F", justify=tk.LEFT, bg = MAC_GREY)	
		lab1.grid(padx=5,pady=15, columnspan=6 ,sticky=tk.W)
		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(column=0,columnspan=6,padx=1,sticky=tk.EW,pady=(0,4)) 
		lab2 = tk.Label(self.cont, text='Upload a file containing values that should be used for correlations.\nNote that each row of the selected columns will be correlated against your input.\nThe number of rows/columns determines on which axis the correlation will be calculated.',justify=tk.LEFT,bg=MAC_GREY)
		combo_coeff = ttk.Combobox(self.cont, textvariable=self.coefficient , values = coefficients)

		self.corr_but = ttk.Button(self.cont, text = 'Correlate', command = self.calculate_correlation, state=tk.DISABLED)
		close_but = ttk.Button(self.cont, text = 'Close', command = self.close_toplevel)
		upload_but = ttk.Button(self.cont, text = 'Load file', command = self.upload_and_evaluate_file)
		lab2.grid(columnspan=5)		
		combo_coeff.grid(columnspan=5, padx=2,pady=2)
		upload_but.grid(row=5, column=0,padx=2,pady=2) 
		
		self.corr_but.grid(row=5, column=1,padx=2,pady=2)
		close_but.grid(row=5, column=2,padx=2,pady=2) 
		
	def close_toplevel(self):
		self.toplevel.destroy() 
		
		
	def determine_correlation(self,row,axis,coeff):
		collect_coeff = []
		if coeff == 'Pearson':
			for i in range(len(self.corr_data.columns)):
				try:
					corr_ = pearsonr(row,self.corr_data.ix[:,i])
				except:
					corr_ = (np.nan,np.nan)
				collect_coeff.extend(corr_)
		
		elif coeff == 'Spearman':
			for i in range(len(self.corr_data.columns)):
				try:
					corr_ = spearmanr(row,self.corr_data.ix[:,i],axis=0)
				except:
					corr_ = (np.nan, np.nan)
				collect_coeff.extend(corr_)

		return collect_coeff
				
		
	def calculate_correlation(self, axis=0):
		
		dat_ = pd.DataFrame() 
		dat_['Collect'] = self.data.apply(lambda row, axis=axis, coeff = self.coefficient.get(): self.determine_correlation(row,axis,coeff), axis=1) 
		name_ = []
		coeff_ = self.coefficient.get()
		for column in self.column:
			n_ = str(column)+'_'+coeff_[0:5]+'_Coeff_'+self.file_name
			n_1 = str(column)+'_'+coeff_[0:5]+'_pVal_'+self.file_name
			name_.extend((n_,n_1))
		
		self.results = pd.DataFrame(dat_['Collect'].values.tolist(), columns = name_, index = self.data.index)
		#print(dat_) 
		
		#self.results = None
		
		self.toplevel.destroy()  
		
                                                         	
		
	
	def get_correlations(self):
	
		return self.results 
		
	def upload_and_evaluate_file(self):
		path_upload = tf.askopenfilename(title="Choose File")
		self.file_name = path_upload.split('/')[-1]
		self.corr_data = pd.read_table(path_upload, low_memory = False)
		
		self.shape_data = self.corr_data.shape
		if self.shape_data[0] == self.n_cols and self.shape_data[1] == self.n_cols:
			quest = tk.messagebox.askquestion('Cols and rows fit', 'InstantClue has detected that correlation can be performed on rows and columns.\nThis is beacuase rows and columns do have the same length as the number of selected columns.\nPlease click YES if you want to correlate each row of your input data with each row of the selected column, or NO if you like to to correlate each row of the selected columns to each column in the uploaded file.')
			if quest == 'yes':
				pass
			else:
				pass ###here we should perform transpose
		if self.shape_data[0] == self.n_cols:
			pass 
			
		elif self.shape_data[1] == self.n_cols:
			self.corr_data = self.corr_data.transpose()
			self.column = self.corr_data.columns
			self.n_cols = len(self.column)
		else:
			tk.messagebox.showinfo('Error..','InstantClue failed to align rows and columns of uploaded file to the number of selected columns')
			return
		self.corr_but.configure(state=tk.ACTIVE)		
		
		
		
		
		#print(self.collection_list_colors)      
	
	