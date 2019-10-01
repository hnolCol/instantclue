import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk  
import tkinter.font as tkFont
import tkinter.simpledialog as ts

import matplotlib.pyplot as plt

from modules.pandastable import core 
from modules.dialogs.VerticalScrolledFrame import  VerticalScrolledFrame


from modules import stats
from modules.pandastable import core
from modules.utils import *


from collections import OrderedDict



class storeAnovaResultsClass(object):
	'''
	'''
	def __init__(self):
	
		self.id = 0
		self.anovaResults = OrderedDict()
	
	
	def save_new_anova_result(self,props,resultDf):
		'''
		'''
		self.id += 1
		
		self.anovaResults[self.id] = {'props':props,
									  'results':resultDf}
		
	def get_anova_results(self):
		'''
		'''
		return self.anovaResults
		
	

class anovaCalculationDialog(object):
	'''
	Class to facilitate Anova calculations. In principal we could 
	calculate any anova model. 
	'''
	
	def __init__(self,anovaType,dependentVariable,
					factors,dfClass,anovaTestsCollection):
		'''
		'''
		
		self.anovaType = anovaType
		self.factorsVariables = OrderedDict()
		
		self.subjectColumns = dfClass.dfsDataTypesAndColumnNames[dfClass.currentDataFile]['int64']
		
		self.categoricalColumns = dfClass.dfsDataTypesAndColumnNames[dfClass.currentDataFile]['object'] + \
		self.subjectColumns
							 
		self.numericalColumns = dfClass.dfsDataTypesAndColumnNames[dfClass.currentDataFile]['float64']
							 
		self.factorsSubmitted = factors
		self.dependentVariable = tk.StringVar(value = dependentVariable)
		self.subjectColumn = tk.StringVar(value = 'Infer')
		
		
		self.dfClass = dfClass
		self.anovaTestCollection = anovaTestsCollection
		
		self.build_toplevel()
		self.build_widgets()
		self.add_selected_factors_to_combo()
		
	
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
		popup.wm_title('Calculate Analysis of Variance') 
		popup.grab_set() 
        
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=420
		h=400
		self.toplevel = popup
		self.center_popup((w,h))
		
			
	def build_widgets(self):
 		
 		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont.grid_columnconfigure(0,weight=1)
 		self.cont.grid_rowconfigure(14,weight=1)
 		
 		
 		labelTitle = tk.Label(self.cont, text= 'Analysis of Variance', 
 								**titleLabelProperties)
 		labelTitle.grid(sticky=tk.W, columnspan=2)
 		
 		ttk.Separator(self.cont, orient =  tk.VERTICAL).grid(sticky=tk.EW,pady=3,padx=2,columnspan=3)

 		depVarLabel = tk.Label(self.cont, text = 'Dependent Variable', 
 								**titleLabelProperties)
 		depVarLabel.grid(sticky=tk.W, columnspan=2,padx=3,pady=5)
 		
 		varLabel = tk.Label(self.cont, text = 'Numeric Column:', bg =MAC_GREY)
 		varLabel.grid(row=3,padx=5,pady=3,sticky=tk.E)
 		
 		comboBox = ttk.Combobox(self.cont, values = self.numericalColumns,
 								textvariable = self.dependentVariable)
 		comboBox.grid(row=3,padx=5,pady=3,sticky=tk.E,column=1)
 		
 		betweenLabel = tk.Label(self.cont, text = 'Between Subject Factors', 
 								**titleLabelProperties)
 		ttk.Separator(self.cont, orient =  tk.VERTICAL).grid(sticky=tk.EW,pady=3,padx=2,columnspan=3)
 		betweenLabel.grid(sticky=tk.W, columnspan=2,padx=3,pady=5)
 		if self.anovaType == '1W-ANVOVA-RepMeas':
 			numOfWithinFactors = 1
 		elif 'RepMeas' in self.anovaType:
 		## check if we have 
 			numOfWithinFactors = int(float(self.anovaType[-5]))
 		else:
 			numOfWithinFactors = 0
 			
 		numOfFactors = int(float(self.anovaType[0]))
 		betweenFactors = numOfFactors - numOfWithinFactors
 		if betweenFactors > 0:
 			betweenFrame = tk.Frame(self.cont, bg = MAC_GREY)
 			betweenFrame.grid(stick=tk.E, columnspan=2)
 			for n in range(1,betweenFactors+1):
 				text = 'Factor {}:'.format(n)
 				var = tk.StringVar()
 				facLabel = tk.Label(betweenFrame, text = text , bg=MAC_GREY)
 				facLabel.grid(row=n,padx=5,pady=3,sticky=tk.E)
 				comboBox = ttk.Combobox(betweenFrame, textvariable=var,
 										values = self.categoricalColumns)
 				comboBox.grid(row=n,padx=5,pady=3,sticky=tk.E,column=1)
 				self.factorsVariables['between_'+text] = var

 		if numOfWithinFactors > 0:

 			withinFrame = tk.Frame(self.cont, bg = MAC_GREY)
 			ttk.Separator(self.cont, orient =  tk.VERTICAL).grid(sticky=tk.EW,pady=3,padx=2,columnspan=3)
 			withinLabel = tk.Label(self.cont, text = 'Within Subject Factors',
 				**titleLabelProperties) 
 			withinLabel.grid(sticky=tk.W, columnspan=2,padx=3,pady=5)
 				 
 			withinFrame.grid(stick=tk.E, columnspan=2)
 			for n in range(1,numOfWithinFactors+1):
 				text = 'Factor {}:'.format(n)
 				var = tk.StringVar()
 				facLabel = tk.Label(withinFrame, text = text , bg=MAC_GREY)
 				facLabel.grid(row=n,padx=5,pady=3,sticky=tk.E)
 				comboBox = ttk.Combobox(withinFrame, textvariable = var,
 								values = self.categoricalColumns, exportselection = 0)
 				comboBox.grid(row=n,padx=5,pady=3,sticky=tk.E,column=1)
 				self.factorsVariables['within_'+text] = var
 				
 		subjectTitle = tk.Label(self.cont, text = 'Subject Identification',
        							 **titleLabelProperties)     
 		ttk.Separator(self.cont, orient =  tk.VERTICAL).grid(sticky=tk.EW,pady=3,padx=2,columnspan=3)
 		subjectTitle.grid(sticky=tk.W, columnspan=2,padx=3,pady=5)
 		
 		subjectLabel = tk.Label(self.cont, text = 'Subject column:',
 											bg = MAC_GREY)
 											
 		subjectCombo = ttk.Combobox(self.cont, textvariable = self.subjectColumn, 
 														values = self.subjectColumns) 
 		subjectLabel.grid(row=13,padx=5,pady=3,sticky=tk.E)
 		subjectCombo.grid(row=13,padx=5,pady=3,sticky=tk.E,column=1)
 		 				        
 		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
 		applyButton = ttk.Button(self.cont, text = 'Calculate', command = self.perform_calculations)
 		
 		applyButton.grid(row=14, column=0,padx=10,pady=5, sticky=tk.W)
 		closeButton.grid(row=14, column=1,padx=10,pady=5, sticky=tk.E)
     
     
	def perform_calculations(self):
		'''
		'''
		
		dataFrame = self.dfClass.df.copy()
		bFactors, wFactors = [], []
		
		depVar = self.dependentVariable.get()
		subj = self.subjectColumn.get()
		
		for key,value in self.factorsVariables.items():
			factorID = key.split('_')[0]
			if factorID == 'between':
				bFactors.append(value.get())
			elif factorID == 'within':
				wFactors.append(value.get())
		
		if len(wFactors) != 0 and subj == 'Infer':
			tk.messagebox.showinfo('Error ..',
								   'Can not infer subject columns when using within subject factor designs.')	
			return
		
		dfColumns = wFactors+bFactors+[depVar]
		if all(column in self.dfClass.df_columns for column in dfColumns):
			if subj in self.dfClass.df_columns:
				dfColumns = dfColumns + [subj]
			
			df = self.dfClass.get_current_data_by_column_list(dfColumns)
			
			props = {'wFactors':wFactors,'bFactors':bFactors,
					'subjectColumn':subj,'dependentVariable':depVar,
					}
			try:
				anovaResult, title = stats.calculate_anova(df, **props)
				props['title'] = title
			
			except Exception as e:
			
				tk.messagebox.showinfo('Error ..','There was an error while calculating anova results. Message '+ str(e) +
										'Most common reasons: a) A combination of factors is missing. b) Subject columns'
										' incorrectly asigned.')
										
				return
				
			self.anovaTestCollection.save_new_anova_result(props,anovaResult)
			
			tk.messagebox.showinfo('Done ..','Calculations performed. Cleaning up ..')
			self.close()
		else:
			tk.messagebox.showinfo('Error ..','Not all given columns are found in data frame.')
				
	def add_selected_factors_to_combo(self):
		'''
		'''
		factorsByDesign = list(self.factorsVariables.keys())
		for n,factor in enumerate(self.factorsSubmitted):
			var = self.factorsVariables[factorsByDesign[n]]
			var.set(factor)
			
		        				
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 
	


class anovaResultDialog(object):
	'''
	Class to display anova results.
	'''

	def __init__(self, anovaTestCollection):
	
		
		self.pandastables = [] 
		
		
		id = anovaTestCollection.id
		self.testProps = anovaTestCollection.anovaResults[id]
		self.selectedTest = tk.StringVar(value = id)
		
		self.collection = anovaTestCollection
		self.build_toplevel() 
		self.build_widgets()
		self.display_results(self.testProps)
		
		self.toplevel.wait_window()

		
	def close(self):
		'''
		Close toplevel
		'''
		for pTable in self.pandastables:
			del pTable
			
		self.toplevel.destroy() 
		
			
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Anova Results')
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=990
		h=680
		self.toplevel = popup		
		self.center_popup((w,h))
		
	def build_widgets(self):
		'''
		'''
		cont = tk.Frame(self.toplevel, bg=MAC_GREY)
		cont.pack(expand=True, fill=tk.BOTH) 
		cont.grid_columnconfigure(0,weight=1)
		cont.grid_rowconfigure(8,weight=1)
		
		self.header = tk.Frame(cont, bg=MAC_GREY) 
		self.header.pack(anchor = tk.W)
		labelTitle = tk.Label(self.header, text= 'Anova results.', 
                                     **titleLabelProperties)		
		
		labelTitle.grid(sticky=tk.W,padx=3,pady=5)
		labelSelect = tk.Label(self.header, text = 'Select Test: ',
											**titleLabelProperties)
		labelSelect.grid(sticky=tk.W,padx=3,pady=5)
		
		comboBoxTests = ttk.Combobox(self.header, textvariable = self.selectedTest,
									 values = list(self.collection.anovaResults.keys()),
									 width = 8)
									 
		comboBoxTests.grid(row=1,sticky=tk.W,padx=(90,0),pady=5)
		comboBoxTests.bind('<<ComboboxSelected>>', self.update_results)
		
		propsButton = ttk.Button(self.header, text = 'Props')#, command = self.show_settings)
		propsButton.grid(row=1,sticky=tk.W,padx= (235,0), pady=5)
		
		txt = self.prepare_prop_string()
		self.toolTip = CreateToolTip(propsButton, title_ = 'Anova Properties', text = txt)
		
		self.resultsCont =  tk.Frame(cont, bg=MAC_GREY)
		self.resultsCont.pack(expand = True , fill = 'x', anchor = tk.N + tk.E)
		self.resultsCont.grid_columnconfigure(1,weight=1)
		
		self.footer = tk.Frame(cont, bg=MAC_GREY)
		self.footer.pack()
		
		labelPlot = tk.Label(self.footer, text = 'Plot - Interactions', bg = MAC_GREY)
		
		plotButton = ttk.Button(self.footer, text = 'Plot', command = '')
		closeButton = ttk.Button(self.footer, text  = 'Close', command = self.close)
		
		
		#ttk.Separator(self.footer, orient = tk.HORIZONTAL).grid(sticky=tk.EW,columnspan = 3)
		#labelPlot.grid()
		#plotButton.grid()
		
		
		closeButton.grid()

	def update_results(self,event):
		'''
		'''
		
		id = int(float(self.selectedTest.get()))
		if id in self.collection.anovaResults:
			self.testProps = self.collection.anovaResults[id]
			## TO DO - instead of redrawing the table again and again, rather substitute only the data
			self.clean_result_frame_up()
			self.display_results(self.testProps)
			txt = self.prepare_prop_string()
			self.toolTip.text = txt
			
	def prepare_prop_string(self):
		'''
		'''
		props = self.testProps['props']
		
		propString = 'Dependent Variable : {}\
		 \nWithin Subjects: {}\nBetween Subjects:\
		  {}\nSubject Column: {}'.format(props['dependentVariable'],
		 						  get_elements_from_list_as_string(props['wFactors']),
		 						  get_elements_from_list_as_string(props['bFactors']),
		 						  props['subjectColumn'])
		
		return propString
		

	def clean_result_frame_up(self):
		'''
		'''
		
		for widget in self.resultsCont.winfo_children():
			widget.destroy()
					
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 
         	
         	
         	 
	def display_results(self,results):
		'''
		Display Anova Results
		'''
		dataCollection = results['results']
		titles = results['props']['title']
		if isinstance(dataCollection,tuple):
			pass
		else:
			dataCollection = (dataCollection,)
			titles = (titles,)
		for dataFrame,title in zip(dataCollection,titles):
			lab = tk.Label(self.resultsCont, text = title, **titleLabelProperties)
			lab.grid(padx=3, pady=2)			
			contTable = tk.Frame(self.resultsCont, bg = 'white')
			contTable.grid(sticky=tk.EW, pady=(3,15), padx=10, columnspan=2)

			table = core.Table(contTable, dataframe = dataFrame, showtoolbar=False, showstatusbar=False)
			table.show() 
			self.pandastables.append(table)
	
	
		
		
		
	
	
	
	
	
	
	
	
	
	