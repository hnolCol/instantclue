import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from modules.pandastable import core 
from modules.dialogs.VerticalScrolledFrame import  VerticalScrolledFrame

from modules.utils import * 



class anovaResultDialog(object):


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
		w=790
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
			
