import tkinter as tk
from tkinter import ttk  
import pandas as pd


from modules.utils import *



class simpleUserInputDialog(object):

	def __init__(self, descriptionValues,initialValueList, optionList, title, infoText):
		
		self.initialValueList = initialValueList
		self.descriptionValues = descriptionValues
		
		if isinstance(optionList[0],list):
			self.optionList = optionList
		else:
			self.optionList = [optionList]
		
		if len(descriptionValues) > len(self.initialValueList):
			for n in range(len(descriptionValues) - len(self.initialValueList)):
				self.initialValueList.append('Please Select')
		
		
		self.title = title
		self.infoText = infoText
		
		self.output = {}
		self.selectionOutput = {}
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
		popup.wm_title(self.title) 
		popup.grab_set() 
        
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=390
		h=130 + 45*len(self.descriptionValues)
		# this could be done better in a scollable frame 
		# but usually not so many values are being asked from the user
		self.toplevel = popup
		self.center_popup((w,h))
	
	def build_widgets(self):

 		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont.grid_columnconfigure(1,weight=1, minsize=250)
 		self.cont.grid_columnconfigure(0,weight=1, minsize=100)
 		 
 		
 		labelTitle = tk.Label(self.cont, text= self.infoText, 
                                     wraplength=320,**titleLabelProperties)
                                     
                                     
 		labelTitle.grid(columnspan=2, sticky=tk.W, padx=3,pady=5)
 		
 		for n, label in enumerate(self.descriptionValues):
 			var = tk.StringVar(value = self.initialValueList[n])
 			lab = tk.Label(self.cont, text = '{}:'.format(label), bg = MAC_GREY)
 			lab.grid(row=n+1, column=0,padx=3,pady=3,sticky=tk.E)
 			comboBox = ttk.Combobox(self.cont, textvariable = var,
 												values = self.optionList[n])
 			comboBox.grid(row=n+1,column=1,padx=3,pady=3,sticky=tk.EW)
 			self.output[label] = var
 												
 			
 		applyButton = ttk.Button(self.cont, text = 'Done', command = self.save_selection)
 		
 		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
 		
 		applyButton.grid(column = 0, sticky=tk.W)
 		row = int(float(applyButton.grid_info()['row']))
 		closeButton.grid(row=row, column = 1, sticky=tk.E)
 		
	def save_selection(self):
		'''
		'''
		for label,var in self.output.items():
			self.selectionOutput[label] = var.get()
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
                                     
                                     
                                     
  