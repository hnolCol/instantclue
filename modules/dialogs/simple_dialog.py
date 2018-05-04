import tkinter as tk
from tkinter import ttk  
import pandas as pd


from modules.utils import *



class simpleUserInputDialog(object):

	def __init__(self, descriptionValues,initialValueList, optionList, title, infoText, h = None):
		
		self.initialValueList = initialValueList
		self.descriptionValues = descriptionValues
		self.h = h
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
		
			
	def close(self, event = None):
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
		popup.bind('<Escape>', self.close) 
		popup.bind('<Return>',self.save_selection)
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=390
		if self.h is None:
			self.h = 60
		h=self.h + 55*len(self.descriptionValues)
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
 												
 			
 		applyButton = ttk.Button(self.cont, text = 'Done', 
 			command = self.save_selection,width=6)
 		
 		closeButton = ttk.Button(self.cont, text = 'Close', 
 			command = self.close, width=6)
 		
 		applyButton.grid(column = 0, sticky=tk.W,padx=4)
 		row = int(float(applyButton.grid_info()['row']))
 		closeButton.grid(row=row, column = 1, sticky=tk.E,padx=4)
 		
	def save_selection(self,event=None):
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
                                     


class simpleListboxSelection(object):
	'''
	Creates a simple toplevel with a Listbox.
	Parameter 
	============
	infoText - text displayed as a title
	data - list as data for Listbox
	title = Title of Toplevel
	
	Properties
	============
	selection - returns user's selection.
	'''
	def __init__(self,infoText,data,title = 'Listbox Selection'):
	
		self.selected = []
		self.infoText = infoText
		self.data = data
		self.title = title
		
		self.selectAll = tk.BooleanVar(value=False)
		
		
		
		self.build_toplevel()
		self.build_widgets()
		self.fill_listbox()
		self.toplevel.wait_window()
	
	def close(self,event = None, discard = True):
		'''
		Closes the toplevel
		If discard ==  True - remove selection
		'''
		if discard:
			self.selected = []
		self.toplevel.destroy()
		
	
	def build_toplevel(self):
		'''
		Builds toplevel
		'''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title(self.title) 
		popup.grab_set() 
		popup.bind('<Escape>', self.close) 
		popup.bind('<Return>',self.define_selection)
		popup.protocol("WM_DELETE_WINDOW", self.close)
		
		w = 390
		h = 450
		self.toplevel = popup
		self.center_popup((w,h))
		
				
	def build_widgets(self):	
		'''
		Builds widgets on toplevel
		'''
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		self.cont.grid_columnconfigure(2,weight = 1)
		self.cont.grid_rowconfigure(2,weight =1)
		labelTitle = tk.Label(self.cont, text= self.infoText, 
                                     wraplength=320,**titleLabelProperties)
 		
		labelTitle.grid(columnspan=2, sticky=tk.W, padx=3,pady=5) 
		self.create_listbox()
		
		self.checkAllCb = ttk.Checkbutton(self.cont, 
								text = 'Select all', 
								variable = self.selectAll,
								command = self.select_all)
		
		self.checkAllCb.grid(row=4,column=0,padx=5,pady=2,sticky=tk.W,columnspan=2)
		applyButton = ttk.Button(self.cont, text = 'Save selection', command = self.define_selection)	
		closeButton = ttk.Button(self.cont, text = 'Discard & Close', command = self.close)
		
		applyButton.grid(row=5,column=0,padx=5,pady=10,sticky=tk.W)
		closeButton.grid(row=5,column=3,padx=5,pady=10,sticky=tk.E)
				
	def create_listbox(self):
 		'''
 		Creates the listbox
 		'''
 		vs = ttk.Scrollbar(self.cont, orient = tk.VERTICAL)
 		hs = ttk.Scrollbar(self.cont, orient = tk.HORIZONTAL)
 		self.listbox = tk.Listbox(self.cont, selectmode = tk.EXTENDED,
 						yscrollcommand = vs.set, xscrollcommand = hs.set)
 		vs.config(command=self.listbox.yview)  
 		hs.config(command=self.listbox.xview)   
 		self.listbox.grid(row=2,column=0,
 						  columnspan=4,sticky=tk.NSEW,
 						  padx = (15,0), pady=(15,0))           
 		vs.grid(row=2,column=4,sticky=tk.NS+tk.W,padx=(0,15),pady=(15,0)) 
 		hs.grid(row=3,column=0,columnspan=4,sticky=tk.EW+tk.N,padx = (15,0),pady=(0,15))
 		                      
 				
	def fill_listbox(self):
		'''
		Fills the listbox with data.
		'''
		for entry in self.data:
			self.listbox.insert(tk.END,entry)
	
	def select_all(self):
		'''
		Select / Deselect all items
		'''
		
		if self.selectAll.get():
		
			self.listbox.select_set(0, tk.END)
			self.checkAllCb.configure(text = 'Deselect all')
			
		else:
			
			self.listbox.selection_clear(0,tk.END)
			self.checkAllCb.configure(text = 'Select all')
					
			
	@property
	def selection(self):
		return self.selected	
		
	def define_selection(self,event=None):
		'''
		Defines selection to accessed using property (selection)
		'''
		self.selected = [self.data[idx] for idx in self.listbox.curselection()]
		if len(self.selected) == 0:
			tk.messagebox.showinfo('Error ..','No items selected..',parent=self.toplevel)
			return
		self.close(discard=False)
			
	
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))		
	
	
	



                                     
                                     
  