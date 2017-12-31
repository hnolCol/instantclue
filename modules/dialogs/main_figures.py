import string 
import os
import sys
try:
	from PIL import Image, ImageTk
except:
	pass
	
	
	
import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
from modules import images
from modules.utils import * 
from modules.dialogs import VerticalScrolledFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg




alphabeticLabel  = list(string.ascii_lowercase)+list(string.ascii_uppercase)
rangePositioning = list(range(1,20))

labelsAndGridInfo = 		dict(
								positionRow = ['Position (row,column):',dict(row=5,column=0, sticky=tk.E, columnspan=2,pady=3),
											dict(row=5, column = 2, sticky=tk.W,pady=3,padx=1)],
								positionColumn = ['','',dict(row=5, column = 3, sticky=tk.W,pady=3,padx=1)],
								columnSpan = ['Column span:',dict(row=5,column = 6, sticky=tk.W,pady=3),
											dict(row=5,column = 7, sticky=tk.W,pady=3) ], 
								rowSpan = ['Row span:',dict(row=5,column = 4, sticky=tk.E,pady=2),
											dict(row=5,column = 5, sticky=tk.W,pady=3)], 
								gridRow = ['Rows:',dict(row=2, column = 0,columnspan=2, sticky =tk.E,pady=3),
											dict(row=2, column = 2, sticky=tk.W,pady=3)],
								gridColumn = ['Columns:',dict(row=2, column =4, sticky=tk.W,pady=3),
												dict(row=2, column = 5, sticky=tk.W,pady=3)],
								subplotLabel = ['Subplot Label:', dict(row=5,column =8, sticky=tk.W,pady=3),
												dict(row=5,column =9, sticky=tk.W,pady=3) ])


MAC_GREY = '#ededed'
LARGE_FONT = ("Verdace", 13, "bold") 
titleFont = dict(font = LARGE_FONT, fg="#4C626F",  justify=tk.LEFT, bg = MAC_GREY)


class mainFigureCollection(object):
	'''
	Class that manages main figures.
	We have decided to split this from the actual creation to achieve easier opening from saved session. 
	'''
	def __init__(self):
		## defining StringVars for grid layout and positioning of axis 
		self.main_fig_add_text = None
		self.positionColumn = tk.StringVar(value='1') 
		self.positionRow = tk.StringVar(value='1') 
		self.columnSpan = tk.StringVar(value='1') 
		self.rowSpan = tk.StringVar(value='1') 	
		self.gridColumn = tk.StringVar(value='3') 
		self.gridRow = tk.StringVar(value='4') 
		
		self.subplotLabel = tk.StringVar(value='a')	
		self.infolabel = tk.StringVar(value = 'Add subplots to build main figure') 
		
				
		self.positionGridDict = OrderedDict(positionColumn = self.positionColumn,positionRow = self.positionRow,
								columnSpan = self.columnSpan, rowSpan = self.rowSpan, gridRow = self.gridRow,
								gridColumn = self.gridColumn, subplotLabel = self.subplotLabel)
								
		
		
		
		self.load_images()	
		self.build_toplevel()
		self.create_frame()
		self.create_widgets()
		
		
	
		
	def close(self):
		'''
		closing the toplevel
		'''
		self.toplevel.destroy() 
		
		
		
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Setup main figure ...') 
         
		popup.protocol("WM_DELETE_WINDOW", self.close)
		
		w = 845
		h = 840
             
		self.toplevel = popup
		self.center_popup((w,h))
		
	def create_frame(self):
		
		'''
		Creates frame to put widgets in
		'''
		self.cont = tk.Frame(self.toplevel,bg=MAC_GREY)
		self.cont.pack(fill='both', expand=True)
		self.cont.grid_columnconfigure(9, weight = 1)
		self.cont.grid_rowconfigure(10, weight=1)
			
	def create_widgets(self):
		'''
		Creates all widgets
		'''
		labelGridSetup = tk.Label(self.cont, text='Define grid layout for main figure',**titleFont)
		labelAxisSetup = tk.Label(self.cont, text='Add subplot to figure',**titleFont)
		labelInfoLab = tk.Label(self.cont, textvariable = self.infolabel,**titleFont)
		labelFigIdLabel = tk.Label(self.cont, text = 'Figure ID ENTER HERE',**titleFont)
		
		
		labelGridSetup.grid(row=1, column = 0, sticky=tk.W, columnspan=6, padx=3,pady=5)       
		for id,variable in labelsAndGridInfo.items():
			
			labelText = variable[0]
			if labelText == 'Position (row,column):':
			
				labelFigIdLabel.grid(row=2,column = 7, sticky = tk.E, columnspan=14, padx=30)
				ttk.Separator(self.cont, orient = tk.HORIZONTAL).grid(sticky=tk.EW, columnspan=15,pady=4, row=3)  
				labelAxisSetup.grid(row=4, column = 0, sticky=tk.W, columnspan=15,padx=3,pady=5)
				
			if labelText != '':
				labCombobox = tk.Label(self.cont, text = labelText, bg=MAC_GREY)
				labCombobox.grid(**variable[1])
			if labelText != 'Subplot Label:':
				valuesCombo = rangePositioning 
			else:
				valuesCombo = alphabeticLabel

			combobox = ttk.Combobox(self.cont, textvariable = self.positionGridDict[id], values = valuesCombo, width=5) 
			combobox.grid(**variable[2])
			
		ttk.Separator(self.cont, orient = tk.HORIZONTAL).grid(sticky=tk.EW, columnspan=15,pady=4)
		
		## crate and grid main buttons  on MAC ttk Buttons with images are buggy in width
		if platform == 'WINDOWS':
			but_add_axis = ttk.Button(self.cont, image = self.add_axis_img, command = self.add_axis_to_figure)	
			but_add_text = ttk.Button(self.cont, image = self.add_text_img)
			but_add_image = ttk.Button(self.cont, image =  self.add_image)
			but_delete_ax = ttk.Button(self.cont, image = self.delete_axis_img)
			but_clear = ttk.Button(self.cont, image = self.clean_up_img, command = self.clean_up_figure)
		elif platform == 'MAC':				
			but_add_axis = tk.Button(self.cont, image = self.add_axis_img, command = self.add_axis_to_figure)	
			but_add_text = tk.Button(self.cont, image = self.add_text_img)
			but_add_image = tk.Button(self.cont, image =  self.add_image)
			but_delete_ax = tk.Button(self.cont, image = self.delete_axis_img)
			but_clear = tk.Button(self.cont, image = self.clean_up_img, command = self.clean_up_figure)
			
		btns= [but_add_axis,but_add_text,but_add_image,but_delete_ax,but_clear]
		
		for n,btn in enumerate(btns):
			btn.grid(row=7,column=n, padx=2,pady=2)
		
		labelInfoLab.grid(row=7,column=n+1, padx=4,pady=2,columnspan=30,sticky=tk.W)
		vertFrame = VerticalScrolledFrame.VerticalScrolledFrame(self.cont)
		vertFrame.grid(row=10,columnspan=20, sticky=tk.NSEW)
		figureFrame = tk.Frame(vertFrame.interior) ##scrlollable
		toolbarFrame = tk.Frame(self.cont,bg=MAC_GREY)
		
		figureFrame.grid(columnspan=20) 
		toolbarFrame.grid(columnspan=16, sticky=tk.W)#+tk.EW)
		self.display_figure(figureFrame,toolbarFrame)
		
             
        
			
		
		
		
		#, command = lambda fig_id = figure_id,info_lab = info_lab: add_text(fig_id,info_lab,popup))  #fig_id = figure_id :  add_text(fig_id))
         #, command = lambda fig_id = figure_id, label=var_subplot_label, row_pos = var_rowpos, col_pos = var_colpos :  clear_fig(fig_id,label,row_pos,col_pos))
        #, command = lambda: add_image(figure_id,popup,info_lab))  
        
        #but_delete_ax.bind('<Button-1>', lambda event, figure_id = figure_id, info_lab = info_lab,rows = var_rowpos,cols= var_colpos,label=var_subplot_label: delete_axis(event,figure_id,info_lab,rows,cols,label))
        
        
        
        
       
       # but_add_axis.grid(row=7,column=0,padx=(4,2),pady=2) 
       # but_add_image.grid(row=7,column=2, padx=2,pady=2)
       # but_add_text.grid(row=7,column = 1,padx=2,pady=2)
       # but_delete_ax.grid(row=7,column=3,padx=2,pady=2) 
       # but_clear.grid(row=7,column = 4,padx=2,pady=2) 
       # 
             
             
             
             #self.main_subs = dict() 
             
        
        
    
		
		 



	
    	
	def load_images(self):
		self.add_axis_img, self.add_text_img,self.add_image,\
		self.delete_axis_img,self.clean_up_img,self.delete_axis_active_img  = images.get_main_figure_images()      
        
        
        


	def display_figure(self, frameFigure,toolbarFrame):
	
		self.figure = plt.figure(figsize=(8.27,11.7))      
		self.figure.subplots_adjust(top=0.94, bottom=0.05,left=0.1,right=0.95)
		
		canvas  = FigureCanvasTkAgg(self.figure,frameFigure)
		canvas.show() 
		self.toolbar_main = NavigationToolbar2TkAgg(canvas, toolbarFrame)
		canvas._tkcanvas.pack(in_=frameFigure,side="top",fill='both',expand=True)
                                                 
		canvas.get_tk_widget().pack(in_=frameFigure,side="top",fill='both',expand=True)
	
         #
	
	def add_axis_to_figure(self):
		'''
		Adss an axis to the figure . Gets the settings from the dictionary that stores self.positionGridDict
		'''
		axisParams =  self.get_axis_parameters()
		gridRow, gridCol,posRow, posCol,rowSpan, colSpan, subplotLabel = axisParams
		
		if posRow-1 + rowSpan > gridRow or posCol -1 + colSpan > gridCol:
			tk.messagebox.showinfo('Invalid input ..','Axis specification out of grid.')
			return 	
		
		grid_spec = plt.GridSpec(gridRow,gridCol) 
		subplotspec = grid_spec.new_subplotspec(loc=(posRow-1,posCol-1),
												rowspan=rowSpan,colspan=colSpan)
										
		ax_ = self.figure.add_subplot(subplotspec)
									
		self.figure.canvas.draw()
		
		self.update_axis_parameters(axisParams)	
		self.infolabel.set('Axis added!')									
												
        
	
	def get_axis_parameters(self):
		'''
		Returns axis and grid parameters
		'''
		gridRow, gridCol = self.positionGridDict['gridRow'].get(), self.positionGridDict['gridColumn'].get()		
		posRow , posCol = self.positionGridDict['positionRow'].get(), self.positionGridDict['positionColumn'].get()	
		rowSpan, colSpan = self.positionGridDict['rowSpan'].get(), self.positionGridDict['columnSpan'].get()
		
		propsStrings = [gridRow, gridCol,posRow, posCol,rowSpan, colSpan]
		propsIntegers = [int(float(item)) for item in propsStrings]
		
		subplotLabel = self.positionGridDict['subplotLabel'].get()
		propsIntegers.append(subplotLabel) 
		
		return propsIntegers
		
	def update_axis_parameters(self, parametersList = None):
		'''
		Updates the comboboxes to provide convenient addition of mroe axes.
		'''	
		if parametersList is None:
			gridRow, gridCol,posRow, posCol,rowSpan, colSpan, subplotLabel  = self.get_axis_parameters() 
		else:
			gridRow, gridCol,posRow, posCol,rowSpan, colSpan, subplotLabel = parametersList 
		## updating 	
		if subplotLabel in alphabeticLabel:
		
			idxLabel = 	alphabeticLabel.index(subplotLabel)
			nextLabelIdx = idxLabel+1
			if nextLabelIdx == len(alphabeticLabel):
				nextLabelIdx = 0
			
			nextLabel = alphabeticLabel[nextLabelIdx]
			self.positionGridDict['subplotLabel'].set(nextLabel)
		# reset position in Grid..
		if posCol + colSpan > gridCol:
			posCol = 1
			posRow = posRow + rowSpan 
		else:
			posCol = posCol + colSpan
		
		self.positionGridDict['positionRow'].set(str(posRow))
		self.positionGridDict['positionColumn'].set(str(posCol))
		




	def clean_up_figure(self):
		
		self.figure.clf()
		self.figure.canvas.draw()
		
		self.update_axis_parameters([4,3,1,0,1,1,'Z'])
		self.infolabel.set('Cleaned up.')	
		
		
		
	def center_popup(self,size):
         	'''
         	Casts the popup in center of screen
         	'''

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))	