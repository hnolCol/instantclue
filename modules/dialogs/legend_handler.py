import string
import os
import sys

import tkinter as tk
from tkinter import ttk
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
import tkinter.filedialog as tf
import seaborn as sns
from modules import images
from modules.utils import *
from modules.dialogs import VerticalScrolledFrame
try:
	#matplotlib 2
	from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
except:
	#matplotlib 3
	from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
import itertools

class legendDialog(object):


	def __init__(self, plotter):
	
	
		self.plotter = plotter
		self.build_toplevel()
		self.build_widgets()
		
	def close(self,event=None):
		'''
		Close toplevel
		'''

		self.toplevel.destroy() 
		
			
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Chart information') 
		popup.bind('<Escape>', self.close)
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=400
		h=650
		self.toplevel = popup
		self.center_popup((w,h))

	
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont.grid_columnconfigure(0, weight = 1)
 		self.cont.grid_rowconfigure(0, weight = 1)
 		vertFrame = VerticalScrolledFrame.VerticalScrolledFrame(self.cont)
 		vertFrame.grid(row=0,columnspan=1, sticky=tk.NSEW)
 		figureFrame = tk.Frame(vertFrame.interior) ##scrlollable
 		toolbarFrame = tk.Frame(self.cont,bg=MAC_GREY)
 		self.display_figure(figureFrame,toolbarFrame)
 		figureFrame.grid(columnspan=2, sticky =tk.NSEW)
 		toolbarFrame.grid(columnspan= 2, sticky=tk.W)#+tk.EW)	
 		self.get_plot_information()
 		
	def get_plot_information(self, nLevels = 8):
		'''
		'''
		scatterPlots = self.plotter.get_scatter_plots()
		ax = self.figure.add_subplot(111)
		for scatterPlot in scatterPlots.values():
			#print(scatterPlot.scatterKwargs)
			
			if 'color' in scatterPlot.scatterKwargs:
				colors = sns.color_palette(scatterPlot.colorMap, nLevels, desat = 0.75)
				print(colors)
				print(scatterPlot.get_numeric_color_data())
				q = np.quantile(scatterPlot.get_numeric_color_data().dropna().values,
								q = np.linspace(0,1,num=nLevels))
				print(q)
				ax.scatter([1] * len(colors), 
								q, 
								color = colors, 
								s = 60)
				
			elif 's' in scatterPlot.scatterKwargs:
				q = np.quantile(scatterPlot.scatterKwargs['s'],q = [0,0.1,0.25,0.5,0.75,0.9,1])
				
				label_x, label_y = [1]*q.size, range(q.size)
				ax.scatter(label_x, label_y, 
							s = q, 
							color = "white", 
							edgecolors = "darkgrey")
				
				for n, text in enumerate(q):
				
					ax.text(
						label_x[n]+0.005,
						label_y[n],
						s=return_readable_numbers(text),
						horizontalalignment='left')
				
				
		self.figure.canvas.draw()
				

	def display_figure(self, frameFigure,toolbarFrame):

		self.figure = plt.figure(figsize = (1.2,8.1), 
								 facecolor = "lightgrey")
		#self.figure.subplots_adjust(top=0.94, bottom=0.05,left=0.1,
		#							right=0.95, wspace = 0.32, hspace=0.32)
		canvas  = FigureCanvasTkAgg(self.figure,frameFigure)
		if hasattr(canvas,'show'):
			canvas.show()
		elif hasattr(canvas,'draw'):
			canvas.draw()
		self.toolbar_main = NavigationToolbar2TkAgg(canvas, toolbarFrame)
		canvas._tkcanvas.pack(in_=frameFigure,side="top",fill='both',expand=True)
		canvas.get_tk_widget().pack(in_=frameFigure,side="top",fill='both',expand=True)
		
						
	def center_popup(self,size):
         '''
         Casts poup and centers in screen mid
         '''

         w_screen = self.toplevel.winfo_screenwidth()
         h_screen = self.toplevel.winfo_screenheight()
         x = w_screen/2 - size[0]/2
         y = h_screen/2 - size[1]/2
         self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))