import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
from modules.pandastable import core 
import numpy as np
import pandas as pd
from itertools import chain
from modules.utils import * 



class dataDisplayDialog(object):


	def __init__(self, data, plotter = None, showOptionsToAddDf = False, 
									dragDropLabel = False, analyzeClass = None,
									dfOutputName = None, topmost = False, waitWindow = True):
	
		self.twodline = None
		self.background = None
		self.dfOutputName = dfOutputName
		
		self.topmost = topmost
		
		self.prev_rows_selected = []
		self.colname, self.catnames, self.plot_type, self.cmap = None, None, None, None
		self.plotter = plotter
		self.analyzeClass = analyzeClass

		
		self.platform = platform
		self.data = data 
		self.data_shape = data.shape
		self.columns = data.columns.values.tolist()

		
		if self.plotter is not None:
			self.figure = plotter.figure
			self.canvas = self.figure.canvas
			id = self.plotter.plotCount
			if id in self.plotter.plotProperties:
				self.colnames, self.catnames, self.plot_type, self.cmap = self.plotter.plotProperties[id]
			

			
		self.build_toplevel() 
		self.initiate_table(self.columns)
		if showOptionsToAddDf:
			self.addDf = False
			self.add_widgets(dragDropLabel)
		
		if dragDropLabel == False and waitWindow:
			self.toplevel.wait_window()

		
	def close(self):
		'''
		Close toplevel
		'''
		if hasattr(self,'identifyAfter'):
			self.toplevel.after_cancel(self.identifyAfter)
		try: ## if error with data selection this will prevent closing..
			self.data = self.pt.model.df
			self.pt.remove()
		except:
			pass 
		if hasattr(self.analyzeClass,'groupedStatsData') and \
		hasattr(self, 'dragDropLabel'):
			## then we are using this to show all Pairwise Comparisons
			del self.analyzeClass.groupedStatsData
		self.toplevel.destroy() 
		
			
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Data   -   Rows: {}  x   Columns: {}'.format(self.data_shape[0],self.data_shape[1]))
		if self.topmost:
			popup.attributes('-topmost', True)
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=790
		h=680
		self.toplevel = popup		
		self.center_popup((w,h))
		
	def add_widgets(self,dragDropLabel = False):
		'''
		'''
		buttonFrame = tk.Frame(self.toplevel,background = MAC_GREY)
		buttonFrame.pack(fill = tk.BOTH)
		buttonFrame.grid_columnconfigure(1,weight=1)
		if dragDropLabel:
			self.dragDropLabel = tk.Label(buttonFrame, text = '   Drop Test Here    ',**titleLabelProperties )	
			self.dragDropLabel.grid(row = 0, column = 1, sticky = tk.EW, padx=30)
		addDfButton = ttk.Button(buttonFrame , text = 'Add to Data Collection', command = self.initiate_add)
		addDfButton.grid(row = 0, padx=3,pady=4, sticky=tk.W) 		
		
		closeButton = ttk.Button(buttonFrame, text = 'Close', command = self.close)
		closeButton.grid(padx=3,pady=4,row=0, column = 1, sticky = tk.E)
	
	def initiate_add(self):
		'''
		'''
		if hasattr(self, 'dragDropLabel'):
			self.add_to_data_collection()
		else:
			self.addDf = True
		self.close()


	def add_to_data_collection(self):
		'''
		'''
		if self.dfOutputName  is not None:
			self.analyzeClass.add_new_dataframe(self.pt.model.df,self.dfOutputName )
				
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 
         	
         	
         	 
	def initiate_table(self,columns):
		'''
		Initiates the table view of the very cool package: pandastable. 
		'''
	
		cont= tk.Frame(self.toplevel, background =MAC_GREY) 
		cont.pack(expand =True, fill = tk.BOTH)

		self.pt = core.Table(cont,dataframe = self.data, showtoolbar=False, showstatusbar=False)
		
		self.pt.show()
		self.identify_data()
	
	def get_data(self):
		dat_ = self.data
		return dat_

#### make nicer code here... 
		
	def identify_data(self):
	
	
		if self.plot_type is None:
			return 
		
		currentRows = self.pt.multiplerowlist
		self.trigger_plotting(currentRows)
		self.identifyAfter = self.toplevel.after(100, self.identify_data)
	
	
			
	def trigger_plotting(self,rows_selected):
	
	
		
		
		if rows_selected == self.prev_rows_selected:
			
			return
		
		
		if len(self.figure.axes) >= 1:
			self.prev_rows_selected = [row for row in rows_selected] 
			df_ = self.pt.model.df.iloc[self.prev_rows_selected,:]
			
			if self.plot_type == 'scatter':
			
				if len(self.catnames) == 0:
					df_.dropna(subset=self.colnames,inplace=True)
					ax_ = self.figure.axes[0]
					if self.twodline is None:
						
						ax_collections = ax_.collections
						ax_collections[0].set_alpha(0.2)
						self.canvas.draw()
						if self.background is None:
							self.background = self.canvas.copy_from_bbox(ax_.bbox)  

							self.twodline = plt.plot(df_[self.colnames[0]],df_[self.colnames[1]],
								'o',markeredgecolor = 'black',markeredgewidth = 0.3, alpha=0.85)
							ax_.draw_artist(self.twodline[0])
							self.canvas.blit(ax_.bbox)
					else:
						self.canvas.restore_region(self.background)
						self.twodline[0].set_xdata(df_[self.colnames[0]].values)
						self.twodline[0].set_ydata(df_[self.colnames[1]].values)
						ax_.draw_artist(self.twodline[0])
						self.canvas.blit(ax_.bbox) 
					
			elif self.plot_type in ['boxplot','violinplot','barplot']:
				if len(self.catnames) == 0:
					line_collection = []
					
					ax_ = self.figure.axes[0]
					
					for n,col in enumerate(self.colnames):
						dat_ = df_[col].dropna()
						dat_length = len(dat_.index) 
						x = np.repeat(n,dat_length)
						y = dat_
						line_collection.append([x,y,'o'])
					
					
					if self.twodline is None:
						self.background = self.canvas.copy_from_bbox(ax_.bbox)
						line_collection = chain.from_iterable(line_collection) 
					
						self.twodline = ax_.plot(*line_collection, markeredgecolor = 'black',markeredgewidth = 0.3, alpha=0.85, markerfacecolor="white")
						for line in self.twodline:
							ax_.draw_artist(line)
						
						self.canvas.blit(ax_.bbox) 
					else:
						self.canvas.restore_region(self.background)
						for n,line in enumerate(self.twodline):
							x,y,marker = line_collection[n]
							line.set_xdata(x)
							line.set_ydata(y) 
							ax_.draw_artist(line)
						self.canvas.blit(ax_.bbox) 
						
						
			

class __PlotHelper_(object):
	def __init__(self):
		pass		