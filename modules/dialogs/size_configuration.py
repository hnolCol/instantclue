import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt

from modules.utils import *

class SizeConfigurationPopup(object):
	
	
	def __init__(self,plotter):
			
						
		self.plot_type = plotter.currentPlotType 
				
		self.catergoricalColumns = plotter.get_active_helper().categoricalColumns
		self.figure = plotter.figure
		self.canvas = self.figure.canvas
		self.size = plotter.sizeScatterPoints		
		
		self.plotter = plotter
		self.size_selected = tk.StringVar() 
		self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
		
		self.build_toplevel() 
		self.build_widgets() 
		
		self.collections = self.get_collections()
		self.r = self.figure.canvas.get_renderer()
		
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
		popup.wm_title('Size configuration') 
         
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=180
		h=120
		self.toplevel = popup
		self.toplevel.attributes('-topmost', True)
		self.center_popup((w,h))
		
			
	def build_widgets(self):
			
             '''
             Building needed tkinter widgets 
             '''
             
             cont = tk.Frame(self.toplevel,background = MAC_GREY)
             cont.pack(fill='both', expand=True)
             cont.grid_columnconfigure(0,weight=1)
             self.size_selected.set(str(self.size))
		
             label_des = tk.Label(cont, text = 'Change size of marker:', font = LARGE_FONT, fg="#4C626F", justify=tk.LEFT, bg = MAC_GREY)
             entry_size = ttk.Entry(cont, textvariable= self.size_selected, width=500) 
             
             entry_size.bind('<Return>', lambda event, val = self.size_selected.get():self.change_size(val,event = event))
             
             self.slider_size = ttk.Scale(cont, from_=5, to=300, value = self.size, command= lambda val : self.change_size(val))
             label_des.grid(pady=5,padx=5,sticky=tk.W)
             entry_size.grid(pady=4,padx=5,sticky=tk.EW)
             self.slider_size.grid(pady=4,padx=5,sticky=tk.EW)
		

	
	def center_popup(self,size):
         	'''
         	Casts the popup in center of screen
         	'''

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))  	

	
	
	def get_collections(self):
		
		
         plot_type = self.plot_type
         catnames = self.catergoricalColumns
         n_categories = len(catnames)
         axColl = []
		
         if plot_type == 'scatter':
         	
             if n_categories > 0:
             	immAxes = self.plotter.categoricalPlotter.scatterWithCategories.inmutableAxes
             	axes = self.figure.axes
             	for ax in axes:
             		if ax not in immAxes:
             			axColl.extend(ax.collections)
             		
             else:
             	axes = self.figure.axes
             	for ax in axes:
             		axColl.extend(ax.collections)
             		
             			
         elif plot_type == 'cluster_analysis':
         
         	ax = self.figure.axes[0]
         	ax_coll = ax.collections
         	for coll in ax_coll:
         		if hasattr(coll,'set_sizes'): #otherwise it is a line collection
         			axColl.append(ax.collections)
           	
		             
         elif plot_type == 'scatter_matrix':
         
             axes = self.figure.axes
             for ax in axes:           
             		axColl.extend(ax.collections)
             		
             			
         elif plot_type == 'pointplot' or plot_type == 'swarm':
         
         	axes = self.figure.axes
         	for ax in axes:
         		axColl.extend(ax.collections)
         		
         return axColl
		
	
	def change_size(self,val,event=None):
         
         '''
         Main execution function of this popup. Changes dynamically the size of collections.
         Collections were extracted before and are stored in self.collections
         '''
		
         if event is not None:
         	val = self.size_selected.get() 
         	val = round(float(val),0)
         	self.slider_size.set(val)
         else:	
         	val = round(float(val),0)
         if val == self.size:
         	return	
         
         self.figure.canvas.restore_region(self.background)
         for coll in self.collections:
             			coll.set_sizes([val])
             			coll.draw(self.r)
             			#ax.draw_artist(coll)
         
         self.size = val
         self.size_selected.set(val)
         self.canvas.blit(self.figure.bbox)		
