import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt

from modules.utils import *

class SizeConfigurationPopup(object):
	
	
	def __init__(self,platform,figure,canvas,plot_type,size,catnames,
						subsets_and_scatter_with_cat,
						axes_scata_dict,filt_source_for_update):
						
		self.plot_type= plot_type
		self.platform = platform
		self.catnames = catnames
		self.figure = figure
		self.canvas = canvas 
		self.size = size
		
		
		self.subsets_and_scatter_with_cat = subsets_and_scatter_with_cat
		self.axes_scata_dict = axes_scata_dict
		self.filt_source_for_update = filt_source_for_update
		self.size_selected = tk.StringVar() 
		
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
	
	
	def change_size(self,val,event=None):
         
         '''
         Main execution function of this popup. Changes dynamically the size of collections.
         It will get the scatter items from 
         '''
		
         if event is not None:
         	val = self.size_selected.get() 
         	val = round(float(val),0)
         	self.slider_size.set(val)
         else:	
         	
         	val = round(float(val),0)
         if val == self.size:
         	return	
         plot_type = self.plot_type
         catnames = self.catnames
         n_categories = len(catnames)
        
         
         if plot_type == 'scatter':
         	
             if n_categories > 0:
             	for key, subset_and_scatter in self.subsets_and_scatter_with_cat.items():
             	
             		subset, ax_, scat = subset_and_scatter
             		scat.set_sizes([val])
             		
             else:
             	axes = self.figure.axes
             	for ax in axes:
             		ax_coll = ax.collections
             		for coll in ax_coll:
             			coll.set_sizes([val])
             			
         elif plot_type == 'cluster_analysis':
         
         	ax = self.figure.axes[0]
         	ax_coll = ax.collections
         	for coll in ax_coll:
         		if hasattr(coll,'set_sizes'): #otherwise it is a line collection
         			coll.set_sizes([val])
           	
		             
         elif plot_type == 'scatter_matrix':
         
             axes = self.figure.axes
             for ax in axes:           
             		ax_coll = ax.collections
             		for coll in ax_coll:
             			coll.set_sizes([val])
             			
         elif plot_type == 'pointplot' or plot_type == 'swarm':
         
         	axes = self.figure.axes
         	for ax in axes:
         		ax_coll = ax.collections
         		for coll in ax_coll:
         			coll.set_sizes([val])         

         self.size = val
        
         self.size_selected.set(val)
         self.canvas.draw()		
