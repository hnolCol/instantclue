import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt

from modules.utils import *



class sizeIntervalDialog(object):
	
	def __init__(self,plotter):
	
	
		self.plotter = plotter
		
		self.define_variables()
		self.build_toplevel()
		self.build_widgets() 		
		
	def define_variables(self):
		'''
		Setup variables.
		'''
		minSize, maxSize = self.plotter.get_size_interval()
		
		self.minVar = tk.IntVar(value = minSize)
		self.maxVar = tk.IntVar(value = maxSize)
		
		self.entries = dict()
		
		
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
		popup.wm_title('Size range') 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		popup.bind('<Escape>',self.close)
		w=230
		h=120
		self.toplevel = popup
		self.center_popup((w,h))

	def build_widgets(self):
		'''
		Puts widgets on toplevel
		'''
		cont = tk.Frame(self.toplevel,background = MAC_GREY)
		cont.pack(fill='both', expand=True)
		labelTitle = tk.Label(cont, text = 'Set size interval', **titleLabelProperties)
		CreateToolTip(labelTitle, text = 'Move slides to adjust level. Chart will be updated upon release.')
		
		labelTitle.grid(row= 0,column=0, sticky = tk.W,
						columnspan=3,padx=3,pady=10)		
		for n,var in enumerate([self.minVar,self.maxVar]):
			entScale = ttk.Entry(cont,width=5) 
			
			if n == 0:
				txt_ = 'Minimum: '
				self.entries['min'] = entScale
				
			else:
				txt_ = 'Maximum: '
				self.entries['max'] = entScale
			
			entScale.insert(0,var.get())
			# create label
			labelScale = tk.Label(cont,text = txt_ , bg = MAC_GREY)
			# add tooltip 
			CreateToolTip(labelScale, text = 'Set the min and max of the size interval.'+
				'If min > max higher values or sorted categories appear smaller.') 
			
			
			sizeScale = tk.Scale(cont, from_ = 10, to = 500, 
								 resolution=1, variable = var, sliderlength = 20,
								 orient =  tk.HORIZONTAL, showvalue = 1, bg = MAC_GREY)			
			sizeScale.bind('<ButtonRelease-1>', self.apply_change)
			entScale.bind('<Return>', lambda event:self.apply_change(event,fromEntry=True))
			
			labelScale.grid(row=n+1,column=0,sticky=tk.S+tk.E)
			sizeScale.grid(row=n+1,column=1)
			entScale.grid(row=n+1,column=2,sticky=tk.S)
		
	def empty_entries(self):
		'''
		'''
		for entry in self.entries.values():
			entry.delete(0,tk.END)
	
	def update_entries(self,min,max):
		'''
		'''
		self.entries['min'].insert(0,min)
		self.entries['max'].insert(0,max)
	
	def update_slider(self,min,max):
		'''
		'''
		self.minVar.set(min)
		self.maxVar.set(max)
			
	def apply_change(self,event,fromEntry = False):	
		'''
		'''
		if fromEntry:
			try:
				min, max = int(float(self.entries['min'].get())),\
				int(float(self.entries['max'].get()))
			except:
				tk.messagebox.showinfo('Error..',
					'Could not convert input to float.',
					parent=self.toplevel)
				return
			self.update_slider(min,max)
			
		else:
			min, max = self.minVar.get(), self.maxVar.get()
			self.empty_entries()
			self.update_entries(min,max)

		
		self.plotter.set_size_interval(min,max)
		if self.plotter.nonCategoricalPlotter is not None:
			self.plotter.nonCategoricalPlotter.update_size_interval_in_chart()
		elif self.plotter.categoricalPlotter is not None:
			if hasattr(self.plotter.categoricalPlotter, 'scatterWithCategories'): 
				self.plotter.categoricalPlotter.scatterWithCategories.update_size_interval_in_chart()
		
		self.plotter.redraw()
		
		
	def center_popup(self,size):
         	'''
         	Casts the popup in center of screen
         	'''

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))  	
             


class SizeConfigurationPopup(object):
	
	
	def __init__(self,plotter):
	
	
		if plotter.get_active_helper() is None:
			return			
			
		self.plot_type = plotter.currentPlotType 
		
		if self.plot_type in ['boxplot','violinplot','density','hclust',
							'corrmatrix','barplot','line_plot']:
							
			tk.messagebox.showinfo('Invisible effect ..','Changing the size will effect the '+
								   'point size for other chart types but are not visibile in'+
								   ' the selected chart. Please note that "added swarms" cannot'+
								   ' be adjusted in size.')
			
							
		
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
		
	def close(self,event=None):
		'''
		Close toplevel
		'''
		scatterPlots = self.plotter.get_scatter_plots()
		for scatterPlot in scatterPlots.values():
			scatterPlot.update_size(self.size)
		self.toplevel.destroy() 
		
			
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Size configuration') 
         
		popup.protocol("WM_DELETE_WINDOW", self.close)
		popup.bind('<Escape>',self.close)
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
		
             label_des = tk.Label(cont, text = 'Change size of marker:', **titleLabelProperties)
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
		       
         elif plot_type == 'PCA':
         	
         	axes = [ax for n,ax in enumerate(self.figure.axes) if n in [0,2]]
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
         	try:
         		val = round(float(val),0)
         	except:
         		tk.messagebox.showinfo('Error..','Could not convert input to float.')
         		return
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
