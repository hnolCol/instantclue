"""
	""CHART CONFIGURATION""
    Instant Clue - Interactive Data Visualization and Analysis.
    Copyright (C) Hendrik Nolte

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""

import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt

from modules.utils import *         
             
class ChartConfigurationPopup(object):
	
	
	def __init__(self,plotter,global_chart_parameter):
				 
		if plotter.plotCount == 0:
			tk.messagebox.showinfo('No chart ..','Please create a chart first ..')
			return	
		self.toplevel = None	
		self.save_entries = dict()
		self.platform = platform
		self.plot_type = plotter.currentPlotType
		self.figure = plotter.figure
		self.fig_axes = plotter.get_axes_of_figure()
		self.show_grid  = plotter.showGrid
		self.show_box = plotter.showSubplotBox
		self.plotter = plotter
		self.global_chart_parameter = global_chart_parameter
		self.fig_axes = self.get_all_axes_to_change()
		self.properties_axis = self.extract_properties_of_first_axis()
		self.build_popup()
		self.add_widgets_to_toplevel()
		
		
		
		self.toplevel.wait_window() 
		
	def close(self, event = None):
		'''
		Closes the toplevel and saves changes made to pyplot global parameters. 
		'''
		self.save_settings()
		self.toplevel.destroy()
		

	def save_settings(self):
		'''
		Store Settings. Settings are stored globally.
		'''
		xy_font = self.global_chart_parameter[0]
		tick_font = self.global_chart_parameter[1]
		legend_font = self.global_chart_parameter[2]
		anno_font = self.global_chart_parameter[3]
		
		plt.rc('legend',**{'fontsize':legend_font})        
		plt.rc('font',size = anno_font)
		plt.rc('axes',titlesize = anno_font)
		plt.rc('axes', labelsize=xy_font)
		plt.rc('xtick', labelsize = tick_font)
		plt.rc('ytick', labelsize= tick_font)

               
			
	def build_popup(self):
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Chart configuration') 
		popup.bind('<Escape>', self.close) 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		if self.platform == "WINDOWS":
			w = 375
		elif self.platform == 'MAC':
			w = 420 
		h=475
		self.toplevel = popup
		self.center_popup((w,h))
		
	def add_widgets_to_toplevel(self):
		'''
		Adds widget to the toplevel.
		Extracting the starting values from the received properties of axis 1 (see function below) 
		'''	
		info = self.properties_axis
		
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(fill='both', expand=True)
		cont.grid_columnconfigure(1,weight=1)
		lab1 = tk.Label(cont, text = 'Change chart settings', **titleLabelProperties)
		lab1.grid(padx=5,pady=15, columnspan=6 ,sticky=tk.W)
		
		settings = ['y-axes [min]:','y-axes [max]:','x-axes [min]:','x-axes [max]:','xy-label font size:',
					'tick font size:','legend font size:','text label font size:']
		for i,sett in enumerate(settings): 
			if i == 0:
				lab_inf = tk.Label(cont, text = 'Adjust sliders\nto change chart\nsettings dynamically:', justify=tk.LEFT, bg = MAC_GREY)
				lab_inf .grid(row=0,padx = 5,rowspan=1,column=2)
			lab_s = tk.Label(cont, text = sett, bg = MAC_GREY)
			lab_s.grid(row=i+1, padx=5,sticky=tk.E,pady=5)
			ent = ttk.Entry(cont, width = 400)
			ent.insert(tk.END, info[i])
			self.save_entries[sett] = ent
			ent.grid(row=i+1,column=1,sticky=tk.W,pady=5)
			if '[max]' in sett or '[min]' in sett:
				if 'x-' in sett:
					delta = info[-2]
				else:
					delta = info[-1]
				if 'max' in sett: 
					from_ = info[i]
					to_ =  info[i] + 6*delta
				else:
					from_ =  info[i] - 6*delta
					to_ = info[i] 
				scale = tk.Scale(cont, from_ = from_ , to = to_, orient = tk.HORIZONTAL,showvalue=0,sliderlength = 20, resolution = 0.000000001 , 
								borderwidth=0.5, width=7, 
								command = lambda value, sort = sett, ent = ent, axes= self.fig_axes: self.change_dynamically(value,sort,ent,self.plot_type,axes))
				scale.set(float(info[i]))  
				scale.grid(row=i+1,column=2,sticky=tk.E,pady=5,padx=3)	
				
			if 	'font size' in sett:
				scale = tk.Scale(cont, from_ = 6, to = 22, orient = tk.HORIZONTAL,showvalue=0,sliderlength = 13, resolution = 1 , borderwidth=0.5, width=7,
                                      command = lambda value, sort = sett, ent = ent, axes=self.fig_axes: self.adjust_font_sizes(value,sort,ent,axes))
				scale.set(int(info[i])) 
				scale.grid(row=i+1,column=2,sticky=tk.E,pady=5,padx=3)                                  		
            
            
            
		for entry in self.save_entries.values(): 
				entry.bind('<Return>', lambda event , axes=self.fig_axes: self.apply_changes(event,self.plot_type,axes))    
		self.cb_grid = ttk.Checkbutton(cont, text = 'Grid', command = self.add_grid_lines_to_plot)
		self.cb_grid.grid(row=i+2, column= 0, columnspan=2,padx=5, sticky=tk.W)
		self.cb_grid.state(['!alternate'])  
		if self.show_grid:
				self.cb_grid.state(['selected'])
		else:
				self.cb_grid.state(['!selected'])
		
		self.cb_box = ttk.Checkbutton(cont, text = 'Show box', command = self.remove_box)
		self.cb_box.grid(row=i+2, column= 1, columnspan=2,padx=5, sticky=tk.W)
		self.cb_box.state(['!alternate'])
		if self.show_box:
				self.cb_box.state(['selected']) 
		else:
				self.cb_box.state(['!selected'])
		but_update = ttk.Button(cont,text = "Update", command = lambda axes=self.fig_axes : self.apply_changes(plot_type = self.plot_type,axes=axes))
		but_close = ttk.Button(cont, text = "Close", command = self.close)
		but_update.grid(row=i+3, column=1,pady=5,padx=5, sticky=tk.E)
		but_close.grid(row=i+3, column = 2 , pady=5, padx=5)	
		
	def add_grid_lines_to_plot(self):
		'''
		Add/Remove gridlines to/from plot.
		'''
		bool_ = self.cb_grid.instate(['selected'])
		if bool_:
			for ax in self.fig_axes:
				ax.grid(color='darkgrey', linewidth = 0.15)
		else:
			for ax in self.fig_axes:
				ax.grid('off')
		self.show_grid  = bool_	
		self.figure.canvas.draw()


	def remove_box(self):
		'''
		Remove box around axis
		'''
	
		bool_ = self.cb_box.instate(['selected'])	
		for ax in self.fig_axes:
			if bool_:
				ax.spines['right'].set_visible(True)
				ax.spines['top'].set_visible(True)
			else:
				ax.spines['right'].set_visible(False)
				ax.spines['top'].set_visible(False) 
		self.show_box = bool_		
		self.figure.canvas.draw()  
			

	def get_all_axes_to_change(self):
		'''
		Filters out axes that should not change: Label axes in scatters with categories 
		or colormap axes.
		Returns new self.fig_axes list
		'''
		fig_axes = self.fig_axes
		ax_cbar, ax_label = [],[]
	
		if self.plotter.categoricalPlotter is not None and self.plot_type == 'scatter':
			ax_label = self.plotter.categoricalPlotter.scatterWithCategories.inmutableAxes
		fig_axes = [ax for ax in fig_axes if ax != ax_cbar and ax not in ax_label]	
		return fig_axes
		
	def extract_properties_of_first_axis(self):
		'''
		Extracts axes limits and labels from axis.
		'''
		if len(self.fig_axes) == 0:
			return None
		for ax in self.fig_axes[:1]:
			ylim, xlim = list(ax.get_ylim()), list(ax.get_xlim())
			#x_label, y_label = ax.get_xlabel() , ax.get_ylabel()
			x_delta, y_delta = xlim[1]-xlim[0], ylim[1]-ylim[0]
			
		combine = ylim + xlim + self.global_chart_parameter + [x_delta,y_delta]
		return combine
		
	def adjust_font_sizes(self,value,sort,ent,axes):

                 '''
                 Adjust font sizes based on slider
                 '''

                 fig = self.figure
                 draw = True
                 fig_axes = axes

                 if 'xy-label' in sort:

                 	
                     if int(self.global_chart_parameter[0]) - int(value) == 0:
                     	
                     	return
                     
                     if int(value) == self.global_chart_parameter[0]:
                         draw = False
                     else:
                         self.global_chart_parameter[0] = str(value)  
                         axis_font = {'size':int(value)}
                         
                         for ax in fig_axes:                            
                             ax.set_xlabel(ax.get_xlabel(), **axis_font)
                             ax.set_ylabel(ax.get_ylabel(), **axis_font)
                                                       
                 elif 'text label' in sort:
                     if int(self.global_chart_parameter[3]) - int(value) == 0:
                     	return
                 
                     self.global_chart_parameter[3] = str(value)
                     
                     for ax in fig.axes:
                             title= ax.get_title()
                             if title != '':
                                 ax.set_title(title,fontsize = int(value))
                             textItems = ax.texts
                             if len(textItems) != 0:
                                 for text in textItems:
                                 	text.set_fontsize(int(float(value)))
                         
                             self.figure.canvas.draw() 
                             
                    
                
                 elif 'tick' in sort:
                 
                     if  int(self.global_chart_parameter[1]) - int(value) == 0: 
                     	return
  
                 
                     
                     if int(value) == self.global_chart_parameter[1]:
                         draw = False
                     else:
                         self.global_chart_parameter[1] = str(value)   
                         for ax in fig_axes:
                             for tick in ax.xaxis.get_major_ticks():
                                 tick.label.set_fontsize(int(value))
                             for tick in ax.yaxis.get_major_ticks():
                                 tick.label.set_fontsize(int(value))    
                 elif 'legend' in sort:
                 
                     if  int(self.global_chart_parameter[2]) - int(value) == 0: 
                     	return 
                 	
                      
                     if int(value) == self.global_chart_parameter[2]:
                         draw = False
                     else:
                         self.global_chart_parameter[2] = str(value)   
                         for ax in fig_axes[:1]:
                             leg = ax.get_legend()
                             if leg is not None:                             
                                 plt.setp(leg.get_title(), fontsize=str(value))       
                                 plt.setp(leg.get_texts(), fontsize = str(value))
                             ax_art = ax.artists
                                    
                             for artist in ax_art:
                                        if str(artist) == 'Legend':
                                            leg = artist 
                                            plt.setp(leg.get_title(), fontsize=str(value))
                                            plt.setp(leg.get_texts(), fontsize = str(value))
                 
                 ent.delete(0,'end')
                 ent.insert(tk.END,value)
                 if draw:
                     self.figure.canvas.draw()   	
             
	def apply_changes(self,event = None,plot_type = None,axes= None):

                 '''
                 Applies porperties via enter and button "update"
                 '''     
                         
                 axis_font = {'size':float(self.save_entries['xy-label font size:'].get())} 	 	
                 for ax in axes:
                 	 xLabel,yLabel = ax.get_xlabel(), ax.get_ylabel()
                 	 if xLabel != '':
                 	 	ax.set_xlabel(xLabel,**axis_font)
                 
                 	 if yLabel != '':
                 	 	ax.set_ylabel(yLabel,**axis_font)                  	 	
                  	                 	
                 	 if plot_type == 'barplot':
                 	 	y_min = 0
                 	 else:
                 	 	y_min = float(self.save_entries['y-axes [min]:'].get())
                 	 ax.set_ylim((y_min,float(self.save_entries['y-axes [max]:'].get())))
                 	 ax.set_xlim((float(self.save_entries['x-axes [min]:'].get()),
                 	 			  float(self.save_entries['x-axes [max]:'].get())))
                 	 
                 	 tickFontSize = int(self.save_entries['tick font size:'].get())
                 	 for tick in ax.xaxis.get_major_ticks():
                 	 	tick.label.set_fontsize(tickFontSize)
                 	 for tick in ax.yaxis.get_major_ticks():
                 	 	tick.label.set_fontsize(tickFontSize)
                 	 	
                 	 if self.cb_box.instate(['selected']):
                    
                            for ax in axes:
                                ax.spines['right'].set_visible(True)
                                ax.spines['top'].set_visible(True)
                 	 else:
                            for ax in axes:
                                ax.spines['right'].set_visible(False)
                                ax.spines['top'].set_visible(False)
                 	 textItems = ax.texts
                 	 if len(textItems) != 0:
                 	 	for txt in textItems:
                 	 		txt.set_fontsize(int(self.save_entries['text label font size:'].get()))
                 leg = ax.get_legend()
                 value = int(self.save_entries['legend font size:'].get())        
                 if leg is not None:                             
                             plt.setp(leg.get_title(), fontsize=str(value))       
                             plt.setp(leg.get_texts(), fontsize = str(value))
                 ax_art = ax.artists
                                
                 for artist in ax_art:
                                    if str(artist) == 'Legend':
                                        leg = artist 
                                        plt.setp(leg.get_title(), fontsize=str(value))
                                        plt.setp(leg.get_texts(), fontsize = str(value))
                 
                 self.figure.canvas.draw() 	

	
		
	def change_dynamically(self,value,sort,ent,plot_type,axes):
		'''
		Apply changes when slider is moved.
		'''
		ent.delete(0,'end')
		ent.insert(tk.END,value)
		fig = self.figure
		value = float(value) 
		fig_axes = self.fig_axes
		if sort == 'y-axes [max]:':   
			for ax in fig_axes:
				ymin,ymax = ax.get_ylim()
				if abs(value - ymax) < ymax*0.001:
					return
				
				ax.set_ylim((ymin,value))
				
			self.figure.canvas.draw()
			return
			
		elif sort == 'y-axes [min]:': 
			for ax in fig_axes:
					ymin,ymax = ax.get_ylim()
					if abs(value - ymin) < ymin * 0.001:
						return
					if plot_type == 'barplot' and ymin == 0:
						ent.delete(0,'end')
						ent.insert(tk.END,0.00)
						return
					ax.set_ylim((value,ymax))
			self.figure.canvas.draw()
			return
		elif sort == 'x-axes [min]:':
			for ax in fig_axes:
				xmin,xmax = ax.get_xlim()
				if abs(value - xmin) < xmin * 0.001:
					return
				ax.set_xlim((value,xmax))
			self.figure.canvas.draw() 
			return 
		elif sort == 'x-axes [max]:':
			for ax in fig_axes:
				xmin,xmax = ax.get_xlim()
				if abs(value - xmax) < xmax* 0.001:
					return 
				ax.set_xlim((xmin,value))
			self.figure.canvas.draw()
			return 
			
	def center_popup(self,size):
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))             
	
	
	
	
        