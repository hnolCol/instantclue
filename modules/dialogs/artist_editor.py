import six
import numpy as np


import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter.colorchooser import *


import matplotlib.pyplot as plt
from matplotlib import colors

from modules.utils import *
ls = ['solid','dashed','dashdot','dotted']


class artistEditorDialog(object):

	def __init__(self,artist_type,artist,artist_list, plot_type=None, axis = None):
	
		self.toplevel = None
		self.cont = None
		self.color = None
		self.legend = None
		self.looks_like_clust = False
		self.QuadMesh  = False        
		self.collection_list_colors  = None
		self.h = 172
      
		self.lineStyle = tk.StringVar()
		self.l_weight = tk.StringVar() 
		
		self.artist = artist
		self.artist_type = artist_type
		self.artist_list = artist_list
		
		
		self.legend = axis.get_legend()
		self.ax = axis
		self.plot_type = plot_type
		
				
		if self.plot_type == 'barplot' and self.artist_type == 'patches':
			x_pos = [art_.get_x() for art_ in self.artist_list]
			self.artist_list = [x for (y,x) in sorted(zip(x_pos,self.artist_list))]  
            #"""This is needed because the rectangles are plotted like : 0,0 ,1,1 per group e.g. not in order """
                       
		self.get_artist_settings()
		self.build_toplevel()
		self.create_widgets()
		self.toplevel.wait_window()

			
	def build_toplevel(self):
		'''
		Create Toplevel
		'''
		self.toplevel = tk.Toplevel() 
		self.toplevel.wm_title('Artist Editor ...')
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close_toplevel)
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(expand=True, fill='both')
		cont.grid_columnconfigure(5,weight=1)
		self.cont = cont
		w = 190
		self.center_popup((w,self.h))
    	
	
	def create_widgets(self):
		'''
		Creates widgets
		'''
		labTitle = tk.Label(self.cont, text = 'Widget editor', **titleLabelProperties)	
		labTitle.grid(padx=5,pady=15, columnspan=6 ,sticky=tk.W)
		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(column=0,columnspan=6,padx=1,sticky=tk.EW,pady=(0,4))  
        
		if self.collection_list_colors is None:
			color_text = tk.Label(self.cont, text ='Color: ',bg=MAC_GREY)
			color_label = tk.Label(self.cont, text = '        ', bg = self.color)
			color_label.bind('<Button-1>', self.ask_for_new_col)	
		else:
			self.collect_coll_labels = [] 
		
			for i,col in enumerate(self.collection_list_colors):
				txt = tk.Label(self.cont, text='Color #{}: '.format(i+1), bg=MAC_GREY)
				lab = tk.Label(self.cont, text = '        ', bg = col)
				lab.bind('<Button-1>', self.ask_for_new_col)	
				txt.grid(row = i+2, column =0,pady=2,padx=2, sticky=tk.E)
				lab.grid(row=i+2, column =1,pady=2,padx=2,sticky=tk.W)
				self.collect_coll_labels.append(lab_)

		applyButton =  ttk.Button(self.cont, text='Apply', 
			command = self.apply_changes, width = 6)
		discardButton =  ttk.Button(self.cont, text= 'Discard', 
			command = self.close_toplevel,width = 6)

		if (self.looks_like_clust == False  and self.collection_list_colors is None) and self.QuadMesh == False:  
			color_text.grid(row=3, column =0,pady=2,padx=2, sticky=tk.E)
			color_label.grid(row=3, column =1,pady=2,padx=2,sticky=tk.W)
		

		if self.artist_type == 'lines':
			line_style =  ttk.OptionMenu(self.cont,self.lineStyle, self.lineStyle.get(),*ls)
			line_label = tk.Label(self.cont, text = 'Linestyle: ',bg=MAC_GREY)
			line_label .grid(row=4, column =0,pady=2,padx=2, sticky=tk.E)
			line_style.grid(row=4, column =1,pady=2,padx=2,sticky=tk.W)
			
			weight_label = tk.Label(self.cont, text = 'Linewidth: ',bg=MAC_GREY)
			line_weight = ttk.OptionMenu(self.cont, self.l_weight , self.l_weight.get(), *np.arange(0.25,6.25,0.25).tolist()) 
			weight_label.grid(row=5, column =0,pady=2,padx=2, sticky=tk.E)
			line_weight.grid(row=5, column =1,pady=2,padx=2,sticky=tk.W)
			
		elif self.QuadMesh:
			cmap_label = tk.Label(self.cont, text = 'Colormap: ',bg=MAC_GREY)
			cmap_label.grid(row=4, column =0,pady=2,padx=2, sticky=tk.E)
            
			cmap_om = ttk.OptionMenu(self.cont, self.cmap, self.cmap.get(),*col_schemes)
			cmap_om.grid(row=4, column =1,pady=2,padx=2,sticky=tk.W)
		else:
			edgecolor_label = tk.Label(self.cont, text = '        ', bg = self.edge_color)
			edgecolor_label.bind('<Button-1>', lambda event, mode='edge':self.ask_for_new_col(event, mode))	
			line_label = tk.Label(self.cont, text = 'Edgecolor: ',bg=MAC_GREY)
			line_label .grid(row=29, column =0,pady=2,padx=2, sticky=tk.E)
			edgecolor_label.grid(row=29, column =1,pady=2,padx=2,sticky=tk.W)
		
		applyButton.grid(row=30, column =0,pady=2,padx=2) 
		discardButton.grid(row=30, column =1, pady=2,padx=2) 
		
	def get_artist_settings(self):
	
		if self.artist_type in ['patches','artists']:

			col = self.artist.get_facecolor()
			self.col_old = col
			edge_col = self.artist.get_edgecolor()
			self.edge_color = col_c(edge_col)
			self.check_for_same_color = [artist for artist in self.artist_list if artist.get_facecolor() == col and artist != self.artist]
			

		elif self.artist_type in ['lines']:
		
			col = self.artist.get_color()
			self.col_old = col
			weight = self.artist.get_linewidth()	
					
			self.check_for_same_color = [artist for artist in self.artist_list \
			if artist.get_color() == col and artist != self.artist]
			
			ls = self.artist.get_ls() 
			weight = self.artist.get_linewidth() 
			self.lineStyle.set(ls) 
			self.l_weight.set(str(weight)) 
			
		else:
			if 'QuadMesh' in str(self.artist):
				self.QuadMesh = True 
				return                     
 
			col = self.artist.get_facecolor()
			cols =  [tuple(col_c(x.tolist()[0:3])) for x in col]
			uniqueColors = return_unique_list(cols) # collection with multiple vals
		
      
			if col.shape[0] > 1 and len(uniqueColors) > 1 and (self.plot_type == 'pointplot' or self.plot_type == 'PCA'):
				if self.plot_type == 'PCA':
					self.collection_list_colors = cols
				else:
	
					self.collection_list_colors = uniqueColors
				edge_col = self.artist.get_edgecolor()
				edge_col = edge_col.tolist()[0]
				self.edge_color = col_c(edge_col)
				self.h = 175 + 18*len(self.collection_list_colors) 
				return
 			
			if col.size == 0:
				self.looks_like_clust = True
				col = self.artist.get_color()
			col = col.tolist()[0]
			self.col_old = col
			if self.looks_like_clust == False:
			
				self.check_for_same_color = [artist for artist in self.artist_list if artist.get_facecolor().tolist()[0] == col and artist != self.artist]
			else:
				self.check_for_same_color = []
			edge_col = self.artist.get_edgecolor()

			edge_col = edge_col.tolist()[0]
			self.edge_color = col_c(edge_col)			
		col  = col_c(col)
		self.color = col

 
	def ask_for_new_col(self,event, mode='main'):
		'''
		Ask for new color. ask colors return rgb and hex (0,1)
		'''
		widget = event.widget
		new_col = askcolor(color = self.color,parent = self.toplevel)
		if new_col is not None:
			if mode == 'main':
				self.color = new_col[1]
			else:
				self.edge_color = new_col[1]
			widget.configure(bg=new_col[1])
			
			
			
	def apply_changes(self):
		self.change_other_artist_and_legend()
		if self.artist_type == 'lines':
			self.artist.set_color(self.color)
			self.artist.set_ls(self.lineStyle.get())
			self.artist.set_linewidth(float(self.l_weight.get()))
			for art_ in self.check_for_same_color:
				art_.set_color(self.color)
				art_.set_ls(self.lineStyle.get())
				art_.set_linewidth(float(self.l_weight.get()))
				
			
		elif self.QuadMesh:
			cmap = get_cmap.get_max_colors_from_pallete(self.cmap.get())  
            
			self.artist.set_cmap(cmap)
            
		elif self.collection_list_colors is not None:
			
			colors = [widget.cget('background') for widget in self.collect_coll_labels]
			self.artist.set_facecolor(colors)
			self.artist.set_edgecolor(self.edge_color)

	
		elif self.artist_type == 'collections':
			if self.looks_like_clust == False:
				self.artist.set_facecolor(self.color) 
			self.artist.set_edgecolor(self.edge_color)
			
			for art_ in self.check_for_same_color:
				
				art_.set_facecolor(self.color)
				art_.set_edgecolor(self.edge_color)
			
		elif self.artist_type in ['patches','artists']:
			self.artist.set_facecolor(self.color)
			self.artist.set_edgecolor(self.edge_color)
			for art_ in self.check_for_same_color:
				art_.set_facecolor(self.color)
				art_.set_edgecolor(self.edge_color)

		self.close_toplevel() 
					
	
	def change_other_artist_and_legend(self):
		
		if self.legend is not None and (len(self.check_for_same_color) != 0 or self.plot_type in ['density','time_series']):
				if self.artist_type == 'lines' and self.plot_type in ['boxplot','violinplot','barplot']:
					return
				if self.QuadMesh:
					return
				if self.collection_list_colors is not None:
					return                
            
				handles, labels = self.ax.get_legend_handles_labels() 
			
				idx_ = [self.artist_list.index(art) for art in self.artist_list if art in  self.check_for_same_color + [self.artist]]
				
				if self.plot_type == 'density':
					idx_min = int(round(min(idx_)/2,0))
				else:
					idx_min = min(idx_) 	
				if self.plot_type == 'pointplot' and idx_min > len(self.legend.legendHandles):
					idx_min = idx_min - len(idx_) + 1
	
				try:
					self.legend.legendHandles[idx_min].set_facecolor(self.color) 
					self.legend.legendHandles[idx_min].set_edgecolor(self.edge_color)
						
				except:
					self.legend.legendHandles[idx_min].set_color(self.color) 
					if self.artist_type == 'lines':
						self.legend.legendHandles[idx_min].set_linestyle(self.lineStyle.get()) 
						self.legend.legendHandles[idx_min].set_linewidth(self.l_weight.get()) 
				
				return 
				
				
	def center_popup(self,size):
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))
	
	def close_toplevel(self):
		self.toplevel.destroy()
