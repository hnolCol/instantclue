import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from collections import OrderedDict
import itertools
import matplotlib.patches as patches

	
class scatter_with_categories(object):


	def __init__(self,data,n_cols,n_categories,colnames,catnames,figure,size,color):
	
		self.grouped_data = None
		self.grouped_keys = None
		
		self.unique_values = OrderedDict() 	
		self.axes = OrderedDict() 
		self.label_axes = OrderedDict() 
		self.axes_combs = OrderedDict()
		self.subsets_and_scatter = OrderedDict() 	
		
		self.data = data.dropna(subset=colnames)
		self.n_cols = n_cols
		self.n_categories = n_categories
		self.colnames = colnames
		self.catnames = catnames
		self.size = size
		self.color = color
		self.figure = figure
		plt.figure(self.figure.number)
		
		self.get_unique_values() 
				
		self.group_data()
		
		n_rows,n_cols = self.calculate_grid_subplot()
		self.prepare_plotting(n_rows,n_cols) 
		
	
	
	def prepare_plotting(self,n_rows,n_cols):
		'''
		Function to plot different groups ...
		'''	
		
		self.figure.subplots_adjust(wspace=0, hspace=0, right=0.96)

		titles = list(self.unique_values[self.catnames[0]][0])
		if self.n_cols == 1:
			# Plots the numeric column against index 
			min_y = self.data[self.colnames[0]].min()
			max_y = self.data[self.colnames[0]].max()
			
		elif self.n_cols == 2:
			# Plot numeric column against numeric column if there are two columns
			min_x, max_x = self.data[self.colnames[0]].min(), self.data[self.colnames[0]].max()
			min_y, max_y = self.data[self.colnames[1]].min(), self.data[self.colnames[1]].max()
			xlim = 	(min_x - 0.1*min_x, max_x + 0.1*max_x )	
		ylim = (min_y - 0.1*min_y, max_y + 0.1*max_y )			
			
			
		if self.n_categories  > 1:
		
			y_labels = list(self.unique_values[self.catnames[1]][0])
			
		if self.n_categories == 3:
		
			levels_3,n_levels_3 = self.unique_values[self.catnames[2]]
			outer = gridspec.GridSpec(n_levels_3, 1, hspace=0.01)
			gs_saved = dict() 
			for n in range(n_levels_3):			
				gs_ = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec = outer[n], hspace=0.0)
				gs_saved[n] = gs_
			
		for i,comb in enumerate(self.all_combinations):
			
			if comb in self.grouped_keys:
				group = self.grouped_data.get_group(comb)

				if self.n_cols == 1:
					n_data_group = len(group.index)
					x_ = range(0,n_data_group) 
					y_ = group[self.colnames[0]]	
					
				
				else:			
					x_ = group[self.colnames[0]]
					y_ = group[self.colnames[1]]

			else:
				### '''This is to plot nothing if category is not in data'''
				group = None
				x_ = []
				y_ = []

			
			pos = self.get_position_of_subplot(comb) 
			
			if self.n_categories < 3:
				n = 0
				ax_ = plt.subplot2grid((n_rows, n_cols), pos)
				
			else:
				n = levels_3.index(comb[2])
				ax_ = self.create_ax_from_grid_spec(comb,pos,n, gs_saved)	
				
			scat = ax_.scatter(x_,y_)#self.color) 
			#ax_.plot(x_,y_,'o',color = self.color,ms = np.sqrt(self.size),
					#markeredgecolor ='black',markeredgewidth =0.3)#,linestyle=None)
					
					
			if ax_.is_last_row() == False and self.n_categories < 3 and self.n_cols > 1:
			
				ax_.set_xticklabels([])
				 
			elif self.n_categories == 3 and self.n_cols > 1:
				if n != n_levels_3-1:
					ax_.set_xticklabels([])
				else:
					if ax_.is_last_row() == False:
						ax_.set_xticklabels([])
			else:
				ax_.set_xticklabels([])

			if ax_.is_last_col():			
				
				ax_.yaxis.tick_right()
				
			else:
			
				ax_.set_yticklabels([])	
			
			ax_.set_ylim(ylim)
			if self.n_cols == 2:
				ax_.set_xlim(xlim)
			self.axes[i] = ax_
			self.axes_combs[comb] = [ax_,scat]
		self.add_labels_to_figure()	
		

	
	def create_ax_from_grid_spec(self,comb,pos, n, gs_saved):
		'''
		Gets the appropiate axis from selected gridspec- In principle we have gridspecs in one big gridspec, 
		this allows hspacing between certain categories on y axis.
		'''
		
		
		gs_ = gs_saved[n]	
		ax = plt.subplot(gs_[pos])

		return ax
				
	
	def add_labels_to_figure(self):
		'''
		Adds labels to figure  - still ugly 
		To DO. make this function cleaner. 
		'''
		
		levels_1,n_levels_1 = self.unique_values[self.catnames[0]]
		self.figure.subplots_adjust(left=0.15, bottom=0.14) 
		
		ax_top = self.figure.add_axes([0.15,0.89,0.81,0.15])
		ax_top.set_ylim((0,4))
		ax_top.axis('off') 
		width_for_rect = 1/n_levels_1
		kwargs_rectangle_main = dict(edgecolor='black',clip_on=False,linewidth=0.1,fill=True)
		kwargs_rectangle = dict(edgecolor='black',clip_on=False,linewidth=0.1,fill=False)
		ax_top.add_patch(patches.Rectangle((0,1),1,1,**kwargs_rectangle_main))
		ax_top.text(0.5, 1.5 , s = self.catnames[0], horizontalalignment='center',verticalalignment = 'center',color="white")
		
		for n,level in enumerate(levels_1):
			
			x = 0 + n * width_for_rect
			y = 0
			width = width_for_rect
			height = 1 
			ax_top.add_patch(patches.Rectangle((x,y),width,height,**kwargs_rectangle))
			ax_top.text(x + width/2 , height/2, s = level, horizontalalignment='center',verticalalignment = 'center')
			
		self.label_axes['top'] = ax_top
		if self.n_categories > 1:
		
			ax_left = self.figure.add_axes([0.02,0.14,0.1,0.74])
			ax_left.axis('off')
			ax_left.set_xlim((0,4)) 
			ax_left.add_patch(patches.Rectangle((2,0),1,1,**kwargs_rectangle_main))
			ax_left.text(2.5, 0.5 , s = self.catnames[1], verticalalignment='center', rotation=90,horizontalalignment='center',color="white")
			levels_2,n_levels_2 = self.unique_values[self.catnames[1]]
			if self.n_categories == 3:
				levels_3,n_levels_3 = self.unique_values[self.catnames[2]]
				n_levels_2 = n_levels_2 * n_levels_3
				levels_2 = levels_2 * n_levels_3
				
			height_for_rect = 1/n_levels_2
			for n,level in enumerate(levels_2):
				
				y = 1 - (n+1) * height_for_rect
				x = 3
				width = 1
				height = height_for_rect 
				
				ax_left.add_patch(patches.Rectangle((x,y),width,height,**kwargs_rectangle))
				ax_left.text(x + width/2 , y + height/2, s = level, verticalalignment='center', rotation=90,horizontalalignment='center')
			if self.n_categories == 3:
			
				
				ax_left.add_patch(patches.Rectangle((0,0),1,1,**kwargs_rectangle_main))
				ax_left.text(0.5, 0.5 , s = self.catnames[2], verticalalignment='center', rotation=90,
									horizontalalignment='center',color="white")
				height_for_rect = 1/n_levels_3
				for n,level in enumerate(levels_3):
					y = 1 - (n+1) * height_for_rect
					x = 1
					height = height_for_rect
					ax_left.add_patch(patches.Rectangle((x,y),width,height,**kwargs_rectangle))					
					ax_left.text(x + width/2 , y + height/2, s = level, verticalalignment='center', rotation=90,horizontalalignment='center')
			
			self.label_axes['left'] = ax_left
		
		
		
	
	def get_position_of_subplot(self, comb):
		'''
		Returns the position of the specific combination of levels in categorical columns. 
		Seems a bit complicated but is needed if a certain combination is missing.
		'''
		levels_1, n_levels_1 = self.unique_values[self.catnames[0]]
		if self.n_categories == 1:
			row = 0
			col = levels_1.index(comb)
		else:
			levels_2,n_levels_2 = self.unique_values[self.catnames[1]]
			col = levels_1.index(comb[0])
			row = levels_2.index(comb[1])
			
			
		return (row,col)
	
	
	
	def calculate_grid_subplot(self):
		'''
		Calculates the subplots to display data
		'''
	
		## get columns of n_cat 1 
		if self.n_categories == 1:
		
			levels_1, n_levels = self.unique_values[self.catnames[0]]
			n_cols = n_levels
			n_rows = 1 
			self.all_combinations = list(levels_1)
			
		elif self.n_categories == 2:
		
			levels_1, n_levels_1 = self.unique_values[self.catnames[0]]
			n_cols = n_levels_1
			levels_2, n_levels_2 = self.unique_values[self.catnames[1]]
			n_rows = n_levels_2
			self.all_combinations = list(itertools.product(levels_1,levels_2))
				
		elif self.n_categories == 3:
		
			levels_1, n_levels_1 = self.unique_values[self.catnames[0]]
			n_cols = n_levels_1
			levels_2, n_levels_2 = self.unique_values[self.catnames[1]]
			n_rows = n_levels_2
			levels_3, n_levels_3 = self.unique_values[self.catnames[2]]
			self.all_combinations = list(itertools.product(levels_1,levels_2,levels_3))	
			
		return n_rows, n_cols	
			
	
	def get_unique_values(self):
		'''
		Determines unique vlaues in each category, that is needed to build the subplots
		'''
		for category in self.catnames:
			
			uniq_levels = self.data[category].unique()
			n_levels = uniq_levels.size
			
			self.unique_values[category] = [list(uniq_levels),n_levels]
			
		
	def group_data(self):
		'''
		Returns a pandas groupby object with grouped data on selected categories.
		'''
		
		self.grouped_data = self.data.groupby(self.catnames, sort = False) 
		self.grouped_keys = self.grouped_data.groups.keys()
		
		
		
	def get_subsets_and_scatters(self):	
		'''
		Return a dictionary that can be used to change color settings and to add a 
		new color category. Also (in the future) to annotate points. 
		'''
		
		for comb in self.all_combinations:
		
			if comb in self.grouped_keys:
				
				subset = self.grouped_data.get_group(comb)
				ax_,scat = self.axes_combs[comb]
			else:
				subset = None
				ax_,scat = self.axes_combs[comb]
				
				
			self.subsets_and_scatter[comb] = [subset,ax_,scat]
	
		return self.subsets_and_scatter
		
	def return_label_axes(self):
		'''
		Return labels axes. We need these to avoid applying ylim and xlim changes.
		'''	
		return self.label_axes
	def return_axes(self):
		'''
		Return the axis. This is useful since achieving similiar styling the axes are needed.
		'''
		return self.axes