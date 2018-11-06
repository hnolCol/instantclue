import tkinter as tk
from tkinter import ttk  
import tkinter.font as tkFont
           

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import sklearn.cluster
from itertools import chain

import seaborn as sns


from modules.utils import *
from modules import images
from modules.pandastable import core 
				


class defineGroupsDialog(object):
	'''	
	'''
	def __init__(self,dimRedCollection,plotter,colorHelper):
		'''
		=====
		
		Define groups for Score plot after dimensional reduction technique has been applied.
		Groups can either be infered by clustering technique, modified individual cells, 
		or by copy pase (Ctrl-v, command-v)
		
		Input
		-----
			- dimRedCollection - Class type. Saved dimensional reduction results
			- plotter		   - Class type. Instant Clue plotter.
			- colorHelper	   - Class type. Color palettes stored.
		Result
		------
		
		Dimensional reduction Score plot is colored according to the defined groups.
		
		=====
		'''
		
		self.dimRedCollection = dimRedCollection
		self.plotter = plotter
		self.colorSchmes = colorHelper.get_all_color_palettes()
		
		result = dimRedCollection.get_last_calculation()
		
		self.colorMap = tk.StringVar(value = 'Blues')
		
		self.data = result['data']['Components'].T
		self.method = result['method']
		self.prepare_data(result)		
		self.build_toplevel()
		self.build_widgets()
		self.display_data(self.data)
		
		self.toplevel.wait_window()
		
	def close(self, event = None):
		'''
		Close toplevel
		'''
		if hasattr(self, 'pt'):
			del self.pt
		self.toplevel.destroy() 	
		

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Define Groups') 
		popup.grab_set() 
		popup.bind('<Escape>', self.close) 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=520
		h=630
		self.toplevel = popup
		self.center_popup((w,h))
		
			
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont_widgets = tk.Frame(self.cont,background=MAC_GREY) 
 		self.cont_widgets.pack(fill=tk.X, anchor = tk.W) 
 		self.cont_widgets.grid_columnconfigure(1,weight=1)
 		self.create_preview_container() 
 		
 		self.doneIcon, self.refreshIcon = images.get_done_refresh_icons()
 		labelTitle = tk.Label(self.cont_widgets, text= 'Define groups to highlight them '+
													'in the projection plot\nNote that you can'+
													' also enter your own color.',  
													**titleLabelProperties)	
														
 		
 		labelMethod = tk.Label(self.cont_widgets, text= 'Components :{}'.format(self.method), 
                                     **titleLabelProperties)
 		labelColorMap = tk.Label(self.cont_widgets,text = 'Color Map:', bg = MAC_GREY)
 		comboboxColorMap = ttk.Combobox(self.cont_widgets, textvariable =self.colorMap,
 											values = self.colorSchmes, width = 10)
 		comboboxColorMap.bind('<<ComboboxSelected>>', self.refresh)
 		## creating buttons for applying 
 		inferButton = ttk.Button(self.cont_widgets, text = 'Infer groups', 
 												command = self.infer_groups)
 												
 		CreateToolTip(inferButton,title_ = 'Infer Grouping',
 					  text = 'Calculates Levenshtein distance and uses affinity'+
 					  ' propagation to group them')
 												
 		applyButton = create_button(self.cont_widgets, 
 									  image=self.doneIcon, 
 									  command = self.apply_coloring)
 									  
 		refreshButton = create_button(self.cont_widgets, 
 									  image = self.refreshIcon, 
 									  command = lambda: self.apply_coloring(close = False))

		## grid widgets
 		labelTitle.grid(row=0,column=0,padx=4, pady = 15, columnspan = 3, sticky=tk.W)
 		labelTitle.grid(row=1,column=0,padx=4, pady = 15, columnspan = 3, sticky=tk.W)
 		
 		labelColorMap.grid(row=2,column=0,padx=4,pady=4,sticky=tk.E)
 		comboboxColorMap.grid(row=2,column=1,padx=4,pady=4,sticky=tk.W)
 		inferButton.grid(row=2,column=1,padx=(15,40),pady=4,sticky=tk.E)
 		
 		applyButton.grid(row=2,column=2,sticky=tk.E,padx=4)
 		refreshButton.grid(row=2,column=2,sticky=tk.E,padx=45)
	
	def display_data(self,df):
		'''
		Displays data in a pandastable. The 
		'''
		self.pt = core.Table(self.cont_preview,
						dataframe = df, 
						showtoolbar=False, 
						showstatusbar=False)
						
		## unbind some events that are not needed
		if platform == 'MAC':			
			self.pt.unbind('<MouseWheel>') # for Mac it looks sometimes buggy
			self.pt.bind('<Command-v>',self.check_clipboard)
		
		self.pt.bind('<Control-v>',self.check_clipboard)
		
		
		self.pt.show()		
				 		
	def check_clipboard(self,event):
		'''
		Copy data for grouping.
		'''
		df = pd.read_clipboard(header=None)
		data = df.values
		try:
			self.data['Group'] = data
		except:
			tk.messagebox.showinfo('Error ..','Paste data do not fit. Pasted data shape was {}'.format(data.shape)+
			' but required shape is ({},1). No header is assumed.'.format(self.data.shape[-1]))
		self.refresh()
		
		
	def create_preview_container(self,sheet = None):
		'''
		Creates preview container for pandastable. 
		'''
		self.cont_preview  = tk.Frame(self.cont,background='white') 
		self.cont_preview.pack(expand=True,fill=tk.BOTH,padx=(1,1))				
	
	def prepare_data(self,result):
		'''
		Checks if user did already annotate groups and restores the previously
		made edits
		'''
		self.data.insert(0,'Features',self.data.index)
		for newCol  in ['Group','Color']:
			key = newCol.lower()
			if key in result:
				data = [result[key][feature] for feature in self.data.index]
			else:
				data = ['' for n in range(len(self.data.index))]
			self.data.insert(0,newCol,data)	
	def refresh(self,event=None):
		'''
		'''
		self.build_colors()
		self.pt.redraw()
		self.apply_coloring(close=False)
		
	def build_colors(self):
		'''
		'''
		uniqueGroups = self.data['Group'].unique()
		rgbColors = sns.color_palette(self.colorMap.get(), uniqueGroups.size, desat = 0.75)
		colorDict = OrderedDict(zip(uniqueGroups ,rgbColors))
		self.data['Color'] = self.data['Group'].apply( lambda x: col_c(colorDict[x]))
			
	
	def infer_groups(self):
		'''
		'''
		inputData = self.data['Features'].values
		## pdist is faster but we do not excpect massive input here
		levSimilarity = -1*np.array([[minimumEditDistance(w1,w2) for w1 in inputData] for w2 in inputData])
		affProp = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
		affProp.fit(levSimilarity.astype('float'))
		self.data.loc[:,'Group'] = affProp.labels_
		self.refresh()
		
	def apply_coloring(self, close = True):
		'''
		'''
		colorDict = dict(zip(self.data['Features'],self.data['Color']))
		proceedBool, badInput,colorDict = self.check_colors(colorDict)
		if proceedBool:
		
			groupDict = dict(zip(self.data['Features'],self.data['Group']))
			self.dimRedCollection.update_dict('color',colorDict)
			self.dimRedCollection.update_dict('group',groupDict)
			calculationResults = self.dimRedCollection.get_last_calculation()
			self.plotter.set_dim_reduction_data(calculationResults)
			self.plotter.nonCategoricalPlotter.update_color_in_projection(calculationResults)
			self.plotter.redraw()
			if close:
				self.close()
		else:
			tk.messagebox.showinfo('Error ..','Could not interpret color input: {}'.format(badInput))
			return

	def check_colors(self,colorDict):
		'''
		col_c function (color_check) returns Error if we could not manage
		to interpret the input
		'''
		for key,color in colorDict.items():	
			
			if '#' not in str(color):
				colHex = col_c(color)
				if '#' not in colHex:
					return False, color, None
				else:
					colorDict[key] = colHex
		return True, None, colorDict 
		  
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 








