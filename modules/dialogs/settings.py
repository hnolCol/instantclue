import tkinter as tk
from tkinter import ttk    

from collections import OrderedDict

from modules.utils import *
from modules.dialogs.VerticalScrolledFrame import VerticalScrolledFrame

hierarchClustering = OrderedDict([('Metric [row]',['None']+pdist_metric),
								  ('Linkage [row]',['None']+linkage_methods),
								  ('Metric [column]',['None']+pdist_metric),
								  ('Linkage [column]',['None']+linkage_methods),
								  ('Row Cluster Color','colorSchemes'),
								  ('Extra Data Color','colorSchemes')]) #

generalSettings = OrderedDict([
							   ('Error bar',['Confidence Interval (95%)','Standard deviation'])])

abbrError = {'Confidence Interval (95%)':95,
			'Standard deviation':'sd'}

dimensionalReduction = OrderedDict([('Components',5)])


allSettings = OrderedDict([('Dimensional Reduction',dimensionalReduction),
						   ('General Settings',generalSettings),
						   ('Hierarchical Clustering',hierarchClustering)])

attrNames = {'Row Cluster Color':'cmapRowDendrogram',
			'Extra Data Color':'cmapColorColumn',
			'Metric [row]':'metricRow',
			'Metric [column]':'metricColumn',
			'Linkage [row]':'methodRow',
			'Linkage [column]':'methodColumn'}


tooltipText = {'Error bar':'Define how the error bars in point- and bar-plots should be calculated. Providing a number between 0-100 will update the confidence interval size.',
			   'Components':'Number of components calculated in dimensional reduction.',
			   'Extra Data Color':'Color of the color map that can represent additional numerical or categorical values that are not used for clustering.',
			   'Row Cluster Color':'Color map for cluster labeling.',
			   'Metric [row]':'Metric to be used for clustering rows in data frame. If you select None no clustering will be performed.',
			   'Metric [col]':'Metric to be used for clustering columns in data frame. If you select None no clustering will be performed.',
			   }



class settingsDialog(object):
	
	def __init__(self,plotter,colorHelper,dimRedCollection):
		
		self.plotter = plotter
		self.dimRedCollection = dimRedCollection
		self.colorPalettes = colorHelper.get_all_color_palettes()
		
		self.define_variables()
		self.build_toplevel()
		self.build_widgets()
	
		
	def close(self, event = None):
		'''
		Close toplevel
		'''
		self.toplevel.destroy() 
		
			
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Settings') 
		popup.bind('<Escape>', self.close) 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=450
		h=500
		self.toplevel = popup
		self.center_popup((w,h))
		
	
	def build_widgets(self):
		'''
		Defines and grids widgets on toplevel
		'''
		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		labelTitle = tk.Label(self.cont, text = 'Main settings\n', **titleLabelProperties)
		labelType = tk.Label(self.cont, text = 'Type: ', bg=MAC_GREY)
		comboSettingType = ttk.Combobox(self.cont, textvariable = self.type, 
							values = list(allSettings.keys())) 
		comboSettingType.bind('<<ComboboxSelected>>', self.refresh_settings)

		labelTitle.grid()
		labelType.grid(row=1,column=0,padx=2)
		comboSettingType.grid(row=1,column=1,padx=2,sticky=tk.EW)	
		ttk.Separator(self.cont, orient = tk.HORIZONTAL).grid(row=2,sticky=tk.EW,columnspan=2,pady=5)
		
		self.contSettings = tk.Frame(self.cont)
		self.contSettings.grid(row=5,sticky=tk.NSEW, columnspan=2, pady=5)
		self.cont.grid_rowconfigure(5,weight=1)
		self.cont.grid_columnconfigure(1,weight=1)
		self.create_setting_options()
		
		applyButton = ttk.Button(self.cont,text='Save & Close', command = self.change_settings)
		closeButton = ttk.Button(self.cont, text = 'Discard & Close', command = self.close)
		
		applyButton.grid(row=6,column=0, padx=5,pady=4,sticky=tk.W)
		closeButton.grid(row=6,column=1, padx=5,pady=4,sticky=tk.E)
				
		
	def create_setting_options(self):
		'''
		'''	
		type = self.type.get()
		if type not in allSettings:
			return	
		self.settingFrame = VerticalScrolledFrame(self.contSettings)
		self.settingFrame.pack(expand=True,fill=tk.BOTH)
		self.build_type_specific_widgets(type)
		
	
	
	def build_type_specific_widgets(self,type):
		'''
		'''
		self.settingsVar = dict()
		n = 0
		for option, values in allSettings[type].items():
			valueList = self.get_values(values)
			var = tk.StringVar()
			label = tk.Label(self.settingFrame.interior, 
				text = '{}: '.format(option),
				bg = MAC_GREY)
			if option in tooltipText:
				CreateToolTip(label,text = tooltipText[option])
			combo = ttk.Combobox(self.settingFrame.interior, textvariable=var,
				values = valueList)
			var.set(self.set_default_value(option))
			label.grid(row=n,column=0,sticky=tk.E)
			combo.grid(row=n,column=1,padx=5,sticky=tk.EW)
			
			self.settingsVar[option] = var
			n+=1
			
		self.settingFrame.interior.grid_columnconfigure(1,weight=1, minsize=100)
		self.settingFrame.interior.grid_columnconfigure(0,weight=1, minsize=150)
		
	def refresh_settings(self,event = None):
		'''
		'''
		clear_frame(self.contSettings)
		self.create_setting_options()
		
				
	def get_values(self,option):
		'''
		This functions checks for strings. 
		And if there is a string (colorScheme) it will return
		the current set of palettes. We need to check this before, because 
		users can add new user defined color palettes.
		'''
		if isinstance(option,str):
			return self.colorPalettes 
		else:
			return option	
	
	def define_variables(self):
		'''
		'''
		self.type = tk.StringVar(value='Hierarchical Clustering')	
	
	def set_default_value(self, option):
		'''
		'''
		if self.type.get() == 'Hierarchical Clustering':
			return getattr(self.plotter,attrNames[option])
		
		elif self.type.get() == 'Dimensional Reduction':
			if option == 'Components':
				return self.dimRedCollection.nComponents	
		elif self.type.get() == 'General Settings':
			if option == 'Error bar':
				return self.plotter.errorBar
				

	def change_settings(self):
		'''
		'''
		type = self.type.get()
			 	
		if type == 'Hierarchical Clustering':
			for key,var in self.settingsVar.items():
				setattr(self.plotter,attrNames[key],var.get())		
		
		elif type == 'General Settings':
		
			self.error_bar_adjustment()
		
		elif type == 'Dimensional Reduction':
		
			ncomps = self.get_number_from_string(self.settingsVar['Components'].get())
			if ncomps is None:
				return
						
			self.dimRedCollection.set_max_comps(ncomps)	
		
		self.close()
	
	
	def get_number_from_string(self,input, type = 'int', addErrorString = ''):
		'''
		'''
		try:
			if type == 'int':
				out = int(float(input))
			else:
				out = float(input)
			return out
		except:
			tk.messagebox.showinfo('Error ..',
					'Could not convert input to float/integer.' + addErrorString,
					parent=self.toplevel)


	def error_bar_adjustment(self):
		'''
		'''
		input = self.settingsVar['Error bar'].get()
		if input in abbrError:
			self.plotter.errorBar = abbrError[input]
		
		else:
				
			ciValue = self.get_number_from_string(input, type='int',
					addErrorString = 'Valid inputs are (0-100] or any option from the drop-down menu.')
			
			if ciValue is None:
					return
					
			if ciValue <= 100:
					self.plotter.errorBar = ciValue
		
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry('%dx%d+%d+%d' % (size + (x, y))) 
         			    	
	
			
