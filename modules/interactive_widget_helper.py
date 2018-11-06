"""
	""INTERACTIVE WIDGET HELPER""
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
from modules.utils import *
from tkinter.colorchooser import *

import pandas as pd
import seaborn as sns

from modules.pandastable import core 
class interactiveWidgetsHelper(object):
	'''
	Interactive Widgets 
	===================
	
	This simply means that the user can interact using these widgets with the program
	that leads automatically to a change in the graph. As an example. 
	If a scatter plot is used to visualize the categorical levels by a color. These 
	colors can be customized by the user. Upon selection of a color, the chart is automatically 
	updated.
	
	===================
	
	Parameters:
		- masterFrame - a frame to build this frame that is used as a container for
		  the 'interactive' widgets. 
	
	===================
	'''	
	def __init__(self, masterFrame, colorHelper = None):
		'''
		'''
		self.master = masterFrame
		self.colorHelper = colorHelper
		self.create_frame()
		self.colorLevelWidgets = dict()
		self.categoryToNewColor = dict()
		self.selectedColors = dict() 
		self.colorMapDict = dict()
		
	def create_widgets(self, mode = 'colorLevel', plotter = None, analyzeData = None ,droppedButton = None):
		'''
		Create tkinter widgets
		'''
		self.mode = mode 
		self.plotter = plotter
		self.analyzeData = analyzeData
		if plotter.currentPlotType == 'line_plot':
			self.helper = plotter.nonCategoricalPlotter.linePlotHelper
		elif plotter.nonCategoricalPlotter is not None:
			self.helper = plotter.nonCategoricalPlotter
		elif plotter.currentPlotType == 'scatter':
			self.helper = plotter.categoricalPlotter.scatterWithCategories
		
		if self.mode == 'colorLevel':
			self.clear_color_helper_dicts()
			self.colorMapDict = self.helper.get_current_colorMapDict()
			if self.colorMapDict is None: return
			
			self.defaultColor = plotter.colorScatterPoints
			self.defaultColorHex = col_c(self.defaultColor)			
			
			droppedButton.config(command = lambda : longLegendColorChanger(self.colorMapDict,self.helper,
										 self.colorHelper,self.plotter, self))			
			
			if len(self.colorMapDict) > 20:
				if droppedButton is not None:
					tk.messagebox.showinfo('Info ..',
						'Too many categorical values to display (>20). No legend drawn.'+
						'\nYou can click on the color  legend icon '+
						'to force a legend and to change colors.')

				return

			self.color_label_widgets(self.colorMapDict)
			
		elif self.mode == 'sizeLevel':
			pass 
		
	def create_frame(self):
		'''
		Creates frame to put labels in indicating color and categorical value
		'''
		self.frame = tk.Frame(self.master,bg=MAC_GREY, relief=tk.GROOVE)
		self.frame.grid(columnspan=2, sticky=tk.EW, padx=5)
			
	def color_label_widgets(self, colorMapDict):
		if len(colorMapDict) > 20:
			return
			
		self.clean_color_frame_up()
		self.colorframe = tk.Frame(self.frame, bg=MAC_GREY, relief=tk.GROOVE)
		self.colorframe.grid(columnspan=2, sticky=tk.EW)			
		
		catColumnList = self.helper.get_size_color_categorical_column()
				
		for column in catColumnList:
			headerLabel = tk.Label(self.colorframe, text = str(column)[:17], bg = MAC_GREY)
			headerLabel.grid(column= 0, padx = 3, sticky=tk.W,pady=1)
			CreateToolTip(headerLabel,title_ = column, text = '')
		
		for group, color in colorMapDict.items():
			if str(group).isspace() == False and str(group) != 'nan':
				hexColorFormat = col_c(color)
				self.create_colorLabel_and_description(hexColorFormat,group)
	
	
	def create_colorLabel_and_description(self, color, group):
		'''
		'''

		colorLabel = tk.Label(self.colorframe, text = '   ', bg = color,
										   height=1, borderwidth=0.5) 
		self.bind_events_to_label(colorLabel)
		
		colorLabel.grid(column= 0, padx = 3, sticky=tk.W,pady=1)
		groupLabel = tk.Label(self.colorframe, text = str(group)[:16], bg = MAC_GREY)
		groupLabel.bind(right_click, lambda event, group = group :self.subset_group(event,group)) 
		CreateToolTip(groupLabel,title_ = group,text = 'The color for each categorical value'+
													   ' can be adjusted:\nYou can either disable'+
													   ' the color highlight (left click)\nOr define'+
													   ' a custom color. (right-click)')
		groupLabel.grid(row = colorLabel.grid_info()['row'], 
						column = 0,padx=(28,0), 
						sticky = tk.W, columnspan=3) 
						
		self.selectedColors[group] = [color,self.defaultColorHex]
		self.colorLevelWidgets[group] = {'colorLabel':colorLabel,
										'groupLabel': groupLabel}
	
	
	def subset_group(self,event,group):
		'''
		'''
		try:
			self.colorMapDict = self.helper.get_current_colorMapDict()
			boolidx = self.helper.data.loc[:,'color'] == self.colorMapDict[group]
			subsetData = self.analyzeData.sourceData.get_data_by_id(\
				self.plotter.get_dataID_used_for_last_chart()).iloc[self.helper.data[boolidx].index]
			
			self.analyzeData.add_new_dataframe(subsetData,'ScatterSubset_color_{}'.format(group))
			tk.messagebox.showinfo('Done..','Subset created. Note that subsetting is done by color.'
			' If the color is not unique for the selected group, results might be unexpected.')
		except:
			tk.messagebox.showinfo('Error..','There was an unknown error.')
	
	def bind_events_to_label(self, widget):
		'''
		'''
		def indicate_if_mouse_over_widget(event, mode = 'Enter'):
			w = event.widget
			if mode == 'Enter':
				w.configure(relief = tk.GROOVE)
			else:
				w.configure(relief = tk.FLAT)
		
		widget.bind('<Enter>', indicate_if_mouse_over_widget)
		widget.bind('<Leave>', lambda event: indicate_if_mouse_over_widget(event,mode='Leave'))
		widget.bind(right_click , self.choose_custom_color) 	
		widget.bind('<Button-1>' , self.set_nan_color)
	
	def choose_custom_color(self,event):
		'''
		'''
		w = event.widget
		for group, labels in self.colorLevelWidgets.items():
			if w == labels['colorLabel']:
				color =  askcolor()
				if color[1] is not None:
					w.configure(bg=color[1]) 
					self.categoryToNewColor[group] = color[1]
					self.selectedColors[group].append(color[1])
					self.apply_changes()
			else:
				pass 
		
	def set_nan_color(self,event):
		'''
		'''
		w = event.widget
		for group, labels in self.colorLevelWidgets.items():
		
			originalColor = col_c(self.colorMapDict[group])

			if w == labels['colorLabel']:
				
				colorInUse = w.cget('bg')
				indexInColorList = self.selectedColors[group].index(colorInUse)
				if indexInColorList + 1 >= len(self.selectedColors[group]):
					indexInColorList = 0
				else:
					indexInColorList += 1
				
				colorToBeUsed = self.selectedColors[group][indexInColorList]
				w.configure(bg = colorToBeUsed)
				if indexInColorList != 0:
					self.categoryToNewColor[group] = colorToBeUsed
				else:
					if group in self.categoryToNewColor:
						del self.categoryToNewColor[group]
				
		self.apply_changes()
	
	def update_new_colorMap(self):
		'''
		'''
		if len(self.colorLevelWidgets) == 0:
			return
			
		self.colorMapDict = self.helper.get_current_colorMapDict()
		if self.colorMapDict is None:
			return
		rawColorMapDict = self.helper.get_raw_colorMapDict()
		# we need the raw colorMap with user's changed to get the correct color
		
		for group, color in rawColorMapDict.items():
			if str(group).isspace() == False and str(group) != 'nan':
				
				hexColor = col_c(color)
				## update color list			
				# get current color (if nan color, it should stay like this)
				colorLabel = self.colorLevelWidgets[group]['colorLabel']
				currentLabelCol = colorLabel.cget('bg')
				indexInColorList = self.selectedColors[group].index(currentLabelCol)
				newColorList = [hexColor]
				for color in self.selectedColors[group][1:]:
					newColorList.append(color)
				self.selectedColors[group] = newColorList
				if indexInColorList == 0:
					colorLabel.configure(bg=hexColor)
				else:
					pass
				# update new standard color in the seelectable colors
				# colors that will show up while clicking throught them
				self.selectedColors[group][0] = hexColor

		
	def apply_changes(self):
		'''
		'''
		self.helper.set_user_def_colors(self.categoryToNewColor)
		self.helper.update_colorMap()
		self.plotter.redraw()
				
	def clear_color_helper_dicts(self):
		'''
		'''
		self.colorLevelWidgets.clear()
		self.categoryToNewColor.clear()
		self.selectedColors.clear()	
		self.colorMapDict.clear()
	
	def clean_color_frame_up(self):
		'''
		'''
		
		if hasattr(self,'colorframe'):
			self.colorframe.destroy()
			del self.colorframe
			self.colorLevelWidgets.clear()
			self.categoryToNewColor.clear()
			self.selectedColors.clear()	
			
		
	def clean_frame_up(self):
		'''
		'''
		widgetsInFrame = self.frame.winfo_children()
		if len(widgetsInFrame) != 0:
			self.frame.destroy()
			self.create_frame()
		
		
			
class longLegendColorChanger(object):
	'''
	'''
	def __init__(self, colorMapDict, plotHelper, colorHelper, plotter, widgetHelper):
		
		self.colorMapDict = colorMapDict
		self.colorHelper = colorHelper
		self.plotter = plotter
		self.plotHelper = plotHelper
		self.widgetHelper = widgetHelper
		self.build_toplevel()
		self.build_widgets()
		self.define_menu()
		self.display_data()


	def apply(self, close = False):
		'''
		Apply changes using the ploter class
		'''	
		
		self.plotHelper.categoricalColorDefinedByUser = self.colorMapDict
		self.plotHelper.update_colorMap()
		self.plotter.redraw()
		self.widgetHelper.color_label_widgets(self.colorMapDict)
		if close:
			self.close()		
		
		
	def get_data(self,colorMapDict):
		'''
		'''	
		df = pd.DataFrame(columns=['Categories','Color (Rgb)','Color (HEX)'])
		keys = list(colorMapDict.keys())
		values = list(colorMapDict.values())

		df.loc[:,'Categories'] = pd.Series(keys, index=range(len(keys)))
		df.loc[:,'Color (Rgb)'] = pd.Series(values, index=range(len(keys)))
		df.loc[:,'Color (HEX)'] = pd.Series([col_c(col) for col in values],index = range(len(keys)))
		return df
 		
	
	def close(self,event = None):
		'''
		Close toplevel
		'''
		self.toplevel.destroy() 	
		

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Legend Handler') 
		popup.grab_set()
		popup.bind('<Escape>', self.close) 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=520
		h=630
		self.toplevel = popup
		self.center_popup((w,h))
		
	def build_widgets(self):	
		'''
		Builds widgets on toplevel
		'''
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		self.cont.grid_columnconfigure(2,weight = 1)
		self.cont.grid_rowconfigure(2,weight =1)
		labelTitle = tk.Label(self.cont, text= 'Double click on item to change color.'+
									 '\nSelect multiple items and choose a color map by right-click menu.', 
                                     wraplength=320,**titleLabelProperties)
 		
		labelTitle.grid(columnspan=2, sticky=tk.W, padx=3,pady=5) 
		self.create_listbox()

		saveButton = ttk.Button(self.cont,text='Apply & Close' ,command = lambda: self.apply(close=True))
		discardButton = ttk.Button(self.cont,text='Discard & Close')
		
		saveButton.grid(sticky=tk.W, padx=4,pady=5)
		discardButton.grid(row=4,column=1,columnspan=2,sticky=tk.E,padx=4,pady=5)


	def create_listbox(self):
 		'''
 		Creates the listbox
 		'''
 		vs = ttk.Scrollbar(self.cont, orient = tk.VERTICAL)
 		hs = ttk.Scrollbar(self.cont, orient = tk.HORIZONTAL)
 		self.listbox = tk.Listbox(self.cont, selectmode = tk.EXTENDED,
 						yscrollcommand = vs.set, xscrollcommand = hs.set)
 		vs.config(command=self.listbox.yview)  
 		hs.config(command=self.listbox.xview)   
 		self.listbox.grid(row=2,column=0,
 						  columnspan=4,sticky=tk.NSEW,
 						  padx = (15,0), pady=(15,0))           
 		vs.grid(row=2,column=4,sticky=tk.NS+tk.W,padx=(0,15),pady=(15,0)) 
 		hs.grid(row=3,column=0,columnspan=4,sticky=tk.EW+tk.N,padx = (15,0),pady=(0,15))
 		self.listbox.bind(right_click,self.post_menu)
 		self.listbox.bind('<Double-Button-1>',self.change_color)
				

	def define_menu(self):
		
		self.menu = tk.Menu(self.toplevel, **styleDict)
		blockMenu = tk.Menu(self.toplevel, **styleDict)
		self.menu.add_cascade(label='Set Color Block', menu = blockMenu)#command = self.get_color_block)
		for paletteType,names in self.colorHelper.preDefinedColors.items():
			subMenu = tk.Menu(self.toplevel, **styleDict)
			blockMenu.add_cascade(label = paletteType, menu = subMenu)
			for name in names:
				subMenu.add_command(label=name, command = lambda colorMap = name: self.get_color_block(colorMap))
				
	def post_menu(self,event):
		'''
		'''
		x = self.toplevel.winfo_pointerx()
		y = self.toplevel.winfo_pointery()
		self.menu.post(x,y)
		
     	
	def get_color_block(self, colorMap):
		'''
		Change multiple selection
		'''
		#colorMap = 'Greens'
		items = self.listbox.curselection()	
		if len(items) == 0:
			return
		colors = sns.color_palette(colorMap,len(items))
		
		for n,n_item in enumerate(items):
			
			self.colorMapDict[self.items[n_item]] = colors[n]
			self.listbox.itemconfig(n_item,bg=col_c(colors[n]))
		self.deselect_items()
			
		
	def change_color(self,event):
		'''
		Change Color by double click on item.
		'''
		items = self.listbox.curselection()	
		n = items[0]
		color = askcolor()
		if color[1] is not None:
			self.listbox.itemconfig(n,bg=color[1])
			self.colorMapDict[self.items[n]] = tuple([col/256 for col in color[0]])
		self.deselect_items()
		
	def deselect_items(self):
		'''
		Remove selection from all items. This is used to make the color change
		visible for users.
		'''
		
		self.listbox.selection_clear(0,tk.END)
			

	def display_data(self):
		'''
		'''
		self.items = []
		
		for item,color in self.colorMapDict.items():
			self.items.append(item)
			self.listbox.insert('end',item)
			self.listbox.itemconfig('end',bg = col_c(color))		
					
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 		
		
		
	
		
		
		
		
		
		
		
		
		




			

		
		
	   