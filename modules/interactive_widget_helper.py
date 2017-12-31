
import tkinter as tk
from tkinter import ttk
from modules.utils import *
from tkinter.colorchooser import *


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
	def __init__(self, masterFrame):
		'''
		'''
		self.master = masterFrame
		self.create_frame()
		self.colorLevelWidgets = dict()
		self.categoryToNewColor = dict()
		self.selectedColors = dict() 
		self.colorMapDict = dict()
		
	def create_widgets(self, mode = 'colorLevel', plotter = None):
		'''
		'''
		self.mode = mode 
		self.plotter = plotter
		
		
		if self.mode == 'colorLevel':
			self.clear_color_helper_dicts()
			self.colorMapDict = plotter.nonCategoricalPlotter.get_current_colorMapDict()
			self.defaultColor = plotter.colorScatterPoints
			self.defaultColorHex = col_c(self.defaultColor)
			self.color_label_widgets(self.colorMapDict)
			
		elif self.mode == 'sizeLevel':
			pass 
		
	def create_frame(self):
		'''
		'''
		self.frame = tk.Frame(self.master,bg=MAC_GREY, relief=tk.GROOVE)
		self.frame.grid(columnspan=2, sticky=tk.EW, padx=5)
			
	def color_label_widgets(self, colorMapDict):
		
		catColumnList = self.plotter.nonCategoricalPlotter.sizeStatsAndColorChanges['change_color_by_categorical_columns']
		
		self.clean_color_frame_up()
		self.colorframe = tk.Frame(self.frame, bg=MAC_GREY, relief=tk.GROOVE)
		self.colorframe.grid(columnspan=2, sticky=tk.EW)
				
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
				self.categoryToNewColor[group] = colorToBeUsed
				
		self.apply_changes()
	
	def update_new_colorMap(self):
		'''
		'''
		if len(self.colorLevelWidgets) == 0:
			return
			
		self.colorMapDict = self.plotter.nonCategoricalPlotter.get_current_colorMapDict()
		rawColorMapDict = self.plotter.nonCategoricalPlotter.rawColorMapDict 
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

	
		
	def apply_changes(self):
		'''
		'''
		self.plotter.nonCategoricalPlotter.categoricalColorDefinedByUser = self.categoryToNewColor
		self.plotter.nonCategoricalPlotter.update_colorMap()
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
			
		
	def clean_frame_up(self):
		'''
		'''
		widgetsInFrame = self.frame.winfo_children()
		if len(widgetsInFrame) != 0:
			self.frame.destroy()
			self.create_frame()
		
	   