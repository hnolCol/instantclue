import tkinter as tk
from tkinter import ttk   
          
import tkinter.simpledialog as ts
from tkinter.colorchooser import *

import seaborn as sns
from matplotlib.colors import ListedColormap
import husl
import numpy as np
import pandas as pd

import pickle
import os
from modules.utils import *
from modules.pandastable import core


#Instant Clue Specific color maps

Set4 = ['#D1D3D4','#6D6E71','#EE3124','#FCB74A','#2E5FAC','#9BD5F4',
          '#068135','#91CA65']
Set5 = ['#7F3F98','#2E5FAC','#27AAE1','#9BD5F4','#017789','#00A14B',
          '#91CA65','#ACD9B2','#FFDE17','#FCB74A','#F26942',
           '#EE3124','#BE1E2D']

## standard repartoire of seabrn/matplotlib color palettes in Instant Clue                                 

oneColorSeq = ['Greys','Blues','Greens','Purples','Reds']
twoColorSeq = ['BuGn','PuBu','PuBuGn','BuPu','OrRd']
diverging = ['BrBG','PuOr','Spectral','RdBu','RdYlBu','RdYlGn']
qualitative = ['Accent','Dark2','Paired','Pastel1','Pastel2','Set1','Set2','Set3']



class colorMapHelper(object):
	'''
	'''
	def __init__(self):
	
		self.path = os.path.join(path_file,'Data','colors','customPalettes.pkl')
		self.customColors = OrderedDict()
		self.add_pre_defined_cols()
		self.open_custom_colors()


	def get_all_color_palettes(self):
		'''
		'''
		colorPalettes = []
		for key,palette in self.preDefinedColors.items():
			colorPalettes.extend(palette)
		return colorPalettes

	
	def add_pre_defined_cols(self):
		'''
		'''
		self.preDefinedColors = OrderedDict([('1-Color-Sequential',oneColorSeq),
											('2-Color-Sequential',twoColorSeq),
											('Diverging',diverging),
											('Qualitative',qualitative)])
	def reset_default(self):
		'''
		Reset the default. 
		'''
		self.customColors['Set4'] = Set4
		self.customColors['Set5'] = Set5
		self.save_custom_colors()
	
	def add_palette(self,palette):
		'''
		Adds the provided palette and saves it. 
		'''
		nCustom = len(self.customColors)-1
		key = 'Custom {}'.format(nCustom)
		self.customColors[key] = palette
		sns.palettes.SEABORN_PALETTES[key] = palette
		self.preDefinedColors['Qualitative (Custom)'].append(key)
		self.save_custom_colors()
		
	def add_gradient(self,cmap,type='1-Color-Sequential'):
		'''
		Adds provided gradient (matplotlib cmap object) and
		saves it with a key that indicates what type of gradient it is. 
		'''
		nCustom = len(self.customColors)-1
		key = '{}GRADIENT_Custom {}'.format(type,nCustom)
		cmapName = key.split('GRADIENT_')[-1]
		matplotlib.cm.register_cmap(name = cmapName,cmap = cmap)
		self.customColors[key] = cmap
		self.preDefinedColors[type].append(cmapName)
		self.save_custom_colors()
		
	def delete_palette(self,key):
		'''
		'''
		for type,palettes in self.preDefinedColors.items():
			if key in palettes:
				idx = palettes.index(key)
				del palettes[idx]
				self.preDefinedColors[type] = palettes
				break
		if type == 'Qualitative (Custom)':
			del self.customColors[key]
		else:
			keyInDict = '{}GRADIENT_{}'.format(type,key)	
			del self.customColors[keyInDict]
			
		self.save_custom_colors()
		return type
					
	def open_custom_colors(self):
		'''
		Opens the file in which the custom colors are stored. 
		If we cannot find the file. Create a new one.
		'''
		if os.path.exists(self.path) == False:
			self.reset_default()
			self.save_custom_colors()
			
		with open(self.path,'rb') as file:
			self.customColors = pickle.load(file)
		
		colorNames = []
		for key,colorList in self.customColors.items():
			if 'GRADIENT' in key:
				keySplit = key.split('GRADIENT_')
				matplotlib.cm.register_cmap(name=keySplit[-1], cmap=colorList)
				self.preDefinedColors[keySplit[0]].append(keySplit[-1])
			else:
				sns.palettes.SEABORN_PALETTES[key] = colorList
				colorNames.append(key)
		
		self.preDefinedColors['Qualitative (Custom)'] = colorNames
	
	def save_custom_colors(self):
		'''
		Pickles the modified dict containing the colors
		(customColors).
		'''
		with open(self.path,'wb') as file:
			pickle.dump(self.customColors, file)


class colorChooseDialog(object):

	def __init__(self,colorHelper, analyzeClass, cmap = 'Blues', alpha = 0.75):
		
		self.colorHelper = colorHelper
		self.selectedCmap = cmap
		self.alpha = tk.StringVar(value=round(float(alpha),2))
		self.analyze = analyzeClass
		
		
		
		self.build_toplevel()
		self.build_widgets()
		self.define_menu()
		
		self.frame = tk.Frame(self.toplevel)
		
	
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
		popup.wm_title('Color Configuration') 
		popup.protocol('WM_DELETE_WINDOW', self.close)
		
		w = 790
		h = 375
		
		self.toplevel = popup
		self.center_popup((w,h))
	
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		settingVertFrames = dict(relief = tk.GROOVE, padx = 5, pady= 10, bg = MAC_GREY)
 		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont.grid_rowconfigure(3,weight=1)
 		
 		labTitle  = tk.Label(self.cont, text = 'Choose Color Palette', **titleLabelProperties)
 		labTitle.grid(row=0,pady=5,column=0,sticky=tk.W,padx=5)
 		
 		customColorButton = ttk.Button(self.cont, text = 'New', 
 			command = self.define_custom_palette,width=5)
		
 		
 		customColorButton.grid(row=0,pady=5,column=4,sticky=tk.E,padx=5)
 		self.cbs = dict()
 		self.frames = dict()
 		colNum = 0
 		for paletteType,names in self.colorHelper.preDefinedColors.items():
 			labelFrame = tk.Label(self.cont,text=paletteType,**titleLabelProperties)
 			labelFrame.grid(row=2,column=colNum, sticky=tk.W,pady=10,padx=3)
 			labFrame = tk.Frame(self.cont, bg=MAC_GREY)#text = paletteType,**settingVertFrames)
 			labFrame.grid(row=3,column=colNum,padx=2,pady=2,sticky=tk.N)
 			self.frames[paletteType] = labFrame
 			colNum += 1

 			for n,name in enumerate(names):
 			
 				self.add_color_labels(name,labFrame,n)
 				self.add_cb_and_label(name,labFrame,n)
 		
 		alphaFrame = tk.Frame(self.cont,bg=MAC_GREY)
 		alphaFrame.grid(columnspan=2,pady=10,padx=3,sticky=tk.W)
 		labAlpha = tk.Label(alphaFrame, text = 'Transparency: ', bg = MAC_GREY)
 		entryAlpha = ttk.Entry(alphaFrame,width=5,textvariable=self.alpha)
 		slider = ttk.Scale(alphaFrame,from_ = 0.0, to = 1.0,
 			value=float(self.alpha.get()), command = self.change_transparency)
 		
 		labAlpha.grid(row=0,column=0)
 		entryAlpha.grid(row=0,column=1)
 		slider.grid(padx=2,pady=2,columnspan=2,sticky=tk.EW)		 		


	
	def add_cb_and_label(self,name,labFrame,n):

 		var = tk.BooleanVar(value=False)
 		cb = ttk.Checkbutton(labFrame, onvalue=1, offvalue=0)
 		cb.grid(row=n, column = 5)
 		cb.configure(command = lambda cmap=name: self.initiate_change(name))
 		cb.state(['!alternate'])
 		if name == self.selectedCmap:
 			cb.state(['selected'])
 		else:
 			cb.state(['!selected']) 			
 		self.cbs[name] = cb
 		lab = tk.Label(labFrame, text=name,bg=MAC_GREY)
 		lab.grid(row=n, column = 6,sticky=tk.W)		
 		if 'Custom' in name:
 			lab.bind(right_click,self.cast_menu)
	
	
	def initiate_change(self,colormap):
		'''
		'''
		for name,cb in self.cbs.items():
			if name != colormap:
				cb.state(['!selected'])
		self.analyze.check_button_handling(colormap)
	
	
			
	def add_color_labels(self,name,frame,row):
		'''
		Adds the a label representing colors from the palette.
		'''
		repColors = sns.color_palette(name,5,desat=0.8)
		for n,color in enumerate(repColors):
			hex = col_c(color)
			
			colLabel = tk.Label(frame, text = '   ', bg=hex,	
								borderwidth=.3, relief='solid')
								
			colLabel.bind('<Button-1>', lambda event, widget = colLabel: \
			self.analyze.change_default_color(widget,event=event))				 
				
			CreateToolTip(colLabel,title_= name, text = hex,
								showcolors=True,
								cm=name,waittime=600,
								master = self.toplevel)
			colLabel.grid(row=row,column=n)
				
	def change_transparency(self,value=None):
		'''
		Changes the transparancy of collections for certain plot types.
		'''
		if value is not None:
			self.alpha.set(round(float(value),2))
		value = float(value)
		self.analyze.alpha_selected.set(value) 
		
		axes = self.analyze.plt.get_axes_of_figure()
		if len(axes) == 0:
			return
			
		plotType = self.analyze.plt.currentPlotType
		if plotType in ['scatter','PCA','swarm','scatter_matrix','cluster_analysis']:
			for ax in axes:
				axColl = ax.collections
				for coll in axColl:
					if hasattr(coll,'set_sizes'): # otherwise it is not a useful collection	
						coll.set_alpha(value)
		
			self.analyze.plt.redraw()
			self.analyze.plt.set_scatter_point_properties(alpha=value)				

						
	def define_custom_palette(self):
		'''
		Opens dialog to open custom made colors.
		'''
		defineColorDialog = createColorMapsDialog(self.colorHelper)	
		if hasattr(defineColorDialog, 'customPaletteType'):
			key = defineColorDialog.customPaletteType
			self.reshow_color_palettes(key)
			
			
	def reshow_color_palettes(self,key):
				
		frame = self.frames[key]
		for widget in frame.winfo_children():
				widget.destroy()
		names = self.colorHelper.preDefinedColors[key]
		for n,name in enumerate(names):
			self.add_color_labels(name,frame,n)
			self.add_cb_and_label(name,frame,n)
	
	
	def define_menu(self):
 
 		'''
 		'''
 		self.menu = tk.Menu(self.toplevel, **styleDict)
 		self.menu.add_command(label = 'Delete Palette', command = self.delete_palette)
     	
	def delete_palette(self):
 		'''
 		'''
 		paletteName = self.widgetClicked.cget('text')
 		type = self.colorHelper.delete_palette(paletteName)
 		self.reshow_color_palettes(type)
 		    	
     												
	def cast_menu(self,event):
		'''
		'''
		self.widgetClicked = event.widget
		x = self.toplevel.winfo_pointerx()
		y = self.toplevel.winfo_pointery()
		self.menu.post(x,y)		
				
	 			
	def center_popup(self,size):
         	'''
         	Casts popup and centers in screen mid
         	'''
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry('%dx%d+%d+%d' % (size + (x, y))) 










class createColorMapsDialog(object):
	'''
	Dialog to create custom color palettes that are:
		a) Qualitative 
		b) 1-Color Sequential or
		c) Diverging
	'''
	def __init__(self, customColorHelper):
	
		
		self.type = tk.StringVar(value='Qualitative')
		self.center = tk.StringVar(value='light')
		self.color = tk.StringVar(value = 'stormy blue')
		self.firstColor = None
		self.slVars = dict(s=tk.StringVar(value='75'),l=tk.StringVar(value='50'))
		self.customColorHelper = customColorHelper
		self.bars = OrderedDict()
		self.saveColors = OrderedDict()
		self.build_toplevel() 
		self.build_widgets()
		self.add_bindings()
		self.toplevel.wait_window() 
		
	def close(self):
		'''
		Close toplevel
		'''
		self.disconnect_bindings()
		self.toplevel.destroy() 
			
	
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Custom Color Palette') 
         
		popup.protocol('WM_DELETE_WINDOW', self.close)
		w = 400
		h = 440
		
		self.toplevel = popup
		self.center_popup((w,h))
		
			
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont.grid_rowconfigure(5,weight=1)
 		self.cont.grid_columnconfigure(1,weight=1)
 		
 		figureCont = tk.Frame(self.cont, background = MAC_GREY)
 		figureCont.grid(row=5,sticky=tk.NSEW,padx=4,pady=5,columnspan=4)

 		labTitle = tk.Label(self.cont, text = 'Custom color palette/map', **titleLabelProperties)
 		labType = tk.Label(self.cont, text  = 'Type: ', bg = MAC_GREY)
 		labCenter = tk.Label(self.cont, text  = 'Center/Low values: ', bg = MAC_GREY)
 		labChoose = tk.Label(self.cont, text  = 'Choose Color: ', bg = MAC_GREY)
 		CreateToolTip(labChoose,title_='Choose wisely',
 				text='Be creative. All xlcd color names and more are accepted.\nhttps://xkcd.com/color/rgb.txt',
 				waittime=900)
 		self.labColorRep = tk.Label(self.cont, text = '     ', bg = '#507b9c',
 			borderwidth=1, relief='solid')
 			
 		addButton = ttk.Button(self.cont, text = 'Add', command = self.add_color, width=5)
 		comboBoxType = ttk.Combobox(self.cont, textvariable = self.type, 
 			values = ['Gradient (sequential)','Gradient (diverging)','Qualitative'])
 		comboBoxType.bind('<<ComboboxSelected>>',self.comboBoxClear)
 		comboBoxCenter = ttk.Combobox(self.cont, textvariable = self.center, 
 			values = ['light','dark'])
 		comboBoxCenter.bind('<<ComboboxSelected>>',self.comboBoxCenterChange)
 			
 		satLightFrame = tk.Frame(self.cont, bg=MAC_GREY)
 		# check input of entry
 		vcmd = (self.toplevel.register(validate_float),
                '%a','%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
 		for n,label in enumerate(['s','l']):
 			lab = tk.Label(satLightFrame,text=label.upper()+': ',bg=MAC_GREY)
 			lab.grid(row=0,column = n * 2)
 			ent = ttk.Entry(satLightFrame,width=4, textvariable = self.slVars[label],
 										validate = 'key', validatecommand = vcmd) 						
 			ent.grid(row=0,column = n * 2 +1)
 			ent.bind('<Return>',self.update_hex_by_husl)
 			CreateToolTip(lab,title_='Husl color code settings',text='Setting for saturation (S)'+
 					' and Lightness (L). Can be in range of [0,100]. Note that for diverging color palettes the '+
 					'saturation and lightness values will be used for both. Otherwise the'+
 					' color levels cannot be compared easily by the human eye.',
 					)
 		
 		CreateToolTip(labType,title_='Type',text='Palette - select multiple colors that'+
 				' will be used in given order.\nDiverging Palette - Choose two colors and if center'+
 				' should be dark or light')
		
 		entryColor = ttk.Entry(self.cont, textvariable = self.color)
 		entryColor.bind('<Return>',self.check_color)
 		
 		self.labColorRep.bind('<Button-1>',self.modify_color)
 		self.labColorRep.bind(right_click,self.modify_color)

 		self.display_figure(figureCont)
 		self.show_axis()
 		
 		doneButton = ttk.Button(self.cont, text='Done', command = self.assemble_palette,width=5)
 		clearButton = ttk.Button(self.cont, text='Clear', command = self.clear_figure,width=5)
 		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close,width=5)
 		
 		labTitle.grid(row=0, padx=5,pady=15, columnspan=6 ,sticky=tk.W)
 		labType.grid(row=1,padx=3,pady=5,sticky=tk.E)
 		comboBoxType.grid(row=1,column=1,sticky=tk.EW,columnspan=3,padx=(0,4))
 		
 		labCenter.grid(row=2,padx=3,pady=5,sticky=tk.E)
 		comboBoxCenter.grid(row=2,column=1,sticky=tk.EW,columnspan=1)
 		satLightFrame.grid(row=2,column=2,sticky=tk.EW,columnspan=2,padx=(0,4))
 		
 		labChoose.grid(row=3,padx=3,pady=5,sticky=tk.E)
 		entryColor.grid(row=3,column=1,sticky=tk.EW,columnspan=1)
 		self.labColorRep.grid(row=3,column=2,sticky=tk.EW,columnspan=1,padx=2)
 		addButton.grid(row=3,column=3,sticky=tk.EW,columnspan=1,padx=(0,4))
 		 		
 		doneButton.grid(row=6, column=0,padx=3,pady=8,sticky=tk.W)
 		clearButton.grid(row=6, column=1,padx=3,pady=8,columnspan=2)
 		closeButton.grid(row=6, column=3,padx=3,pady=8,sticky=tk.E)
 		
	
		
	def add_color(self, color = None):
		'''
		'''
		if color is None:
			color = self.labColorRep.cget('bg')
		
		if self.type.get() == 'Qualitative':		
			if color in list(self.saveColors.values()):
				tk.messagebox.showinfo('Alread iny ..','Color already in '+
					'palette. You can delete colors '+
					'by right-clicking on bars. To change '+
					'the psoition of a color move the bars'+
					' to the correct position in the graph.',
					parent=self.toplevel)
				return
			for n in range(len(self.bars)+5):
				if n not in self.saveColors and n in self.bars:
					break
				elif n not in self.bars:
					self.add_rectangle(n)
					break					
			self.bars[n].set_facecolor(color)
			self.saveColors[n] = color
			self.figure.canvas.draw()
		
		elif 'Gradient' in self.type.get():
			if 'sequential' in self.type.get():
				self.get_cmap_by_color(color)
				self.color_bars_by_gradient()
			else:
				if self.firstColor is None:
					self.firstColor = color
					self.firstColorSaved = color
					self.get_cmap_by_color(color,reverse=True)
					self.color_bars_by_gradient(how='half')
				else:
					col1 = husl.hex_to_husl(self.firstColor)
					col2 = husl.hex_to_husl(color)						
					s,l = self.get_sl_from_entry()
					if s is None:
						return
						
					cmapOutput = sns.diverging_palette(col1[0],col2[0], s = s, l = l,
									as_cmap = True,center=self.center.get())
					self.cmap = matplotlib.cm.get_cmap(cmapOutput)
					
					self.firstColor = None
					self.color_bars_by_gradient()
	
	def get_sl_from_entry(self):
		'''
		'''
		try:
			s,l = float(self.slVars['s'].get()),float(self.slVars['l'].get())
			return s,l
		except:
			tk.messagebox.showinfo('Error..',
											'Could not convert input for L or S as float.',
											parent=self.toplevel)
			return None,None
		
					
	def get_cmap_by_color(self,color,reverse=False):
		'''
		'''
		if self.center.get() == 'light':
					cmapOutput = sns.light_palette(color, as_cmap = True, reverse=reverse)
					self.cmap = matplotlib.cm.get_cmap(cmapOutput)	
							
		elif self.center.get() == 'dark':
			
					cmapOutput = sns.dark_palette(color, as_cmap = True,reverse=reverse)
					self.cmap = matplotlib.cm.get_cmap(cmapOutput)			
			
		
	def color_bars_by_gradient(self,how='full'):
		'''
		'''
		nBars = len(self.bars)
		colorRatio = np.linspace(0,1,num=nBars,endpoint=True)
		n=0
		for id,bar in self.bars.items():
			if n > len(self.bars)/2 and how=='half':
				bar.set_facecolor('white')
				n+=1
				continue
			color = self.cmap(colorRatio[n])
			bar.set_facecolor(color)
			n+=1
			
		self.figure.canvas.draw()

	def update_hex_by_husl(self,event):	
		'''
		'''
		color = self.labColorRep.cget('bg')
		huslColorHue = husl.hex_to_husl(color)[0]
		s,l = self.get_sl_from_entry()
		if s is None:
			return
		hex = husl.husl_to_hex(huslColorHue,s,l)
		self.labColorRep.configure(bg=hex)
		paletteType = self.type.get()
		if  paletteType != 'Qualitative':
			if 'diverging' in paletteType:
				self.firstColor = str(self.firstColorSaved)
				self.add_color()
			else:
				self.add_color()	
									
	def modify_color(self,event):
		'''
		'''
		color =  askcolor(parent=self.toplevel)
		if color is None:
			return
		else:
			self.update_sl(color[1])
			self.labColorRep.configure(bg=color[1])
			self.add_color()	

	def update_sl(self,color):
		'''
		'''
		if color is not None:
			huslColor = husl.hex_to_husl(color)
			self.slVars['s'].set(huslColor[1])
			self.slVars['l'].set(huslColor[2])
			
				
	def check_color(self,event):
		'''
		Evaluate entry input and add to selection. 
		'''
		hex_ = col_c(self.color.get())
		if hex_ == 'error':
			tk.messagebox.showinfo('Error..','Could not interpret color input.',parent=self.toplevel)
			return
		if isinstance(hex_,tuple):
			hex_ = col_c(hex_)
		self.update_sl(hex_)
		self.labColorRep.configure(bg=hex_)
		self.add_color()
				        
	def add_bindings(self):
		'''
		'''
		self.onPress = self.figure.canvas.mpl_connect('button_press_event',self.identify_rect)
		self.onRelease = self.figure.canvas.mpl_connect('button_release_event',self.on_release)

	def disconnect_bindings(self):
		
		self.figure.canvas.mpl_disconnect(self.onPress)
		self.figure.canvas.mpl_disconnect(self.onRelease)
		self.disconnect_motion()

	def disconnect_motion(self):
		'''
		'''
		if hasattr(self,'onMotionEvent'):
			self.figure.canvas.mpl_disconnect(self.onMotionEvent)
		
	def identify_rect(self,event):
		'''
		'''
		
		for id,rect in self.bars.items():
			if rect.contains(event)[0]:
				
				if event.button == 1:
					width = rect.get_width()
					self.rect = rect
					self.rect.set_zorder(10)
					self.onMotionEvent = self.figure.canvas.mpl_connect('motion_notify_event', \
					lambda event, width=width: self.move_rectangle(event,width))
					
				elif event.button != 1:
				
					if id in self.saveColors:
						rect.set_facecolor('white')
						del self.saveColors[id]
						self.figure.canvas.draw()
	
	def on_release(self,event = None):
		'''
		'''
		if event.button == 1:
			self.reorder_rects()
			self.disconnect_motion()
			if hasattr(self,'rect'):
				self.rect.set_zorder(5)
			self.figure.canvas.draw()
					
	def move_rectangle(self,event,width):
		'''
		'''
		
		if event.inaxes is None:
			self.on_release()
			return
				
		x = event.xdata	
		xLeft = x-width/2	
		self.rect.set_x(xLeft)
		
		self.figure.canvas.draw()		

	def reorder_rects(self):
		'''
		'''
		collectX = []
		collectRects = OrderedDict()
		collectColors = OrderedDict() 
		for id,rect in self.bars.items():
			collectX.append((id,rect.get_x()))
		sortedX = sorted(collectX , key = lambda tup: tup[1])
		width = rect.get_width()
		for n,tup in enumerate(sortedX):
			self.bars[tup[0]].set_x(n-width/2)
			collectRects[n] = self.bars[tup[0]]
			if tup[0] in self.saveColors:
				collectColors[n] = self.saveColors[tup[0]]
		self.bars = collectRects
		self.saveColors = collectColors	
		
		
	def add_rectangle(self,id):
		'''
		'''
		newYBars = min_x_square(len(self.bars)+1)
		newBar = self.ax.bar(id,newYBars[-1], color = 'white', edgecolor='darkgrey',linewidth=0.72)
		for n,bar in enumerate(self.bars.values()):
			bar.set_height(newYBars[n])
		self.bars[id] = newBar[0] 
		self.ax.set_xlim(-0.7,id+0.7)
             
	def show_axis(self):
		'''
		'''
		self.ax = self.figure.add_subplot(111)
		y_bar = min_x_square(8)		
		bars = self.ax.bar(range(8), y_bar, color = 'white', edgecolor='darkgrey',linewidth=0.72)
		for n,patch in enumerate(bars):
			self.bars[n] = patch
		
		self.ax.set_ylim(-0.1,max(y_bar)+0.5)
		self.ax.set_xlim(-0.7,8.6)
		self.ax.axhline(0, color = 'darkgrey', linewidth=0.72)
		self.ax.axvline(-0.6,color='darkgrey',linewidth=0.72)
		plt.axis('off')
		

	def assemble_palette(self):
		'''
		'''
		if self.type.get() == 'Qualitative':
			palette = sns.color_palette(list(self.saveColors.values()))
			self.customColorHelper.add_palette(palette)
			self.customPaletteType = 'Qualitative (Custom)'
		else:
			if 'sequential' in self.type.get():
				self.customColorHelper.add_gradient(self.cmap)
				self.customPaletteType = '1-Color-Sequential'
			else:
				self.customColorHelper.add_gradient(self.cmap,'Diverging')
				self.customPaletteType = 'Diverging'
		self.close()			 
	
	def comboBoxCenterChange(self,event):
		'''
		'''
		if self.type.get() != 'Qualitative':
			self.add_color()

	def comboBoxClear(self,event):
		'''
		'''
		self.bars.clear()
		self.saveColors.clear() 
		self.figure.clf()
		self.show_axis() 
		self.figure.canvas.draw()  
						   	
		
	        
	def clear_figure(self):
		'''
		'''
		quest = tk.messagebox.askquestion('Confirm ..',
			'This will remove added unsaved colors. Proceed?',parent=self.toplevel)
		if quest == 'yes': 
			self.bars.clear()
			self.saveColors.clear() 
			self.figure.clf()
			self.show_axis() 
			self.figure.canvas.draw()     	
        	
	def display_figure(self, frameFigure):
	
		self.figure = plt.figure()      
		self.figure.subplots_adjust(top=.95, bottom=.05,left=.05,
									right=.95, wspace = 0, hspace=0)
		canvas  = FigureCanvasTkAgg(self.figure,frameFigure)
		canvas.show() 
		canvas._tkcanvas.pack(in_=frameFigure,side='top',fill='both',expand=True)
		canvas.get_tk_widget().pack(in_=frameFigure,side='top',fill='both',expand=True)


	

	
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry('%dx%d+%d+%d' % (size + (x, y))) 
         			
         			

def get_max_colors_from_pallete(cmapName):

    if cmapName in ['Greys','Blues','Greens','Purples','Reds','BuGn','PuBu','PuBuGn',
    				'BuPu','OrRd','BrBG','PuOr','Spectral','RdBu','RdYlBu','RdYlGn']:
        n = 60
    elif cmapName in ['Accent','Dark2','Pastel2','Set2','Set4']:
        n=8
    elif cmapName in ['Paired','Set3']:
        n = 12
    elif cmapName in ['Pastel1','Set1']:
        n = 9
    elif cmapName == 'Set5':
        n = 13
    
    cmap = ListedColormap(sns.color_palette(cmapName, n_colors = n ,desat = 0.85))  
    
    return cmap            
