	
import tkinter as tk
import tkinter.font as tkFont

from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
import tkinter.filedialog as tf
from tkinter.colorchooser import *

from modules import images
from modules.utils import * 




default = {'font':'Arial','size':12,'weight':'normal','style':'normal',
			'rotation':'0','color':'black','ha':'left'}


class textEditorDialog(object):

	def __init__(self, inputText = None, props = None, extraCB = {}):
		'''
		extraCB - dict. Can handle addition checkbox widgets creation. key = key, 
					items = text,initial Value (example extraCB = {ticks:['Apply to all x-ticks',True]}
		'''
		self.styleDict	= None
		self.extraCbs = extraCB
		if props is None:
			self.itemProps = default
		else:
			self.itemProps = props
			self.evaluate_itemProps()		
		
		if inputText is None:
			self.inputText = 'Enter text here ..'
		else:
			self.inputText = inputText
			
		self.get_images()
		self.build_toplevel()
		self.build_widgets()
		
		self.toplevel.wait_window()
				
		
	def close(self):
		'''
		closing the toplevel
		'''
		
		self.toplevel.destroy() 
		
		
	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Text editor ...') 
         
		popup.protocol("WM_DELETE_WINDOW", self.close)
		
		w = 620
		h = 380
             
		self.toplevel = popup
		self.center_popup((w,h))
		
	def build_widgets(self):
		'''
		'''
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		self.cont.grid_columnconfigure(0,weight=1)
		
		labelTitle = tk.Label(self.cont, text= 'Text Editor', 
 		
                                     **titleLabelProperties)
		
		m = 0
		for k,textAndBool in self.extraCbs.items():
			var = tk.BooleanVar(value=textAndBool[-1])
			cb = ttk.Checkbutton(self.cont, text = textAndBool[0], variable = var)
			cb.grid(row=m,column=1, sticky=tk.E,columnspan=9,padx=6)
			self.extraCbs[k] = var
			m+=1
					
		
		
		self.comboSettings = self.combobox_setup()
		n=0
		for key, settings in self.comboSettings.items():

			combo = ttk.Combobox(self.cont,**settings)
			combo.grid(row=m+1,column=n,sticky=tk.EW,pady=3,padx=1)
			combo.bind('<<ComboboxSelected>>',self.update_text_window)
			n+=1
		
		colorLabel = tk.Label(self.cont, text = '     ', bg = self.itemProps['color'])
		colorLabel.grid(row=m+1,column=n,sticky=tk.W,pady=3,padx=3)
		colorLabel.bind('<Button-1>',self.choose_color)
		colorLabel.bind(right_click,self.choose_color)
		n+=1
		self.haAligns = dict()
		for key,im in zip(['left','center','right'],[self.left_align,self.center_align,self.right_align]):
			lab = tk.Label(self.cont,image=im,bg=MAC_GREY,relief=tk.FLAT)
			lab.grid(row=m+1,column=n,sticky=tk.W,pady=3)
			self.haAligns[key] = lab
			lab.bind('<Button-1>',self.adjust_ha_setting)
			n+=1
		self.adjust_ha_setting()
		
		self.txtWindow = tk.Text(self.cont, undo=True, background='white')
		self.txtWindow.insert(1.0,self.inputText)
		self.update_text_window()
		self.txtWindow.grid(row=m+3,column=0,columnspan=10,pady=(2,2),padx=1,sticky=tk.NSEW)
		self.cont.grid_rowconfigure(m+3,weight=1)
		
		applyButton = ttk.Button(self.cont, text = 'Apply', command =self.prepare_text_props)
		applyButton.grid(row=m+4,column=0,columnspan=10,padx=1,sticky=tk.EW)
		labelTitle.grid(row=0, column=0, padx=3,pady=5,columnspan=2,sticky=tk.W)
		self.toplevel.bind('<Shift-Return>', self.prepare_text_props)
	
	def evaluate_itemProps(self):
		'''
		'''
		for key,item in default.items():
			if key not in self.itemProps:
				self.itemProps[key] = default[key]
		
	def choose_color(self,event):
		'''
		'''
		w = event.widget
		color = askcolor(color=w.cget('bg'),parent=self.toplevel)
		if color[1] is not None:
			w.configure(bg=color[1])
			self.txtWindow.configure(fg=color[1])
			self.itemProps['color'] = color[1]
	
	def check_variant(self,input=None):
		'''
		'''
		variant = self.comboSettings['variant']['textvariable'].get()
		if  variant != 'None':
			if input is None:
				inputText = self.txtWindow.get('1.0',tk.END)
			else:
				inputText = input
			if variant == 'Title':
				textOut = inputText.title()
			elif variant == 'UPPER':
				textOut = inputText.upper()
			elif variant == 'lower case':
				textOut = inputText.lower()
			if input is None:
				self.txtWindow.delete('1.0',tk.END)
				self.txtWindow.insert(tk.END,textOut)
			else:
				return textOut
		else:
			return
		
			
	def update_text_window(self,event=None):
		'''
		'''
		style = self.comboSettings['style']['textvariable'].get()
		if style not in ['roman','italic']:
			style = 'roman'
			
		self.check_variant()
		## matplotlib has much more setting for this option than the text widget from tkinter - careful!
		weight = self.comboSettings['weight']['textvariable'].get()
		if weight not in ['normal','bold']:
			if weight in self.comboSettings['weight']['values']:
				idx = self.comboSettings['weight']['values'].index(weight)
				if idx > 2:
					weight = 'bold'
				else:
					weight = 'normal'
			else:
				weight = 'normal'
				
		self.txtWindow.configure(fg = self.itemProps['color'],
						font = tkFont.Font(family = self.comboSettings['font']['textvariable'].get(), 
		 				size = int(float(self.comboSettings['size']['textvariable'].get()))+2,
		 				weight = weight, 
		 				slant = style))
		 
	
	
	def adjust_ha_setting(self,event = None):
		'''
		'''
		if event is None:
			self.haAligns[self.itemProps['ha']].configure(relief=tk.SUNKEN)
		else:
			w = event.widget
			for key, label in self.haAligns.items():
				if label == w:
					w.configure(relief=tk.SUNKEN)
					self.itemProps['ha'] = key
				else:
					label.configure(relief=tk.FLAT)
		
	
		 				
	def prepare_text_props(self,event=None):
		'''
		Prepare text props and other settings
		'''
		self.styleDict = self.define_style_dict()
		self.close()
	
	
	
					
	def define_style_dict(self):
		'''
		'''
		weight = self.comboSettings['weight']['textvariable'].get()
		size = self.comboSettings['size']['textvariable'].get()
		try:
			sizeFont = float(size)
		except:
			tk.messagebox.showinfo('Error..','Could not interpret size input. Please provide a number.', 
					parent=self.toplevel)
			
		if weight in self.comboSettings['weight']['values']:
			weightFont = self.comboSettings['weight']['textvariable'].get()
		else:
			try:
				weightFont = int(float(weight))
			except:
				tk.messagebox.showinfo('Error..','Could not interpret weight input.', 
					parent=self.toplevel)
				return
		
		style = {'s':self.txtWindow.get('1.0',tk.END).strip(),
				'fontdict' : {'weight':weightFont,
							 'size':sizeFont,
							 'style':self.comboSettings['style']['textvariable'].get(),
							 'family':self.comboSettings['font']['textvariable'].get(),
							 'color':self.itemProps['color'],
							 'ha':self.itemProps['ha'],
							 'rotation':self.comboSettings['rotation']['textvariable'].get().split(' ')[0]}}
		return style 


	def get_variant(self):
		'''
		'''
		return self.comboSettings['variant']['textvariable'].get()
	
	def get_results(self):
		'''
		'''
		if len(self.extraCbs) > 0:
			for k,var in self.extraCbs.items():
				self.extraCbs[k] = var.get()
				
		return self.styleDict, self.extraCbs
	
	def combobox_setup(self):
		'''
		'''
		font = {'textvariable':tk.StringVar(value=self.itemProps['font']),
				'width':15,
				'values':['Verdana','Arial','Calibri','Cambria','Courier New','Corbel','Helvetica','Magneto','Times New Roman']}
		size = {'textvariable':tk.StringVar(value=self.itemProps['size']),
				'values':list(range(3,40)),
				'width':4}
		weight = {'textvariable':tk.StringVar(value=self.itemProps['weight']),
				'values':['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'],
				'width':7}
		style = {'textvariable':tk.StringVar(value=self.itemProps['style']),
				'values':['normal', 'italic', 'oblique'],
				'width':7}
		rotation = {'textvariable':tk.StringVar(value=self.itemProps['rotation']),
				'values':[str(x)+' °' for x in range(0,360,15)],
				'width':5}	
		variant = {'textvariable':tk.StringVar(value='None'),
				'values':['None','Title','UPPER','lower case'],
				'width':6}
		comboSettings = OrderedDict([('font',font),('size',size),('weight',weight),
							('style',style),('rotation',rotation),('variant',variant)])
	
		return comboSettings



	def get_images(self):
		'''
		'''
		self.back_icon, self.center_align,self.left_align,self.right_align, \
           					self.config_plot_icon, self.config_plot_icon_norm = images.get_utility_icons()           
		
	def center_popup(self,size):
         	'''
         	Casts the popup in center of screen
         	'''

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))	


