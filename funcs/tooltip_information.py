# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 19:29:51 2017

@author: https://stackoverflow.com/questions/3221956/what-is-the-simplest-way-to-make-tooltips-in-tkinter
THANKS TO THE AUTHOR ## modofied to add a title Label and show colors
"""
import tkinter as tk
import matplotlib
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
def min_x_square(cm):
	n = np.linspace(-np.pi,np.pi/2, num=cm)
	
	out = [np.cos(x)+1.4 for x in n]
	return out

class CreateToolTip(object):

    def __init__(self, widget,
                 *,
                 title_ = None,
                 bg='white',
                 pad=(5, 3, 5, 3),
                 text= None,
                 waittime=300,
                 wraplength=250, platform = 'WINDOWS',
                 showcolors = False,cm = None):

        self.waittime = waittime  # in miliseconds, originally 500
        self.wraplength = wraplength  # in pixels, originally 180
        self.widget = widget
        self.text = text
        self.title = title_
        self.widget.bind("<Enter>", self.onEnter)
        self.widget.bind("<Leave>", self.onLeave)
        self.widget.bind("<ButtonPress>", self.onLeave)
        self.bg = bg
        self.pad = pad
        self.id = None
        self.tw = None
        self.plat = platform
        self.showcolors = showcolors
        self.cm = cm
        self.f_tw  = None
        self.canvas = None

    def onEnter(self, event=None):
        self.schedule()

    def onLeave(self, event=None):
    	if self.showcolors == True:
    		if self.f_tw  is not None and self.canvas is not None:
    			self.f_tw.clear()
    			plt.close(self.f_tw)
    			try:
    				self.canvas.get_tk_widget().delete("all")
    				self.canvas.get_tk_widget().destroy()
    			except:
    				pass 	
    			
    	self.unschedule()
    	self.hide()
    		
       
    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.show)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def show(self):
        def tip_pos_calculator(widget, label,
                               *,
                               tip_delta=(10, 5), pad=(5, 3, 5, 3)):

            w = widget

            s_width, s_height = w.winfo_screenwidth(), w.winfo_screenheight()

            width, height = (pad[0] + label.winfo_reqwidth() + pad[2],
                             pad[1] + label.winfo_reqheight() + pad[3])

            mouse_x, mouse_y = w.winfo_pointerxy()

            x1, y1 = mouse_x + tip_delta[0], mouse_y + tip_delta[1]
            x2, y2 = x1 + width, y1 + height

            x_delta = x2 - s_width
            if x_delta < 0:
                x_delta = 0
            y_delta = y2 - s_height
            if y_delta < 0:
                y_delta = 0

            offscreen = (x_delta, y_delta) != (0, 0)

            if offscreen:

                if x_delta:
                    x1 = mouse_x - tip_delta[0] - width

                if y_delta:
                    y1 = mouse_y - tip_delta[1] - height

            offscreen_again = y1 < 0  # out on the top

            if offscreen_again:
                y1 = 0

            return x1, y1

        bg = self.bg
        pad = self.pad
        widget = self.widget
        self.tw = tk.Toplevel(widget)
        

        if self.plat == 'WINDOWS':
        # Leaves only the label and removes the app window
            self.tw.wm_overrideredirect(True)
            font_size = 9
        else:
            self.tw.tk.call("::tk::unsupported::MacWindowStyle","style",self.tw._w, "plain", "none")
            font_size = 11
        win = tk.Frame(self.tw,
                       background=bg,
                       relief=tk.GROOVE,
                       )
        if self.text is not None:
        	title_label = tk.Label(win, text = self.title,background=bg,
                               justify =tk.LEFT,
                               font = ("Helvetica", font_size,'bold'))
        	label = tk.Label(win,
                          text=self.text,
                          justify=tk.LEFT,
                          background=bg,
                          relief=tk.SOLID,
                          borderwidth=0.0,
                          wraplength=self.wraplength,
                          font = ("Helvetica", font_size))
        	title_label.grid(sticky = tk.W) 
        	label.grid(padx=(pad[0], pad[2]),
                   pady=(pad[1], pad[3]),
                   sticky=tk.W)
        if self.showcolors == True:
                
        	self.f_tw = plt.figure(figsize = (4.0,1.2), facecolor='white')
        	self.f_tw.subplots_adjust(wspace=0.0, hspace=0.0,
        							 right = 1, top = 1, left = 0.0, bottom = 0.0)
        	ax = self.f_tw.add_subplot(111)
        	cm_ = sns.color_palette(self.cm,15)
        	cm = []
        	for color in cm_:
        		if color not in cm:
        			cm.append(color)
        	y_bar = min_x_square(len(cm)) 

        	ax.bar(range(len(cm)),y_bar, color= cm, edgecolor="darkgrey",linewidth=0.72)
        	ax.set_ylim(-0.1,max(y_bar)+0.5)
        	ax.set_xlim(-0.7,len(cm))
        	ax.axhline(0, color = "darkgrey", linewidth=0.72)
        	ax.axvline(-0.6,color="darkgrey",linewidth=0.72)
        	self.canvas = FigureCanvasTkAgg(self.f_tw,win)
        	plt.axis('off')
        	self.canvas._tkcanvas.grid(in_=win, sticky = tk.W,padx=10,pady=10)
        	self.canvas.get_tk_widget().grid(in_=win, sticky = tk.W,padx=10,pady=10)
        	if self.cm not in ['Accent','Pastel1','Pastel2','Set1','Set3','Spectra','RdYlGn']:
        		col_blind = 'True'
        	else:
        		col_blind = 'False'
        	if self.cm in ["Greys","Blues","Greens","Purples",'Reds',"BuGn","PuBu","PuBuGn","BuPu","OrRd"]:
        		type = 'sequential'
        		n = np.inf
        	elif self.cm in ["BrBG","PuOr","Spectral","RdBu","RdYlBu","RdYlGn"]:
        		type = 'diverging'
        		n = np.inf
        	else:
        		type = 'qualitative'
        		n = len(cm)
        	text = self.text + '\nMaximum number of colors: {}\nColorblind safe: {}\nType: {}'.format(n,col_blind,type)
        	label.configure(text=text)
        	if self.plat == 'WINDOWS':
                        self.tw.attributes('-topmost',True)
            
        win.grid()
        
		
        x, y = tip_pos_calculator(widget, label)
        

        self.tw.wm_geometry("+%d+%d" % (x, y))

    def hide(self):
        tw = self.tw
        if tw:
            tw.destroy()
        self.tw = None
