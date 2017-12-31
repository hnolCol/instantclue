# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:22:01 2017

@author: hnolte-101
"""

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from funcs import determine_platform
import matplotlib.pyplot as plt
ax_out = object()
#sys.path.append('.../funcs/')
MAC_GREY = '#ededed'
def center(toplevel,size):
         	#toplevel.update_idletasks()
         	w_screen = toplevel.winfo_screenwidth()
         	h_screen = toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))
             
             
             
def close_(popup,names,ax,ax_l):
    global ax_out
    
    ax_n_idx = names.index(ax.get())
    ax_out = ax_l[ax_n_idx]
    
    popup.destroy() 
    
    
def determine(popup):
    global ax_out
    ax_out = None    
    popup.destroy() 
    
def ask_for_subplot(dict_with_axes,fig_id,platform):
    
    if platform == "WINDOWS":                             
        LARGE_FONT = ("Helvetica", 11, "bold")
        w = 310
    else:
        LARGE_FONT = ("Helvetica", 13, "bold")
        w = 355

    
    h = 200
    subplot_editor = tk.Toplevel()
    subplot_editor.wm_title('Choose axis for Image Upload')
    subplot_editor.attributes('-topmost', True)
    subplot_editor.protocol("WM_DELETE_WINDOW", lambda: determine(subplot_editor))
    cont = tk.Frame(subplot_editor, background =MAC_GREY)
    cont.pack(expand=True, fill='both')
    #cont.grid_rowconfigure(10,weight=1)
    cont.grid_columnconfigure(4, weight=1)
    #print(tk.font.families())
    var_axes = tk.StringVar() 
    lab1 = tk.Label(cont, text = 'Choose axis for image upload', font = LARGE_FONT, fg="#4C626F", justify=tk.LEFT, bg = MAC_GREY)
    lab1.grid(padx=6,pady=15, columnspan=6 ,sticky=tk.W)
    lab2 = tk.Label(cont, text = 'Please note that so far only .png files are supported.\nThe quality will appear much better after export.', fg="#4C626F", justify=tk.LEFT, bg = MAC_GREY)
    lab2.grid(pady=3,padx=3)
    ax_l = []
    names = [] 
    for key, values in dict_with_axes.items() :
        if 'Figure__'+str(fig_id) in key:
            ax_l.append(key)
            names.append('Figure__'+str(fig_id)+'_'+values[1])
            
            
    var_axes.set(names[0])  
    if platform == 'WINDOWS':      
    	combo_ = ttk.Combobox(cont, textvariable = var_axes, values = names,width=200) 
    else:
    	combo_ = ttk.OptionMenu(cont, var_axes, names[0], *names)       
    combo_.grid(sticky=tk.W+tk.EW,padx=3,pady=3,columnspan=6) 
    but_okay = ttk.Button(cont, text= 'Done', command = lambda popup=subplot_editor, names=names, ax = var_axes,ax_l = ax_l: close_(popup,names,ax,ax_l))
    but_okay.grid(row=4,sticky=tk.W,padx=3,pady=3) 
    
    close_but = ttk.Button(cont, text ='Close', command = lambda: determine(subplot_editor))
    close_but.grid(row=4,sticky=tk.E,column=0,columnspan=6,padx=3)
    center(subplot_editor, (w,h))
    subplot_editor.wait_window()
    
    return (var_axes.get(),ax_out)
    
            
            
            
    #axes = 
    
    
    