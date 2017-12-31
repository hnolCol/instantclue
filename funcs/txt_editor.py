# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 23:34:43 2017

@author: hnolte-101
"""
import six
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from funcs import determine_platform
import matplotlib.pyplot as plt
from matplotlib import colors
from tkinter.colorchooser import *
#import matplotlib
#sys.path.append('.../funcs/')
MAC_GREY = '#ededed'

#left_align                            =        tk.PhotoImage(file=os.path.join(path_file,'icons','left_align.png'))
def center(toplevel,size):
         	#toplevel.update_idletasks()
         	w_screen = toplevel.winfo_screenwidth()
         	h_screen = toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))
 
colors_ = list(six.iteritems(colors.cnames))
def col_c(color):
              y = tuple([int(float(z * 255)) for z in color])
              hex_ = "#{:02x}{:02x}{:02x}".format(y[0],y[1],y[2])
              return hex_   
          

#print(cols_)
            
#def change_font(widget,family,size,weight,fig):
#    
#    widget.configure(font = tkFont.Font(family = family.get(), size = int(float(size.get())),weight = weight.get()))
#    tx_  = widget.get('1.0',tk.END)
#    
#    plt.figtext(0.5,0.5,tx_, figure=fig, fontdict = {'weight':weight.get(), 'size':int(float(size.get())), 'family':family.get()})
save_txt = ''  
color_out = 'black'  
def close_(popup):
    
    global save_txt
    save_txt = 'InstantSaysNone___***444'
    popup.destroy()
def close_and_return(txt_,popup,col_label,event = ''):
    global save_txt, color_out
    save_txt = txt_.get('1.0',tk.END).strip()
    color_out = col_label.cget('background')
   
    #print(save_txt)
    popup.destroy()
    
def choose_color(event,txt,family,size,weight,style,popup):  
    global color_out
    widget = event.widget    
    color = askcolor(color = widget.cget('background'),parent = popup)
    
    if color is not None:
        widget.configure(bg=color[1])
        txt.configure(fg = color[1])
        color_out = color[1]
def change_ha(event,mode,left,right,center,var_ha_align):
    set_ = var_ha_align.get()
    
    if mode == 'left' and set_ != 'left':
    
        left.configure(relief = tk.SUNKEN)
        right.configure(relief = tk.FLAT)
        center.configure(relief = tk.FLAT)
        var_ha_align.set('left')
    elif mode == 'right' and set_ != 'right':
        right.configure(relief = tk.SUNKEN)
        left.configure(relief = tk.FLAT)
        center.configure(relief = tk.FLAT)
        var_ha_align.set('right')
    elif mode == 'center' and set_ != 'center':
        left.configure(relief = tk.FLAT)
        right.configure(relief = tk.FLAT)
        center.configure(relief = tk.SUNKEN)
        var_ha_align.set('center')
def change_var(var_apply_all,var_axis_all,mode = None):

	if mode == 'axis':
		if var_axis_all.get() == False:
			var_apply_all.set(False)
	else:
		if var_apply_all.get() == True:
			var_axis_all.set(True)
	#elif var_apply_all.get() == False:
		
	#print(val)
	#if val == False:
	#	var_apply_all.set(False)
	#else:
	#	pass

        
    
def update_widget(event, widget,family,size,weight,style):
    
    widget.configure(font = tkFont.Font(family = family.get(), size = int(float(size.get()))+2,weight = weight.get(), slant = style.get()))
    
    
def create_txt_widget(platform,fig,canvas = None, edit_in_figure = False, input_string = None, font = 'Arial', size = '12', weight = 'bold', style = 'roman',color = 'black',rotation = '0',ha_align = 'left',
                      fig_left = None, fig_right = None, fig_center = None, key = None):
    global save_txt
    if input_string is None:
        save_txt = '' 
    if platform == "WINDOWS":                             
        LARGE_FONT = ("Helvetica", 11, "bold")
    else:
        LARGE_FONT = ("Helvetica", 13, "bold")
    if platform == 'MAC':
    	w = 650
    else:    
    	w = 500
    h = 400
    txt_editor = tk.Toplevel()
    txt_editor.wm_title('Enter and format text')
    if platform != 'MAC':
    	txt_editor.attributes('-topmost', True)
    txt_editor.protocol("WM_DELETE_WINDOW", lambda: close_(txt_editor))
    cont = tk.Frame(txt_editor, background =MAC_GREY)
    cont.pack(expand=True, fill='both')
    cont.grid_rowconfigure(10,weight=1)
    cont.grid_columnconfigure(10, weight=1)
    #print(tk.font.families())
    
    lab1 = tk.Label(cont, text = 'Text editor', font = LARGE_FONT, fg="#4C626F", justify=tk.LEFT, bg = MAC_GREY)
    lab1.grid(padx=5,pady=15, columnspan=6 ,sticky=tk.W)
    var_font = tk.StringVar()
    var_font.set(font)
    var_size = tk.StringVar()
    var_size.set(size)
    var_weight = tk.StringVar() 
    var_ha_align= tk.StringVar() 
    var_ha_align.set(ha_align)
    var_apply_all = tk.BooleanVar()
    var_axis_all = tk.BooleanVar() 
    var_axis_all.set(False)
    if key is not None:
    
    	if 'tick' in key or 'subplots' in key or 'legend_texts' in key:
        	var_apply_all.set(True)
    	else:
        	var_apply_all.set(False)
    
    
    if key is not None:
        s_ = 'Apply to all '+key
        
        cb_ = tk.Checkbutton(cont, text = s_, variable = var_apply_all, background = MAC_GREY, command = lambda: change_var(var_apply_all,var_axis_all)  )  
        cb_.grid(row=0, column = 4, sticky=tk.E, padx=10, pady=(15,5), columnspan=7)
        if key == 'legend_texts':
            cb_.config(state=tk.DISABLED)
        
        if 'tick' in key:
        	cb_ax = tk.Checkbutton(cont, text = 'Apply to all ticks on axis', variable=var_axis_all, background = MAC_GREY, command = lambda: change_var(var_apply_all,var_axis_all,'axis'))
        	var_axis_all.set(True)
        	#var_axis_all.trace('w',check_other_var) 
        	cb_ax.grid(row=0, column = 2, sticky=tk.W, padx=4, pady=(15,5), columnspan=7)
    if weight not in ['bold','normal']:
        if weight > 600:
            weight = 'bold'
        else:
            weight = 'normal'
    if color[0] == '#':     
        pass
    elif color[0] == '.':
       rgb_ = colors.ColorConverter.to_rgba(color) 
       #print(rgb_)
       color = col_c(rgb_)
        
    elif color in colors.cnames:
        color = colors.cnames[color]
        #print('worked')
    
        
        
    var_weight.set(weight)
    var_style = tk.StringVar() 
    if style not in ['roman','italic']:
        style = 'roman'
    var_style.set(style)
    var_color = tk.StringVar()

    var_color.set(color)
    var_rotation = tk.StringVar() 
    var_rotation.set(rotation+' °')
    
    
    txt_ = tk.Text(cont, width=600, height=600, undo=True)
    if platform == 'WINDOWS':
    
    	combo_fonts = ttk.Combobox(cont, textvariable = var_font,values = ['Arial','Calibri','Cambria','Courier New','Corbel','Helvetica','Magneto','Times New Roman','Verdana'], width=15)
    	combo_size = ttk.Combobox(cont, textvariable = var_size,values = list(range(3,40)), width=4)
    	combo_weight = ttk.Combobox(cont,textvariable = var_weight, values = [ 'normal','bold'], width=7)       
    	combo_style = ttk.Combobox(cont, textvariable = var_style , values = ['roman','italic'], width=7)
    	combo_rotation = ttk.Combobox(cont, textvariable = var_rotation, values = [str(x)+' °' for x in range(0,360,15)], width=5)
    else:
    	combo_fonts =  tk.OptionMenu(cont,var_font,*['Arial','Calibri','Cambria','Courier New','Corbel','Helvetica','Magneto','Times New Roman','Verdana'],command = lambda event,widget= txt_,family = var_font,size = var_size,weight = var_weight, style = var_style : update_widget(event,widget,family,size,weight,style))
    	combo_fonts.configure(width=15,bg=MAC_GREY)
		
    	combo_size = tk.OptionMenu(cont, var_size,*list(range(4,42,2)),command = lambda event,widget= txt_,family = var_font,size = var_size,weight = var_weight, style = var_style : update_widget(event,widget,family,size,weight,style))#, width=4)
    	combo_size.configure(width=4,bg=MAC_GREY)
    	combo_weight = tk.OptionMenu(cont,var_weight,*[ 'normal','bold'],command = lambda event,widget= txt_,family = var_font,size = var_size,weight = var_weight, style = var_style : update_widget(event,widget,family,size,weight,style))#, width=7) 
    	combo_weight.configure(width=7,bg=MAC_GREY)  
    	combo_style = tk.OptionMenu(cont, var_style, *['roman','italic'],command = lambda event,widget= txt_,family = var_font,size = var_size,weight = var_weight, style = var_style : update_widget(event,widget,family,size,weight,style))#, width=7)
    	combo_style.configure(width=7,bg=MAC_GREY)
    	combo_rotation = tk.OptionMenu(cont, var_rotation, *[str(x)+' °' for x in range(0,360,15)],command = lambda event,widget= txt_,family = var_font,size = var_size,weight = var_weight, style = var_style : update_widget(event,widget,family,size,weight,style))#, width=5)
    	combo_rotation.configure(width=5,bg=MAC_GREY)
	
    	#combo_size = 
    	#combo_weight =     
    	#combo_style = 
    	#combo_rotation = 
    
    if edit_in_figure:
        txt_.insert(1.0,input_string)
    
    labels_text = ['Font: ','Size: ','Weight: ','Style: ']
    combos = [combo_fonts,combo_size,combo_weight,combo_style,combo_rotation]
    for n,combo in enumerate(combos):
			#combo.grid(row= 2, column = 0+n, sticky=tk.W+tk.EW,pady=3,padx=2)
            combo.grid(row= 2, column = 0+n, sticky=tk.W,pady=3,padx=2)
            
            if combo != combo_rotation and platform != 'MAC':
                combo.bind('<<ComboboxSelected>>', lambda event,widget= txt_,family = var_font,size = var_size,weight = var_weight, style = var_style : update_widget(event,widget,family,size,weight,style))
    color_ = tk.Label(cont, text = '       ', bg = color)
    color_.grid(row=2, column = 6, sticky=tk.W,padx=2,pady=3)
    color_.bind('<Button-1>', lambda event,txt = txt_,family = var_font,size = var_size,weight = var_weight, style = var_style, popup = txt_editor : choose_color(event, txt,family,size,weight,style,popup))
    
    align_left = tk.Label(cont, image = fig_left)
    align_left.grid(row=2,column=7, sticky=tk.W, padx=(2,1),pady=3)
    align_right = tk.Label(cont, image = fig_right)
    align_right.grid(row=2,column=8, sticky=tk.W, padx=(1,2),pady=3)
    align_center = tk.Label(cont, image = fig_center)
    align_center.grid(row=2,column=9, sticky=tk.W, padx=(1,2),pady=3)
    if ha_align == 'left':
        align_left.configure(relief=tk.SUNKEN)
    elif ha_align == 'right':
        align_right.configure(relief=tk.SUNKEN)
    else:
        align_center.configure(relief=tk.SUNKEN)
        
    align_left.bind('<Button-1>', lambda event,mode = 'left', left = align_left, right = align_right,center = align_center, var_ha_align= var_ha_align: change_ha(event,mode,left,right,center,var_ha_align))
    align_right.bind('<Button-1>', lambda event,mode = 'right', left = align_left, right = align_right , center = align_center, var_ha_align= var_ha_align: change_ha(event,mode,left,right,center,var_ha_align))
    align_center.bind('<Button-1>', lambda event,mode = 'center', left = align_left, right = align_right ,center = align_center,  var_ha_align= var_ha_align: change_ha(event,mode,left,right,center,var_ha_align))
    
    txt_.grid(row=10,sticky=tk.NSEW,columnspan=500,padx=3)
    if input_string is None:
        txt_button = 'Add text'
    else:
        txt_button = 'Apply Change'
    but_add_ = ttk.Button(cont, text = txt_button, command = lambda txt_ = txt_ ,popup=txt_editor: close_and_return(txt_,popup,color_))
    but_add_.grid(sticky=tk.EW, columnspan=15) 
    
    txt_.configure(font = tkFont.Font(family = var_font.get(), size = int(float(var_size.get()))+2,weight = var_weight.get(), slant = var_style.get()))
    txt_.configure(fg = color)
    
    txt_.focus_set()
    
    txt_.bind('<Shift-Return>', lambda event,txt_ = txt_ ,popup=txt_editor: close_and_return(txt_,popup,color_,event=event))
    
    center(txt_editor, (w,h))
    txt_editor.wait_window()
    return [save_txt,var_font.get(), var_size.get(),var_weight.get(),var_style.get(),color_out,var_rotation.get().split(' ')[0],var_ha_align.get(),var_apply_all.get(),var_axis_all.get()]


























