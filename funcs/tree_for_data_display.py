# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:12:17 2017

@author: hnolte-101
"""


import tkinter as tk 
from tkinter import ttk 
import tkinter.font as tkFont
import pandas as pd

def display_data_in_tree(popup, headers, cols_to_center=None, data_to_fill_tree = None, kind_of_data = None, show_headers_only = True):
    
    
    tree = __setup_trees(popup, headers, show_headers_only)
    __fill_tress(tree, headers, cols_to_center, data_to_fill_tree, kind_of_data)
    
    return tree


def __setup_trees(popup, headers,  show_headers_only):
    cont = ttk.Frame(popup)
    cont.pack(fill="both", expand = True)
    cont.grid_columnconfigure(0, weight=1)
    cont.grid_rowconfigure(0, weight=1)
    
    
    if show_headers_only == True:
        data_tree = ttk.Treeview(cont, columns  = headers,show='headings')
        
    else:    
        data_tree = ttk.Treeview(cont, columns  = headers)
    
    scroll_vert = ttk.Scrollbar(cont,orient='vertical',
                                                command = data_tree.yview)
    scroll_hor = ttk.Scrollbar(cont,orient='horizontal',
                                               command = data_tree.xview)
    
    data_tree.configure(yscrollcommand = scroll_vert.set,
                                             xscrollcommand = scroll_hor.set)
    data_tree.grid(column = 0, row = 0,
                                        sticky ='nsew', in_=cont)
    
    scroll_vert.grid(column=1, row=0, sticky = 'ns', in_=cont)
    scroll_hor.grid(column=0, row=1, sticky = 'ew', in_=cont)
    
    return data_tree

    
    
def __fill_tress(tree, headers, cols_to_center, data_to_fill_tree, kind_of_data):
    
    
    
    for col in headers:
                   tree.heading(col, text=col) 
    
                   col_w = tkFont.Font().measure(col) + 40
                       # if  dtyp_col == np.float64 or dtyp_col == np.int64:
                   if col in cols_to_center:
                       anch = tk.CENTER
                   else:
                       anch = tk.W
                   tree.column(col, anchor=anch, width=col_w)
                   
    if kind_of_data == 'dict':
        i = 0
        for key, values in data_to_fill_tree.items():
          
            if key & 1:
                tag = ''
            else:
                tag = 'odd'
            for item in values:
                item = [key] + item
                tree.insert('', 'end', iid = str(i), text = str(key), values = item, tag=tag)
                i += 1
            
            
            




               
                   

                   
                   
                   