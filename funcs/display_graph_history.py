# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:34:20 2017

@author: hnolte-101
"""

import tkinter as tk 
from tkinter import ttk 
import tkinter.font as tkFont
import pandas as pd

columns_in_tree = ['Numeric value','Categories','Chart type','Colormap','Swarm','Data']
collect_list = []


def display_graph_history(popup,plotted_charts, performed_stats):
    
    
    tree = __setup_trees(popup)
    df_to_export = __fill_tress(tree,plotted_charts, performed_stats)
    
    return tree, df_to_export
    
        
def __setup_trees(popup):
    cont = ttk.Frame(popup)
    cont.pack(fill="both", expand = True)
    cont.grid_columnconfigure(0, weight=1)
    cont.grid_rowconfigure(0, weight=1)
    
    
    
    data_tree = ttk.Treeview(cont, columns  = columns_in_tree)
    
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

    
def __fill_tress(tree, plotted_charts, performed_stats):
    for col in columns_in_tree:
                   tree.heading(col, text=col) 
    
                   col_w = tkFont.Font().measure(col) + 40
                       # if  dtyp_col == np.float64 or dtyp_col == np.int64:
                   if col in ['Chart type','Colormap','Swarm']:
                       anch = tk.CENTER
                   else:
                       anch = tk.W
                   tree.column(col, anchor=anch, width=col_w)
        
    n_charts = len(plotted_charts)
    for key, plot_info in plotted_charts.items():
        if key & 1:
            tag = ''
        else:
            tag = 'odd'
        plot_info = [str(x) for x in plot_info]   

        
        main = tree.insert('', 'end',str(key), text = str(key), values = plot_info, tag=tag)
        #
        for_df = [key] + plot_info
        collect_list.append(for_df)
        if key in performed_stats:
            stat_val = ['Group1','Group2','Test','p-value','test_statistic']
            tree.insert(main, 'end', 'head_'+str(key), values = stat_val, tag = 'header')
            head_df =  [''] + stat_val
            collect_list.append(head_df)
            
            stats = performed_stats[key]
            for stat_ in stats:
                
                fill_stats = stat_[:2]
                if stat_[-1] != '':
                    if stat_[-1] == True:
                        paired = 'paired'
                    else:
                        paired = 'unpaired'
                else:
                    paired = ''
                test = '{}_{}_{}'.format(stat_[-3],stat_[-2],paired)

                fill_stats = fill_stats + [test, "{:.2E}".format(stat_[4][1]),stat_[4][0]]
                
                
                tree.insert(main,'end',str(stat_),values=fill_stats) 
                
                
                for_df = [''] + fill_stats
                collect_list.append(for_df)
                
                  
    tree.tag_configure("header", background = "#58A1D8", foreground="white")
    tree.tag_configure("odd", background = "#58A1D8", foreground="white")            
    print(collect_list)
    df_to_export = pd.DataFrame(collect_list, columns = ['Index']+columns_in_tree)
    
    
    return df_to_export





    
    