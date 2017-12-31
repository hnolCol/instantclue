# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:12:01 2017

@author: hnolte-101
"""
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt

def fill_axes_with_plot(data, x , y , hue, ax, cmap, plot_type = 'boxplot', order = None):
    
         if plot_type == 'boxplot':
                         sns.boxplot(x= x, y=y,data=data, hue = hue, palette = cmap, order = order, fliersize = 3, linewidth=0.65, ax = ax)
         elif plot_type == 'swarm':
                         
                         sns.swarmplot(x= x, y=y,data=data, palette = cmap, hue = hue, split = True, order=order, ax = ax)
         elif plot_type == 'barplot':
                         sns.barplot(x= x, y=y, hue = hue, data=data, order = order, palette = cmap,errwidth=0.5, capsize=0.09, edgecolor=".25", ax = ax)
         else:
                         sns.violinplot(x= x, y=y, hue = hue, data=data, palette = cmap ,order=order, linewidth=0.65, ax = ax)
                         give_violins_edge_color(ax)
       


def give_violins_edge_color(ax):
         
         #if plot_type == 'violinplot':
                           ax_coll = ax.collections
                          
                           ax_coll = ax_coll[::2]
                          
                           num_patches  = len(ax_coll)
                    
                           for collection in ax_coll:
     
                              collection.set_edgecolor("black")
                              collection.set_linewidth(0.55)
                              
def add_draggable_legend(ax, patches = [] , leg_title = '', leg_font_size = '8',handles= None , labels = None, ncols=2, collection_list_legend = None):
    if len(patches) > 0: 
        leg = ax.legend(handles = patches, bbox_to_anchor=(0., 1.04, 1., .102), loc=3, title = leg_title,
                                   ncol=ncols, borderaxespad=0.)
    elif handles is not None and labels is not None:
    	leg = ax.legend(handles,labels, bbox_to_anchor=(0., 1.04, 1., .102), loc=3, title = leg_title,
                                   ncol=ncols, borderaxespad=0.)
    
    else:
        
         leg = ax.legend(bbox_to_anchor=(0., 1.06, 1., .102), loc=3, title = leg_title,
                                   ncol=2, borderaxespad=0.)
    
    if leg is not None:     
        leg.draggable(use_blit = True)     
        plt.setp(leg.get_title(), fontsize=leg_font_size)         
        plt.setp(leg.get_texts(), fontsize=leg_font_size)
          
    if collection_list_legend is not None and leg is not None:
    	collection_list_legend[ax] = [leg,ax] 
    	return collection_list_legend




def calculate_new_ylim_from_data(data):
                 get_max_val = data.max().max()
                 get_min_val = data.min().min()
                 add = 0.12
                 y_min = round(get_min_val-abs(get_min_val*add),2)
                 y_max = round(get_max_val+get_max_val*add,2)
                 return (y_min,y_max)                     
