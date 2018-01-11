import sys

import start_page

# tkinter import
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tkinter.simpledialog as ts
import tkinter.filedialog as tf
import tkinter.font as tkFont
from tkinter.colorchooser import *



# internal imports

from modules import data
from modules import sourceDataTreeView 
from modules import plotter
from modules import images
from modules import color_changer
from modules import stats
from modules import save_and_load_sessions
from modules import interactive_widget_helper

from modules.plots import scatter_with_categories

from modules.dialogs import chart_configuration
from modules.dialogs import size_configuration
from modules.dialogs import display_data
from modules.dialogs import excel_import
from modules.dialogs import txt_file_importer
from modules.dialogs import main_figures
from modules.dialogs import change_columnName
from modules.dialogs import numeric_filter 
from modules.dialogs import categorical_filter
from modules.dialogs import curve_fitting
from modules.dialogs import clustering
from modules.dialogs import findAndReplace
from modules.dialogs import classification
from modules.dialogs import custom_sort
from modules.dialogs import mergeDataFrames
from modules.dialogs import anova_calculations
from modules.dialogs import anova_results
from modules.dialogs import define_groups_dim_reduction
from modules.dialogs import pivot_table
from modules.dialogs import dimRed_transform
from modules.dialogs import VerticalScrolledFrame
from modules.dialogs import custom_filter
from modules.dialogs import color_configuration
from modules.dialogs import correlations

from modules.dialogs.simple_dialog import simpleUserInputDialog
from modules.utils import *


import os
import time
import textwrap as tw 
import string 
import webcolors as wb

from decimal import Decimal

import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly



import warnings
warnings.simplefilter('ignore', np.RankWarning)

import itertools
from collections import OrderedDict


import gc



import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Lasso
from matplotlib import colors
from matplotlib import path
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

##
import seaborn as sns
sns.set(style="ticks",font=defaultFont)
sns.axes_style('white')
##

import multiprocessing
from multiprocessing import Process
from threading import Timer,Thread,Event
import concurrent.futures
from multiprocessing import Pool 



from scipy import interpolate
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd

import fastcluster
from scipy.stats import linregress
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import ranksums
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import zscore
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import f
from scipy.stats import gaussian_kde

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.libqsturng import psturng


cat_error = '''The category has more than 100 levels. This results in a confusing plot and is not supported.
                \nYou can use the filters to filter out categories by right-clicking on the items in the treeview.'''


last_called_plot = [] 
t_dict =dict() 


class analyze_data(tk.Frame):
	 
            
     def __init__(self,parent,controller):
           tk.Frame.__init__(self,parent, background = MAC_GREY)
           
           default_font = tkFont.nametofont("TkDefaultFont")
           # parameters are defined in utils.py 
           default_font.configure(size=defaultFontSize,family=defaultFont)
           
            
           self.get_images()
           self.define_variables()
           self.build_menus()
           self.build_label_frames()
           self.grid_widgets(controller)
           
           ## sourceData holds all data frames , self.plt is the plotter class,
           ## anovaTestCollection saves all anova tests made
           ## curveFitCollection saves all curve fittings made by the user
           self.sourceData = data.DataCollection()
           self.plt = plotter._Plotter(self.sourceData,self.f1)
           self.mainFigureCollection = main_figures.mainFigureCollection(self)
           self.anovaTestCollection = anova_calculations.storeAnovaResultsClass()
           self.dimensionReductionCollection = stats.dimensionReductionCollection()
           self.curveFitCollection = curve_fitting.curveFitCollection()
           self.clusterCollection = clustering.clusterAnalysisCollection()
           self.classificationCollection =classification.classifierAnalysisCollection()
           self.colorHelper = color_configuration.colorMapHelper()
           
           
           self.interactiveWidgetHelper = interactive_widget_helper.interactiveWidgetsHelper(self.mark_sideframe)
           # add empty figure to GUI 
           self.display_graph(self.f1)
           # actions on resizing the window (e.g. changing the icons to smaller/bigger ones)
           self.bind("<Configure>", self.icon_switch_due_to_rescale) 
            
     def define_variables(self):
     
               
           self.folder_path = tk.StringVar()
           self.size_selected = tk.StringVar(value = '50')
           ## stats Test 
           self.twoGroupstatsClass = None
           
           ###
           self.old_width = None
           self.old_height = None
           
           #self.sourceData.df = None
           self.closest_gn_bool = False
           ## dics
           self.a = dict()
           ## dicts to save dropped column names 
           
           self.selectedNumericalColumns  = OrderedDict() 
           self.selectedCategories  = OrderedDict()
           
           self.count = 0 
           self.save_tree_view_to_dict = OrderedDict()
           self.colormaps = dict()
           self.data_set_information = OrderedDict()
           self.tooltip_data = None
           
           ##scatter matrix dicts
           self.data_scata_dict = dict()
           self.axes_scata_dict = dict()
           self.performed_stats = OrderedDict() 
           #lists
           self.data_types_selected = [] 
           ##
           self.cmap_in_use = tk.StringVar(value = 'Blues') 
           self.alpha_selected = tk.StringVar(value = '0.75')
           
           self.original_vals = []
           self.add_swarm_to_new_plot =False
           self.swarm_but = 0
           self.split_on_cats_for_plot = tk.BooleanVar(value = True)
           self.split_in_subplots = tk.BooleanVar(value = False)
           
           self.id_tt = None
                  
           self.selection_press_event = None 
           self.tooltip_inf = None 
           self.pick_label = None 
           self.pick_freehand  = None 
           self._3d_update = None
           self.filt_source_for_update = None
           self.mot_adjust = None
           self.mot_adjust_ver = None
           self.label_button_droped = None
           self.tooltip_button_droped = None
           self.size_button_droped = None
           self.color_button_droped = None
           self.stat_button_droped = None
           self.cursor_release_event  = None
           
           self.hclust_dat_ = None
           self.hclust_axes = dict()
           self.calculate_row_dendro = True
           self.calculate_col_dendro = True  
           
           self.release_event = None
           self.hclust_move_level = None
           self.annotation_main = OrderedDict() 
           self.scat = None
           
           self.col_map_keys_for_main  = None
           self.col_map_keys_for_main_double  = None
           self.mot_button_dict = dict()
           self.color_added = None
           
           self.mot_button = None
           self.dimReduction_button_droped = None
           self.color_pca_dict = dict() 
           self.subsets_and_scatter_with_cat = OrderedDict() 
           self.regression_in_plot = dict()
           self.label_axes_for_scatter = OrderedDict() 
           self.legend_collecion_for_drag = []           
                     
     def build_menus(self):

           self.build_main_drop_down_menu_treeview()
           self.def_split_sub_menu()
           self.build_selection_export_menu() 
           self.build_merge_menu()
           self.build_datatype_menu()
           self.build_main_figure_menu()
           self.build_pca_export_menu()
           self.build_corrMatrix_menu()
           self.build_hclust_menu()   
           
     def build_label_frames(self):
                  
		   ## define labelframes to have a nicely separated layout in the gui 
		   	
           settingHorizontFrames = dict(relief = tk.GROOVE, padx = 4, pady= 4, bg = MAC_GREY)
           settingVertFrames = dict(relief = tk.GROOVE, padx = 5, pady= 10, bg = MAC_GREY)
           
           self.receiverFrame = {}
              
           self.column_sideframe = tk.LabelFrame(self, text='Numeric Data', **settingHorizontFrames)
           self.category_sideframe = tk.LabelFrame(self, text='Categories', **settingHorizontFrames)
           self.plotoptions_sideframe = tk.LabelFrame(self, text='Plot Options', relief=tk.GROOVE, padx=3, pady=7, bg=MAC_GREY)   
           self.sideframe_upload = tk.LabelFrame(self, text='Data', **settingVertFrames)
           self.source_sideframe = tk.LabelFrame(self, text='Source Data', **settingVertFrames)
           self.analysis_sideframe = tk.LabelFrame(self, text='Analysis', **settingVertFrames)       
           self.mark_sideframe = tk.LabelFrame(self, text = "Slice and Marks", **settingVertFrames)
           self.receiverFrame['numeric'] = self.column_sideframe
           self.receiverFrame['category'] = self.category_sideframe

           
           ## store frames in list 
           self.label_frames = [self.sideframe_upload ,
								self.column_sideframe ,
								self.category_sideframe ,
								self.plotoptions_sideframe,
								self.source_sideframe,
    							self.analysis_sideframe ,    
    							self.mark_sideframe]  
    							           
           self.sideframe_upload.grid(in_=self,
                                     row=1,
                                     column =0,
                                     rowspan = 4,
                                     sticky=tk.EW+tk.NW,
                                     padx=5)
           
           self.source_sideframe.grid(in_=self,
                                     row=5,
                                     column =0,
                                     pady=(0,6),
                                     sticky=tk.NS,
                                     padx=5)
           self.mark_sideframe.grid(in_=self,
                                     row=1,
                                     column = 2,
                                     rowspan = 20,
                                     sticky=tk.NW,
                                     padx=5)
           self.analysis_sideframe.grid(in_=self,
                                     row=11,
                                     column =0,
                                     rowspan = 4,
                                     sticky=tk.NW+tk.S,
                                     padx=5,
                                     pady=0)
           self.column_sideframe.grid(in_=self,
                                     row=1,
                                     column =3,
                                     rowspan=3,                                     
                                     sticky=tk.EW+tk.NW,
                                     padx=5)

           self.category_sideframe.grid(in_=self,
                                     row=4,
                                     column =3,
                                     rowspan = 3,                                     
                                     sticky=tk.EW+tk.NW,
                                     padx=5)
#           
           
           self.plotoptions_sideframe.grid(in_=self,
                                     row=1,
                                     column = 5,
                                     rowspan=15,
                                     sticky=tk.NW,
                                     padx=5)           
           
     def build_main_drop_down_menu_treeview(self):
         
           '''
           This has grown historically and the code needs to be re-written
           '''
           menuDict = {}
           menus = ['main','column','dfformat','sort','split','replace','dataType','logCalc',\
           'rolling','smoothing','rowColCalc','export','multTest','curvefit','categories',\
           'predict','transform']
           
           for menu in menus:
           	menuDict[menu] = tk.Menu(self, **styleDict)
                               
           self.menu = menuDict['main']
           
           rowColumnCalculations = ['Mean [row]','Median [row]','Stdev [row]','Sem [row]',
           									'Mean & Stdev [row]','Mean & Sem [row]',
           									'Square root [row]','x * N [row]','x ^ N [row]',
           									'N ^ x [row]']  
           									
           splitOptions = ["Space [ ]","Semicolon [;]","Comma [,]","U-Score [_]",
           					"Minus [-]","Slash [/]","B-Slash [\]"]
           replace_options = ['0 -> NaN','NaN -> 0','NaN -> Constant','NaN -> Mean[col]',
           					  'NaN -> Median[col]']   
           rollingOptions = ['mean','median','quantile','sum','max','min','std']
           
           mult_opt = ['FWER','bonferroni','sidak','holm-sidak','holm','simes-hochberg','hommel',
           				'FDR - methods','benjamini-hochberg','benjamini-yekutieli',
           				'2-stage-set-up benjamini-krieger-yekutieli (recom.)',
           				'gavrilov-benjamini-sarkar','q-value','storey-tibshirani']           				
           				           									         
           self.menu.add_command(label ="Data management ..",state=tk.DISABLED, foreground ="darkgrey")
           self.menu.add_separator()
           self.menu.add_cascade(label='Column operation ..', menu = menuDict['column'])
           
           self.menu.add_cascade(label="Sort data by ..", menu = menuDict['sort'])
           menuDict['column'].add_command(label = 'Rename', command = self.rename_columns)
           menuDict['column'].add_command(label='Duplicate', command = self.duplicate_column)           
           menuDict['column'].add_command(label='Delete', command = self.delete_column)
           menuDict['column'].add_command(label="Combine",command = self.combine_selected_columns)
           menuDict['column'].add_cascade(label="Split on ..", menu = menuDict['split'])
           menuDict['column'].add_cascade(label='Change data type to..', menu = menuDict['dataType'])
           menuDict['column'].add_cascade(label="Replace", menu = menuDict['replace'])
           menuDict['column'].add_command(label='Count through', command = self.create_count_through_column)
           menuDict['column'].add_command(label='Drop rows with NaN', command = self.remove_rows_with_na)
           
           menuDict['dataType'].add_command(label='Float', command = lambda: self.change_column_type('float64'))
           menuDict['dataType'].add_command(label='Category', command = lambda: self.change_column_type('object')) 
           menuDict['dataType'].add_command(label='Integer', command = lambda: self.change_column_type('int64')) 
           
           menuDict['sort'].add_command(label="Value", command = lambda s = "Value":  self.sort_source_data(s))
           menuDict['sort'].add_command(label="String length", command = lambda s = "String length":  self.sort_source_data(s)) 
           menuDict['sort'].add_command(label="Custom order", command = self.custom_sort_values)#lambda : tk.messagebox.showinfo('Under revision','Currently under revision. Will be available in the next minor update.'))#)self.design_popup(mode='Custom sorting'))

           for splitString in splitOptions:
               menuDict['split'].add_command(label=splitString, 
               							   command = lambda splitString = splitString:  self.split_column_content_by_string(splitString)) 
           menuDict['replace'].add_command(label='Find & Replace', command = lambda: findAndReplace.findAndReplaceDialog(dfClass = self.sourceData, dataTreeview = self.DataTreeview))        
           for i,replaceOption in enumerate(replace_options):
               menuDict['replace'].add_command(label=replaceOption, command = lambda replaceOption = replaceOption:  self.replace_data_in_df(replaceOption))   
           
           menuDict['main'].add_cascade(label='Change data format', menu = menuDict['dfformat'])
                                 
           
           menuDict['dfformat'].add_command(label = 'To long format (melt)', command = self.melt_data)
           menuDict['dfformat'].add_command(label = 'To wide format (pivot)', command = self.pivot_data)
           menuDict['dfformat'].add_command(label = 'Transpose', command = self.transpose_data)
	        
		
           for text in mult_opt:
           		if text in ['FWER','FDR - methods','q-value']:
           			menuDict['multTest'].add_command(label=text, 
           					state = tk.DISABLED, foreground="darkgrey") 
           			menuDict['multTest'].add_separator()
           		else:
           			menuDict['multTest'].add_command(label=text, \
           			command = lambda proc = text: self.multiple_comparision_correction(proc))
           
           ## Data 
           menuDict['main'].add_command(label ="Data Transformation ..",state=tk.DISABLED,foreground='darkgrey')
           menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label='Smoothing', menu = menuDict['smoothing'])
           menuDict['smoothing'].add_cascade(label='Rolling', menu = menuDict['rolling'])
           for rolling in rollingOptions:
               menuDict['rolling'].add_command(label=rolling, command = lambda rolling = rolling: self.rolling_mod_data(rolling))
           menuDict['smoothing'].add_command(label='IIR filter', command = self.iir_filter)
           
           menuDict['main'].add_cascade(label='Row & column calculations', menu = menuDict['rowColCalc']) 
           menuDict['rowColCalc'].add_command(label='Summary Statistics', command = self.summarize)
            
           menuDict['rowColCalc'].add_cascade(label="Logarithmic", menu = menuDict['logCalc'])
           for logType in ['log2','-log2','ln','log10','-log10']:
               menuDict['logCalc'].add_command(label=logType, 
               				command = lambda logType = logType : self.transform_selected_columns(logType))               
           menuDict['rowColCalc'].add_command(label='Z-Score [row]', 
           		command = lambda transformation = 'Z-Score_row': self.transform_selected_columns(transformation)) 
           menuDict['rowColCalc'].add_command(label='Z-Score [columns]', 
           		command = lambda transformation = 'Z-Score_col': self.transform_selected_columns(transformation)) 
           
           for metric in rowColumnCalculations :
                    menuDict['rowColCalc'].add_command(label=metric, 
                    	command = lambda metric = metric: self.calculate_row_wise_metric(metric))
           menuDict['rowColCalc'].add_command(label='Kernel Density Estimation [col]', 
           												command = self.calculate_density)
           menuDict['main'].add_command(label ="Filters ..",state=tk.DISABLED,foreground='darkgrey')
           menuDict['main'].add_separator()
           menuDict['main'].add_command(label="Annotate Numeric Filter", command = self.numeric_filter_dialog)
           menuDict['main'].add_cascade(label="Categorical Filters", menu = menuDict['categories'])
           menuDict['categories'].add_command(label="Find Category & Annotate", 
           	command = lambda : self.categorical_column_handler('Find category & annotate'))
           menuDict['categories'].add_command(label="Find String(s) & Annotate", 
           	command = lambda: self.categorical_column_handler('Search string & annotate'))
           menuDict['categories'].add_command(label = 'Custom Categorical Filter', 
           	command = self.custom_filter) 
           menuDict['categories'].add_command(label = 'Subset Data on Category', 
           	command = lambda: self.categorical_column_handler('Subset data on unique category'))
           
           menuDict['main'].add_command(label ="Time series ..",state=tk.DISABLED,foreground='darkgrey')
           menuDict['main'].add_separator()
           
           menuDict['main'].add_command(label = 'Base line correction', command = self.correct_baseline)    
           menuDict['main'].add_command(label='Add as error' ,command = self.add_error)   
                      
           menuDict['main'].add_command(label ="Statistic ..",state=tk.DISABLED,foreground='darkgrey')
           menuDict['main'].add_separator()
           #menuDict['main'].add_command(label='Pairwise Comparision', command = self.get_all_combs)
           menuDict['main'].add_cascade(label='Multiple Testing Correction', menu = menuDict['multTest'])

           
           #menuDict['main'].add_command(label ="Fit, Correlate and Predict..",state=tk.DISABLED,foreground='darkgrey')
           menuDict['main'].add_separator()
           menuDict['main'].add_command(label="Correlate rows to ..." , command = self.calculate_correlations)
           
           menuDict['main'].add_cascade(label='Curve fit', menu= menuDict['curvefit'] )    
           menuDict['curvefit'].add_command(label="Curve fit of rows to...", command = self.curve_fit)
           menuDict['curvefit'].add_command(label="Display curve fit(s)", command = self.display_curve_fits)
           menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label='Predictions', menu= menuDict['predict'] )    
           menuDict['predict'].add_command(label = 'Predict Cluster', command = lambda: clustering.predictCluster(self.clusterCollection, self.sourceData, self.DataTreeview))
           menuDict['predict'].add_command(label = 'Predict Class', command = '')
           menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label='Transform by ..', menu= menuDict['transform'] )    
           menuDict['transform'].add_command(label = 'Dimensional Reduction Model', command = self.apply_dimRed)
           menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label='Export', menu= menuDict['export'] )    
           menuDict['export'].add_command(label='To excel', command = lambda: self.export_data_to_file(data = self.sourceData.df))
           menuDict['export'].add_command(label='To txt', command = lambda: self.export_data_to_file(data = self.sourceData.df, format_type = 'txt'))
           
           
     def build_pca_export_menu(self):
     
     	self.pca_export_menu = tk.Menu(self,**styleDict)
     	self.pca_export_menu.add_command(label='Define groups ..', 
     				command = self.define_groups_in_dimRed)
     	self.pca_export_menu.add_command(label='Remove/show feature names',  
     				command = lambda : self.plt.nonCategoricalPlotter.hide_show_feature_names())     				 
     	self.pca_export_menu.add_command(label='Export Components', 
     				command = lambda :  self.export_dimRed_results('Export Components')) 
     	self.pca_export_menu.add_command(label='Export Loadings', 
     				command = lambda: self.export_dimRed_results('Export Loadings')) 
     	self.pca_export_menu.add_command(label='Add Loadings To Source Data', 
     				command = lambda: self.export_dimRed_results('Add Loadings To Source Data'))  
     				    
     def build_hclust_menu(self):

         self.hclust_menu =  tk.Menu(self,**styleDict)      
         #self.hclust_menu.add_command(label='Cluster map', command = lambda: self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),
         #self.hclust_menu.add_command(label='Show row dendrogram',  foreground="black",command = lambda: self.calculate_dendogram(mode='row')) 
         #self.hclust_menu.add_command(label='Show column dendrogram',  foreground="black", command = self.calculate_dendogram)   
         self.hclust_menu.add_command(label='Add cluster # to source file', command = self.add_cluster_to_source)         
         self.hclust_menu.add_command(label='Find entry in hierarch. cluster', command = lambda: self.categorical_column_handler('Find entry in hierarch. cluster'))
         #self.hclust_menu.add_command(label='Modify hclust settings', command = lambda: self.design_popup(mode = 'Hierarchical Clustering Settings')) 
    
     def def_split_sub_menu(self):
         
         self.split_sub_menu =  tk.Menu(self, **styleDict)
         self.plotCumulativeDist = tk.BooleanVar()    
            
         self.split_sub_menu.add_checkbutton(label='Split Categories', 
         								variable = self.split_on_cats_for_plot , 
         								command = lambda: self.plt.set_split_settings(self.split_on_cats_for_plot.get()))    
         								   
         self.split_sub_menu.add_separator()
         self.split_sub_menu.add_checkbutton(label='Use Cumulative Density Function', variable = self.plotCumulativeDist , command = lambda: self.plt.set_dist_settings(self.plotCumulativeDist.get()))    
     
     def build_merge_menu(self):

         self.merge_data_frames_menu = tk.Menu(self, **styleDict)
         
         self.merge_data_frames_menu.add_command(label='Dataframe drop-down menu', state = tk.DISABLED,foreground="darkgrey")
         self.merge_data_frames_menu.add_separator()
         self.merge_data_frames_menu.add_command(label='Re-Sort columns', command = lambda: self.design_popup(mode='Re-Sort Columns'))
         self.merge_data_frames_menu.add_command(label='Concatenate', command = lambda: self.join_data_frames('Concatenate'))
         self.merge_data_frames_menu.add_command(label='Merge', command = lambda: self.join_data_frames('Merge')) 
         self.merge_data_frames_menu.add_command(label = "Delete", command = self.delete_data_frame_from_source)
     	 	
     	 		
     def build_datatype_menu(self):

         self.data_type_menu = tk.Menu(self, **styleDict)
         
         self.data_type_menu.add_command(label='Sort and rename columns', state = tk.DISABLED)
         self.data_type_menu.add_separator()
         self.data_type_menu.add_command(label = 'Custom column order', command  =  self.custom_column_order)#lambda: tk.messagebox.showinfo('Under revision','Currently under revision. Will be available in the next minor update.'))#self.design_popup(mode='Re-Sort'))
         self.data_type_menu.add_command(label='Sort columns alphabetically', command = self.re_sort_source_data_columns)
         self.data_type_menu.add_command(label='Colum names - Find and replace', command = lambda : findAndReplace.findAndReplaceDialog('ReplaceColumns',
         																				self.sourceData,self.DataTreeview))
         
     def build_selection_export_menu(self):
        
         self.selection_sub_menu =  tk.Menu(self, **styleDict)
         
         self.selection_sub_menu.add_command(label='Create sub-dataset', 
         									 command = self.create_sub_data_frame_from_selection)
         self.selection_sub_menu.add_command(label='Add annotation column', command = self.add_annotation_column_from_selection)
         self.selection_sub_menu.add_command(label='Exclude from Dataset', command = lambda: self.drop_selection_from_df())
         self.selection_sub_menu.add_command(label='Copy to clipboard', command = lambda: self.copy_file_to_clipboard(self.data_selection))
         self.selection_sub_menu.add_command(label='Export Selection [.txt]', 
         								command = lambda: self.export_data_to_file(self.data_selection, 
         																			initialfile='Selection', format_type = 'txt'))
         self.selection_sub_menu.add_command(label='Export Selection [.xlsx]', 
         								command = lambda: self.export_data_to_file(self.data_selection,
         																			initialfile='Selection',  format_type = 'Excel',sheet_name = 'SelectionExport'))
     	         
     def build_main_figure_menu(self):
         
         self.main_figure_menu = tk.Menu(self, **styleDict)         
         self.main_figure_menu.add_command(label='Add in main figure to ..',foreground='darkgrey')  
         self.main_figure_menu.add_separator()

         
     def build_corrMatrix_menu(self):
     	'''
     	'''
     	self.variableDict = {'pearson':tk.BooleanVar(value=True),
     					'spearman':tk.BooleanVar(value=False),
     					'kendall':tk.BooleanVar(value=False)}
     	
     	self.corrMatrixMenu = tk.Menu(self, **styleDict)
     	
     	self.corrMatrixMenu.add_command(label = ' Corr. method ', 
     									state = tk.DISABLED,foreground='darkgrey')
     	self.corrMatrixMenu.add_separator()
     	for method, variable in self.variableDict.items():
     		self.corrMatrixMenu.add_checkbutton(label = method, 
     											variable = variable, 
     											command = lambda method=method: self.update_corr_matrix_method(method))							
     	self.corrMatrixMenu.add_separator()
     	self.corrMatrixMenu.add_command(label='Results', command = self.display_corrmatrix_results)
     	
     	
     def post_menu(self, event = None, menu = None):
     	'''
     	Posts any given menu at the mouse x y coordinates
     	'''
     	x = self.winfo_pointerx()
     	y = self.winfo_pointery()
     	menu.post(x,y)

     def display_corrmatrix_results(self):
     	'''
     	'''
     	numColumns,_,plot_type,_ = self.plt.current_plot_settings
		
     	if self.plt.nonCategoricalPlotter._hclustPlotter is not None or \
     	plot_type != 'corrmatrix':
     		
     		data = self.plt.nonCategoricalPlotter._hclustPlotter.export_data_of_corrmatrix()
     		dataDialog = display_data.dataDisplayDialog(data,showOptionsToAddDf=True)
     		     		
     		if dataDialog.addDf:
     			nameOfDf = 'Corrmatrix Results {}'.format(get_elements_from_list_as_string(numColumns))
     			self.add_new_dataframe(dataDialog.data,nameOfDf)
     	
     	else:
     		tk.messagebox.showinfo('Error ..','No clustering performed yet.',parent=self)


     def rename_columns(self):
     	'''
     	'''
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		renameDialog = change_columnName.ColumnNameConfigurationPopup(self.DataTreeview.columnsSelected,
     														self.sourceData, self.DataTreeview)
     		if renameDialog.renamed: #indicates if any renaming was done (or closed)
     			tk.messagebox.showinfo('Done..','Column names were replaced.',parent=self)
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.',parent=self)
     	# reset before selected df
     	self.sourceData.set_current_data_by_id(currentDataFrameId)	

         	
         	
                            	     	       		
     def update_corr_matrix_method(self,method):
     	'''
     	Correlation matrix can be constructed from different correlation coefficients.
     	To ensure the selected one is used. We update the corrMatrixCoeff in the PlotterClass
     	'''
     	for corrMethod, variable in self.variableDict.items():
     		if method == corrMethod:
     			pass
     		else:
     			variable.set(False)
     			
     	self.plt.corrMatrixCoeff = method
     	
     	
     def apply_dimRed(self):
     	'''
     	'''
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		dimRedDialog = dimRed_transform.transformDimRedDialog(self.dimensionReductionCollection,
     															self.sourceData, self.DataTreeview)
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.',parent=self)
     															

     	#dimRedCollection, dfClass, dataTreeview,
     	
     def export_dimRed_results(self, which):
     	'''
     	Can be used to export results of a dimensional reduction either to file 
     	or to be added to source treeview and data collection
     	'''
     	if 'Export Components' in which:
     	
     		_,components,columns, dataID = \
     		self.dimensionReductionCollection.get_drivers_and_components(which='Components')
     		mainString = 'Components'
     		data = components.T
     		data['Feature'] = data.index 
     		
     	elif 'Export Loadings' in which or 'Source Data' in which:
     	
     		data,_,columns, dataID = \
     		self.dimensionReductionCollection.get_drivers_and_components(which='Drivers')
     		mainString = 'Loadings'
     		data.columns = ['Comp_{}'.format(n+1) for n in range(len(data.columns.values.tolist()))]
     		
     	if 'Source Data' in which:
     	
     		self.sourceData.set_current_data_by_id(dataID)
     		columnsAdded = \
     		self.sourceData.join_df_to_currently_selected_df(data, exportColumns = True)
     		self.DataTreeview.add_list_of_columns_to_treeview(dataID, 'float64', columnsAdded)
     	
     	else:
     		self.add_new_dataframe(data,
     						'{}: [{}]'.format(mainString,
     						get_elements_from_list_as_string(columns,
     						maxStringLength = 10)))
     
     def define_groups_in_dimRed(self):
     	'''
     	Dialog to define grouping in dimensional reduction procedure
     	'''
     	define_groups_dim_reduction.defineGroupsDialog(self.dimensionReductionCollection,
     												   self.plt,
     												   self.colorHelper)
     def custom_sort_values(self):
     	'''
     	'''
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     	
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		selectedColumns = self.DataTreeview.columnsSelected
     		inputValues = OrderedDict()
     		
     		uniqueValueList = self.sourceData.get_unique_values(selectedColumns,forceListOutput=True) 
     		
     		for n,column in enumerate(selectedColumns):
     			inputValues[column] = uniqueValueList[n]
     		dialog = custom_sort.customSortDialog(inputValues) 
     		
     		if dialog.resortedValues is not None:
     			idxColNames = []
     			for key,values in dialog.resortedValues.items():
     				factors = range(len(values))
     				#kind of factorize for sorting
     				orderMapDict = dict(zip(values,factors))
     				idxColName = self.sourceData.evaluate_column_name('instantClueSort{}'.format(key))
     				idxColNames.append(idxColName)
     				self.sourceData.df[idxColName] = self.sourceData.df[key].astype(str).map(orderMapDict)     				
     				del orderMapDict
     			
     			self.sourceData.df.sort_values(idxColNames,kind='mergesort',inplace=True) 
     			self.sourceData.delete_columns_by_label_list(idxColNames)
     			tk.messagebox.showinfo('Done ..','Customary sorting done.',parent=self)
     
     		else:
     			pass
     			
     		del dialog
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.',parent=self)


     def custom_column_order(self):
     	'''
     	'''
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     	
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		
     		selectedDataTypes = self.DataTreeview.allItemsSelected
     		inputValues = OrderedDict()     		
     		for item in selectedDataTypes:
     			dataType = item.split('{}_'.format(selectionDataFrameId))[-1]
     			inputValues[dataType] = self.sourceData.get_columns_data_type_relationship()[dataType]
     			
     		dialog = custom_sort.customSortDialog(inputValues) 
     		
     		if dialog.resortedValues is not None:
     			columnOrder = []
     			for key,values in dialog.resortedValues.items():
     				columnOrder.extend(values)
     			df1 = self.sourceData.df[columnOrder]	
     			dfOut = self.sourceData.join_missing_columns_to_other_df(df1,id=selectionDataFrameId)     			
     			self.sourceData.update_data_frame(selectionDataFrameId,dfOut)
     			self.update_all_dfs_in_treeview()
     			tk.messagebox.showinfo('Done ..','Customary sorting done.',parent=self)
     		else:
     			pass
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only data types from one file.',parent=self)



     	
     def summarize(self):
     	'''
     	Summarize Table 
     	'''
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		selectedColumns = self.DataTreeview.columnsSelected  
     		summarizedData = self.sourceData.df[selectedColumns].describe(
     													percentiles = [.25, .5, .75],
     													include = 'all')
     		countNanValues = self.sourceData.df[selectedColumns].isnull().sum()
     		summarizedData.loc['nan count',:] = countNanValues
     		summarizedData.insert(0,'Measure',summarizedData.index)
     		dataDialog = display_data.dataDisplayDialog(summarizedData,showOptionsToAddDf=True)
     		     		
     		if dataDialog.addDf:
     			nameOfDf = 'Summary of {}'.format(get_elements_from_list_as_string(selectedColumns))
     			self.add_new_dataframe(dataDialog.data,nameOfDf)
     	else:
     		
      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.',parent=self)
     
     
     def transpose_data(self):
     	'''
     	Transpose Data
     	'''
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		selectedColumns = self.DataTreeview.columnsSelected  
     		
     		
     		selectColumnDialog = \
     		pivot_table.transformColumnDialog(selectedColumns[0],self.sourceData.df_columns)
     		
     		columnForColumns = selectColumnDialog.columnForColumns
     		if columnForColumns  is None:
     			del selectColumnDialog
     			return
     		if columnForColumns not in self.sourceData.df_columns:
     			newColumns = [str(x) for x in self.sourceData.df.index.tolist()]
     		else:
     			uniqueValues = self.sourceData.get_unique_values(columnForColumns) 
     		
     			if uniqueValues.size != len(self.sourceData.df.index):
     				quest = tk.messagebox.askquestion('Error ..','Number of unique values in selected column does not'+
     									' match the number of rows.\nWould you like to make them unique '+
     									'by adding the value index?')
     				if quest == 'yes':
     					columnValues = self.sourceData.df[columnForColumns].values
     					newColumns = ['{}_{}'.format(column,n) for n,column in enumerate(columnValues)]
     				else:
     					return
     			else:
     				newColumns = self.sourceData.df[columnForColumns]
     		# transpose data
     		data = self.sourceData.df.transpose()
     		# add new column names (selected by user)
     		data.columns = newColumns
     		# add index as pure numbers
     		data.index = np.arange(0,len(data.index)) 
     		# inser a column with index holding old columns
     		indexName = self.sourceData.evaluate_column_name('Index',newColumns)
     		data.insert(0,indexName,self.sourceData.df_columns)
     		
     		dataDialog = display_data.dataDisplayDialog(data,showOptionsToAddDf=True)
     		     		
     		if dataDialog.addDf:
     			nameOfDf = 'Transpose - {}'.format(self.sourceData.get_file_name_of_current_data())
     			self.add_new_dataframe(data,nameOfDf)
     	else:
     		
      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.',parent=self)     
     	     	
	     
     def pivot_data(self):
     	'''
     	Perform pivot Table 
     	'''
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		pivotDialog = pivot_table.pivotDialog(self.sourceData,selectionDataFrameId)
     		data = pivotDialog.pivotedDf
     		if data.empty:
     			return
     		dataDialog = display_data.dataDisplayDialog(data,showOptionsToAddDf=True)
     		     		
     		if dataDialog.addDf:
     			nameOfDf = 'Pivot - {}'.format(self.sourceData.get_file_name_of_current_data())
     			self.add_new_dataframe(data,nameOfDf)
     	else:
     		
      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.',parent=self)     
     	
     	
	
     def calculate_correlations(self):
     	'''
     	Calculates correlation of rows against given values 
     	'''
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData: 
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     	else:
     		tk.messagebox.showinfo('Error ..',
     			'Please select only columns from one data frame',parent=self)
     		return
     	if self.DataTreeview.onlyNumericColumnsSelected == False:
     		tk.messagebox.showinfo('Error ..','Please select only numeric data.',parent=self) 
     		return
     		   	
     	corrDialog = correlations.correlationDialog(dfClass=self.sourceData, 
     										selectedColumns = self.DataTreeview.columnsSelected)
     	
     	cor_data = corrDialog.get_correlations() 
     	if cor_data.empty:
     		return
     	
     	columnsAddedToDf = self.sourceData.join_df_to_currently_selected_df(cor_data, exportColumns = True)
     	self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,'float64',columnsAddedToDf)
     	     	
     	tk.messagebox.showinfo('Done..','Calculations were performed and data were added to the source data treeview.',parent=self) 
     	             
             
     def custom_filter(self):
     ## seleceted data ?
     
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData: 
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     	else:
     		tk.messagebox.showinfo('Error ..',
     			'Please select only columns from one data frame',parent=self)
     		return
     	
     	customFilter = custom_filter.customFilterDialog(self.sourceData, self.DataTreeview.columnsSelected)
     	
     	
     	data,mode, match_annotation = customFilter.get_data() 
     	del customFilter
     	currentFileName = self.sourceData.get_file_name_of_current_data()
     	
     	if mode == 'remove':
     		self.sourceData.update_data_frame(self.sourceData.currentDataFile, data)     		
     		
     	elif mode == 'subset':
			
         	nameOfSubset = 'Filter_Subset: '+currentFileName
         	self.add_new_dataframe(data,nameOfSubset)
         	
     	elif mode == 'annotate':
     	
     		nameOfColumn =  \
     					get_elements_from_list_as_string(self.DataTreeview.columnsSelected, 
     													addString = 'CustomFilter: ',
     													newLine = False)
     													
     		columnName = self.sourceData.evaluate_column_name(nameOfColumn)
     	     		
     		if match_annotation == False:
        
        		idx_ = pd.Series(self.sourceData.df.index).isin(data.index)
        		replace_dict = {True : "+",
                         	   False: self.sourceData.replaceObjectNan
                         	   }
        		outputColumn = idx_.map(replace_dict)
        		self.sourceData.add_column_to_current_data(columnName,outputColumn)
        		
     		else:
     		
        		joinHelper = pd.DataFrame(data, columns= [columnName], index= data.index)
        		self.sourceData.join_df_to_currently_selected_df(joinHelper)
        		del joinHelper
     		self.sourceData.change_data_type_in_current_data(columnName,'object')
     		self.DataTreeview.add_list_of_columns_to_treeview(id = self.sourceData.currentDataFile,
        													dataType = 'object',
        													columnList = [columnName])
            
     	if mode is not None:

        	tk.messagebox.showinfo('Done..','Custom filter was applied successfully. Well done.',parent=self) 
          	

     def perform_export(self, axExport,axisId,figureTemplate, exportId = None):
     		### to do - log transofrm of axes .. 
     		## 
         '''
         Performs the export.
         Parameter
         ==========
         axExport - matplotlib axis. The axis that should be used for export
         axisId   - id for given axExport in the main figure template collection
         figureTemplete  - dict that contains the figure (key = 'figure') on which the
         				   axis axExport is on and the template (key = 'template') which
         				   is the class that stores all axes and texts etc in a main
         				   figure template
         				   
         '''
     		
         figureTemplate['template'].clear_axis(axExport)
		
         plotExporter = self.plt.get_active_helper()
         
         plotExporter.export_selection(axExport,self.axNum,figureTemplate['figure'])
         
         if hasattr(plotExporter,'_hclustPlotter'):
         	if plotExporter._hclustPlotter is not None:
         		exportId = plotExporter._hclustPlotter.exportId
         		axes = plotExporter._hclustPlotter.exportAxes[exportId]
         		figureTemplate['template'].associate_axes_with_id(axisId,axes)
         figureTemplate['template'].extract_artists(axExport,axisId)
         figureTemplate['template'].add_axis_label(axExport,
         										   axisId,
         										   label=figureTemplate['template'].figureProps[axisId]['axisLabel'])
         figureTemplate['figure'].canvas.draw()
         
         self.mainFigureCollection.store_export(axisId,
         										figureTemplate['template'].figureId,
         										self.plt.plotCount,
         										self.axNum,
         										exportId,
         										bool(self.plt.showSubplotBox),
         										bool(self.plt.showGrid))
                   
         
     def display_curve_fits(self):      
      	'''
      	Display curve fits that were made. The user can define a name for each curve
      	fit. 
      	'''
      	selectFitAndGrid = curve_fitting.displayCurveFitting(self.sourceData,self.plt,self.curveFitCollection) 
      	fitsToPlot = selectFitAndGrid.curve_fits_to_plot  
      	categoricalColumns = self.curveFitCollection.get_columns_of_fitIds(fitsToPlot)
      	if len(categoricalColumns) > 0:
      		self.plt.set_selectedCurveFits(fitsToPlot)
      		self.plt.initiate_chart(numericColumns = [], categoricalColumns = categoricalColumns ,
      								 selectedPlotType = 'curve_fit', colorMap = self.cmap_in_use.get())
      	
      	else:
      		pass	
      	
      
     def export_selected_figure(self,event):
         '''
         Cast a menu to export subplots from the main window into a 
         main figure template.
         '''
     	 
         if event.inaxes is None:
             return
         if event.button == 1:
             return
         
         if event.button in [2,3]:
         	
             if self.selection_press_event is not None:
             	return         
             numb = self.plt.get_number_of_axis(event.inaxes)
             if numb is None:
                 return
                 
             self.axNum = numb
             self.ax_export_ax = event.inaxes
             
             if self.plt.castMenu == False:
             	return                             
             self.post_menu(menu = self.main_figure_menu)        
        
        
       
     def calculate_density(self):
     	'''
     	Calculates kernel density estimate and adds new column to the datatreeview
     	'''
     	if self.DataTreeview.onlyNumericColumnsSelected == False:
     		tk.messagebox.showinfo('Error ..','Please select only numerical columns for this type of calculation.',parent=self)
     		return
     	
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		
     		numericColumns = self.DataTreeview.columnsSelected  
     		
     		densityColumnName = self.sourceData.calculate_kernel_density_estimate(numericColumns)
     		
     		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,dataType = 'float64',
     														columnList = [densityColumnName])
     											
     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		
     		tk.messagebox.showinfo('Done ..','Representation of a kernel-density estimated using '+
     								'Gaussian kernels calculated. Column was added.', parent = self)
     	else:
     		
      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.',parent=self)
     		     	        
     		     	        
         
     def destroy_tt(self,event):
     	try:
     		self.tooltip_data.destroy()   
     	except:
     		pass
     
     	
     def show_tt(self,text = None):
     	
     			if self.tooltip_data is not None:
     				self.tooltip_data.destroy()
     			if text is None:
     				text = self.tt_text	
     			self.tooltip_data = tk.Toplevel(background="white")
     			if platform == 'MAC':     			
     				self.tooltip_data.tk.call("::tk::unsupported::MacWindowStyle","style",self.tooltip_data._w, "plain", "none")
     			else:
     				self.tooltip_data.wm_overrideredirect(True)
     				
     			self.tooltip_data.attributes('-topmost',True,'-alpha',0.955)
     				 
     			label = tk.Label(self.tooltip_data, text=text, font = NORM_FONT,relief=tk.SOLID,
                          borderwidth=0.0,
                          wraplength=250, justify=tk.LEFT,
                                         background = "white")
     			label.pack(padx=8,pady=8)
     			x = self.winfo_pointerx()
     			y = self.winfo_pointery()
     			self.focus_force()
     			self.source_treeview.focus_set() 

     			self.tooltip_data.wm_geometry("+%d+%d" % (x+8, y+8))
     			self.tooltip_data.bind('<Motion>', self.destroy_tt)
     			self.tooltip_data.bind('<ButtonPress>',self.destroy_tt)
     			
     def identify_item_and_start_tooltip(self,event):
     	
     
         iid = self.source_treeview.identify_row(event.y)
         
         for iid_ in self.DataTreeview.columnsIidSelected:
         	self.source_treeview.selection_remove(iid_)
         self.source_treeview.selection_add(iid) 	
         #if iid in self.items_selected and len(self.items_selected) > 1:
             
                 
            #    
#          if iid:
#           
#           im = self.source_treeview.item(iid)['image']
#           if im != '':
#               info = self.data_set_information[iid] 
#               if info[1] in  [np.float64,np.int64]:
#                   text = 'Name: {}\nTotal: {}\nMean: {}\nMedian: {}\nStd. dev.: {}\n[Min,Max]: {}\nValid values: {}'.format(info[0],info[2],info[3],info[4],info[5],info[7],info[6]) 
#               else:
#                   text = 'Name: {}\nTotal: {}\nUnique categories: {}'.format(info[0],info[2],info[5]) 
#               self.tt_text = text
#               self.show_tt(text)
#           else:
#               if self.source_treeview.parent(iid) == '':
#                   pass
#               else:
#                   pass
              
     
     def select_data(self):
     	'''
     	Triggers data free-hand selection. 
     	'''
     	if self.plt.currentPlotType != 'scatter':
     		tk.messagebox.showinfo('Error ..','Only useful for scatter plots.', parent=self)
     		return
     	dataId = self.plt.get_dataID_used_for_last_chart()
     	self.sourceData.set_current_data_by_id(dataId)
     	
     	numericColumns = list(self.selectedNumericalColumns.keys())
     	# important to take data from the plot because it might resort the data
     	# to display categorical columns
     	self.slectionDataAsTuple = self.sourceData.get_data_as_list_of_tuples(numericColumns,
     											data = self.plt.nonCategoricalPlotter.data)
     	self.selection_press_event = \
     	self.canvas.mpl_connect('button_press_event', lambda event: \
     	self.on_selection(event))
     	
     
     def drop_selection_from_df(self):
         '''
         Drops rows from data frame and reinitiates chart.
         '''
         self.sourceData.delete_rows_by_index(self.data_selection.index)
         self.plt.figure.canvas.mpl_disconnect(self.selection_press_event)
         self.plt.initiate_chart(*self.plt.current_plot_settings)
         ## update data if more selection should be performed
         self.select_data()
         #self.slectionDataAsTuple =  self.sourceData.get_data_as_list_of_tuples(numericColumns,
     									#		data = self.plt.nonCategoricalPlotter.data)                 
      #   print(self.slectionDataAsTuple)
         
     def on_selection(self,event):
             '''
             On free-hand selection - left click starts Lasso,
             right - click starts menu. 
             
             Parameter
             ===========
             event - matplotlib button press event
             '''
             if event.button == 1:
                 if self.canvas.widgetlock.locked():
                     return
                 if event.inaxes is None:
                     return
                 self.lasso = Lasso(event.inaxes,
                               (event.xdata, event.ydata),
                               lambda verts: 
                               self.selection_callback(verts))
                               
                 self.lasso.line.set_linestyle('--')
                 self.lasso.line.set_linewidth(0.3)
 
             elif event.button in [2,3]:
                 self.post_menu(menu=self.selection_sub_menu)
 
     
     def selection_callback(self, verts):
             '''
             Get indices of selected data and marks them in current scatter
             plot.
             '''
             p = path.Path(verts)
             if p is None:
                 return 
             indexList = p.contains_points(self.slectionDataAsTuple)
             self.data_selection = self.plt.nonCategoricalPlotter.data.iloc[indexList]
             columns = self.plt.nonCategoricalPlotter.numericColumns
             ax = self.plt.nonCategoricalPlotter.axisDict[0]
             ## plot selection
             self.plt.add_scatter_collection(ax,self.data_selection[columns[0]],
             								self.data_selection[columns[1]],
             								color = 'red')
             del self.lasso   
             self.plt.redraw()       
   
     
     def add_cluster_to_source(self):
     	'''
     	Cluster being identified in a hclust plot can be added to the source file.
     	'''
     	if self.plt.currentPlotType in ['hclust','corrmatrix']:
     		plotterInUse = self.plt.nonCategoricalPlotter
     		idData = plotterInUse._hclustPlotter.dataID
     		self.sourceData.set_current_data_by_id(idData) 
     		columnName = plotterInUse._hclustPlotter.export_cluster_number_to_source()
		
     		self.DataTreeview.add_list_of_columns_to_treeview(idData,
     													dataType = 'object',
     													columnList = [columnName])
     		
     		tk.messagebox.showinfo('Done ..','Cluster numbers were added.', parent=self)
     
             
     def copy_file_to_clipboard(self, data):
         '''
         Copies data to clipboard
         '''
         data.to_clipboard(excel=True, na_rep = "NaN",index=False, encoding='utf-8', sep='\t') 

         
     def create_sub_data_frame_from_selection(self):
         '''
         When user has defined data selection by Lasso, a new data frame is created
         and added to the treeview
         '''
         sub_data = self.sourceData.df[self.sourceData.df.index.isin(self.data_selection.index)]
         currentFileName = self.sourceData.fileNameByID[self.plt.get_dataID_used_for_last_chart()]
         nameOfSubset = 'selection_{}'.format(currentFileName)
         self.add_new_dataframe(sub_data,nameOfSubset)
         
         
     def add_annotation_column_from_selection(self):
     	'''
     	When User uses the selection tool. 
     	This can be used to 
     	annotate these in the source
     	data for further analysis.
     	'''
     	# check if data Id has been changed before
     	colnames = list(self.selectedNumericalColumns.keys())
     	dataID = self.plt.get_dataID_used_for_last_chart()
     	self.sourceData.set_current_data_by_id(dataID)
     	selectionIndex = self.data_selection.index
     	
     	columnName = 'Select_{}_{}'.format(len(selectionIndex),
     								get_elements_from_list_as_string(colnames))
     	columnName = self.sourceData.evaluate_column_name(columnName)
     	
     					
     	true_false_map = dict(zip([False,True], [self.sourceData.replaceObjectNan,'+']))
     	
     	boolIndicator = pd.Series(self.sourceData.df.index.isin(selectionIndex), index = self.sourceData.df.index,
     							name = columnName)
     	annotationColumn = boolIndicator.apply(lambda x: true_false_map[x]) 
     	self.sourceData.join_series_to_currently_selected_df(annotationColumn)
     	self.DataTreeview.add_list_of_columns_to_treeview(dataID,'object',[columnName])
     	tk.messagebox.showinfo('Done ..', \
     	'Categorical column ({}) has been added. Indicating if row was in selection.'.format(columnName))

     
     def add_error(self):
     	'''
     	
     	'''
     	if self.plt.currentPlotType != 'time_series':
     		tk.messagebox.showinfo('Error..','Only useful for time series.')
     		return
   		
     	if self.DataTreeview.onlyNumericColumnsSelected:
     		currentDataFrameId = self.sourceData.currentDataFile
     		selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     		if selectionIsFromSameData:
     			
     			dataId = self.plt.get_dataID_used_for_last_chart()
     			print(dataId, selectionDataFrameId) 
     			if dataId != selectionDataFrameId:
     				tk.messagebox.showinfo('Error..','Error data must be from the same data used for plotting')
     				return
     			self.sourceData.set_current_data_by_id(selectionDataFrameId)     	     	
     			numericColumnsPlotted = self.plt.nonCategoricalPlotter.numericColumns[1:]
     			selectedColumns = self.DataTreeview.columnsSelected 
     			numColumns = self.sourceData.get_numeric_columns()
     			options = [] #options * len(selectedColumns)
     			for label in numericColumnsPlotted:
     				options.append(numColumns)
     			dialog = simpleUserInputDialog(numericColumnsPlotted,selectedColumns,options,
     								 title = 'Select Error Columns',
     								 infoText = 'Select columns that hold the error. For'+
     								 ' example the standard deviation of several signals over'
     								 ' time.\nIf you do not want to plot the error for one of'+
     								 ' the plotted columns simply enter "None".'
     								 )
     			selection = dialog.selectionOutput
     			if len(selection) != 0:
     				self.plt.nonCategoricalPlotter.timeSeriesHelper.add_error_to_lines(selection)
     																		
     																					
     				self.plt.redraw()
     								 
     			
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only numerical columns (floats, and integers)')
     	
     		
      

         
         
         
     def reduce_tick_to_n(self,ax,tick,n, rotation = 0):
         
         def get_lab(label,i,idx_):
             if i in idx_:
                 return label
             else:
                 return '' 
         if tick == 'y':    
             labels = ax.get_yticklabels() 
         else:
             labels = ax.get_xticklabels()        
         n_labs = len(labels)
         idx_ = [int(round(x,0)) for x in np.linspace(0,n_labs-1,num=n)]
         
         new_labs = [get_lab(label,i,idx_) for i,label in enumerate(labels)] 

         if tick == 'y':
             ax.set_yticklabels(new_labs)
         else:
              ax.set_xticklabels(new_labs, rotation =rotation)         

       

     def iir_filter(self):
     	'''
     	Smoothing data by iir filter. 
     	'''
     	if self.DataTreeview.onlyNumericColumnsSelected == False:
     		tk.messagebox.showinfo('Error ..','Please select only numerical columns for this type of calculation.')
     		return
     		
     	n = ts.askinteger('IIR Filter - N',prompt='Provide number n for filtering.\nThe higher the number the smoother the outcome.', initialvalue = 20, minvalue = 1, maxvalue = len(self.sourceData.df.index)) 
     	if n is None:
     		return
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
		
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		newColumnNames = self.sourceData.iir_filter(self.DataTreeview.columnsSelected  ,n)  
     		
     		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,
     													dataType = 'float64',
     													columnList = newColumnNames)
     		
     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		tk.messagebox.showinfo('Done ..','IIR Filter calculated. New columns were added.')
	
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     		return
     													   

         
     def create_count_through_column(self):
     	'''
     	Counts through the data in current order.
     	'''
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
		     	
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		columnName = self.sourceData.add_count_through_column()
     		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,
     													dataType = 'int64',
     													columnList = [columnName],
     													startIndex = -1)
     		
     		
     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		tk.messagebox.showinfo('Done ..','Index column was added to the treeview.')
	
         
         
     def multiple_comparision_correction(self,method,alpha = 0.05):
     	 '''
     	 Checks if column is numerical. And then computes the selected method.
     	 ''' 
     	 numericalColumns = self.DataTreeview.columnsSelected  
     	 if self.DataTreeview.onlyNumericColumnsSelected == False:
     	 	tk.messagebox.showinfo('Select float ..',
     	 						   'Please select a numerical column or change the data type.')
     	 	return
     	 currentDataFrameId = self.sourceData.currentDataFile
     	 selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	 if selectionIsFromSameData:	
     	 	self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     	 	method = multCorrAbbr[method]
     	 	if method == 'fdr_tsbky':
     	 		alpha = ts.askfloat(title = 'Set alpha..',
     	 						prompt='You have to provide an alpha for the two stage FDR'+
     	 						' calculations a priori. Note that the corrected p-values '+
     	 						'are not valid for other alphas. You have to compute them '+
     	 						'again when switchting to another alpha!',
     	 						initialvalue = 0.05, minvalue = 0, maxvalue = 1)
     	 		if alpha is None:
     	 			return
     	 	corrColumns = []
     	 	for col in numericalColumns:
     	 		if self.sourceData.df[col].min() < 0 or self.sourceData.df[col].max() > 1:
     	 			tk.messagebox.showinfo('Error..','You need to select an untransformed p-value column with data in [0,1]. If you have -log10 transformed p-values please transform them first using 10^p.') 
     	 			continue
     	 		data_ = self.sourceData.df.dropna(subset=[col]) 
     	 		if 'storey' not in method:
     	 			reject, corr_pvals,_,_ = multipletests(data_[col], alpha = alpha, 
     	 									 method = method, is_sorted= False, returnsorted=False) 	
     	 		else:
     	 			corr_pvals, pi0 = stats.estimateQValue(data_[col].values)
     	 		if method =='fdr_tsbky':
     	 			newCol = 'Alpha_'+str(alpha)+'_corr_pVal_'+col
     	 		elif 'storey' in method:
     	 			newCol = 'qValue_pi_'+str(pi0)+'_'+col
     	 		else:
     	 			newCol = 'corr_pVal_'+method+'_'+col 
     	 		
     	 		toBeJoined = pd.DataFrame(corr_pvals,
     	 							  columns=[newCol],
     	 							  index= data_.index)
     	 							  
     	 		self.sourceData.join_df_to_currently_selected_df(toBeJoined)
     	 	
     	 		del toBeJoined	
     	 		corrColumns.append(newCol) 
     	 	
     	 	self.DataTreeview.add_list_of_columns_to_treeview(id = self.sourceData.currentDataFile,
     	 												   dataType = ['float64'],
     	 												   columnList = corrColumns)
     	 	tk.messagebox.showinfo('Done ..','Calculations performed. Corrected p'
     	 									 '-values were added.')
     	 												   
     	 else:
     	 	tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     	 	return
     	 
     def calculate_row_wise_metric(self,metric,columns= None,promptN = 0):
     	'''
     	calculates row-wise data transformations/metrices
     	'''
     	if self.DataTreeview.onlyNumericColumnsSelected == False:
     		tk.messagebox.showinfo('Error ..','Please select only numerical columns for this type of calculation.')
     		return

     	if columns is None:
     		columns = self.DataTreeview.columnsSelected   
     	
     	askFloatTitlePrompt = dict(
     	
     					[('x ^ N [row]',['Power ..','Pleaser enter N to calculate: x^N: ']),
						('x * N [row]',['Multiply ..','Please enter N to calcuate: x *(N): ']), 
						('N ^ x [row]',['N^x','Please enter N to calcuate: N^x: '])]
						)
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:

     		if metric in askFloatTitlePrompt:
     			title, prompt =  askFloatTitlePrompt[metric]
     			## changed to use askstring instread of asfloat because 
     			## the function float() can also interprete entered strings
     			## like 1/400
     			promptN = ts.askstring(title,prompt,initialvalue='2')
     			promptN = float(promptN)
     			
     			if promptN is None:
     				return
     	
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		newColumnNames = self.sourceData.calculate_row_wise_metric(metric,columns,promptN) 
     		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,
     													dataType = 'float64',
     													columnList = newColumnNames)
     		tk.messagebox.showinfo('Done ..','Calculations performed. Columns added.')
     	
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     		return     
                  
       
       
             
                    
     def correct_baseline(self):
     	'''
     	'''
     	if self.plt.currentPlotType != 'time_series':
     		tk.messagebox.showinfo('Error..','Only useful for time series.')
     		return
     	if self.DataTreeview.onlyNumericColumnsSelected	== False:
     		tk.messagebox.showinfo('Error ..','Only numeric columns allowed.')
     		return
     	selectedColumns = self.DataTreeview.columnsSelected
     	selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:	
     	 	self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     	 	dataId = self.plt.get_dataID_used_for_last_chart()
     	 	if dataId != selectionDataFrameId:
     	 		tk.messagebox.showinfo('Error ..','Data frame of selected columns and the one used for plotting do not match!')
     	 		return
     	 	
     	 	self.plt.nonCategoricalPlotter.timeSeriesHelper.activate_baselineCorr_or_aucCalc(columns = selectedColumns,
     	 																					DataTreeview = self.DataTreeview)
 
     	 												   
     	else:
     	 	tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     	 	return     		
             
         
     def rolling_mod_data(self, rollingMetric, columns = None, quantile=0.5):
     	'''
     	Rolling window metric calculation.
     	'''
     	if columns is None:
     		columns =  self.DataTreeview.columnsSelected  
     	window = ts.askinteger('Set window size...',prompt = 'Please set window for rolling.\nIf window=10, 10 following values are\nused to calculate the '+str(rollingMetric)+'.', initialvalue = 10)
     	if window is None:
     		return
     		
     	if rollingMetric == 'quantile':
     		quantile = ts.askfloat('Set quantile...',prompt = 'Please set quantile\nMust be in (0,1):', initialvalue = 0.75, minvalue = 0.0001, maxvalue = 0.99999)
     		
     		if quantile is None:
     			return
		                       
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     	
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     	
     		newColumnNames = self.sourceData.calculate_rolling_metric(columns,window,rollingMetric,quantile)
		
     		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,'float64',newColumnNames)

     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		tk.messagebox.showinfo('Done ..','Rollling performed. New columns were added.')

     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     		return
     													   
        
             

         
     def save_current_session(self):
     	'''
     	There are x main classes that we need to restart a session.
     	'''
     	tk.messagebox.showinfo('Note ..','Please not style changes on axis labels/ticks in main'+
     							' figure templates are currently not saved.', parent=self)
     	## to save the last axis limits change
     	self.plt.save_axis_limits()
     	## create collection to be saved
     	saveCollectionDict = {'plotter':self.plt,
     						  'sourceData':self.sourceData,
     						  'mainFigureCollection':self.mainFigureCollection,
     						  'curveFitCoellection':self.curveFitCollection,
     						  'clusterAnalysis':self.clusterCollection,
     						  'classificationAnalysis':self.classificationCollection,
     						  'anovaTests':self.anovaTestCollection,
     						  'dimReductionTests':self.dimensionReductionCollection}
     						  
     	try:					  
     		performed = save_and_load_sessions.save_session(saveCollectionDict)
     		if performed != False:
     			tk.messagebox.showinfo('Done..','Session has been saved.')
     	except Exception as e:
     		tk.messagebox.showinfo('Error ..','Session not saved.\nError {}'.format(e))               
         
           
     def clean_up_dropped_buttons(self, mode = 'all', replot = True):
         if mode == 'all':
             for button in self.selectedNumericalColumns.values():
                      button.destroy() 
             for button in self.selectedCategories.values():
                      button.destroy()    
             self.plt.clean_up_figure()
             self.interactiveWidgetHelper.clean_frame_up()
             self.selectedNumericalColumns.clear()
             self.selectedCategories.clear()

         elif mode == 'num':
             for button in self.selectedNumericalColumns.values():
                      button.destroy() 
             
             self.selectedNumericalColumns.clear() 
             self.but_stored[9].configure(image= self.add_swarm_icon)  
             if replot:       
             	plot_type = self.estimate_plot_type_for_default() 
             	self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),
                                             catnames = list(self.selectedCategories.keys() ),
                                                            plot_type = plot_type)          
         elif mode == 'cat':              
                for button in self.selectedCategories.values():
                      button.destroy()   
                self.selectedCategories.clear() 
                if replot:     
                	plot_type = self.estimate_plot_type_for_default() 
                	self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),
                                             catnames = list(self.selectedCategories.keys() ),
                                                            plot_type = plot_type)       
                  
     def open_saved_session(self):
     	

         savedSession = save_and_load_sessions.open_session()
         if savedSession is None:
         	return
         elif savedSession == 'Not pckl found':
         	tk.messagebox.showinfo('Not found ..','Could not find a saved session file in selected directory..')
         	return
         self.plt = savedSession['plotter']
         self.sourceData = savedSession['sourceData']
         self.mainFigureCollection = savedSession['mainFigureCollection']
         self.curveFitcollection = savedSession['curveFitCoellection']
         self.clusterCollection = savedSession['clusterAnalysis']
         self.classificationCollection = savedSession['classificationAnalysis']
         self.anovaTestCollection = savedSession['anovaTests']
         self.dimensionReductionCollection = savedSession['dimReductionTests']
         self.plt.define_new_figure(self.f1)
         self.plt.reinitiate_chart()
         if self.plt.nonCategoricalPlotter is not None:
         	if self.plt.nonCategoricalPlotter.createIntWidgets:
         		self.interactiveWidgetHelper.create_widgets(plotter=self.plt)
         
         dataTypeColumCorrelation = self.sourceData.dfsDataTypesAndColumnNames
         file_names = self.sourceData.fileNameByID
         self.DataTreeview.add_all_data_frame_columns_from_dict(dataTypeColumCorrelation,file_names)
         if self.plt.plotCount in self.plt.plotHistory:
         	numericColumns, categoricalColumns  = self.plt.get_active_helper().columns
         	self.place_buttons_in_receiverbox(numericColumns, dtype = 'numeric')
         	self.place_buttons_in_receiverbox(categoricalColumns, dtype = 'category')
         
         self.open_main_figures()
         tk.messagebox.showinfo('Done ..','Session loaded. Happy working.', parent=self)
     	
     def open_main_figures(self): 
     	'''
     	'''
     	## reinitiate self.mainFigures dict since we had to delete it before save
     	self.mainFigureCollection.mainFigures = dict()
     	self.mainFigureCollection.analyze = self
     	## 
     	createdFigs = self.mainFigureCollection.mainFigureTemplates     	
     	for figureId, axisParameters in createdFigs.items():
     		
     		main_figures.mainFigureTemplateDialog(self.mainFigureCollection,figureId)
     		
     		self.mainFigureCollection.mainFigures[figureId]['template'].restore_axes(axisParameters)
     		axisDict = self.mainFigureCollection.exportDetails[figureId]
     		
     		figText = self.mainFigureCollection.figText[figureId] 
     		if len(figText) > 0:
     			for id , props in figText.items():
     				text = self.mainFigureCollection.mainFigures[figureId]['figure'].text(**props)
     				self.mainFigureCollection.mainFigures[figureId]['template'].textsAdded[id] = text
     		
     		self.unpack_exports(axisDict,figureId)
     		self.mainFigureCollection.mainFigures[figureId]['figure'].canvas.draw()		     		
     		
     		     		
     def unpack_exports(self,axisDict,figureId,specAxisId=None,specificAxis = None,transferAxisId = None): 
     		for axisId,exportDetails in axisDict.items():
     			if specAxisId is not None:
     				if axisId != specAxisId:
     					continue
     				else:
     					axisId = transferAxisId
     			if 'path' in exportDetails:
     				try:
     					self.mainFigureCollection.mainFigures[figureId]\
     					['template'].add_image_to_axis(pathToFile = exportDetails['path'], \
     					axisId =axisId)
     				except:
     					tk.messagebox.showinfo('Error..',
     						'Could not load image. Moved to another place?',
     						parent = self)
     				
     				if transferAxisId  is not None:
     					break
     					
     					
     				
     			else:
     				plotter = [plotter for plotter in self.plt.plotHistory[exportDetails['plotCount']]\
     				if plotter is not None]
     				_,_,plotType,_ = self.plt.plotProperties[exportDetails['plotCount']]
     				
     				# check if we had a hclust or corrmatrix 
     				# special because they have inmutable axes
     				if plotType in ['hclust','corrmatrix']:
     						storeId = plotter[0]._hclustPlotter.exportId
     						plotter[0]._hclustPlotter.fromSavedSession = True 
     						plotter[0]._hclustPlotter.exportId =  exportDetails['exportId']
     				else:
     					storeId = None
     				## limits -> axis limits used when exported on main figure template
     				limits = exportDetails['limits']
     				if specificAxis is None:
     					ax = self.mainFigureCollection.mainFigures[figureId]['template'].figureProps[axisId]['ax']
     				else:
     					ax = specificAxis
     				# check how export was done (grids, boxes..)
     				boxBool, gridBool = exportDetails['boxBool'], exportDetails['gridBool']
     				## export again with saved settings
     				plotter[0].export_selection(ax,exportDetails['subplotNum'],
     					self.mainFigureCollection.mainFigures[figureId]['figure'], 
     					limits = limits,boxBool = boxBool, gridBool = gridBool)
     				self.mainFigureCollection.mainFigures[figureId]['template'].extract_artists(ax,axisId)
     				
     				if plotType in ['hclust','corrmatrix']:
     					plotter[0]._hclustPlotter.exportId  = storeId
     					axes = plotter[0]._hclustPlotter.exportAxes[exportDetails['exportId']]
     					self.mainFigureCollection.mainFigures[figureId]['template'].associate_axes_with_id(axisId,axes)
     				
     				# indicates transfer
     				if transferAxisId is not None:
     					self.mainFigureCollection.store_export(transferAxisId,
         										figureId,
         										exportDetails['plotCount'],
         										exportDetails['subplotNum'],
         										storeId,
         										exportDetails['boxBool'],
         										exportDetails['gridBool'])
     					break
 	
 		
 
  
     		
     def setup_main_figure(self): 
     	'''
     	Setup main figure
     	'''
     	main_figures.mainFigureTemplateDialog(self.mainFigureCollection)
   
     def melt_data(self): 
     	'''
     	Melts the data using selected columns. Enters new data columns into the source treeview.
     	'''
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     	
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     	
     		fileID,fileName,columnNameDataTyperRelationship = \
     		self.sourceData.melt_data_by_column(self.DataTreeview.columnsSelected )
     		
     		self.DataTreeview.add_new_data_frame(fileID,fileName,columnNameDataTyperRelationship)

     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		tk.messagebox.showinfo('Done ..','Melting done. New data frame was added to the treeview.')

     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     		return

    
     def get_all_combs(self):
     	'''
     	Shows all comparisions within the categorical values. 
     	The user can drag & drop statistical tests from the analysis frame / treeview 
     	onto the label called (Drop Statistic here)
     	'''
     	if hasattr(self,'groupedStatsData'):
     		## user cannot open new window
     		return 
     		
     	categoricalColumns = list(self.selectedCategories.keys())
     	if len(categoricalColumns) == 0:
     		tk.messagebox.showinfo('No categories..',
     							'Please load categorical columns into the receiver box.')
     		return
     	
     	self.groupedStatsData = self.sourceData.get_groups_by_column_list(categoricalColumns)
     	
     	self.groupedStatsKeys = self.groupedStatsData.groups.keys()
     	
     	dataDict = OrderedDict([('id',[]),('Group 1',[]), ('Group 2',[])])
     	
     	for n,combination in enumerate(itertools.combinations(self.groupedStatsKeys,2)):
     		dataDict['id'].append(str(n+1))
     		dataDict['Group 1'].append(get_elements_from_list_as_string(combination[0]))
     		dataDict['Group 2'].append(get_elements_from_list_as_string(combination[1]))
     	
     	
     	optionalDfName = 'MultComp_{}'.format(get_elements_from_list_as_string(categoricalColumns))
     	dataFrame = pd.DataFrame.from_dict(dataDict)
     	self.statsDataDialog = display_data.dataDisplayDialog(dataFrame,
     														  showOptionsToAddDf=True,
     														  dragDropLabel = True,
     														  analyzeClass = self,
     														  dfOutputName = optionalDfName,
     														  topmost=True)
     	 									 
     	 									 	    	
     def calculated_droped_stats_for_all_combs(self):
     	'''
     	Calculates the droped statitic.
     	'''
     	dataDict = OrderedDict([('id',[]),('Group 1',[]), ('Group 2',[])])
     	numericColumns = list(self.selectedNumericalColumns.keys()) 
     	
     	if len(numericColumns) != 0:
     	
     		columnsToAdd = [col for col in self.DataTreeview.columnsSelected if col not in numericColumns]
     		
     		columnsForTest = numericColumns + columnsToAdd
     		
     		for column in columnsForTest:
     			dataDict['{} p-value'.format(column)] = []
     			dataDict['{} test statistic'.format(column)] = []
     			
     		
     		testSettings = {'paired':self.paired,
     						'test':self.test,
     						'mode':self.mode}
     		print(testSettings)	
     		for n,combination in enumerate(itertools.combinations(self.groupedStatsKeys,2)):
     			valuesGroup1 = self.groupedStatsData.get_group(combination[0])
     			valuesGroup2 = self.groupedStatsData.get_group(combination[1])
     			## we do this again, because user could change the df and resort 
     			## to redo seems easier than matching 
     			dataDict['id'].append(str(n+1))
     			dataDict['Group 1'].append(get_elements_from_list_as_string(combination[0]))
     			dataDict['Group 2'].append(get_elements_from_list_as_string(combination[1]))
     			
     			for column in columnsForTest:
     				data1 = valuesGroup1[column].dropna().values
     				data2 = valuesGroup2[column].dropna().values
     				try:
     					testResult = stats.compare_two_groups(testSettings,[data1,
     											data2])
     					t , p = round(testResult[0],4), testResult[1]
     				except:
     					t, p = np.nan, 1
     				
     				dataDict['{} p-value'.format(column)].append(p)
     				dataDict['{} test statistic'.format(column)].append(t)
     		
     		for column in columnsForTest:
     			pValColumn = '{} p-value'.format(column)
     			data = np.array(dataDict[pValColumn])
     			reject, corr_pvals,_,_ = multipletests(data, alpha = 0.05, 
     	 									 method = 'fdr_bh', is_sorted= False, returnsorted=False)
     			dataDict['{} adj. p-values'.format(column)] = corr_pvals
     			
     		resultDf = pd.DataFrame.from_dict(dataDict)
     		self.statsDataDialog.pt.model.df = resultDf
     		self.statsDataDialog.pt.redraw()     				
     	else:
     		tk.messagebox.showinfo('No data ..','Please load numerical columns into the receiver box.')
     	
     	if hasattr(self,'groupedStatsData'):
     		del self.groupedStatsData
     	


     def replace_data_in_df(self,replaceOption):
     	'''
     	replaces NaNs or 0s with constant or metric
     	'''
     	if 'Constant' in replaceOption:
     		value = ts.askfloat('Constant ..',prompt='Please provide constant to be used for NaN replacement')
     		if value is None:
     			return
     	
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		if self.DataTreeview.onlyNumericColumnsSelected == False:
     			tk.messagebox.showinfo('Error ..','Please select only columns containing floats and/or integers.')
     			return
     		numericColumns = self.DataTreeview.columnsSelected 
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		if 'NaN -' in replaceOption:
     			if '0' in replaceOption:
     				replaceValue = 0
     			elif 'Constant' in replaceOption:
     				replaceValue = value ##from ask float above
     			elif 'Mean':
     				replaceValue = self.sourceData.df[numericColumns].mean(axis=0)
     			elif 'Median' in replaceoption:
     				replaceValue = self.sourceData.df[numericColumns].median(axis=0)
     				
     			self.sourceData.fill_na_in_columnList(numericColumns,replaceValue)
     		else:
     			for numColumn in numericColumns:
     				self.sourceData.df[self.sourceData.df[numColumn]==0] = np.nan
     			
     		tk.messagebox.showinfo('Done ..','Calculations done.')
     		
     		
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     		return	
     
                  
     def build_analysis_tree(self):
        seps_tests =   ['Model fitting',
                        'Compare-two-groups',
                        'Compare multiple groups',
                        'Two-W-ANOVA',
                        'Three-W-ANOVA',
                        'Cluster Analysis',
                        'Classification',
                        'Dimensional reduction',
                        'Miscellaneous',
                        'Curve fitting']

        self.options_for_test = dict({'Model fitting':['linear',
                                         # 'logarithmic',
                                          'lowess',
                                          #'expontential',
                                          ],
                        'Compare-two-groups':['t-test','Welch-test',
                                              'Wilcoxon [paired non-para]',
                                              'Whitney-Mann U [unparied non-para]'
                                              ],
                        'Compare multiple groups':['1W-ANOVA','1W-ANOVA-RepMeas','Kruskal-Wallis'],
                        'Miscellaneous':['Pairwise Comparision','AUC','Density'],
                        'Classification':classification.availableMethods,
                        'Cluster Analysis':clustering.availableMethods,
                         'Two-W-ANOVA':['2W-ANOVA','2W-ANOVA-RepMeas(1fac)','2W-ANOVA-RepMeas(2fac)'],#,'Interactions'],
                         'Three-W-ANOVA':['3W-ANOVA','3W-ANOVA-RepMeas(1fac)','3W-ANOVA-RepMeas(2fac)','3W-ANOVA-RepMeas(3fac)'],
                         'Dimensional reduction':list(stats.dimensionalReductionMethods.keys())
                                       ,
                                                  'Curve fitting':['Curve fit ..','Display curve fit(s)']})
        opt_two_groups = ['paired',
                          'unpaired']
        direct_test = ['less','two-sided [default]',
                       'greater']
        self.stats_tree = ttk.Treeview(self.analysis_sideframe, height=8, show='tree',style='source.Treeview')
        self.stats_tree.bind('<<TreeviewSelect>>', 
                             lambda event, stats_tree=self.stats_tree: self.retrieve_test_from_tree(event,stats_tree)) 
        self.stats_tree.bind("<B1-Motion>", lambda event,analysis = True: self.on_motion(event, analysis)) 
        self.stats_tree.bind("<ButtonRelease-1>", lambda event, analysis =True: self.release(event,analysis)) 
        self.stats_tree.column("#0",minwidth=800)
        
        for heads in seps_tests:
            main = self.stats_tree.insert('','end',str(heads), text=heads) 
            
            for opt_test in self.options_for_test[heads]:
                    
                    sub1 = self.stats_tree.insert(main, 'end', str(opt_test), text = opt_test)
                    if heads == 'Compare-two-groups':
                        if opt_test in  ['t-test','Welch-test']:
                            for sub_opt in opt_two_groups:
                                 sub2 = self.stats_tree.insert(sub1, 'end','%s_%s' % (opt_test,sub_opt), text=sub_opt)
                                 for direction in direct_test:
                                    self.stats_tree.insert(sub2, 'end', '%s_%s_%s' %  (opt_test,sub_opt,direction), text=direction) 
                                 if opt_test == 'Welch-test':
                                 	break
                                
                        else:
                                for direction in direct_test:
                                    self.stats_tree.insert(sub1, 'end', '%s_%s' % (str(opt_test),str(direction)), text=direction)
        sourceScroll = ttk.Scrollbar(self, orient = tk.HORIZONTAL, command = self.stats_tree.xview)
        sourceScroll2 = ttk.Scrollbar(self,orient = tk.VERTICAL, command = self.stats_tree.yview)
        self.stats_tree.configure(xscrollcommand = sourceScroll.set,
                                          yscrollcommand = sourceScroll2.set)
        sourceScroll2.pack(in_=self.analysis_sideframe, side = tk.LEFT, fill=tk.Y, anchor=tk.N)
        self.stats_tree.pack(in_=self.analysis_sideframe, padx=0, expand=True, fill=tk.BOTH) 
        sourceScroll.pack(in_=self.analysis_sideframe, padx=0,anchor=tk.N, fill=tk.X) 
         
         
         
     def  retrieve_test_from_tree(self,event,tree):
         '''
         Get the test properties from the selection.
         '''
         self._drag_and_drop = True
         self.mot_button = None
         curItem = tree.focus()
         itx = curItem.split('_')
         self.mode = ''
         self.paired = ''
         self.test = itx[0]
         if len(itx) == 3:
             if itx[1] == 'paired':
                 self.paired=True
             else:
                 self.paired=False
             self.mode = itx[-1]   
         elif len(itx) == 2:
            self.mode = itx[-1]
         else:
            pass   
   
     		
     def combine_selected_columns(self):
     	'''
     	Combines the content of selected columns.
     	'''
     	if len(self.DataTreeview.columnsSelected  ) < 2:
     		tk.messagebox.showinfo('Error..','Please select at least two columns')
     		return
     		
     	
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		combinedColumnName = self.sourceData.combine_columns_by_label(self.DataTreeview.columnsSelected  )  
     		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,
     													dataType = 'object',
     													columnList = [combinedColumnName])
     													      
     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		
     		
     		tk.messagebox.showinfo('Done ..','Selected columns were combined in a newly added column.')
     	else:
     		
      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
      		return
     		     	        
     	
     def duplicate_column(self):
     
     	'''
     	Duplicates selected columns. Changes dataframe selection if needed. Eventually
     	it will change back to the previous selected dataframe.
     	'''
     	
     	currentDataFrameId = self.sourceData.currentDataFile
     	
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		
     		columnLabelListDuplicate  = self.sourceData.duplicate_columns(self.DataTreeview.columnsSelected  )
     		dataTypes = self.sourceData.get_data_types_for_list_of_columns(columnLabelListDuplicate)
     	
     		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,
     													dataTypes,
     													columnLabelListDuplicate)
     		
     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		
     		tk.messagebox.showinfo('Done ..','Selected column(s) were duplicated and added to the source data treeview.')
     			
     	else:
     		
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     		return
     	
     
     def delete_column(self):
     	'''
     	Removes selected columns. Changes dataframe selection if needed. Eventually
     	it will change back to the previous selected dataframe.
     	'''
     	
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		self.sourceData.delete_columns_by_label_list(self.DataTreeview.columnsSelected  ) 
     		
     		self.DataTreeview.delete_selected_entries()
     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		

     		tk.messagebox.showinfo('Done ..','Selected columns were removed.')
     	else:
     		
      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
      		return
     		     	
     def remove_rows_with_na(self):
     
     	'''
     	Drops rows from selected columns. Changes, if necessary, to the selected DataFrame.
     	'''
     	
     	currentDataFrameId = self.sourceData.currentDataFile
     	
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		
     		self.sourceData.drop_rows_with_nan(self.DataTreeview.columnsSelected)    
     
     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		
     		tk.messagebox.showinfo('Done ..','NaN were removed from selected columns.')
 
     	else:   
     		
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     		return
 
             
     def change_column_type(self, changeColumnTo = 'float64'):
     	'''
     	Changes the column type of the selected one.
     	'''
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	
     	columns = self.DataTreeview.columnsSelected  
     	
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		status = self.sourceData.change_data_type_in_current_data(columns,changeColumnTo)
     		if status == 'worked':
     		
     			self.DataTreeview.change_data_type_by_iid(self.DataTreeview.columnsIidSelected,changeColumnTo)
     			tk.messagebox.showinfo('Done..','Column type changed.')
     		else:
     			if changeColumnTo == 'int64':
     				addToMsg = ' If the column contains NaN you cannot change it type to integer.Please remove NaN and try again.'
     			else:
     				addToMsg = '.'
     			tk.messagebox.showinfo('Error..','An error occured trying to change the column type.' + addToMsg)
     	else:   
     		
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
     		return		

         
         
     def transform_selected_columns(self, transformation):
     	'''
     	Transform data. Adds a new column to the data frame.
     	'''
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		
     		transformedColumnName = self.sourceData.transform_data(self.DataTreeview.columnsSelected  ,transformation) 
     		
     			
     		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,
     													dataType = 'float64',
     													columnList = transformedColumnName,
     													)
     		
     		self.sourceData.set_current_data_by_id(currentDataFrameId)

     		tk.messagebox.showinfo('Done ..','Calculations performed.')
     		
     	else:
     		
      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
      		return
     		     	
     		
         
     def split_column_content_by_string(self,splitStringCommand):
     	'''
     	Splits the content of a column row-wise with given splitString. 
     	For example: KO_10min would be split by '_' into two new columns KO , 10min
     	'''
     	 
     	splitString = splitStringCommand[-2]
     	
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		
     		splitColumnName, indexStart = self.sourceData.split_columns_by_string(self.DataTreeview.columnsSelected  ,splitString) 
     		
     		if splitColumnName is None:
     		
     			tk.messagebox.showinfo('Error..','Split string was not found in selected column.')
     			self.sourceData.set_current_data_by_id(currentDataFrameId)
     			return
     			
     		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,
     													dataType = 'object',
     													columnList = splitColumnName,
     													startIndex = indexStart)
     		
     		self.sourceData.set_current_data_by_id(currentDataFrameId)

     		tk.messagebox.showinfo('Done ..','Selected column(s) were split and added to the source data treeview.')
     		
     	else:
     		
      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
      		return
         
      
         
     def sort_source_data(self,sortType):
     	'''
     	Sort columns either according to the value or by string length. Value can handle mutliple
     	columns while string length is only able to sort for one column. Note that the 
     	sort is ascending first, then upon second sort it will be descending. 
     	'''
     	
     	currentDataFrameId = self.sourceData.currentDataFile
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
     		if sortType == 'Value':
     			ascending = self.sourceData.sort_columns_by_value(self.DataTreeview.columnsSelected  )
     		elif sortType == 'String length':
     			if len(self.DataTreeview.columnsSelected  ) > 1:
     				tk.messagebox.showinfo('Note..','Please note that this sorting can handle only one column. The column: {} will be used'.format(self.DataTreeview.columnsSelected  [0]))
     			self.sourceData.sort_columns_by_string_length(self.DataTreeview.columnsSelected  [0])
     			
     		self.sourceData.set_current_data_by_id(currentDataFrameId)
     		if ascending:
     			infoString = 'in ascending order. Sort again to get descending order.'
     		else:
     			infoString = 'in descending order.'

     		tk.messagebox.showinfo('Done ..','Selected column(s) were sorted {}'.format(infoString))
     		
     	else:
     		
      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
      		return
     		     	     	
     	
     def change_plot_style(self, plot_type = ''):
         '''
         Function that handles the event triggered by plot options. 
         Very important step is to set the data selection back to the one
         that was used to generate the last chart. Otherwise you might 
         experience difficulties that column headers are not present. 
         '''
         
         if len(self.plt.plotHistory) == 0:
             return
             
         dataID = self.plt.get_dataID_used_for_last_chart()
         self.sourceData.set_current_data_by_id(dataID)
        
         numericColumns = list(self.selectedNumericalColumns.keys())
         categoricalColumns = list(self.selectedCategories)
        
         underlying_plot = self.plt.currentPlotType
        
         if plot_type  not in ['boxplot','violinplot','barplot','add_swarm']:
             
             self.but_stored[9].configure(image = self.add_swarm_icon)
             self.swarm_but = 0
             self.plt.addSwarm = False                 
                          
         if plot_type == 'add_swarm':


             if underlying_plot not in ['boxplot','violinplot','barplot']:
             
                 tk.messagebox.showinfo('Error..','Not useful to add swarm plot to this '+
                 						'type of chart. Possible chart types: Boxplot, '+
                 						'Violinplot and Barplot')
                 return
                 
             if self.swarm_but == 0:
                 self.but_stored[9].configure(image= self.remove_swarm_icon)
                 self.add_swarm_to_figure()  
                 self.swarm_but = 1
                 
             else:
                 self.but_stored[9].configure(image = self.add_swarm_icon)
                 self.swarm_but = 0
                 help = self.plt.get_active_helper() 
                 help.remove_swarm()
                 self.plt.redraw()
		 		                       	
         else:    
         	if plot_type in ['hclust','corrmatrix'] and len(numericColumns) > 1\
         	and len(categoricalColumns) > 0:
         	## forces removable of categories upon selection
         		self.clean_up_dropped_buttons('cat',replot=False)
         		categoricalColumns = []	
         	self.prepare_plot(colnames = numericColumns, 
             				   catnames = categoricalColumns, 
             				   plot_type = plot_type )
       
     def add_swarm_to_figure(self):
     	'''
     	Helper function to trigger the addition of 
     	swarm plot onto the underlying graph.
     	Please note that if the data get bigger stripplot instead of swarm 
     	will be used (less computing time) the difference is that in
     	swarm plots you can estimate the distribution much better.
     	'''
     	help = self.plt.get_active_helper() 
     	help.add_swarm_to_plot()
     	self.plt.redraw()
     	             
         
         
     def re_sort_source_data_columns(self):
     	'''
     	Resorts column in currently selected data frame. 
     	(Alphabetical order)
     	'''
     	self.sourceData.resort_columns_in_current_data()
     	dict_ = self.sourceData.dfsDataTypesAndColumnNames
     	file_names = self.sourceData.fileNameByID
     	self.DataTreeview.add_all_data_frame_columns_from_dict(dict_,file_names) 
     	  
        

         
     def delete_data_frame_from_source(self):
     	'''
     	Removes data frames from sourceDataCollection and from DataTreeview.
     	The data cannot be restored.
     	'''
     	
     	for fileIid in self.DataTreeview.dataFramesSelected:
     		
     		fileName = self.sourceData.fileNameByID[fileIid]
     		dataFrameIid = '{}_{}'.format(fileIid,fileName)     	
     		self.DataTreeview.delete_entry_by_iid(dataFrameIid)
     		self.sourceData.delete_data_file_by_id(fileIid)
     		
     	tk.messagebox.showinfo('Done..','Selected data frame(s) delted.')
      
        
        
     def on_slected_treeview_button3(self, event):
         
         if self.DataTreeview.onlyDataFramesSelected:
         	self.post_menu(menu = self.merge_data_frames_menu)
         elif self.DataTreeview.onlyDataTypeSeparator:
         	self.post_menu(menu=self.data_type_menu)
         else:
         	self.post_menu(menu = self.menu)
         

     def on_motion(self,event, analysis = False):
         '''
         checks current widget under mouse event and colors the motion button
         if it is over an appropiate widget that accepts a drag & drop event
         '''
           
         if self.DataTreeview.stopDragDrop:
         	return
        
         self.widget = self.winfo_containing(event.x_root, event.y_root)
         
         if self.mot_button is None:
         
             self.DataTreeview.columnsSelected   = self.DataTreeview.columnsSelected  
             self.data_types_selected   =   self.DataTreeview.dataTypesSelected
             if analysis:
             	but_text = str(self.test) 
             else:
             	but_text=str(self.DataTreeview.columnsSelected)[1:-1]

             self.mot_button = tk.Button(self, text=but_text, bd=1,
                                     		   fg="darkgrey", bg=MAC_GREY)
                                    
             if  len(self.mot_button_dict) != 0:
                                     
             	for item in self.mot_button_dict.values():
             		item.destroy() 
             	
             	self.mot_button_dict.clear()	
             
             self.mot_button_dict[self.mot_button] = self.mot_button
             
         x = self.winfo_pointerx() - self.winfo_rootx()
         y = self.winfo_pointery() - self.winfo_rooty()
         
         self.mot_button.place( x= x-20 ,y = y-30) ## offset because otherwise dropped widget will always be the same button
                         
         if analysis:
            
             if self.widget == self.canvas.get_tk_widget():
                 self.mot_button.configure(fg = "blue")
             else:
                 self.mot_button.configure(fg = "darkgrey")
                 
         else:
                 if len(self.data_types_selected) == 0:
                 	return
                 unique_dtypes_selected = self.data_types_selected[0]
                 

                 if unique_dtypes_selected == 'float64':
                     
                     if self.widget in [self.tx_space,self.column_sideframe] or self.widget in self.sliceMarkButtonsList: 
                         self.mot_button.configure(fg = "blue")
                         return
      
                 elif unique_dtypes_selected == 'int64':
                     
                     if self.widget in [self.tx_space,
                     					self.cat_space,self.color_button_droped,
                     					self.column_sideframe,self.category_sideframe] or self.widget in self.sliceMarkButtonsList:
                         self.mot_button.configure(fg = "blue")
                         return
                         
                 elif unique_dtypes_selected == 'object':      

                     if self.widget in [self.cat_space,self.color_button_droped,
                     		 			self.category_sideframe] or self.widget in self.sliceMarkButtonsList:

                         self.mot_button.configure(fg = "blue")
                         return
                
                 self.mot_button.configure(fg = "darkgrey") 
                     
                         
     def delete_dragged_buttons(self, event, but_name, columns=False):
     
         if columns:
             self.selectedNumericalColumns[but_name].destroy() 
             del self.selectedNumericalColumns[but_name]
         else:
             self.selectedCategories[but_name].destroy() 
             del self.selectedCategories[but_name]
             
         if len(self.selectedCategories) == 0 and len(self.selectedNumericalColumns) == 0:
             self.plt.clean_up_figure()
             self.interactiveWidgetHelper.clean_frame_up()  
             self.plt.redraw()
             return 
             
         _,_, plot_type, cmap = self.plt.current_plot_settings
         if plot_type in ['hclust','corrmatrix'] and len(list(self.selectedNumericalColumns.keys())) == 1:
             plot_type = 'boxplot'
         if columns == False and plot_type in ['scatter_matrix','hclust','corrmatrix']:
             return
         if plot_type == 'PCA':
             plot_type = 'boxplot'
             
         numericColumns = list(self.selectedNumericalColumns.keys())
         categoricalColumns = list(self.selectedCategories.keys())
         if len(numericColumns) == 0:
         	plot_type = 'countplot'
         self.plt.initiate_chart(numericColumns,categoricalColumns,
         							plot_type, cmap)
     
     
         
     def remove_mpl_connection(self, plot_type = ''):
         
         try:
                              self.canvas.mpl_disconnect(self.pick_label)
                              self.canvas.mpl_disconnect(self.pick_freehand) 
                              self.annotations_dict.clear() 
                              
         except:
                                  pass
         mpl_connections = [                      
                            self.selection_press_event,
                            self.tooltip_inf,
                            self.pick_label,
                            self.pick_freehand,
                            self._3d_update,
                            self.mot_adjust,
                            self.mot_adjust_ver,
                            self.release_event,
                            self.hclust_move_level]
         for con in mpl_connections:
                if con is not None:
                    self.canvas.mpl_disconnect(con) 
                    con = None
         buttons_dropped = [           
                     self.tooltip_button_droped,
                     self.label_button_droped,
                     self.size_button_droped,
                     self.color_button_droped,
                     self.stat_button_droped     ] 
         if plot_type == 'PCA':
             buttons_dropped = buttons_dropped[:-1]
         for but in buttons_dropped:
                if but is not None:
                    
                    but.destroy() 
                    but = None    
           
                    
     def check_input(self):
     	'''
     	'''
     	dataFrames = self.DataTreeview.dataFramesSelected
     	lastUsedDf = self.plt.get_dataID_used_for_last_chart()
     	if lastUsedDf is not None and len(dataFrames) != 0:
            	if lastUsedDf == dataFrames[0]:
            		pass
            	else:
            		self.clean_up_dropped_buttons()
     
     def place_buttons_in_receiverbox(self,columnNames,dtype):
     	'''
     	Receiver boxes do receive drag & dropped items by the user. 
     	We separate between numeric data and categorical data. This function allows
     	place button into these receiver boxes
     	'''
     	self.check_input()
     	for column in columnNames:
     		
     		button = tk.Button(self.receiverFrame[dtype], text = column)
     		if dtype == 'numeric':
     			self.selectedNumericalColumns[column] = button
     			numeric = True
     		else:
     			self.selectedCategories[column] = button
     			numeric = False
     		button.pack(side=tk.LEFT,padx=2)
     		
     		button.bind(right_click, lambda event, column = column: \
     		self.delete_dragged_buttons(event,column,columns=numeric))
     		

     def estimate_plot_type_for_default(self):
         
         colnames = list(self.selectedNumericalColumns.keys())
         catnames = list(self.selectedCategories.keys())
         used_plot_style = self.plt.currentPlotType
         n_col = len(colnames)
         n_categories = len(catnames)
         if used_plot_style == 'hclust' and n_categories > 0:
             return 'boxplot'
         if used_plot_style in ['hclust','corrmatrix'] and n_col == 1:
             return 'boxplot'
         if used_plot_style == 'PCA':
             return 'boxplot'
         if n_col == 1 and n_categories == 0:
             plot_type = 'density'
         elif n_categories != 0 and n_col == 0:
             plot_type = 'countplot'
         elif used_plot_style == 'density_from_scatter':
             if n_col == 1:
                 return 'boxplot'
             else:
                 return 'density_from_scatter'
         elif n_col == 2 and n_categories == 0 and used_plot_style != 'time_series':    
             plot_type = 'scatter'
         else:
             if self.plt.plotCount > 0:
                 if used_plot_style not in ['density','countplot','scatter','PCA']:
                     plot_type = used_plot_style
                 else:
                     plot_type = 'boxplot'
             else:
                 plot_type = 'boxplot'
         return plot_type
         
                      
     def release(self,event,analysis=''):
     	
         if len(self.data_types_selected) == 0:
             return
            
         widget = self.winfo_containing(event.x_root, event.y_root)


         if self.mot_button is not None:
         	self.mot_button.destroy()
         	self.mot_button = None
         
         dataFrames = self.DataTreeview.dataFramesSelected
         self.sourceData.set_current_data_by_id(dataFrames[0])
                  
         if analysis == '':
             if widget == self.source_treeview:
             	return
             try:	
                 self.cat_filtered = [col for col in self.DataTreeview.columnsSelected   if col not in list(self.selectedCategories) and self.sourceData.df[col].dtype != np.float64] 
                 self.col_filtered = [col for col in self.DataTreeview.columnsSelected   if col not in list(self.selectedNumericalColumns) and (self.sourceData.df[col].dtype == np.float64 or self.sourceData.df[col].dtype == np.int64)]
             finally:
                 if self.mot_button is not None:
                 	self.mot_button.destroy()
                 	self.mot_button = None  
                 	  
         if widget == self.tx_space or widget == self.column_sideframe:
             
             if len(self.col_filtered) == 0:
                 return 
             else:
                 self.place_buttons_in_receiverbox(self.col_filtered,dtype='numeric')
                         
             plot_type = self.estimate_plot_type_for_default() 
             self.update_idletasks() # update here otherwise the windows starts flashing
             
             self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),
                                             catnames = list(self.selectedCategories.keys() ),
                                             plot_type = plot_type) 
             
         elif widget == self.cat_space or widget == self.category_sideframe:
             
             if len(self.cat_filtered) == 0:
                 return 
             else:
                 self.place_buttons_in_receiverbox(self.cat_filtered,dtype='category')
                 

             plot_type = self.estimate_plot_type_for_default() 
             self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),
                                             catnames = list(self.selectedCategories.keys() ),
                                                            plot_type = plot_type)
                     
         elif widget ==  self.data_button:
         	
             self.show_droped_data() 
                              
         elif widget == self.sliceMarkFrameButtons['label']:
 
                 
             last_plot_type = self.plt.currentPlotType
             if last_plot_type not in ['scatter','hclust','PCA']:
                 return
             if self.label_button_droped is not None:
                 self.label_button_droped.destroy() 
                 self.label_button_droped = None 
        
             self.anno_column = self.DataTreeview.columnsSelected  
             if last_plot_type in ['scatter','PCA']:
                 if all(self.sourceData.df[col].dtype == 'object' for col in self.DataTreeview.columnsSelected  ):
                     if all(self.sourceData.df[col].unique().size == 2 for col in self.DataTreeview.columnsSelected  ):
                         
                         if all('+' in self.sourceData.df[col].unique() for col in self.DataTreeview.columnsSelected  ):
                             
                                 quest = tk.messagebox.askyesno('Annotate ..',
                                 	'Would you like to annotate all rows having a "+"? You can choose the desired column in the next step.')
                                 if quest:
                                     dataSubset = self.sourceData.df[self.sourceData.df[self.DataTreeview.columnsSelected[0]].str.contains(r'^\+$')]
                                 	
                                     dialog = categorical_filter.categoricalFilter(self.sourceData,self.DataTreeview,self.plt,
                                     									  operationType = 'Annotate scatter points',
                                     									  columnForFilter = self.DataTreeview.columnsSelected[0], 
                                     									  dataSubset = dataSubset)
                                     if dialog.closed:
                                     	return
                          
                                 else:
                                    self.make_labels_selectable() 
                         else:
                             self.make_labels_selectable()
                     else:
                         self.make_labels_selectable()                 
                 else:
                     self.make_labels_selectable()
             else:
                 if len(self.anno_column) > 1:
                     tk.messagebox.showinfo('Info..','Please note that only column can be used for labeling rows in a h-clust.\nHowever you can also merge columns with the function: Combine columns from the drop-down menu'
                                            )
                                            
                 self.plt.nonCategoricalPlotter._hclustPlotter.add_label_column(self.anno_column)
                 
             s =  self.return_string_for_buttons(self.DataTreeview.columnsSelected[0])
             self.label_button_droped  = create_button(self.interactiveWidgetHelper.frame, text = s, 
             											image= self.but_label_icon, 
             											compound=tk.CENTER)
             
             if last_plot_type != 'hclust':
             	self.label_button_droped.bind(right_click, 
             					self.remove_annotations_from_current_plot)
             else:
             	self.label_button_droped.bind(right_click,
             					lambda event, label = self.label_button_droped:
             					self.plt.nonCategoricalPlotter._hclustPlotter.remove_labels(event,label))    
           
             self.label_button_droped.grid(columnspan=2, padx=1, pady=1)
             
         elif widget == self.sliceMarkFrameButtons['filter']:
             
             if any(self.sourceData.df[col].dtype not in [np.float64,np.int64] for col  in self.DataTreeview.columnsSelected  ):
              		
                     self.categorical_column_handler('Find category & annotate')
             else:    
                 
                 self.numeric_filter_dialog()
                          
         elif widget == self.sliceMarkFrameButtons['tooltip']:
         
             try:
                 self.tooltip_button_droped.destroy()
             except:
                 pass
                 
             self.add_tooltip_information()
            
             s =  self.return_string_for_buttons(self.DataTreeview.columnsSelected[0])
             
             self.tooltip_button_droped = create_button(self.interactiveWidgetHelper.frame, text = s, 
             											image= self.but_tooltip_icon, 
             											compound=tk.CENTER)
             self.tooltip_button_droped.bind(right_click, self.remove_tool_tip_active) 
             self.tooltip_button_droped.grid(columnspan=2, padx=1, pady=1) 
                
                
         elif widget == self.sliceMarkFrameButtons['size']:
         
             if self.size_button_droped is not None:
                 self.size_button_droped.destroy()
                 self.size_button_droped = None 
             
             colorChangeWorked = self.update_size()
             if colorChangeWorked == False: 
             	return 
             	
             s =  self.return_string_for_buttons(self.DataTreeview.columnsSelected [0])
             
             self.size_button_droped = create_button(self.interactiveWidgetHelper.frame, text = s, 
             											image= self.but_size_icon, 
             											compound=tk.CENTER)
             self.size_button_droped.bind(right_click, self.remove_sizes_)
             self.size_button_droped.grid(columnspan=2, padx=1, pady=1)
             
             
         elif widget == self.sliceMarkFrameButtons['color']:
         
            if len(self.plt.plotHistory) == 0:
                return
            last_plot_type = self.plt.currentPlotType
                             
            if self.color_button_droped is not None:
                 self.color_button_droped.destroy()
                 self.color_button_droped = None 
                 
            s =  self.return_string_for_buttons(self.DataTreeview.columnsSelected  [0])
            
            self.color_button_droped = create_button(self.interactiveWidgetHelper.frame, text = s, 
             											image= self.but_col_icon, 
             											compound=tk.CENTER)
            
            if last_plot_type != 'hclust':
                    self.color_button_droped.bind(right_click, 
                    		self.remove_color_)
                    self.interactiveWidgetHelper.clean_color_frame_up()
                    
                    		
            else:
                    self.color_button_droped.bind(right_click, 
                    		lambda event, label = self.color_button_droped: 
                    		self.plt.nonCategoricalPlotter._hclustPlotter.remove_color_column(event,label))
                 
            self.color_button_droped.grid(columnspan=2, padx=1, pady=1)
            
            if last_plot_type == 'hclust':
                 self.plt.nonCategoricalPlotter._hclustPlotter.add_color_column(self.DataTreeview.columnsSelected)
            else:
                ret_ = self.update_color()
                if ret_ == False: 
                 	return 

         elif widget == self.color_button_droped:
             
             
             columnSelected = self.DataTreeview.columnsSelected
             if self.plt.nonCategoricalPlotter is not None:
             	if 'change_color_by_categorical_columns' in self.plt.nonCategoricalPlotter.sizeStatsAndColorChanges:
             		alreadyUsedColors = self.plt.nonCategoricalPlotter.sizeStatsAndColorChanges['change_color_by_categorical_columns']
             else:
             	alreadyUsedColors = []
             self.DataTreeview.columnsSelected = alreadyUsedColors + columnSelected
             proceed = self.update_color(add_new_cat = True)
             
             if proceed:
                 self.color_button_droped.configure(text = "Multiple")
            
         elif analysis:
         
             if len(self.plt.plotHistory) == 0:
                     return
             else:
                 last_plot_type = self.plt.currentPlotType
                 
                 if hasattr(self,'statsDataDialog'):
                 	if widget == self.statsDataDialog.dragDropLabel:
                         self.calculated_droped_stats_for_all_combs()

             if widget == self.canvas.get_tk_widget():

                     try:
                         self.stat_button_droped.destroy()
                     except:
                       pass
                     
                     s =  self.return_string_for_buttons(self.test)
                     
                     if  self.test not in stats.dimensionalReductionMethods: 
                        
                         self.stat_button_droped = create_button(self.interactiveWidgetHelper.frame, text = s, 
                         										image= self.but_stat_icon, compound=tk.CENTER)
                        
                         self.stat_button_droped.grid(columnspan=2, padx=0, pady=1)
                         self.stat_button_droped.bind(right_click, self.remove_stat_)
                                              
                     #print(self.test)
                     if self.test in ['linear','lowess'] and last_plot_type not in ['scatter','scatter_matrix']:
                         tk.messagebox.showinfo('Error..','This operation can only be performed on scatter plots.')
                         return 
                         
                     if self.test == 'linear':
                          self.add_linear_regression()
                     elif self.test == 'lowess':
                         self.add_lowess() 
                     elif self.test == 'Density':
                         
                          self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),
                                             catnames = list(self.selectedCategories.keys() ),
                                                            plot_type = 'density_from_scatter')
                         
                     elif self.test == 'AUC':
                         if last_plot_type != 'time_series':
                             tk.messagebox.showinfo('Error..','Area under the courve can only be calculated in time series charts')
                             return
                         else:
                             pass 
                         self.plt.nonCategoricalPlotter.timeSeriesHelper.activate_baselineCorr_or_aucCalc('aucCalculation')
                         self.stat_button_droped.configure(command = self.show_auc_calculations)
                         
                         
                         #self.calculate_area_under_courve()
                      
                     elif self.test in clustering.availableMethods:
                     
                     	clustering.clusteringDialog(self.sourceData,self.plt, self.DataTreeview, 
                     								self.clusterCollection , self.interactiveWidgetHelper,
                     								numericColumns = list(self.selectedNumericalColumns.keys()),
                     								initialMethod = self.test,
                     								cmap = self.cmap_in_use.get())
                     	
                     								
                     elif self.test in classification.availableMethods:
                     	tk.messagebox.showinfo('Under revision','Currently under revision. Will be available in the next minor update.')
                     	return
                     	classification.classifyingDialog(self.sourceData,self.plt, self.DataTreeview, 
                     								self.classificationCollection, 
                     								numericColumns = list(self.selectedNumericalColumns.keys()),
                     								initialMethod = self.test)
                     								                       	
                      
                     elif 'ANOVA' in self.test:  
                      
                     	anova_calculations.anovaCalculationDialog(self.test,
                     											  list(self.selectedNumericalColumns.keys())[0],
                     											  list(self.selectedCategories.keys()),
                     											  self.sourceData,
                     											  self.anovaTestCollection)
                     	
                     	self.stat_button_droped.configure(command = lambda : \
                     	self.show_anova_results(id = self.anovaTestCollection.id))
                    	                    	
                     elif self.test =='Kruskal-Wallis':
                         self.perform_one_way_anova_or_kruskall()
                     
                     ## check if test should be used to compare two groups   
                     elif self.test =='Pairwise Comparision':
                     	self.get_all_combs()
                      
                     elif self.test in self.options_for_test['Compare-two-groups']:
                     	## note that the statistic results are being saved in self.plt associated with the plot
                     	## count number

                        statTestInformation = {'test':self.test, 'paired':self.paired, 'mode':self.mode}
                        if self.twoGroupstatsClass is not None:
                        	## if it exists already -> just update new stat test settings
                        	self.twoGroupstatsClass.selectedTest = statTestInformation
                        else:
                        	self.twoGroupstatsClass = stats.interactiveStatistics(self.plt,
                        								self.sourceData,statTestInformation)
                        self.stat_button_droped.configure(command = self.show_statistical_test)
                        #self.stat_button_droped.bind(right_click, self.statsClass.delete_all_stats)
                                                
                     elif self.test == 'Display curve fit(s)':
                         
                          self.display_curve_fits()
                          
                     elif self.test == 'Curve fit ..':
                     
                     	self.curve_fit(from_drop_down=False)
                          
                         #self.design_popup(mode='Curve fit ..', from_drop_down = False)
                         
                     elif self.test in stats.dimensionalReductionMethods:
                     	
                         
                         self.interactiveWidgetHelper.clean_frame_up()  
                         self.dimReduction_button_droped = create_button(self.interactiveWidgetHelper.frame, text = s, 
                         									image= self.but_stat_icon, compound=tk.CENTER, 
                         									command = lambda: self.post_menu(menu=self.pca_export_menu))#self.post_pca_menu)
                         self.dimReduction_button_droped.grid(columnspan=2, padx=0, pady=1) 
                         self.perform_dimReduction_analysis() 
                         
     def curve_fit(self,from_drop_down = True):
     	'''
     	Dialogue window to calculate curve fit. 
     	'''
     	if  from_drop_down:
     		selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     		if selectionIsFromSameData: 
     			self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		else:
     			tk.messagebox.showinfo('Error ..','Please select only columns from one data frame',parent=self)
     			return
     		columns = self.DataTreeview.columnsSelected  
     	else:
     		id = self.plt.get_dataID_used_for_last_chart()
     		self.sourceData.set_current_data_by_id(id)
     		columns = list(self.selectedNumericalColumns.keys())
     	  
     	curve_fitting.curveFitter(columns,self.sourceData,self.DataTreeview,self.curveFitCollection)
        
        
      
     
     def show_anova_results(self, id):
     	'''
     	Display anova results
     	'''
     	anova_results.anovaResultDialog(self.anovaTestCollection)
	
	                   
     def show_statistical_test(self):
     	'''
     	Displays calculated statistics in a pandastable. 
     	Allows the user to add these data to the data collection and 
     	treeview. Which then can be could be used for plotting. 
     	'''
     	data = self.twoGroupstatsClass.performedTests
     	dataDialog = display_data.dataDisplayDialog(data,showOptionsToAddDf=True)
     	
     	if dataDialog.addDf:
     		nameOfDf = 'StatResults_plotID:{}'.format(self.plt.plotCount)
     		self.add_new_dataframe(dataDialog.data,nameOfDf)
     	del dataDialog
     	
     
     def show_auc_calculations(self):
     	'''
     	Display determined AUC in pandastable.
     	Users can also add the data frame to the source
     	'''
     	df = self.plt.nonCategoricalPlotter.timeSeriesHelper.get_auc_data()
     	dataDialog = display_data.dataDisplayDialog(df,showOptionsToAddDf=True)
     	if dataDialog.addDf:
     		del dataDialog
     		nameOfDf = 'aucResults_plotID:{}'.format(self.plt.plotCount)
     		self.add_new_dataframe(df,nameOfDf)     	                         
                         	
     def remove_stat_(self,event):
     	'''
     	Deletes results of statistical tests.
     	'''
     	quest = tk.messagebox.askquestion('Confirm ..','This will remove all statistical test results from the current chart. Proceed?')
     	if quest == 'yes':
     		if self.twoGroupstatsClass is not None:
     		
     			self.twoGroupstatsClass.delete_all_stats()
     			self.twoGroupstatsClass.disconnect_event()
     			del self.twoGroupstatsClass
     			self.twoGroupstatsClass = None
     			self.plt.redraw()
     		
     def remove_color_(self,event):
         '''
         Removes color levels added to the chart (can only be - scatter,pca,hclust,corrmatrix)
         '''
     	
         quest = tk.messagebox.askquestion('Confirm ..', 'This will remove all color changes.'+
         									' Proceed?')
         if quest == 'yes':
         
         	self.plt.remove_color_level()
         	self.color_button_droped.destroy()
         	self.color_button_droped = None 
         	self.interactiveWidgetHelper.clean_color_frame_up()
         	self.plt.redraw()
         	
    
    
     def remove_sizes_(self,event):    
         '''
         Resets the size of scatter points to the basic level.
         '''
              
         _,catnames,plot_type,_ = self.plt.current_plot_settings
         n_categories = len(catnames) 
         
         quest = tk.messagebox.askquestion('Confirm ..', 'This will remove all size changes.'+
         									' Proceed?')
         if quest == 'yes':
         	
         	self.plt.remove_size_level()
         	self.size_button_droped.destroy()
         	self.size_button_droped = None 
         	
         	self.plt.redraw()
         	     	
         return
         
         
         size = self.size_selected.get()
         val = float(size)
         if plot_type == 'scatter': 
         	if n_categories == 0:
         		self.scat.set_sizes([val])  
         	else:
         		for key, subset_and_scatter in self.subsets_and_scatter_with_cat.items():
         			_,_,scat = subset_and_scatter
         			scat.set_sizes([val])	
           
         if self.size_leg is not None:        
             self.size_leg.remove()         
         self.canvas.draw()
         self.size_button_droped.destroy()
         self.size_button_droped = None 
         self.size_leg = None
         
     def remove_tool_tip_active(self,event):
         
         self.tooltip_button_droped.destroy() 
         self.canvas.mpl_disconnect(self.tooltip_inf)
         
     def remove_annotations_from_current_plot(self,event):
         '''
         Removes all annotations from a plot
         '''
         quest = tk.messagebox.askquestion('Deleting labels..','This step will remove all labels from your plot.\nPlease confirm..')
         if quest == 'yes':
          	self.plt.nonCategoricalPlotter.annotationClass.remove_all_annotations()
          	self.plt.nonCategoricalPlotter.annotationClass.disconnect_event_bindings()
          	self.plt.nonCategoricalPlotter.annotationClass = None
          	self.plt.redraw()
          	if self.label_button_droped is not None:
          		self.label_button_droped.destroy()
          		self.label_button_droped = None
         else:
             return
         
         
         

     def generate_pandas_data_frame_from_pariwise(self, hsd, cat_levels):
         groups = [cor for cor in itertools.combinations(hsd.groupsunique,2)]
         source_df = pd.DataFrame.from_records(groups, columns = ['Group1', 'Group2'])

         Q = hsd.meandiffs / hsd.std_pairs
         p_vals = psturng(np.abs(Q), cat_levels, hsd.df_total)
         source_df.loc[:,'Mean diff.'] = hsd.meandiffs

             
         intervals = pd.DataFrame.from_records(hsd.confint, columns = ['Lower','Upper'])
         source_df = pd.concat([source_df,intervals], axis=1)
         source_df.loc[:,'Std. pairs'] = hsd.std_pairs     
         source_df.loc[:,'Q'] = Q
         source_df.loc[:,'Reject'] = hsd.reject
         source_df.loc[:,'adj. p-val'] = p_vals
         decimals = pd.Series([2,2,2,2,2,5], index = ['Mean diff.','Lower','Upper','Std. pairs','Q','adj. p-val']) 
         source_df = source_df.round(decimals)
         return source_df
     
        
     def perform_dimReduction_analysis(self, open_session = False):

         numericColumns = list(self.selectedNumericalColumns.keys())
         	
         if len(numericColumns) == 0:
             tk.messagebox.showninfo('Error..','Please add columns to the numeric data receiver box.')
             return 
         
         	
         dimRedResult = stats.get_dimensionalReduction_results(self.sourceData.df[numericColumns],
         													method = self.test)
         # save PCA results (a pca can also be used to fit other unseen data)
         self.dimensionReductionCollection.save_calculation(dimRedResult,
         													numericColumns,
         													self.test,
         													self.sourceData.currentDataFile) 
         # set the current data for plotting
         self.plt.set_dim_reduction_data(self.dimensionReductionCollection.get_last_calculation()) 
         # plot the results 
         self.plt.initiate_chart(numericColumns = numericColumns,categoricalColumns = [] ,
         							selectedPlotType = 'PCA', 
         							colorMap = self.cmap_in_use.get())

     	    
     def perform_one_way_anova_or_kruskall(self):
         cols = list(self.selectedNumericalColumns.keys())
         cats = list(self.selectedCategories.keys())
         self.output_df_hsd = OrderedDict() 
         used_plot_type = self.plt.currentPlotType
         list_for_testing = list() 
         if len(cats) == 0:
             
             for col_ in cols:
                 list_for_testing.append(self.sourceData.df[col_].dropna())
             if self.test == '1W-ANOVA':
                 stat, p = f_oneway(*list_for_testing)
                 stat_ = 'F'
             else:
                 stat, p = kruskal(*list_for_testing, nan_policy = 'omit')
                 stat_ = 'H'
             gn=self.plt.add_annotationLabel_to_plot(ax = self.plt.figure.axes[0],text=\
             '{}\n{}.stat: {}\np.val: {}\nColumns: {}'.format(self.test,stat_,round(stat,2),round(p,5),len(cols)))
            
             
             if len(cols) > 2:
                 id_vars = [col for col in self.sourceData.df_columns if col not in cols]
                 long_pd = pd.melt(self.sourceData.df, id_vars = id_vars, value_vars = cols, value_name = 'Value', var_name = 'Variable')
                 long_pd.dropna(inplace=True, subset=['Value'])
                 hsd = pairwise_tukeyhsd(long_pd['Value'],long_pd['Variable'])
                 df = self.generate_pandas_data_frame_from_pariwise(hsd,len(cols))
                          
         else:
             n_levels_0 = self.sourceData.df[cats[0]].unique() 
                  
         
         if len(cats) == 1:
             anova_group = self.sourceData.df.groupby(cats[0], sort=False)
             if anova_group.size().min() < 3:
                 messagebox.showinfo('Error..','Less than 3 observations in one group.')
                 return
             for col_ in cols:  
                 for name, group  in anova_group:
                     list_for_testing.append(group[col_].dropna())
                
                 if self.test == '1W-ANOVA':
                     stat, p = f_oneway(*list_for_testing)
                     
                     data = np.array(self.sourceData.df[col_])
                     groups = np.array(pd.factorize(self.sourceData.df[cats[0]])[0]+1)
                     FF = spm_stats.anova1(data,groups, equal_var=True)
                     Fi = FF.inference(alpha=0.05) 
                     
                     stat_ = 'F'
                 else:
                     stat, p = kruskal(*list_for_testing, nan_policy = 'omit')
                     stat_ = 'H'
                 ln_s = np.linspace(0.025,0.74,num=len(cols))
                 x_pos = ln_s[cols.index(col_)]
                 gn=self.plt.add_annotationLabel_to_plot(ax = self.plt.figure.axes[0],text = \
             	 '{}\n{}.stat: {}\np.val: {}\nColumns: {}'.format(self.test,stat_,round(stat,2),round(p,5),len(cols)))
            	 
                 
                 if n_levels_0.size >= 2:
                     source = self.sourceData.df[[col_]+cats].dropna() 
                     hsd = pairwise_tukeyhsd(source[col_],source[cats[0]])
                     df = self.generate_pandas_data_frame_from_pariwise(hsd,n_levels_0.size)
                     self.output_df_hsd[cats[0]+'_'+col_] = df
               
                 list_for_testing = []
                 
         self.plt.redraw()
         display_data.dataDisplayDialog(df,showOptionsToAddDf = True)
             
                 
  
     def return_string_for_buttons(self, items_for_col, lim = 12):
         
         
         string_length  = len(items_for_col)
         if string_length > lim:
                s = items_for_col[:lim-1]+'..'
         else:
                s = items_for_col
         return s
         
               
     			
     def categorical_column_handler(self,mode):
     	'''
     	Open categorical filter dialog. Please note that this is also used 
     	to annotate scatter plot points which looks strange. But since the annotation
     	can also only be considered as a categorical value, the dialog window is used as well.
     	
     	Parameter
     	==========
     	mode - Can only be one of : 
     	
     			- Find category & annotate
				- Search string & Annotate
				- Subset data on unique category
				- Annotate scatter points 
				- Find entries in hierarch clustering
		Output
		==========
		None - But new data frames are entered automatically from within the dialog
     	'''
     	self.annot_label_scatter = False
     	if mode == 'Annotate scatter points' and len(self.DataTreeview.columnsSelected) == 0:
     		filterColumn  = None
     		dataSubset = None
     	elif mode == 'Find entry in hierarch. cluster':
     		if self.plt.currentPlotType != 'hclust':
     			tk.messagebox.showinfo('Error ..','Please plot a hierarchical clustermap.')
     			return
     		else:
     			filterColumn  = None
     			dataSubset = self.plt.nonCategoricalPlotter._hclustPlotter.df
     	else:
     		filterColumn = self.DataTreeview.columnsSelected[0]
     		dataSubset = None
     		
     	categorical_filter.categoricalFilter(self.sourceData,self.DataTreeview,
             										self.plt,operationType = mode,
             										columnForFilter = filterColumn,
             										dataSubset = dataSubset)
     	            
     def add_linear_regression(self):
         '''
         Adds linear regression to scatter plot.
         '''
         plotHelper = self.plt.get_active_helper()
         plotHelper.add_regression_line(list(self.selectedNumericalColumns.keys())) 
         self.plt.redraw()

         
     def add_lowess(self):
     	'''
     	Adds lowess line to current plot.
     	'''
     	
     	plotHelper = self.plt.get_active_helper()
     	plotHelper.add_lowess_line(list(self.selectedNumericalColumns.keys()))
     	self.plt.redraw() 
        	
	
     def show_data(self):
     	'''
     	Shows all data and allows update of source data if anything as been done
     	'''
     	currentDataSelection = self.DataTreeview.get_data_frames_from_selection()
     	if len(currentDataSelection) == 0:
     		currentDataSelection = [self.sourceData.currentDataFile]
     		if currentDataSelection is None:
     			tk.messagebox.showinfo('No data ..','No data loaded.',parent=self)
     			return
     	else:
     		pass
     	datToInspect = self.sourceData.get_data_by_id(currentDataSelection[0]).copy() ## this is needed otherwise the self.df will be updated instantly
     	dataDialogue = display_data.dataDisplayDialog(datToInspect,self.plt)
     	data_ = dataDialogue.get_data()
     	del dataDialogue
     	if data_.equals(self.sourceData.df):
     		pass
     	else:
     		quest = tk.messagebox.askquestion('Confirm ..','Data changed. Would you like to update?')
     		if quest == 'yes':
     			self.sourceData.update_data_frame(id=currentDataSelection[0],
         													dataFrame = data_)
     			self.update_all_dfs_in_treeview()     	
     		else:
     			pass
     	return
	
     def show_droped_data(self):
         '''
         Shows only selection of data when drag & drop was performed on the SourceData button.
         '''
         numericalColumns = list(self.selectedNumericalColumns.keys())
         categoricalColumns = self.sourceData.get_columns_data_type_relationship()['object']
         columSelection = categoricalColumns + numericalColumns + self.DataTreeview.columnsSelected  
         display_data.dataDisplayDialog(self.sourceData.df[columSelection],self.plt)

         
     def normalize_size(self, size_data,min = None, max = None):
         if min is None and max is None:
         	min = size_data.min()
         	max = size_data.max()

         if max/(min+1) > 100000:
             sizes = np.log2(size_data)
             sizes = sizes.replace('-inf',0)
         else:
             sizes = size_data
         sizes = (sizes-min)/(max-min)*100  + 15  
         return sizes    
     
     def add_legend(self, name):
         if self.size_leg is None:
             s1= self.a[1].scatter([],[],s=15, color ="grey")
             s2= self.a[1].scatter([],[],s=65, color ="grey")
             s3= self.a[1].scatter([],[],s=115, color ="grey")
             self.size_leg = plt.legend((s1,s2,s3),(0,0.5,1),title = 'Scaled: '+str(name), bbox_to_anchor=(0., 1.02, 1., .102),
                           loc=1, 
                           ncol=3, 
                           borderaxespad=0.)
             self.size_leg.draggable(state=True)
             
             plt.setp(self.size_leg.get_title())        
         else:
             self.size_leg.get_title().set_text('Scaled: '+str(name))
         
         
     def update_size(self, col_ = None):
         
         if len(self.plt.plotHistory) == 0:
         	return False
         	
         	
         _, categoricalColumns, plotType, _  = self.plt.current_plot_settings
         n_categories = len(categoricalColumns) 
         
         if plotType not in ['scatter','scatter_matrix','PCA']:
             tk.messagebox.showinfo('Error..','Not useful for any other chart type than Scatter and Scatter Matrix')
             if self.size_button_droped is not None:
                 self.size_button_droped.destroy()         
             return False
         
         if col_ is None:
             col_ = self.DataTreeview.columnsSelected[0]
         dtype = self.sourceData.df[col_].dtype
         if dtype == np.float64 or dtype == np.int64:
             if plotType == 'scatter_matrix': 
             
             	#print(self.plt.nonCategoricalPlotter._scatterMatrix.data)
             	
             	self.plt.nonCategoricalPlotter._scatterMatrix.change_size_by_numeric_column(self.DataTreeview.columnsSelected[0])

                     
             else:    

                 
                 self.plt.nonCategoricalPlotter.change_size_by_numerical_column(self.DataTreeview.columnsSelected[0])
                 self.plt.redraw()
                 return
                 
                 size_data = self.sourceData.df[col_]
                 
                 if n_categories != 0:
                 	min = size_data.min()
                 	max = size_data.max()
                 	for key, subset_and_scatter in self.subsets_and_scatter_with_cat.items():
                 		subset,ax,scat = subset_and_scatter
                 		
                 		sizes = self.normalize_size(subset[col_],min = min, max=max) 
                 		scat.set_sizes(sizes) 
                 		subset['size'] = sizes               		
                 		self.subsets_and_scatter_with_cat[key] = [subset,ax,scat]
                 		
                 	
                 else:
                 	self.plt.nonCategoricalPlotter.change_size_by_categorical_column(self.DataTreeview.columnsSelected[0])
                 	self.plt.redraw()
                 	return                 
             
         else:
             
                    
             if plotType in ['scatter','PCA']:       
                 if n_categories == 0:
                              
                 	self.plt.nonCategoricalPlotter.change_size_by_categorical_column(self.DataTreeview.columnsSelected[0])
                 	self.plt.redraw()
                 	return
                
                 else:
                 	for key, subset_and_scatter in self.subsets_and_scatter_with_cat.items():
                 		
                 		subset,ax,scat = subset_and_scatter
                 		sizes = subset[col_].apply(lambda x: size_map_update[x])
                 		scat.set_sizes(sizes)
                 
                 
             elif plot_type == 'scatter_matrix':
                 for key, scatter in self.axes_scata_dict.items():
                     data = self.data_scata_dict[key]
                     size_data = data[col_]
                     sizes = size_data.apply(lambda x: size_map_update[x])
                     
                                                

                        
     def update_color(self, update=False, update_cat = False, add_new_cat = False, from_save = False):
		         
         if self.DataTreeview.onlyNumericColumnsSelected:
         	colorColumn = self.DataTreeview.columnsSelected[0]
         	if len(self.DataTreeview.columnsSelected) > 1:
         		tk.messagebox.showinfo('Note..','Numerical columns cannot be combined. Only {} will be used.'.format(colorColumn))
         		
         	if self.plt.currentPlotType in ['scatter','PCA']:	
         	
         		self.plt.nonCategoricalPlotter.change_color_by_numerical_column(colorColumn)
         		
         	elif self.plt.currentPlotType == 'scatter_matrix':
         	
         		self.plt.nonCategoricalPlotter._scatterMatrix.change_color_by_numeric_column(colorColumn)
         else:
         	if self.plt.currentPlotType == 'scatter_matrix':
         	
         		self.plt.nonCategoricalPlotter._scatterMatrix.change_color_by_categorical_column(self.DataTreeview.columnsSelected)
         		
         	elif self.plt.currentPlotType in ['scatter','PCA']:
         	
         		self.plt.nonCategoricalPlotter.change_color_by_categorical_columns(self.DataTreeview.columnsSelected, updateColor=False)
         		self.interactiveWidgetHelper.clean_color_frame_up()
         		self.interactiveWidgetHelper.create_widgets(plotter = self.plt)         		     		
		
         self.plt.redraw()
         

     def remove_curs(self,event,fig):
         self.frame.config(cursor='arrow')
         
         if self.mot_adjust is not None:
             fig.canvas.mpl_disconnect(self.mot_adjust)
             self.mot_adjust = None
         if self.mot_adjust_ver is not None:
             fig.canvas.mpl_disconnect(self.mot_adjust_ver)
             self.mot_adjust_ver = None
         if self.cursor_release_event is not None:    
             fig.canvas.mpl_disconnect(self.cursor_release_event)

         
     def adjust_subplots_by_motion(self,event,fig,side,speed = False):
         if event.button != 1:
             self.remove_curs(event,fig)
             return
         left_ = fig.subplotpars.left
         right_ = fig.subplotpars.right
         if side == 'left':
             rat = event.x/self.last_x
             left_new = rat * left_
             try:    
                 self.f1.subplots_adjust(left=left_new)
             except ValueError:
                self.remove_curs(event,fig)
             self.last_x = event.x
         elif side == 'right' :
             rat = event.x/self.last_x
             left_new = rat * right_
             try:    
                 self.f1.subplots_adjust(right=left_new)
             except ValueError:
                self.remove_curs(event,fig)
             self.last_x = event.x
             
         elif side == 'bottom':
             bottom_ = fig.subplotpars.bottom
             rat = event.y/self.last_y

             n_subs = len(fig.axes)
             if speed:
                 bottom_new = (rat) * bottom_ + ((rat-1) * 1.2)
             else:
                 bottom_new = (rat) * bottom_
             try:    
                 self.f1.subplots_adjust(bottom=bottom_new)
             except ValueError:
                self.remove_curs(event,fig)
             self.last_y = event.y
         elif side == 'top':
             top_ = fig.subplotpars.top
             rat = event.y/self.last_y

             n_subs = len(fig.axes)
             if speed:
                 top_new = (rat) * top_ #+ ((rat-1) * 1.2)
             else:
                 top_new = (rat) *top_
             try:    
                 self.f1.subplots_adjust(top=top_new)
             except ValueError:
                self.remove_curs(event,fig)
             self.last_y = event.y
         self.canvas.draw()
         try:
        # make sure that the GUI framework has a chance to run its event loop
        # and clear any GUI events.  This needs to be in a try/except block
        # because the default implementation of this method is to raise
        # NotImplementedError
            self.f1.canvas.flush_events()
         except NotImplementedError:
            pass
        
  
     def is_just_outside(self,fig, event):
        if event.inaxes != None or self.count == 0:
             return 
        if len(self.a) == 0:
             return
        if len(self.original_vals) == 0:
                 self.original_vals = [self.a[1].get_ylim(),self.a[1].get_xlim()]
                 self.center_x = False
                 self.center_y = False
        if len(fig.axes) == 0:
            return
        
        
        
        x,y = event.x, event.y
        self.remove_curs(event = '',fig = fig)
        self.cursor_release_event = fig.canvas.mpl_connect('button_release_event', lambda e, fig=fig: self.remove_curs(e,fig))
        tup_fig  = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
        for ax in fig.axes:
            
            xAxes, yAxes =  ax.transAxes.inverted().transform([x, y])
            
            if (-0.20 < xAxes < -0.04):
                if event.dblclick and event.button == 1:
                    axes  = fig.axes
                    if self.center_y == True:
                              for ax in axes:
                                  ax.set_ylim(list(self.original_vals[0]))
                                  self.center_y = False
                    elif self.center_y == False:
                                 ymin, ymax  = self.original_vals[0]
                                 max_y = max(abs(ymin),abs(ymax))
                                 self.center_y = True
                                 for ax in axes:
                                     ax.set_ylim([-max_y,max_y])
                   
                    
                elif event.button == 3:
                     axes  = fig.axes
                     if self.log_y == False:
                         for ax in axes:
                             ax.set_yscale("symlog", subsy = [2,  4,  6,  8,])
                         
                         self.log_y = True
                     else:
                         for ax in axes:
                             ax.set_yscale('linear')
                         self.log_y = False
                         
                self.canvas.mpl_disconnect(self.cursor_release_event   )
                self.canvas.draw()
                break
                return         
            elif  (-0.20 < yAxes < -0.04):  
                
                if event.dblclick and event.button == 1:
                    axes  = fig.axes
                    if self.center_x == True:
                              for ax in axes:
                                  ax.set_xlim(list(self.original_vals[1]))
                                  self.center_x = False
                    elif self.center_x == False:
                                 xmin, xmax  = self.original_vals[1]
                                 max_x = max(abs(xmin),abs(xmax))
                                 self.center_x = True
                                 for ax in axes:
                                     ax.set_xlim([-max_x,max_x])
                   
                    
                elif event.button == 3:
                     axes  = fig.axes
                     if self.log_x == False:
                         for ax in axes:
                             ax.set_xscale("symlog", subsy = [2,  4,  6,  8,])
                         
                         self.log_x = True
                     else:
                         for ax in axes:
                             ax.set_xscale('linear')
                         self.log_x = False
                         
                self.canvas.mpl_disconnect(self.cursor_release_event   )
                self.canvas.draw()
                break
                return         
             
            if  (-0.04 < xAxes < 0) | (1 < xAxes < 1.04):
                ##print ("just outside x-axis"    )
                
                if xAxes < 0:
                    if ax.is_first_col() == True:
                        side = 'left'
                    else:
                        return
                else:
                    
                    if ax.is_last_col() == True or len(fig.axes) == 1:
                        side = 'right'
                    else:
                        return
                self.frame.config(cursor = 'right_side')
                self.last_x = x
                self.mot_adjust = fig.canvas.mpl_connect('motion_notify_event', lambda e, fig=fig,side =  side: self.adjust_subplots_by_motion(e,fig,side))
            elif  (-0.04 < yAxes < 0) | (1 < yAxes < 1.04):
                if yAxes < 0:
                    if ax.is_last_row()  or len(fig.axes) < tup_fig[1]+1:
                        side = 'bottom'
                    else:
                        return
                else:
                    if ax.is_first_row():
                        side = 'top'
                    else:
                        return
                  
                self.frame.config(cursor = 'top_side')
                self.last_y = y
                
                if tup_fig != (1,1):
                    speed_up = True
                else:
                    speed_up = False
                self.mot_adjust_ver = fig.canvas.mpl_connect('motion_notify_event', lambda e, fig=fig,side = side,speed = speed_up : self.adjust_subplots_by_motion(e,fig,side,speed))
            else:
                return                                           

        
     def evaluate_ymin_for_plot(self, plot_type, ylims, data, split_cols = None):
         if plot_type == 'barplot':
             ymin,ymax = ylims
             if split_cols is not None:
                 groups = data.groupby(split_cols)
                 if groups.size().min() == 1:
                     
                     y_max = ymax
                 else:
                 
                     max_of_mean = groups.mean().max().max()
                     max_of_stdev = groups.std().max().max() 
                     y_max = max_of_mean + 2.2 * max_of_stdev
             else:    
                 
                 max_of_mean = data.mean().max()
                 max_of_stdev = data.std().max() 
                 
             if ymin > 0:
                 ymin = 0
             ylims = (ymin,y_max)   
             return ylims
         else:
             return ylims
     def define_some_stuff_before_plot(self):
     		
             self.hclust_axes.clear() 
             self.center_x = False
             self.center_y = False
             self.log_x = False
             self.log_y = False
             self.col_num_for_stat = None
             self.size_leg = None
             self.label_button_droped = None
             self.tooltip_button_droped = None
             self.size_button_droped = None
             self.color_button_droped = None
             self.stat_button_droped = None
             self.size_column = None
             self.data_as_tuple = None
             self.col_map_keys_for_main  = None
             self.col_map_keys_for_main_double  = None
             self.hclust_dat_ = None
             
     def prepare_plot(self, colnames = [], catnames = [], plot_type = "" , cmap_ = ''):
          
         n_cols = len(colnames)
         n_categories = len(catnames)
         
         if self.twoGroupstatsClass is not None:
         	self.twoGroupstatsClass.disconnect_event()
         	del self.twoGroupstatsClass
         	self.twoGroupstatsClass = None         	
         
         if True:
         	
             self.colormaps.clear() 
             self.annotations_dict.clear() 
             self.remove_mpl_connection(plot_type = plot_type) 
             self.selection_press_event = None
             self.subsets_and_scatter_with_cat.clear() 
             self.original_vals = []
             self.define_some_stuff_before_plot()
             gc.collect() 
             
             if cmap_ == '':
                 cmap_ = self.cmap_in_use.get()
             else:
                 cmap_ = cmap_

                         
             if n_cols == 0  and n_categories == 0:
                 
                 self.plt.clean_up_figure()
                 self.plt.redraw()
                 return 
                      
                 self.count += 1 
                     
                                  
             if catnames != []:
                 if any(len(self.sourceData.df[cat].unique()) > 100 for cat in catnames) \
                 and plot_type not in ['network','hclust']:
                 
                     quest = tk.messagebox.askquestion('More than 100 categories',
                     	'You have selected a categorical column with more than 100 unique values.'+
                     	' Usually this produces very complex (unreadable) plots. Are you sure you'+
                     	' want to proceed?')
                     if quest == 'yes':
                         pass
                     else:                     
                         but = self.selectedCategories[catnames[-1]]
                         but.destroy()
                         del self.selectedCategories[catnames[-1]]
                         return             
             
             self.f1.subplots_adjust(right = 0.88, top = 0.88, bottom = 0.12)  

     			
         ## clean up
         self.interactiveWidgetHelper.clean_frame_up()
                 
         if  n_categories != 0 and n_cols == 0: 
         
             self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
        
        
         elif plot_type == 'scatter' and n_categories > 0 and n_cols  > 0:
         
         	scatter_with_cat = scatter_with_categories.scatter_with_categories(self.sourceData.df,n_cols, n_categories,colnames,
         										catnames,self.f1,50,GREY) 
         										
         	self.a = scatter_with_cat.return_axes()
         	self.subsets_and_scatter_with_cat =  scatter_with_cat.get_subsets_and_scatters()
         	self.label_axes_for_scatter = scatter_with_cat.return_label_axes()
         	
         	
         elif plot_type == 'display_fit':
                 
 
                 curve_fitting.displayCurveFitting(self.sourceData,
                 								   self.plt,self.courveFitCollection) 
                 self.plt.initiate_chart(numericColumns = [], categoricalColumns = [] ,
                 						  selectedPlotType = 'curve_fit', 
                 						  colorMap = self.cmap_in_use.get())

         elif plot_type == 'density_from_scatter':
             
            if n_cols == 2:
                 pass
            elif n_cols == 1:
                return 
            else: 
                tk.messagebox.showinfo('Info..',
                	'Please note that only the first two numeric columns will be used.')
                
            if True: 
                 data = self.sourceData.df[colnames+catnames].dropna(subset = colnames)
                 
                 if True:
                     ax = self.f1.add_subplot(111)
                     self.f1.subplots_adjust(wspace=0.03, hspace=0.27, right = 0.88)
                     self.a[1] = ax
                 else:
                     ax = ax_export
                 patches = [] 
                 
                 cmaps = ['Blues','Reds','Greys','Greens','Purples']
                 if n_categories == 1:
                         
                         uniq = self.sourceData.df[catnames[0]].unique()
                         n_uniq = uniq.size
                         if n_uniq > 4: 
                             cmaps = cmaps * 50
                         for i,cat in enumerate(uniq):
                             c_ = cmaps[i]
                             colors = sns.color_palette(c_,4)
                             patches.append(mpatches.Patch(edgecolor='black',
                                                           linewidth=0.4,
                                                           facecolor=colors[-2], 
                                                           label=cat))   
                             
                             dat = data[data[catnames[0]] == cat]
                             sns.kdeplot(dat[colnames[0]], dat[colnames[1]], shade = True, shade_lowest = False, cmap = c_, ax = ax, alpha=0.5)
                         fill_axes.add_draggable_legend(ax,patches=patches, leg_title = 'Split on: '+str(catnames[0]))    
                 elif n_categories  == 2:
                     uniq_1 = list(data[catnames[0]].unique())
                     uniq_2 = list(data[catnames[1]].unique())
                     collect_unique_vals = [uniq_1,uniq_2]
                     theor_combi = list(itertools.product(*collect_unique_vals))
                     data['combi'] = data[catnames].apply(tuple, axis=1)
                     uniq_combs_in_data = data['combi'].unique()
                     
                     theor_combi = [comb for comb in theor_combi if comb in list(uniq_combs_in_data)]
              
                     for i,combi in enumerate(theor_combi):
                         c_  = cmaps[i]
                         colors = sns.color_palette(c_,4)
                         patches.append(mpatches.Patch(edgecolor='black',
                                                           linewidth=0.4,
                                                           facecolor=colors[-2], 
                                                           label=combi))  
                         dat = data[data['combi']==combi]
                         sns.kdeplot(dat[colnames[0]], dat[colnames[1]], shade = True, shade_lowest = False, cmap = c_, ax = ax, alpha=0.5)
                     fill_axes.add_draggable_legend(ax,patches=patches, leg_title = 'Split on: '+str(catnames))    
                     
                     
                 elif n_categories > 2:
                     tk.messagebox.showinfo('Error..','Density calculations is not supported for more than 2 categorical columns.'
                                            )
                     return
                 else:
                     c_ = self.cmap_in_use.get()
                     sns.kdeplot(data[colnames[0]], data[colnames[1]], shade = True, shade_lowest = False, cmap = c_, ax = ax,alpha=0.5)
                 
 
         elif (plot_type == 'density') or (n_cols == 1 and n_categories == 0 and (plot_type == '' or plot_type == 'scatter')):
             
             self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
             return
             	
             	             	
                 
                 
               #               
#                    ###IMPLEMENT THIS?? 
   
#                  elif self.split_on_cats_for_plot.get() == False:
#                      colors = palette_new = sns.color_palette(self.cmap_in_use.get(), 
#                                                           n_categories+1,
#                                                          desat=0.75)
#                      
#                      dat_ = self.sourceData.df[[column]+catnames].dropna(how='any', subset=[column])
#                      sns.distplot(dat_[column], color = colors[0], ax = ax_, hist=False)
#                      for cat_ in catnames:
#                          sub_ = dat_[dat_[cat_]=="+"]
#                          
#                          idx_ = catnames.index(cat_)
#                          if len(sub_.index) > 0:
#                              sns.distplot(sub_[column], color = colors[(idx_+1)], hist=False, ax = ax_)
#                      i = 0
#                      for cat_ in ['Compl.']+catnames:
#                          ax_.plot([],[], label = cat_, c = colors[i])
#                          i += 1
# 
#                          
                                  
                
         elif plot_type in ['hclust','corrmatrix']:
             
             if n_cols == 1:
                 tk.messagebox.showinfo('Error..','You need at least 2 numeric columns for this plot type.')
                 return 
             else:
             		self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
             		return
                
         elif plot_type == 'scatter_matrix' and n_cols > 1:


             self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)

			
             #cor_matrix = self.sourceData.df[colnames].corr()
             #self.display_hclust(cols = colnames, plot_in_main = plot_in_main, ax_export = ax_export, fig_id = fig_id, corr_m = cor_matrix)
             
             
        
         elif plot_type == 'time_series':
             
             self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                 
    
             
             
             
             
         elif (n_cols>2 and  n_categories == 0 and plot_type != 'pointplot' and plot_type != 'scatter' ) or plot_type in ['boxplot','violinplot','swarm','barplot']:
             
             
             
             if n_categories == 0:

                 	self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)                 		
                 	return
		
                 
             elif (n_categories > 3 and n_cols > 0) or self.split_on_cats_for_plot.get() == False:
             	                 
                 if any('+' not in uniqueValues.tolist() or uniqueValues.size > 2 for \
                 uniqueValues in self.sourceData.get_unique_values(catnames, forceListOutput = True)):
                 	tk.messagebox.showinfo('Error ..','This type of chart can only handle two categorical values (for example "+" and "-") of'+
                 										' which one must be "+". If you do not have such a column you can apply the categorical filters'+
                 										' from the drop down menu.')
                 	return
                 	
                 self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                 return
             
             elif n_categories > 0:
             	self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
             	return
             	
             else:
                          
             		
                 if  n_categories == 1 and n_cols == 1: 
                 	self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                 	return

                 elif n_cols > 1 and n_categories == 1:
                 
                     n_in_cat = self.sourceData.df[catnames[0]].unique().size
                     self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                     return                              
                 
                                     
                 elif n_cols > 0 and n_categories > 1:  

                    
                     self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                     return
                   	
                 else:
                     messagebox.showinfo('Error..','Not yet supported.')
                     
                                  
                 
            #  if n_cols > 1:
#                  rotation = 90 
#             
#              
#                  for ax in self.f1.axes:  
#                      self.reduce_tick_to_n(tick = 'x',ax = ax, n=20, rotation = rotation)
#                      
#              if plot_type == 'swarm':
#              	if plot_in_main == False:
#              		for ax in self.f1.axes:
#              			ax_coll = ax.collections
#              			for coll in ax_coll:
#              				coll.set(**self.settings_points)
#              				#print(coll.get_hatch())
#              	else:
#              		ax_coll = ax_export.collections
#              		for coll in ax_coll:
#              			coll.set(**self.settings_points)
   
                 
             
         elif  plot_type == 'pointplot' :
         
             self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                                            
                
         else:
             
             if n_cols == 2:
             	
                 self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                 return
                 	
                 
                 
             elif  n_cols == 3:
                 self.a[1] = self.f1.add_subplot(111, projection = '3d')
                 self.f1.subplots_adjust(right = 0.8, top = 0.8, bottom = 0.05)  
                 self.scat = self.a[1].scatter( self.filt_source_for_update[colnames[0]],self.filt_source_for_update[colnames[1]],self.filt_source_for_update[colnames[2]],
                                     alpha= round(float(self.alpha_selected.get()),2), color=(211/255,211/255,211/255), edgecolor = "black", linewidth=0.3, s = int(float(self.size_selected.get()))) 
                 
             if plot_in_main == False:   
                 for ax in self.a.values():
                             value = self.global_chart_parameter[2]
                             leg = ax.get_legend()
                             if leg is not None:                             
                                 plt.setp(leg.get_title(), fontsize=str(value))       
                                 plt.setp(leg.get_texts(), fontsize = str(value))
                             ax_art = ax.artists
                                    
                             for artist in ax_art:
                                        if str(artist) == 'Legend':
                                            leg = artist 
                                            plt.setp(leg.get_title(), fontsize=str(value))
                                            plt.setp(leg.get_texts(), fontsize = str(value))              
    
                              
   
     
     def measure_time_diff_thread(self):
             delta_t = time.time()-self.start_time
             if delta_t > 0.2 and self.fire == 0:
                self.add_annotation_to_plot() 
             elif delta_t < 0.2:
                 self.fire = 0
                
     def add_annotation_to_plot(self):
          
          self.fire = 1
          x,y = xdata, ydata
          try:
              self.names_of_draged_items = list(self.selectedNumericalColumns.keys())
              abs_x = (self.filt_source_for_update[self.names_of_draged_items[0]]-x).abs()
              abs_y = (self.filt_source_for_update[self.names_of_draged_items[1]]-y).abs()       
              
              
              data_hovered_ = self.filt_source_for_update.iloc[(abs_x+abs_y).argsort()[:1]]
              xy = tuple(data_hovered_[self.names_of_draged_items].iloc[0]) 
          except:
              return
          if abs(xy[0]-x) > abs(x*0.02) or abs(xy[1]-y) > abs(y*0.02):
              return 
          

          x_text = xy[0]+ xy[0]*0.03
          y_text = xy[1]+ xy[1]*0.03
          xy_text =(x_text, y_text)
          
          if len(self.DataTreeview.columnsSelected  ) == 1:
              self.anno_column = self.DataTreeview.columnsSelected  [0]
              text_annot = str(self.anno_column) +': '+str(data_hovered_[self.anno_column].iloc[0]) 
          else:
              text_annot = str() 
              
              for col in self.DataTreeview.columnsSelected  :
                  text_annot = text_annot+ '\n'+str(col) +': '+str(data_hovered_[col].iloc[0]) 
          
                   
         
          
          
          self.hover_annot = self.a[1].annotate(text_annot, xy = xy, xytext = xy_text, 
                              bbox=bbox_args,
                              arrowprops=arrow_args,
                              size=9, ha='left')
          self.canvas.draw()
          
     def add_tooltip_information(self):
         try:
             self.label_button_droped.destroy()
             self.label_button_droped = None
             self.canvas.mpl_disconnect(self.pick_label)
             self.canvas.mpl_disconnect(self.pick_freehand)
         except:
             pass 
         self.tooltip_inf = self.canvas.mpl_connect('motion_notify_event', self.onHover)
        
         
                 
         
     def make_labels_selectable(self):
         
         try:
             self.canvas.mpl_disconnect(self.tooltip_inf)
             self.tooltip_button_droped.destroy()
         except:
             pass
             
         try:
             self.canvas.mpl_disconnect(self.selection_press_event)
         except:
             pass 
                  
         if len(self.plt.plotHistory) == 0:
         	return
         plotExporter =  self.plt.get_active_helper()
         plotExporter.bind_label_event(self.anno_column)
         
          
         #self.pick_label = self.canvas.mpl_connect('pick_event', self.onclick)
        # self.pick_freehand = self.canvas.mpl_connect('button_press_event', self.onPress)
         
         
         
     def onPress(self,event):
             
             if self.canvas.widgetlock.locked():
                 return
             if event.inaxes is None:
                 return
             if self.closest_gn_bool:
                 return
             try:
                 for key,artist in self.annotations_dict.items():
                     
                     if artist is not None:
                         if artist.contains(event)[0]:
                             return
                         elif artist.contains(event)[0] and event.button in [3,2]:
                             artist.remove()
                             del self.annotations_dict[key]
                             self.clean_up_global_anno_dict(artist)
                             break 
                             
                     else:
                         return
                         
             except:
                         
                         pass
             self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.callback)
             self.lasso.line.set_linestyle('--')
             self.lasso.line.set_linewidth(0.3)
            
             #self.canvas.widgetlock(self.lasso)   
                         

         
     def callback(self,verts):

             p = path.Path(verts)
             if p is None:
                 del self.lasso 
                 self.canvas.draw_idle()
                 return
             if self.closest_gn_bool:
                 del self.lasso 
                 self.canvas.draw_idle()
                 return
             x = list(self.selectedNumericalColumns.keys())[0]
             y = list(self.selectedNumericalColumns.keys())[1]
             self.data_as_tuple = list(zip(self.filt_source_for_update[x], self.filt_source_for_update[y]))
             ind = p.contains_points(self.data_as_tuple)
             
             data_to_annotate = self.filt_source_for_update.iloc[ind]
             if data_to_annotate.empty:                
                 return 
             r = self.canvas.get_renderer()
             annotations_positions = dict() 
             ymin, ymax = self.a[1].get_ylim()
             xmin, xmax = self.a[1].get_xlim() 
             y_delta = ymax-ymin 
             x_delta = xmax-xmin 
             if len(data_to_annotate.index) > 0:
                 n_labels = len(self.anno_column)
                 colnames = list(self.selectedNumericalColumns.keys())
                 collect_texts = [] 
                 collect_names = [] 
                 for sel_row in range(len(data_to_annotate.index)):
                     if n_labels == 1:
                             col = self.anno_column[0]
                             text_annot = str(data_to_annotate[col].iloc[sel_row]) 
                     else:
                         text_annot = str() 
                         for col in self.anno_column:
                                        text_annot = text_annot+ '\n'+str(col) +': '+str(data_to_annotate[col].iloc[sel_row]) 
                     xy = tuple(data_to_annotate[colnames].iloc[sel_row])                   
                     if text_annot+str(xy) in self.annotations_dict:
                             
                            pass
                     else:
                         if len(data_to_annotate.index) <=2000:
                             x_text = xy[0]+ x_delta*0.02
                             y_text = xy[1]+ y_delta*0.02
                         
                             xy_text = (x_text,y_text)
                             an_gn = self.a[1].annotate(text_annot,xy=xy,xytext = xy_text,
                                           bbox=bbox_args,
                                           arrowprops=arrow_args,
                                            ha='left')
                             
                             self.artist_list.append(an_gn)                       
                             self.annotations_dict[text_annot+str(xy)] = an_gn
                             self.add_annotation_to_global_dict(an_gn)                     
                                                  
                         else:
                             an_gn = self.a[1].text(xy[0],xy[1], s= text_annot)   
                             collect_texts.append(an_gn)
                             collect_names.append(text_annot+str(xy))
                         
                            
                         
             else:
                     return
             del self.lasso 
             self.canvas.draw_idle()
             
             

     def onHover(self,event):
            
             global t, started, xdata, ydata
             started = 1 
   
             try:
                 self.hover_annot.remove() 
             except:
                 pass 
             
             xdata = event.xdata
             ydata = event.ydata
             self.start_time = time.time() 
             
             t = perpetualTimer(0.1, self.measure_time_diff_thread)
             if started == 1:

                 for t_in_dict in t_dict.values():                         
                         t_in_dict.cancel()
                        
                 t_dict[str(xdata)+str(t)] = t     
                 t.start() 
                 
             else:
                 pass 
             
             if 'Axes' not in str(event.inaxes):
                 t.cancel()
                 started = 0
                 return 
         
            
     def configure_chart(self):
        '''
        Helper function to open Chart Configuration Dialog that allows
        easy adjustment of several chart properties. It also then
        upgrades import things in the plotter class to maintain changes like
        box around subplots or grid lines
        '''
        plot_type = self.plt.currentPlotType
        if plot_type in ['PCA','corrmatrix','hclust']:
            tk.messagebox.showinfo('Not supported..','Configuration of this plot type is currently not supported.')
            return

         	
        chart_configurator = chart_configuration.ChartConfigurationPopup(platform,self.f1,self.canvas,plot_type,self.colormaps,
             											self.label_axes_for_scatter,
             											self.global_chart_parameter,self.plt.showGrid,
             											self.plt.showSubplotBox) 

        self.global_chart_parameter = chart_configurator.global_chart_parameter
        self.plt.showSubplotBox = chart_configurator.show_box
        self.plt.showGrid = chart_configurator.show_grid
                      
     def display_graph(self,f1, hover=False, label_selection=False, main_figure= False, fig_id = None):
         ''' 
         Grids/Packs the widgets for a figure either in the main GUI or as 
         a main figure template. 
         '''
         plt.rcParams['pdf.fonttype'] = 42
         matplotlib.rcParams['pdf.fonttype'] = 42
         matplotlib.rcParams['svg.fonttype'] = 'none'   
         plt.rcParams['svg.fonttype'] = 'none'
         
         if main_figure == False:      
             self.annotations_dict = dict() 
             self.fire = 0
             self.artist_list = list() 
             self.canvas = FigureCanvasTkAgg(f1,self.frame)
             self.col_num_for_stat = None
             f1.canvas.mpl_connect('button_press_event',lambda e: self.export_selected_figure(e))
             self.canvas.show()
             self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.frame)
             self.canvas.get_tk_widget().pack(in_=self.frame,
                                                 side="top",fill='both',expand=True)
             self.canvas._tkcanvas.pack(in_=self.frame,
                                                 side="top",fill='both',expand=True)

     
     def sortby(self,tree, col, descending, data_file):
                        """sort tree contents when a column header is clicked on"""
                        # grab values to sort
                        try:
                            data_type = data_file[col].dtype
                        except:
                            pass 
                       # #print(col)
                        if isinstance(data_file, str):
                            
                            data = [(tree.set(child, col), child) \
                                    for child in tree.get_children('')]
                        elif  data_type == np.float64:
                        
                            data = [(float(tree.set(child, col)), child) \
                                    for child in tree.get_children('')]
                            data = pd.DataFrame(data, columns = ['Value','idx'])
                        elif  data_type == np.int64:
                        
                            data = [(float(tree.set(child, col)), child) \
                                    for child in tree.get_children('')]
                            data = pd.DataFrame(data, columns = ['Value','idx'])
                        else:
                                                   
                            data = [(tree.set(child, col), child) \
                                    for child in tree.get_children('')]
                            data = pd.DataFrame(data, columns = ['Value','idx'])
                        
                        data.sort_values('Value',ascending=descending, inplace=True)
                            
                        for ix, item in enumerate(data['idx']):
                                tree.move(item, '', ix)
                        
                        # switch the heading so it will sort in the opposite direction
                        tree.heading(col, command=lambda col=col: self.sortby(tree, col, \
                            int(not descending),  data_file))     
             

     def change_default_color(self, button, event = None):
         '''
         Changing the default color means that this color is used if the 
         hue is not reserved by a categorical data separation/grouping
         '''
         col_get = button.cget('background')
         if len(self.colormaps) > 0:
             for key, cb in self.colormaps.items():
                 cb[0].remove() 
             self.colormaps.clear() 
         if len(self.plt.plotHistory) == 0:
         	return
         plotHelper = self.plt.get_active_helper()
         plotHelper.change_nan_color(col_get)
         self.plt.set_scatter_point_properties(color=col_get)
         self.plt.redraw()
         
         
         
     def export_data_to_file(self, data, format_type = 'Excel',sheet_name = 'ExportInstantClue', initialfile = 'Untitled',set_current = True):
         if set_current:
         	pass
         	##set_data_to_current_selection()
         if format_type == 'txt':
             file_name_saving = tf.asksaveasfilename(title='Select name for saving file',defaultextension = '.txt' ,initialfile=initialfile,filetypes = [('text files', '.txt')])
             data.to_csv(file_name_saving, index=None, na_rep ='NaN', sep='\t')
         else:
              file_name_saving = tf.asksaveasfilename(title='Select name for saving file', defaultextension='.xlsx',initialfile=initialfile,filetypes = [('Excel files', '.xlsx')])
              n_col = len(data.columns)
              data.to_excel(file_name_saving, index=None, sheet_name = sheet_name, na_rep = 'NaN')
         tk.messagebox.showinfo('Done..','File has been saved!')     
   
             
             
     def check_button_handling(self, colormap):

         
         self.cmap_in_use.set(colormap)
         
         if self.plt.plotCount != 0:
         	color_changer.colorChanger(self.plt,self.sourceData,colormap, self.interactiveWidgetHelper)
         
        
     def numeric_filter_dialog(self):
         	'''
         	Numeric Filter. 
         	Checks if all columns are from one data type.
         	If yes  - Opens dialog to set up parameter for filtering
         	Adds a new column indicating matches by a "+" sign.
         	'''
                 
         	currentDataFrameId = self.sourceData.currentDataFile
         	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
         	if selectionIsFromSameData:
         		self.sourceData.set_current_data_by_id(selectionDataFrameId) 
         		if self.DataTreeview.onlyNumericColumnsSelected:
         			numFilter = numeric_filter.numericalFilterDialog(self.sourceData,self.DataTreeview.columnsSelected )
         			filterColumnName = numFilter.columnName
         			if filterColumnName is None:
         				return
         		else:
         			tk.messagebox.showinfo('Error..','Please select only columns with floats or integeres.')
         			return
         		self.DataTreeview.add_list_of_columns_to_treeview(selectionDataFrameId,
     													dataType = 'object',
     													columnList = [filterColumnName],
     													)
     				
         		
         		self.sourceData.set_current_data_by_id(currentDataFrameId)
         		tk.messagebox.showinfo('Done ..','Filtering performed. Column was added.')
     		
     			
         	else:
         		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
         		return         
         
	         
     def open_color_configuration(self):
     	'''
     	Opens a dialog window to change color/alpha settings.
     	'''
     	color_configuration.colorChooseDialog(self.colorHelper,self, 
     			self.cmap_in_use.get(),float(self.alpha_selected.get()))     

         
     def create_frame(self,popup):

         cont = tk.Frame(popup, background =MAC_GREY)
                 
         return cont    

         
     def design_popup(self, mode, event ='', excel_sheets = None, from_drop_down = True):
         
         def fill_lb(lb,data_to_enter):
                  try:
                      lb.delete(0,END)
                  except:
                      pass 
                  dat = list(data_to_enter)
                  for indx in range(len(data_to_enter)):
                
              
                      lb.insert(indx, dat[indx])      
         
         w = 560
         h = 520
         
         def center(toplevel,size):
         	w_screen = toplevel.winfo_screenwidth()
         	h_screen = toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))
         	         
         if mode not in ['Chart configuration','Size setting','Data','Change column name']:

         	popup=tk.Toplevel(bg=MAC_GREY)
         	popup.wm_title(str(mode))
         

         
         if mode == 'Size setting':
         	
             _,catnames,plot_type,_ = self.plt.current_plot_settings
             size = self.settings_points['sizes'][0]
             size_handle = size_configuration.SizeConfigurationPopup(platform,self.f1,self.canvas,plot_type,size, catnames,
             				self.subsets_and_scatter_with_cat, self.axes_scata_dict,self.filt_source_for_update)
			
             self.settings_points['sizes'] = [size_handle.size]
             self.plt.set_scatter_point_properties(size = size_handle.size)
             self.plt.redraw()
             return 
         
                  
         elif mode == 'Multiple comparision':
             popup.attributes('-topmost', True)
             w = 830
             h = 680
             vs_frame = VerticalScrolledFrame.VerticalScrolledFrame(popup)
             vs_frame.pack(fill='both', expand=True) 
             
             lab = tk.Label(vs_frame.interior, text = 'Multiple comparision of '+str(self.test), font = LARGE_FONT, fg="#4C626F", bg = MAC_GREY)
           
             lab.pack(pady=10,padx=10, anchor='w') 

             for key, df in self.output_df_hsd.items():
                 cols = df.columns.values.tolist()
                 lab_widget = tk.Label(vs_frame.interior, text = key, font = LARGE_FONT, fg="#4C626F")
                 
                 if self.test == 'Two-W-ANOVA':
                     
                     lab_2 = tk.Label(vs_frame.interior, text = 'Result of 2-Way ANOVA and Tukey HSD', font = LARGE_FONT, fg="#4C626F", bg = MAC_GREY)
                    
                         
                     lab_2.pack(pady=10,padx=10, anchor='w')
                     text_ins = self.anova_panda_output[key]
                     text = Text(vs_frame.interior, height=12)
                     text.insert(tk.END, str(text_ins))
                     text.pack(fill='x', expand=True, pady=20, padx=40)
                 
                 lab_widget.pack(pady=10,padx=10, anchor='w')
                 self._initiate_tree_widget(vs_frame.interior, col_sel = cols, save_to_dict = True, dict_key = key)                 
                 self._fill_trees(col_sel = cols, data = df, tree_from_dict = True, dict_key = key)
             
             
         elif 'Data' in mode:
         		
         		
         		currentDataSelection = self.DataTreeview.get_data_frames_from_selection()
         		if len(currentDataSelection) == 0:
         			currentDataSelection = [self.sourceData.currentDataFile]
         		else:
         			pass
         		
         		datToInspect = self.sourceData.get_data_by_id(currentDataSelection[0]).copy() ## this is needed otherwise the self.df will be updated instantly
         		dataDialogue = display_data.dataDisplayDialog(datToInspect,self.plt)
         		data_ = dataDialogue.get_data()
         		del dataDialogue
         		if data_.equals(self.sourceData.df):
         			pass
         		else:
         			quest = tk.messagebox.askquestion('Confirm ..','Would you like to update?')
         			if quest == 'yes':
         				self.sourceData.update_data_frame(id=currentDataSelection[0],
         													dataFrame = data_)
         				self.update_all_dfs_in_treeview()
         			else:
         				pass
         		return
            
             
             

         elif mode == 'Chart configuration':
         

             plot_type = self.plt.currentPlotType
             if plot_type in ['PCA','corrmatrix','hclust']:
             	tk.messagebox.showinfo('Not supported..','Configuration of this plot type is currently not supported.')
             	return

         	
             chart_configurator = chart_configuration.ChartConfigurationPopup(platform,self.f1,self.canvas,plot_type,self.colormaps,
             											self.label_axes_for_scatter,
             											self.global_chart_parameter,self.plt.showGrid,
             											self.plt.showSubplotBox) 

             self.global_chart_parameter = chart_configurator.global_chart_parameter
             self.plt.showSubplotBox = chart_configurator.show_box
             self.plt.showGrid = chart_configurator.show_grid


         
         elif mode == 'Hierarchical Clustering Settings':
         
             def close_and_save(popup,vars_,cbs):
                 for i,met in enumerate(self.hclust_metrices):
                     self.hclust_metrices[i] = vars_[i].get() 
                 self.calculate_row_dendro = cbs[0].get() 
                 self.calculate_col_dendro = cbs[1].get()    
                 tk.messagebox.showinfo('Done..','Settings are saved and will be used for the next clustering', parent=popup)
                 popup.destroy()
                 
             def update(vars_,cbs):
                 popup.destroy()
                 items_that_differ = [i for i,var_  in enumerate(vars_) if var_.get() != self.hclust_metrices[i]]
                 for i,met in enumerate(self.hclust_metrices):
                     self.hclust_metrices[i] = vars_[i].get()  
                 
                 if self.calculate_row_dendro != cbs[0].get() or self.calculate_col_dendro != cbs[1].get():
                 
                     self.calculate_row_dendro = cbs[0].get() 
                     self.calculate_col_dendro = cbs[1].get()   
                     update_needed = True
                 else:
                     update_needed = False
                 
                 if all(x > 3 for x in items_that_differ) and update_needed == False:
                     add_draw = True
                     if 6 in items_that_differ:

                         cmap = get_cmap.get_max_colors_from_pallete(self.hclust_metrices[6])  
                         self.hclust_axes['im'].set_cmap(cmap)
                         self.reduce_tick_to_n(self.hclust_axes['colormap'],'y',3)
                         
                     if 5 in items_that_differ:                         
                          cmap = get_cmap.get_max_colors_from_pallete(self.hclust_metrices[5])  
                          if 'color_im' in self.hclust_axes:
                              self.hclust_axes['color_im'].set_cmap(cmap)  
                              tk.messagebox.showinfo('Error..','No color axis yet.')
                     if self.color_added is not None:
                          self.add_new_color_column(column = self.color_added, redraw= False)
                          if self.color_added  is not None:
                             self.add_information_to_labeling_rows()
                          ax = self.hclust_axes['map']
                          self.on_ylim_panning(ax, just_rename = True, redraw=False)
                          
 
                     if 4 in items_that_differ:
                          add_draw = False
                          line = self.hclust_axes['cluster_line_left']
                          x_data = line.get_xdata()[0]
                          self.hclust_axes['left'].clear()  
                          Y_row = self.hclust_axes['row_link']  
                          cmap = self.hclust_metrices[4]
                          rgb_vals = sns.color_palette(cmap,len(list(set(self.clust_h))),desat=0.75)
                          colors = [col_c(color) for color in rgb_vals]     
                          sch.set_link_color_palette(colors)
                          Z_row = sch.dendrogram(Y_row, orientation='left', color_threshold= x_data, leaf_rotation=90, ax = self.hclust_axes['left'])
                          self.add_cluster_number_to_dendo(self.hclust_axes['left'])
                          self.adjust_lines_in_hclust(self.hclust_axes['left'])
                          self.hclust_axes['left'].set_xticks([])
                          self.canvas.draw()                        
                          self.background_hclust = self.canvas.copy_from_bbox(self.hclust_axes['left'].bbox)
                          self.hclust_axes['cluster_line_left'] = self.hclust_axes['left'].axvline(x_data, linewidth=1.5, color = '#1f77b4')        
                          self.hclust_axes['left'].draw_artist(self.hclust_axes['cluster_line_left'])
                          self.canvas.blit(self.hclust_axes['left'].bbox)
                                                   
                     if add_draw == False:
                         pass
                     else:
                         
                         self.canvas.draw()
                     tk.messagebox.showinfo('Done..','Hierarchichal Cluster updated.')
                 else:
                     
                     self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),
                                   catnames = list(self.selectedCategories.keys() ),
                                                plot_type = 'hclust')    
                     tk.messagebox.showinfo('Done..','Hierarchichal Cluster had to be re-calculated for your changes. Additional colormaps as well as labels are not preserved.')
                     
                 
                 
                 
             
             w = 500
             popup.attributes('-topmost', True)
             popup.grab_set() 
             cont = self.create_frame(popup)  
             cont.pack(fill='both', expand=True)
             cont.grid_columnconfigure(1, weight=1)
             
             
             lab_text =  'Change settings for hierarchical clustering'
             
             lab_main = tk.Label(cont, text= lab_text, 
                                     font = LARGE_FONT, 
                                     fg="#4C626F", 
                                     justify=tk.LEFT, bg = MAC_GREY)
             
             lab_main.grid(padx=10, pady=15, columnspan=6, sticky=tk.W)
             
             
             vb_dist_row = tk.StringVar()
             vb_linkage_row= tk.StringVar()
             vb_dist_col = tk.StringVar()
             vb_linkage_col= tk.StringVar()
             calc_dendro_row = tk.BooleanVar()
             calc_dendro_col = tk.BooleanVar()
             calc_dendro_row.set(self.calculate_row_dendro)
             calc_dendro_col.set(self.calculate_col_dendro)
             cbs = [calc_dendro_row ,calc_dendro_col ]
             vars_ = []
             m = 0
             for dendo in ['row','column']:
                 title = tk.Label(cont, text = 'Settings for '+dendo, bg=MAC_GREY,font = LARGE_FONT, 
                                     fg="#4C626F")
                 title.grid(padx=10, pady=6, columnspan=2, sticky=tk.W,column=0)
                 sep = ttk.Separator(cont, orient =  tk.HORIZONTAL)
                 sep.grid(sticky=tk.EW, columnspan=2)
                 
                 cbs_ = ttk.Checkbutton(cont, text = "Calculate {} dendrogram".format(dendo), variable = cbs[m] )
                 cbs_.grid(padx=10, pady=3, sticky=tk.W,column=0)
                 m+=1
                 dist_label = tk.Label(cont, text = 'Distance metric: ', bg=MAC_GREY)
                 dist_label.grid(padx=10, pady=3, sticky=tk.W,column=0)
                 row_ = int(float(dist_label.grid_info()['row']))
                 if dendo == 'row':
                     idx_met = 0
                     vb_dist = vb_dist_row
                     vb_linkage = vb_linkage_row
                 else:
                     idx_met= 2
                     vb_dist = vb_dist_col
                     vb_linkage = vb_linkage_col
                 om_dist = ttk.OptionMenu(cont, vb_dist ,self.hclust_metrices[idx_met], *pdist_metric)
                 om_dist.grid(padx=10,pady=3,column=1, sticky=tk.E, row=row_ )
                 link_label = tk.Label(cont, text = 'Linkage method: ', bg=MAC_GREY)
                 link_label.grid(padx=10, pady=3, sticky=tk.W,column=0, row=row_ +1)
                 om_dist = ttk.OptionMenu(cont, vb_linkage , self.hclust_metrices[idx_met+1], *linkage_methods)
                 om_dist.grid(padx=10,pady=3,column=1, sticky=tk.E,row=row_ +1)
                 vars_.append(vb_dist)
                 vars_.append(vb_linkage)
                 
             col_pal_ = tk.Label(cont, text = 'Color palettes', bg=MAC_GREY,font = LARGE_FONT, 
                                     fg="#4C626F")  
             col_pal_.grid(padx=10, pady=15, columnspan=2, sticky=tk.W,column=0)
             sep = ttk.Separator(cont, orient =  tk.HORIZONTAL)
             sep.grid(sticky=tk.EW, columnspan=2,padx=10)
             
             col_lab = tk.Label(cont, text = 'Choose color palette for clusters in dendrogram: ', bg=MAC_GREY)
             col_lab.grid(padx=10, pady=3, sticky=tk.W,column=0)
             vb_col_dendo = StringVar()
             om_dist = ttk.OptionMenu(cont, vb_col_dendo , self.hclust_metrices[4], *color_schemes)
             om_dist.grid(padx=10,pady=3,column=1, sticky=tk.E, row= col_lab.grid_info()['row'])
             vars_.append(vb_col_dendo)
             
             vb_col_color_column = StringVar()
             col_lab = tk.Label(cont, text = 'Choose color palette for additional colormap: ', bg=MAC_GREY)
             col_lab.grid(padx=10, pady=3, sticky=tk.W,column=0)
            
             om_dist = ttk.OptionMenu(cont, vb_col_color_column , self.hclust_metrices[5], *color_schemes)
             om_dist.grid(padx=10,pady=3,column=1, sticky=tk.E, row= col_lab.grid_info()['row'])
             vars_.append(vb_col_color_column)
             col_lab = tk.Label(cont, text = 'Choose color palette for heatmap: ', bg=MAC_GREY)
             col_lab.grid(padx=10, pady=3, sticky=tk.W,column=0)
             vb_col_clust = StringVar()
             om_dist = ttk.OptionMenu(cont, vb_col_clust, self.hclust_metrices[-1], *color_schemes)
             om_dist.grid(padx=10,pady=3,column=1, sticky=tk.E, row= col_lab.grid_info()['row'])
             vars_.append(vb_col_clust)
             okay_but = ttk.Button(cont, text='Update', command = lambda vars_=vars_, cbs=cbs: update(vars_,cbs))    
             close_but = ttk.Button(cont, text = 'Close & Save', command =lambda popup=popup,vars_=vars_, cbs = cbs: close_and_save(popup,vars_,cbs))    
             okay_but.grid(padx=50, pady=8, sticky=tk.NS+tk.W,column=0)
             close_but.grid(padx=10, pady=8, sticky=tk.NS+tk.W,column=1, row = okay_but.grid_info()['row'])
             
             cont.grid_rowconfigure(okay_but.grid_info()['row'], weight=1)
             

             
         elif mode == 'Custom sorting' or mode == 'Re-Sort Columns' or mode == 'Re-Sort':
         
             def move_items_around(event,lb):
                 
                 selection = lb.curselection()
                 if len(selection) == 0:
                     return 
                 pos_item = selection[0]
                 curs_item = lb.nearest(event.y)
                 if pos_item == curs_item:
                    return
                 else:
                     text = lb.get(pos_item)
                     lb.delete(pos_item)
                     lb.insert(curs_item,text)
                     lb.selection_set(curs_item)
             def resort_source_column_data(lb,idx_,popup):
                 new_sort = list(lb.get(0, END))
                 self.sourceData.df_columns = new_sort
                 self.sourceData.df = self.sourceData.df[new_sort]
                 
                 index = range(0,len(new_sort))
                 dict_for_sorting = dict(zip(new_sort,index))
                 
                 for col in new_sort:
                     item = idx_+str(col)
                     parent = self.source_treeview.parent(item)
                     
                     index = dict_for_sorting[col]
                     self.source_treeview.move(item,parent,index)
                 tk.messagebox.showinfo('Done..','Columns were resorted. This order will also appear when you export the data frame.',parent=popup)    
             
             def resort_dtype(lb,idx_,pop,dtype):
                 new_sort = list(lb.get(0, END))
                 old_cols = self.sourceData.df_columns
                 old_cols_s = [col for col in old_cols if col not in new_sort]
                 self.sourceData.df_columns = new_sort + old_cols_s
                 self.sourceData.df = self.sourceData.df[self.sourceData.df_columns]
                  
                 for i,col in enumerate(new_sort):
                     item = idx_+str(col)
                     parent = self.source_treeview.parent(item)                    
                     index = i
                     self.source_treeview.move(item,parent,index)
                     

                 
                 tk.messagebox.showinfo('Done..','Columns were resorted in the provided order.Please note that the newly sorted columns are placed at the beginning of the source file. Visible upon export.',
                                        parent = popup) 

                 
             def resort_source_data(lb,col,popup):
                 new_sort = lb.get(0, END)
                 index = range(0,len(new_sort))
                 dict_for_sorting = dict(zip(new_sort,index))
                 self.sourceData.df['sorting_idx_instant_clue'] = self.sourceData.df[col].replace(dict_for_sorting)
                 self.sourceData.df.sort_values(by = 'sorting_idx_instant_clue',inplace=True)
                 self.sourceData.df.drop('sorting_idx_instant_clue',1, inplace=True)
                 
                 if len(last_called_plot) > 0:
                     self.prepare_plot(*last_called_plot[-1])
                 tk.messagebox.showinfo('Done..','Custom re-sorting performed.',parent=popup)
                 
             if  (len(self.DataTreeview.columnsSelected  ) > 1 and mode == 'Custom sorting') or (mode == 'Re-Sort Columns' and len(self.data_sources_selected) > 1) or (self.only_datatypes_selected and len(self.DataTreeview.columnsSelected  ) > 1 and mode == 'Re-Sort'):
                 if mode == 'Re-Sort Columns':
                     typ = 'Dataframe'
                     sel = self.data_sources_selected[0]
                 elif mode == 'Re-Sort':
                     typ = 'Data type'
                     sel = self.data_types_selected[0]
                 else:
                     typ = 'Column'
                     sel = str(self.DataTreeview.columnsSelected  [0])
                 tk.messagebox.showinfo('Note..','Can perform this action only on one {}.\nFirst one selected: {}\nSorting is stable, meaning that you can perform sorting on columns sequentially.'.format(typ,sel))
        
             if mode == 'Re-Sort Columns':
                 sel = self.data_sources_selected[0]
                 for key,value in self.idx_and_file_names.items():
                         if value == sel:
                             idx_ = key
                 self.set_source_file_based_on_index(idx_)            
                 uniq_vals = self.sourceData.df_columns
                 
             elif mode == 'Re-Sort':
                 if self.only_datatypes_selected:
                     sel = self.items_selected[0]
                     idx_,dtype = sel.split('_')[0]+'_',sel.split('_')[-1]
                     
                     
                 else:
                     return
                
                 self.set_source_file_based_on_index(idx_)
                 
                 ##set_data_to_current_selection()
                 
                 uniq_vals = [col for col in self.sourceData.df_columns if self.sourceData.df[col].dtype == dtype]
             else:
                 ##set_data_to_current_selection()
                 col_to_sort = self.DataTreeview.columnsSelected  [0]
                 uniq_vals = list(self.sourceData.df[col_to_sort].unique()) 
                 
                 

             
             popup.attributes('-topmost', True)
             
             cont = self.create_frame(popup)  
             cont.pack(fill='both', expand=True)
             cont.grid_rowconfigure(2, weight=1)
             cont.grid_columnconfigure(0, weight=1)
             lab_text =  'Move items in listbox in custom order'
             
             lab_main = tk.Label(cont, text= lab_text, 
                                     font = LARGE_FONT, 
                                     fg="#4C626F", 
                                     justify=tk.LEFT, bg = MAC_GREY)
             lab_main.grid(padx=10, pady=15, columnspan=6, sticky=tk.W)
             scrollbar1 = ttk.Scrollbar(cont,
                                          orient=VERTICAL)
             scrollbar2 = ttk.Scrollbar(cont,
                                          orient=HORIZONTAL)
             lb_for_sel = Listbox(cont, width=1500, height = 1500,  xscrollcommand=scrollbar2.set,
                                      yscrollcommand=scrollbar1.set, selectmode = tk.SINGLE)
             lb_for_sel.bind('<B1-Motion>', lambda event, lb=lb_for_sel : move_items_around(event,lb))
             lb_for_sel.grid(row=2, column=0, columnspan=3, sticky=tk.E, padx=(20,0))
             scrollbar1.grid(row=2,column=4,sticky = 'ns'+'e')
             scrollbar2.grid(row=5,column =0,columnspan=3, sticky = 'ew', padx=(20,0))
             
             scrollbar1.config(command=lb_for_sel.yview)
             scrollbar2.config(command=lb_for_sel.xview)
             
             fill_lb(lb_for_sel,uniq_vals)
             if mode == 'Re-Sort Columns':
                 but_okay = ttk.Button(cont, text = 'Sort', command = lambda lb=lb_for_sel,idx_=idx_, pop = popup: resort_source_column_data(lb,idx_,pop))
             elif mode == 'Re-Sort':
                 but_okay = ttk.Button(cont, text = 'Sort', command = lambda lb=lb_for_sel,idx_=idx_, pop = popup: resort_dtype(lb,idx_,pop,dtype))
             
             else:   
                 but_okay = ttk.Button(cont, text = 'Sort', command = lambda lb=lb_for_sel, col = col_to_sort, pop = popup: resort_source_data(lb,col,pop))
             but_close = ttk.Button(cont, text = 'Close', command = popup.destroy)
             but_okay.grid(row = 6, column = 1, pady=5)
             but_close.grid(row = 6, column = 2, pady=5)
                         
       
         center(popup,size=(w,h))
         if 'excel' in mode or 'add_error' in mode or 'choose subject column' in mode or 'choose subject and repeated measure column' in mode or 'Color' in mode: 
              self.wait_window(popup)
         

     def open_label_window(self):
         '''
         Opens a popup dialog to annotate desired rows.
         '''
         if len(self.plt.plotHistory) > 0:
             if self.plt.currentPlotType != 'scatter':
                     return 
             self.categorical_column_handler('Annotate scatter points')
         
     
     
     def reset_dicts_and_plots(self):
          if True:
              self.source_treeview.delete(*self.source_treeview.get_children())     
              for button in self.selectedNumericalColumns.values():
                  button.destroy() 
              for button in self.selectedCategories.values():
                  button.destroy()    
              self.selectedCategories.clear()
              self.selectedNumericalColumns.clear() 
              self.but_stored[9].configure(image= self.add_swarm_icon)
              

              self.swarm_but = 0
              self.add_swarm_to_new_plot = False
              self.remove_mpl_connection()
              self.performed_stats.clear() 
              self.count = 0 
              self.data_set_information.clear()

              self.f1.clf() 
              self.canvas.draw()               
              
          
     def source_file_upload(self, pathUpload = None, resetTreeEntries = True):
          """Upload file, extract data types and insert columns names into the source data tree"""

			
          if pathUpload is None:
              pathUpload = tf.askopenfilename(initialdir=self.folder_path.get(),
                                                         title="Choose File")
              if pathUpload == '':
                  return
                  
          fileName = pathUpload.split('/')[-1]
          
          if '.xsl' in fileName or '.xlsx' in fileName:
          	
          	dataFile = pd.ExcelFile(pathUpload)
          	sheets = dataFile.sheet_names 
          	excelImporter = excel_import.ExcelImporter(sheets,dataFile)
          	if excelImporter.data_to_export == False:
          		return
          	uploadedDataFrame = excelImporter.get_files()
          	if uploadedDataFrame is None:
          		return
          	naString = excelImporter.replaceObjectNan
          	del excelImporter

          elif '.txt' in fileName or '.csv' in fileName:
          
          	fileImporter = txt_file_importer.fileImporter(pathUpload)
          	uploadedDataFrame = fileImporter.data_to_export
          	if uploadedDataFrame is None:
          		return
          	naString = fileImporter.replaceObjectNan
          	del fileImporter
          	
          else:
          	tk.messagebox.showinfo('Error..','File format not supported yet.')
          	return
          
          if resetTreeEntries:
          	del self.sourceData
          	self.sourceData = data.DataCollection()
          	self.plt.clean_up_figure()
          	self.plt.redraw()	

          if isinstance(uploadedDataFrame,dict):
          		## if users selected all sheets
          	for sheetName, dataFrame in uploadedDataFrame.items():
          			
          		id = self.sourceData.get_next_available_id()
          		fileName = '{}_{}'.format(sheetName,fileName)
          		self.sourceData.add_data_frame(dataFrame, id = id, fileName = fileName)
          		self.sourceData.set_current_data_by_id(id)
          		self.update_all_dfs_in_treeview()
          		objectColumnList = self.sourceData.get_columns_data_type_relationship()['object']
          		self.sourceData.fill_na_in_columnList(objectColumnList,naString)
          
          else:
          		## add data frame to the sourceData class 
          	
          	id = self.sourceData.get_next_available_id()
          	self.sourceData.add_data_frame(uploadedDataFrame, id=id, fileName=fileName)
          	self.sourceData.set_current_data_by_id(id)
		  
		  ### extracts data type columns relationship and fills the source data
          	self.update_all_dfs_in_treeview()
          	## avoid nan in data categorical columns
          	objectColumnList = self.sourceData.get_columns_data_type_relationship()['object']
          	self.sourceData.fill_na_in_columnList(objectColumnList,naString)
         
          self.sourceData.replaceObjectNan = naString
          
          if resetTreeEntries:

          	self.plt = plotter._Plotter(self.sourceData,self.f1)

          	self.plt.set_scatter_point_properties(GREY,round(float(self.alpha_selected.get()),2),
          								int(float(self.size_selected.get())))
          if resetTreeEntries:
          	self.clean_up_dropped_buttons()
     
     def update_all_dfs_in_treeview(self):
     	'''
     	Updates the data frames added in the treeview
     	'''
     	dict_ = self.sourceData.dfsDataTypesAndColumnNames
     	file_names = self.sourceData.fileNameByID
     	self.DataTreeview.add_all_data_frame_columns_from_dict(dict_,file_names)
     	
     	
	    
 	
     def join_data_frames(self,method):
     	'''
     	Open Dialog to merge two or more dataframes.
     	'''
     	mergeDialog = mergeDataFrames.mergeDataFrames(self.sourceData, 
     												  self.DataTreeview,method,
     												  images = self.mergeImages)   
     	del mergeDialog
 	 	
 	         
     def add_new_dataframe(self,newDataFrame,fileName):
     	'''
     	Add new subset to source data collection and treeview
     	'''
     	id = self.sourceData.get_next_available_id()
     	self.sourceData.add_data_frame(newDataFrame, id=id, fileName=fileName)
     	dict_ = self.sourceData.dfsDataTypesAndColumnNames
     	file_names = self.sourceData.fileNameByID
     	
     	self.DataTreeview.add_all_data_frame_columns_from_dict(dict_,file_names) 
     	

          
	
     def icon_switch_due_to_rescale(self,event):

            new_width = event.width
            new_height = event.height
            n = 0
            global NORM_FONT
            if self.old_width is None:
                # defining the resoltion from start 
                
                self.old_width = event.width
                self.old_height = event.height 
                n = 1 ## to trigger rescaling if screen resolution is very small from beginning (e.g. laptop)
            
            icon_ =  check_resolution_for_icons(new_width,
                                                         new_height,
                                                         self.old_width,
                                                         self.old_height,
                                                         n)
            
            if icon_ is not None:
           ## data / session load and save 

           		
                if icon_ == 'NORM':
                    self.uploadFrameButtons['upload'].configure(image=self.open_file_icon_norm)
                    self.uploadFrameButtons['saveSession'].configure(image=self.save_session_icon_norm)
                    self.uploadFrameButtons['openSession'].configure(image=self.open_session_icon_norm )
                    self.uploadFrameButtons['addData'].configure(image=self.add_data_icon_norm) 
                    
                    self.sliceMarkFrameButtons['size'].configure(image = self.size_icon_norm)
                    self.sliceMarkFrameButtons['filter'].configure(image = self.filter_icon_norm)
                    self.sliceMarkFrameButtons['color'].configure(image = self.color_icon_norm) 
                    self.sliceMarkFrameButtons['label'].configure(image = self.label_icon_norm)
                    self.sliceMarkFrameButtons['tooltip'].configure(image = self.tooltip_icon_norm)
                    self.sliceMarkFrameButtons['selection'].configure(image = self.selection_icon_norm)
                    
                    self.but_col_icon = self.but_col_icon_norm
                    self.but_size_icon = self.but_size_icon_norm
                    self.but_tooltip_icon = self.but_tooltip_icon_norm
                    self.but_label_icon = self.but_label_icon_norm
                    self.but_stat_icon = self.but_stat_icon_norm
                    
                    if self.color_button_droped is not None:
                        self.color_button_droped.configure(image= self.but_col_icon)
                    if self.size_button_droped is not None:
                        self.size_button_droped.configure(image= self.but_size_icon)
                    if self.label_button_droped is not None:
                        self.label_button_droped.configure(image= self.but_label_icon)
                    if self.stat_button_droped is not None:
                         self.stat_button_droped.configure(image = self.but_stat_icon)
                    if self.tooltip_button_droped is not None:
                        self.tooltip_button_droped.configure(image=self.but_tooltip_icon)
                    
                    
                    if platform == 'WINDOWS': 
                        	NORM_FONT   = ("Helvetica", 8)
                    else:
                        	NORM_FONT =  ("Helvetica",11) 
                    
                    ###### LIST FOR PLOT OPTIONS 
                    icon_list = [self.point_plot_icon_norm,self.scatter_icon_norm,self.time_series_icon_norm ,self.matrix_icon_norm,self.dist_icon_norm,self.barplot_icon_norm ,
                                 self.box_icon_norm,self.violin_icon_norm, self.swarm_icon_norm ,self.add_swarm_icon_norm
                                 ,self.hclust_icon_norm,self.corr_icon_norm,self.config_plot_icon_norm] 
                    for i, icon in enumerate(icon_list):
                        self.but_stored[i].configure(image = icon)
                   
                    self.remove_swarm_icon = self.remove_swarm_icon_norm      
                    self.add_swarm_icon = self.add_swarm_icon_norm   
                    
                    
                    self.main_fig.configure(image=self.main_figure_icon_norm)
                       
                   # self.fig_history_button.configure(image=self.figure_history_icon_norm)
                    self.data_button.configure(image = self.streteched_data_norm )
                    self.mark_sideframe .configure(pady=2,padx=1)
                     
                          
                    self.sideframe_upload.configure(pady=2)
                    if platform == 'MAC':
                    	self.grid_columnconfigure(0, weight=1, minsize=213)
                    else:
                    	self.grid_columnconfigure(0, weight=1, minsize=213)
                    if platform == 'MAC':
                    	
                    	font_ = ('Helvertica',12)
                    	self.tx_space.configure(font = font_)
                    	self.cat_space.configure(font = font_)
                    	
                    for frame in self.label_frames:
                    	frame.configure(font=NORM_FONT)
                    self.label_marks1.configure(font=NORM_FONT)
                    self.label_marks2.configure(font=NORM_FONT)
                    self.label_nav.configure(font=NORM_FONT)
                    for but in self.selectedNumericalColumns.values():
                    	but.configure(font=NORM_FONT)
                    for but in self.selectedCategories.values():
                    	but.configure(font=NORM_FONT)
                    self.delete_all_button_num.configure(image = self.delete_all_cols_norm)
                    self.delete_all_button_cat.configure(image = self.delete_all_cols_norm)
                    	
                    	
              		
                    #self.label_nav.configure(font = ('Helvetica',5))
                elif icon_ == 'LARGE':
                    
                    self.uploadFrameButtons['upload'].configure(image=self.open_file_icon)
                    self.uploadFrameButtons['saveSession'].configure(image=self.save_session_icon)
                    self.uploadFrameButtons['openSession'].configure(image=self.open_session_icon)            
                    self.uploadFrameButtons['addData'].configure(image=self.add_data_icon) 
                                        
                    self.sliceMarkFrameButtons['size'].configure(image = self.size_icon)
                    self.sliceMarkFrameButtons['filter'].configure(image = self.filter_icon)
                    self.sliceMarkFrameButtons['color'].configure(image = self.color_icon) 
                    self.sliceMarkFrameButtons['label'].configure(image = self.label_icon)
                    self.sliceMarkFrameButtons['tooltip'].configure(image = self.tooltip_icon)
                    self.sliceMarkFrameButtons['selection'].configure(image = self.selection_icon)
                    
                    self.but_col_icon = self.but_col_icon_
                    self.but_size_icon = self.but_size_icon_
                    self.but_tooltip_icon = self.but_tooltip_icon_
                    self.but_label_icon = self.but_label_icon_
                    self.but_stat_icon = self.but_stat_icon_
                    
           
           
                    if self.color_button_droped is not None:
                        self.color_button_droped.configure(image= self.but_col_icon)
                    if self.size_button_droped is not None:
                        self.size_button_droped.configure(image= self.but_size_icon)
                    if self.label_button_droped is not None:
                        self.label_button_droped.configure(image= self.but_label_icon)
                    if self.stat_button_droped is not None:
                         self.stat_button_droped.configure(image = self.but_stat_icon)
                    if self.tooltip_button_droped is not None:
                        self.tooltip_button_droped.configure(image=self.but_tooltip_icon)
                    
                    self.main_fig.configure(image=self.main_figure_icon)
                    
                    self.sideframe_upload.configure(pady=5)
                    self.data_button.configure(image = self.streteched_data)
                    self.grid_columnconfigure(0, weight=0)
                    self.mark_sideframe.configure(pady=10,padx=5)
                    self.remove_swarm_icon = self.remove_swarm_icon_  
                    self.add_swarm_icon = self.add_swarm_icon_ 
                    
                    if platform == 'WINDOWS': 
                    	NORM_FONT   = (defaultFont, 9)
                    else:
                    	NORM_FONT =  (defaultFont,12) 
                    
                  
                    icon_list = [self.point_plot_icon,self.scatter_icon,self.time_series_icon ,self.matrix_icon,self.dist_icon,self.barplot_icon ,
                                 self.box_icon,self.violin_icon, self.swarm_icon ,self.add_swarm_icon
                                 ,self.hclust_icon,self.corr_icon,self.config_plot_icon] 
                    for i, icon in enumerate(icon_list):
                        self.but_stored[i].configure(image = icon)
                    #self.fig_history_button.configure(image=self.figure_history_icon)    
                   
                    if platform == 'MAC':
                    	
                    	font_ = (defaultFont,15)
                    	self.tx_space.configure(font = font_)
                    	self.cat_space.configure(font = font_)
                    for frame in self.label_frames:
                    	frame.configure(font=NORM_FONT)
                    self.label_marks1.configure(font=NORM_FONT)
                    self.label_marks2.configure(font=NORM_FONT)
                    self.label_nav.configure(font=NORM_FONT)
                    for but in self.selectedNumericalColumns.values():
                    	but.configure(font=NORM_FONT)
                    for but in self.selectedCategories.values():
                    	but.configure(font=NORM_FONT)
                    self.delete_all_button_num.configure(image = self.delete_all_cols)
                    self.delete_all_button_cat.configure(image = self.delete_all_cols)
                
            self.old_width = new_width
            self.old_height = new_height
            
            
            
     def get_images(self):
            
           '''
           Images are stored in base64 code in the module 'images'
           '''
          
           
           self.size_icon, self.color_icon, self.label_icon, \
           				self.filter_icon, self.selection_icon, self.tooltip_icon  = images.get_slice_and_mark_images()
           				
           self.size_icon_norm, self.color_icon_norm, self.label_icon_norm, \
           				self.filter_icon_norm, self.selection_icon_norm, self.tooltip_icon_norm  = images.get_slice_and_mark_images_norm()    
           				                          
           self.open_file_icon,self.save_session_icon,self.open_session_icon ,self.add_data_icon   = images.get_data_upload_and_session_images() 
              
           self.back_icon, self.center_align,self.left_align,self.right_align, \
           					self.config_plot_icon, self.config_plot_icon_norm = images.get_utility_icons()              
                              
           self.box_icon,self.barplot_icon,self.scatter_icon,self.swarm_icon,self.time_series_icon\
           					,self.violin_icon,self.hclust_icon,self.corr_icon,self.point_plot_icon, \
           					self.matrix_icon, self.dist_icon, self.add_swarm_icon_,self.remove_swarm_icon_  =   images.get_plot_options_icons()                   

           self.box_icon_norm,self.barplot_icon_norm ,self.scatter_icon_norm ,self.swarm_icon_norm ,self.time_series_icon_norm \
           					,self.violin_icon_norm ,self.hclust_icon_norm ,self.corr_icon_norm ,self.point_plot_icon_norm , \
           					self.matrix_icon_norm , self.dist_icon_norm, self.add_swarm_icon_norm , self.remove_swarm_icon_norm     = images.get_plot_options_icons_norm()
           
           self.open_file_icon_norm,self.save_session_icon_norm,self.open_session_icon_norm,self.add_data_icon_norm =  images.get_norm_data_upload_and_session_images()  
           
           self.streteched_data ,self.streteched_data_norm  = images.get_data_images()
           
           self.delete_all_cols, self.delete_all_cols_norm = images.get_delete_cols_images()
           
           self.right, self.outer, self.left, self.inner = images.get_merge_images()
           self.mergeImages = {'right':self.right,'left':self.left,
						'outer':self.outer,'inner':self.inner}
           
           self.but_col_icon_, self.but_col_icon_norm, self.but_label_icon_, \
        					self.but_label_icon_norm,  self.but_size_icon_,self.but_size_icon_norm, \
           					self.but_stat_icon_, self.but_stat_icon_norm, self.but_tooltip_icon_, self.but_tooltip_icon_norm  =  images.get_drop_button_images()
           
		
           self.main_figure_icon, self.main_figure_icon_norm = images.get_main_figure_button_images()
           
           self.but_col_icon = self.but_col_icon_
           self.but_size_icon = self.but_size_icon_
           self.but_tooltip_icon = self.but_tooltip_icon_
           self.but_label_icon = self.but_label_icon_
           self.but_stat_icon = self.but_stat_icon_
           self.add_swarm_icon = self.add_swarm_icon_
           self.remove_swarm_icon = self.remove_swarm_icon_
			
           
     def grid_widgets(self,controller):
           '''
           Grid widgets
           '''
        	
           self.hclust_metrices = ['euclidean','complete','euclidean','complete','Paired','Paired','RdYlBu']

           self.items_selected = []
           
           self.settings_points = dict(edgecolor='black',linewidth=0.4,zorder=5,alpha=float(self.alpha_selected.get()),
             					sizes = [float(self.size_selected.get())])    
            
            
            
            
             					    
           labelMain = tk.Label(self, text='Interactive Data Analysis',
                           font=LARGE_FONT,
                           fg="#4C626F",
                           justify=tk.LEFT,bg=MAC_GREY)
            
           labelMain.grid(row=0, pady=5, padx=20, sticky=tk.NW, columnspan=5)

           self.f1 = plt.figure(figsize = (19.5,12.8), facecolor='white')
    
           
           #### styles
           style = ttk.Style()
           style.configure("Grey.TLabel", foreground='grey', font = (defaultFont, 12))
           style.configure("White.TButton" , background ="white")
           
          
           self.global_chart_parameter = [8,9,8,8]  



    							            
           self.frame = tk.Frame(self,highlightbackground=MAC_GREY, \
           highlightcolor=MAC_GREY, highlightthickness=1)
           
           
           ## data / session load and save 
           self.uploadFrameButtons = OrderedDict() 
           imagesAndFunctionsUpload = OrderedDict([('upload', [self.open_file_icon,self.source_file_upload]),
           						 ('saveSession',[self.save_session_icon,self.save_current_session]),
           						 ('openSession',[self.open_session_icon,self.open_saved_session]),
           						 ('addData',[self.add_data_icon,lambda: self.source_file_upload(resetTreeEntries = False)])])
           						 
           for key,values in imagesAndFunctionsUpload.items():
           		button = create_button(self.sideframe_upload, image = values[0], command = values[1]) 
           		self.uploadFrameButtons[key] = button 
           		
           
           ## mark/slice frame buttons
           self.sliceMarkFrameButtons = OrderedDict() 	
           imageAndFunctionsSliceMark = OrderedDict([('size',[self.size_icon,lambda: self.design_popup(mode = 'Size setting')]),
           											 ('color',[self.color_icon,self.open_color_configuration]),
           											 ('label',[self.label_icon,self.open_label_window]),
           											 ('tooltip',[self.tooltip_icon,'']),
           											 ('selection',[self.selection_icon,self.select_data]),
           											 ('filter',[self.filter_icon,''])
           											])	
           											
           for key,values in imageAndFunctionsSliceMark.items():
           		button = create_button(self.mark_sideframe, image = values[0], command = values[1]) 
           		self.sliceMarkFrameButtons[key] = button  
           
           self.sliceMarkButtonsList = list(self.sliceMarkFrameButtons.values()) 
           sep_marks1 = ttk.Separator(self.mark_sideframe, orient = tk.HORIZONTAL)
           sep_marks2 = ttk.Separator(self.mark_sideframe, orient = tk.HORIZONTAL)
           
           self.label_marks2 = tk.Label(self.mark_sideframe, text = "Marks", bg=MAC_GREY)
           self.label_marks1 = tk.Label(self.mark_sideframe, text = "Slice Data", bg=MAC_GREY)
           
           ## numeric and categorical receiver boxes
           
                      
           self.delete_all_button_num = create_button(self.column_sideframe, 
           											  image = self.delete_all_cols, 
           											  command = lambda: self.clean_up_dropped_buttons(mode = 'num'))
           self.delete_all_button_cat = create_button(self.category_sideframe, 
           											  image = self.delete_all_cols, 
           											  command = lambda: self.clean_up_dropped_buttons(mode = 'cat'))        
        

           
           
           #self.fig_history_button = create_button(self.plotoptions_sideframe, 
           	#										image = self.figure_history_icon, 
           	#										command = self.show_graph_history)
           	
           
           self.main_fig = create_button(self, image = self.main_figure_icon, command = self.setup_main_figure) 
           sep_nav_ = ttk.Separator(self.plotoptions_sideframe, orient = tk.HORIZONTAL)
           self.label_nav = tk.Label(self.plotoptions_sideframe, text = "Navigation", bg=MAC_GREY)
           
           
           receiverBoxStyle = dict(justify=tk.CENTER, background =MAC_GREY, foreground ="darkgrey", font = (defaultFont,defaultFontSize+2)) 
           receiverBoxText = '                                            Drag & Drop here                        '
           
           self.tx_space = tk.Label(self.column_sideframe,text=receiverBoxText,
                                    **receiverBoxStyle) 
                                    
           self.cat_space = tk.Label(self.category_sideframe,text=receiverBoxText,
                                     **receiverBoxStyle)       

           
           back_button = create_button(self,image = self.back_icon, 
           								command =  lambda: controller.show_frame(start_page.StartPage)) 
           
           back_button.grid( in_=self,
                                         row=0,
                                         column = 5,
                                         rowspan=15,
                                         sticky=tk.N+tk.E,
                                         padx=4)


           
           chartTypes = ['pointplot','scatter','time_series','scatter_matrix','density','barplot','boxplot','violinplot', 'swarm','add_swarm','hclust','corrmatrix','configure']
           tooltip_info = tooltip_information_plotoptions
           # we are using the icon in desired order to create plot/chart options
           iconsForButtons = [
                                    self.point_plot_icon,
                                    self.scatter_icon,                                 
                                    self.time_series_icon,
                                    self.matrix_icon,   
                                    self.dist_icon,
                                    self.barplot_icon,                       
                                    self.box_icon, 
                                    self.violin_icon, 
                                    self.swarm_icon, 
                                    self.add_swarm_icon,                                                                                             
                                    self.hclust_icon,
                                    self.corr_icon,
                                    self.config_plot_icon]
           i = 0
           self.but_stored = []
           
           for n, buttonIcon in enumerate(iconsForButtons):

            	chartType = chartTypes[n]
            	if chartType in ['density','hclust','configure',
            					'barplot','corrmatrix']:
            		pady = (5,1)
            	else:
            		pady = 1
            	
            	if chartType == 'configure':
            	
            		commandButton = self.configure_chart
            	else:
            		commandButton = lambda plotChartType = chartType: self.change_plot_style(plot_type = plotChartType)
            	
            	chartButton = create_button(self, 
            								command = commandButton)
            								
            	text, title = tooltip_info[n]	
            							
            	CreateToolTip(chartButton, text  = text, title_ = title,   wraplength=230, bg ="white") 
            	
            	self.but_stored.append(chartButton)
            	if chartType in ['density','barplot','violinplot','boxplot','swarm']:
            		chartButton.bind(right_click, lambda event: self.post_menu(event,self.split_sub_menu))
            	elif chartType == 'hclust':
            		chartButton.bind(right_click, lambda event: self.post_menu(event,self.hclust_menu))
            	elif chartType == 'corrmatrix':
            		chartButton.bind(right_click, lambda event: self.post_menu(event,self.corrMatrixMenu))
            	if n & 1:
            		columnPos = 1
            	else:
            		columnPos = 0
            	chartButton.grid(in_ = self.plotoptions_sideframe, row = i ,column = columnPos, pady=pady)
            	if columnPos == 1:
           			i += 1
           self.main_fig .grid(in_=self.plotoptions_sideframe)	
           
           style_tree = ttk.Style(self)

           if platform == 'WINDOWS':
               style_tree.configure('source.Treeview', rowheight = 19, font = (defaultFont,8))
           else:
               style_tree.configure('source.Treeview', rowheight = 21, font = (defaultFont,11))
               

           self.source_treeview = ttk.Treeview(self.source_sideframe, height = "4", show='tree', style='source.Treeview')
           
           ## make the source treeview part of the sourceDataTreeview class that detects dataframes selected,
           ## data types selected, and handles adding new columns, as well as sorting
           
           self.DataTreeview = sourceDataTreeView.sourceDataTreeview(self.source_treeview)
           self.source_treeview.heading("#0", text="")
           self.source_treeview.column("#0",minwidth=800)
           self.source_treeview.bind("<B1-Motion>", self.on_motion) 
           self.source_treeview.bind("<ButtonRelease-1>", self.release) 
           self.source_treeview.bind("<Double-Button-1>", self.identify_item_and_start_tooltip)
           #self.source_treeview.bind('<Motion>', self.destroy_tt)
           
           self.source_treeview.bind(right_click, self.on_slected_treeview_button3)
                          
           sourceScroll = ttk.Scrollbar(self, orient = tk.HORIZONTAL, command = self.source_treeview.xview)
           sourceScroll2 = ttk.Scrollbar(self,orient = tk.VERTICAL, command = self.source_treeview.yview)
           self.source_treeview.configure(xscrollcommand = sourceScroll.set,
                                          yscrollcommand = sourceScroll2.set)
           
           self.build_analysis_tree()
           
           
           self.data_button = create_button(self.source_sideframe, 
           						image = self.streteched_data, 
           						command = self.show_data)#
           
           padDict = dict(padx=8,pady=1.5) 
           
           self.sliceMarkFrameButtons['filter'].grid(in_=self.mark_sideframe, row=2, column = 1 , sticky=tk.W, **padDict)
           self.sliceMarkFrameButtons['selection'].grid(in_=self.mark_sideframe, row=2, column = 0, **padDict)
           self.sliceMarkFrameButtons['size'].grid(in_=self.mark_sideframe, row=5 , column=1, sticky=tk.W,**padDict)
           self.sliceMarkFrameButtons['color'].grid(in_=self.mark_sideframe, row=5 , column=0, **padDict)
           self.sliceMarkFrameButtons['label'].grid(in_=self.mark_sideframe, row=6 , column=0, **padDict)
           self.sliceMarkFrameButtons['tooltip'].grid(in_=self.mark_sideframe, row=6 , column=1, sticky=tk.W,**padDict)
           

           sep_marks1.grid(in_=self.mark_sideframe, row=1, column = 0 , columnspan=2, sticky = tk.EW)
           self.label_marks1.grid(in_=self.mark_sideframe, row=0, column = 0 , sticky = tk.W)
           sep_marks2.grid(in_=self.mark_sideframe, row=4, column = 0 , columnspan=2, sticky = tk.EW)
           self.label_marks2.grid(in_=self.mark_sideframe, row=3, column = 0 , sticky = tk.W)
           self.delete_all_button_num.pack(in_=self.column_sideframe,   anchor=tk.NE, side=tk.RIGHT)
           self.delete_all_button_cat.pack(in_=self.category_sideframe,   anchor=tk.NE, side=tk.RIGHT)
           self.tx_space.pack(in_=self.column_sideframe,  fill = tk.BOTH,expand=True )#f,
           self.cat_space.pack(in_=self.category_sideframe, fill = tk.BOTH,expand=True)
           recBox = ['numeric','category']
           if platform == 'MAC':
           	pady_ = 1
           else:
           	pady_ = 4
           
            
        
           for dtype in recBox:
           		placeHolder = tk.Label(self.receiverFrame[dtype], text='',
           			bg = MAC_GREY)
           		placeHolder.pack(side=tk.LEFT,padx=2,pady=pady_)
           
           for column,button in enumerate(self.uploadFrameButtons.values()):
           		button.grid(in_=self.sideframe_upload, row=1, column=column, padx=4)              
           self.data_button.pack(in_= self.source_sideframe,  pady=2, anchor=tk.W)
           sourceScroll2.pack(in_= self.source_sideframe, side = tk.LEFT, fill=tk.Y, anchor=tk.N)
           self.source_treeview.pack(in_= self.source_sideframe, padx=0, expand=True, fill=tk.BOTH, anchor = tk.NE) 
           sourceScroll.pack(in_= self.source_sideframe, padx=0,anchor=tk.N, fill=tk.X) 
           
           self.frame.grid(in_=self,
                                     row=5,
                                     column =3,
                                     rowspan=25,
                                     pady=(90,20),
                                     sticky=tk.NSEW,
                                     padx=5)
     	 

     