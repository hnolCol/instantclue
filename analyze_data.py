"""
	""MODULE to define Graphical User Interface and its functions""
	
    Instant Clue - Interactive Data Visualization and Analysis.
    Copyright (C) Hendrik Nolte

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""
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
from modules import workflow
from modules import find_cat_overlap
# internal dialog window imports
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
from modules.dialogs import define_groups_dim_reduction
from modules.dialogs import pivot_table
from modules.dialogs import dimRed_transform
from modules.dialogs import VerticalScrolledFrame
from modules.dialogs import custom_filter
from modules.dialogs import color_configuration
from modules.dialogs import correlations
from modules.dialogs import settings
from modules.dialogs import mask_filtering
from modules.dialogs import multi_block
from modules.dialogs import import_TDT
from modules.dialogs import shift_data
from modules.dialogs import compare_groups
from modules.dialogs import legend_handler


from modules.plots.time_series_helper import aucResultCollection
from modules.dialogs.simple_dialog import simpleUserInputDialog, simpleListboxSelection
from modules.utils import *

from modules.calculations.normalize import dataNormalizer, quantileNormalize
from modules.calculations.feature_selection import selectFeaturesFromModel, estimators, estimatorSettings,checkDataType
import os
import time
import string
import re

from decimal import Decimal

import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
from urllib.request import urlopen

import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore", 'This pattern has match groups')

import itertools
from collections import OrderedDict
import gc


#matplotlib imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Lasso
from matplotlib import colors
from matplotlib import path
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
	#matplotlib 2
	from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
except:
	#matplotlib 3
	from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
	
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

## Import Seaborn and set the default settings
import seaborn as sns
sns.set(style="ticks",font=defaultFont)
sns.axes_style('white')
##

from scipy.stats import wilcoxon
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import f

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.libqsturng import psturng



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
           self.initiate()

           # add empty figure to GUI
           self.display_graph(self.f1)
           # actions on resizing the window (e.g. changing the icons to smaller/bigger ones)
           #self.bind("<Configure>", self.icon_switch_due_to_rescale)
           self.scale_icons_to_small() #default for all screen resolutions now
           self.grid_columnconfigure(0, weight=1, minsize=50)
           self.bind("<Button-1>", self.app_has_focus)

     def initiate(self):

           ## sourceData holds all data frames , self.plt is the plotter class,
           ## anovaTestCollection saves all anova tests made
           ## curveFitCollection saves all curve fittings made by the user
           self.workflow = workflow.workflowCollection()
           self.sourceData = data.DataCollection(workflow = self.workflow)
           self.plt = plotter._Plotter(self.sourceData,self.f1,workflow = self.workflow)
           self.mainFigureCollection = main_figures.mainFigureCollection(self)
           self.anovaTestCollection = anova_calculations.storeAnovaResultsClass()
           self.dimensionReductionCollection = stats.dimensionReductionCollection()
           self.curveFitCollection = curve_fitting.curveFitCollection()
           self.clusterCollection = clustering.clusterAnalysisCollection()
           self.classificationCollection =classification.classifierAnalysisCollection()
           self.colorHelper = color_configuration.colorMapHelper()
           self.statResultCollection = stats.statisticResultCollection()
           self.aucResultCollection = aucResultCollection()
           self.interactiveWidgetHelper = interactive_widget_helper.interactiveWidgetsHelper(self.mark_sideframe, 
           																			colorHelper = self.colorHelper)
           
           self.workflow.add_handles(self.sourceData, self.plt, self.DataTreeview, self)
           

     def define_variables(self):			
			
           self.menuCollection = dict()
           
           self.circulizeDendrogram = tk.BooleanVar(value=False)
           self.showCluster = tk.BooleanVar(value=True)
           self.enforceLabel = tk.BooleanVar(value=False)
           self.size_selected = tk.StringVar(value = '50')

           ## stats Test
           self.twoGroupstatsClass = None

           ###
           self.old_width = None
           self.old_height = None
           self.currentMenu  = None
           
           self.pathUpload = path_file
           self.tooltipFirstTime = True

			
           ## dicts to save dropped column names

           self.selectedNumericalColumns  = OrderedDict()
           self.selectedCategories  = OrderedDict()

           self.colormaps = dict()

           ##scatter matrix dicts
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
           self.selection_press_event = None
           self.pick_label = None
           self.pick_freehand  = None
           self.label_button_droped = None
           self.tooltip_button_droped = None
           self.size_button_droped = None
           self.color_button_droped = None
           self.stat_button_droped = None
           self.release_event = None
           self.mot_button_dict = dict()
           self.mot_button = None
           self.dimReduction_button_droped = None

     def build_menus(self):
           '''
           Build menus to be used.
           '''
           self.build_main_drop_down_menu_treeview()
           self.def_split_sub_menu()
           self.build_selection_export_menu()
           self.build_merge_menu()
           self.build_datatype_menu()
           self.build_main_figure_menu()
           self.build_pca_export_menu()
           self.build_corrMatrix_menu()
           self.build_hclust_menu()
           self.build_scatter_menu()
           self.build_analysis_menu()
           self.build_lineplot_menu()
           self.build_receiverBox_menu()
           self.build_addit_file_menu()



     def build_label_frames(self):

           '''
           Builds label frames.
           '''
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
           self.general_settings_sideframe = tk.LabelFrame(self, text = 'Settings', relief=tk.GROOVE, padx=3, pady=7, bg=MAC_GREY)
           self.receiverFrame['numeric'] = self.column_sideframe
           self.receiverFrame['category'] = self.category_sideframe
           
           self.numTool = CreateToolTip(self.column_sideframe,
           								title_ = 'Numerical Receiver Box', 
           								text = '',
           								waittime = 1500)
           self.catTool = CreateToolTip(self.category_sideframe,
           								title_ = 'Categorical Receiver Box', 
           								text = '',
           								waittime = 1500)


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
                                     column = 0,
                                     rowspan = 4,
                                     sticky=tk.EW,#+tk.NW,
                                     padx=5)

           self.source_sideframe.grid(in_=self,
                                     row=5,
                                     column =0,
                                     pady=(0,6),
                                     sticky=tk.EW+tk.NS,
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
                                     sticky=tk.EW+tk.NW+tk.S,
                                     padx=5,
                                     pady=(0,5))
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

           self.plotoptions_sideframe.grid(in_=self,
                                     row=1,
                                     column = 5,
                                     rowspan=15,
                                     sticky=tk.NW,
                                     padx=5)

           self.general_settings_sideframe.grid(in_=self,
                                     row=11,
                                     column = 5,
                                     rowspan=4,
                                     sticky=tk.EW+tk.SW,
                                     pady=(0,5),
                                     padx=5)

     def build_main_drop_down_menu_treeview(self):
           '''
           This has grown historically and the code needs to be re-written
           '''
           menuDict = {}
           menus = ['main','column','dfformat','sort','split','replace','dataType','logCalc',\
           'rolling','smoothing','rowColCalc','multTest','curvefit','categories',\
           'predict','transform','correlation','nanReplace','basicCalc','normalization',\
           'featureSelection','columnBasicDiv','columnBasicSub','aggRows','time_series',
           'compareGroups','featureSelection','classification','modelFeatSel','nanReplaceCol']
           
          # _, self.filterIcon,self.calcAddColumn,_ = images.get_workflow_images()

           for menu in menus:
           	menuDict[menu] = tk.Menu(self, **styleDict)

           rowColumnCalculations = ['Mean [row]','Median [row]','Stdev [row]','Sem [row]',
           									'Mean & Stdev [row]','Mean & Sem [row]',
           									'Square root [row]','x * N [row]','x ^ N [row]',
           									'N ^ x [row]']

           splitOptions = ["Space [ ]","Semicolon [;]","Comma [,]","U-Score [_]",
           					"Minus [-]","Slash [/]","B-Slash [\]","Custom String"]

           replace_options = ['0 -> NaN','NaN -> 0','NaN -> Constant','NaN -> Mean[col]','NaN -> Mean[row]',
           					  'NaN -> Median[col]','NaN -> Gauss Distribution']
           
           normOptions = ['Standardize','Quantile (25,75)','0->1']

           rollingOptions = ['mean','median','quantile','sum','max','min','std']

           nanDroppingOptions = ['all == NaN','any == NaN','Threshold']

           mult_opt = ['FWER','bonferroni','sidak','holm-sidak','holm','simes-hochberg','hommel',
           				'FDR - methods','benjamini-hochberg','benjamini-yekutieli',
           				'2-stage-set-up benjamini-krieger-yekutieli (recom.)',
           				'gavrilov-benjamini-sarkar','q-value','storey-tibshirani']

           menuDict['main'].add_command(label ="Data management ..",state=tk.DISABLED, foreground ="darkgrey")
           menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label='Column operation ..', menu = menuDict['column'])
           menuDict['main'].add_cascade(label="Sort data by ..", menu = menuDict['sort'])
           menuDict['column'].add_command(label = 'Rename', command = self.rename_columns)
           menuDict['column'].add_command(label='Duplicate', command = self.duplicate_column)
           menuDict['column'].add_command(label='Delete', accelerator = 'Delete',
           		command = self.delete_column)
           menuDict['column'].add_command(label="Combine",accelerator = "{}+M".format(ctrlString),
           		command = self.combine_selected_columns)
           menuDict['column'].add_cascade(label="Split on ..", menu = menuDict['split'])
           menuDict['column'].add_cascade(label='Change data type to..', menu = menuDict['dataType'])
           menuDict['column'].add_cascade(label="Replace", menu = menuDict['replace'])
           menuDict['column'].add_command(label='Count through', command = self.create_count_through_column)
           menuDict['column'].add_command(label='Count valid values', command = self.count_valid_values)
           menuDict['column'].add_command(label='Factorize', command = self.factorize_column)
            
           menuDict['column'].add_cascade(label='Drop rows with NaN ..', menu = menuDict['nanReplace'])
           menuDict['column'].add_cascade(label='Drop columns with NaN ..', menu = menuDict['nanReplaceCol'])
           #menuDict['column'].add_cascade(label='Feature selection', menu = menuDict['featureSelection'])
           
          
           for opt in nanDroppingOptions:
           	menuDict['nanReplace'].add_command(label=opt, command = lambda how = opt: self.remove_rows_with_nan(how))
			
          
           for opt in nanDroppingOptions:
           	menuDict['nanReplaceCol'].add_command(label=opt, command = lambda how = opt, columns = True: self.remove_rows_with_nan(how,columns))
			
           menuDict['dataType'].add_command(label='Float', command = lambda: self.change_column_type(changeColumnTo ='float64'))
           menuDict['dataType'].add_command(label='Category', command = lambda: self.change_column_type(changeColumnTo ='str'))
           menuDict['dataType'].add_command(label='Integer', command = lambda: self.change_column_type(changeColumnTo = 'int64'))

           menuDict['sort'].add_command(label="Value", command = lambda s = "Value":  self.sort_source_data(s))
           menuDict['sort'].add_command(label="String length", command = lambda s = "String length":  self.sort_source_data(s))
           menuDict['sort'].add_command(label="Custom order", command = self.custom_sort_values)#lambda : tk.messagebox.showinfo('Under revision','Currently under revision. Will be available in the next minor update.'))#)self.design_popup(mode='Custom sorting'))
           for splitString in splitOptions:
     
           	menuDict['split'].add_command(label=splitString,
               	command = lambda splitString = splitString:  self.split_column_content_by_string(splitString))
               	
           menuDict['replace'].add_command(label='Find & Replace', accelerator  = "{}+R".format(ctrlString), command = lambda: findAndReplace.findAndReplaceDialog(dfClass = self.sourceData, dataTreeview = self.DataTreeview))
           for i,replaceOption in enumerate(replace_options):
               menuDict['replace'].add_command(label=replaceOption, command = lambda replaceOption = replaceOption:  self.replace_data_in_df(replaceOption))

           menuDict['main'].add_cascade(label='Change data format', menu = menuDict['dfformat'])
           menuDict['dfformat'].add_command(label = 'To long format (melt)', command = self.melt_data)
           menuDict['dfformat'].add_command(label = 'To grouped long format (melt)', command = self.melt_data_by_groups)
           menuDict['dfformat'].add_command(label = 'To wide format (pivot)', command = self.pivot_data)
           menuDict['dfformat'].add_command(label = 'Transpose', command = self.transpose_data)
           menuDict['dfformat'].add_command(label = 'Create Annotation Modules',
           									command = self.create_categorical_module) 
           menuDict['dfformat'].add_command(label = 'Unstack Column', command = self.unstack_column)           	
           for text in mult_opt:
           		if text in ['FWER','FDR - methods','q-value']:
           			menuDict['multTest'].add_command(label=text,
           					state = tk.DISABLED, foreground="darkgrey")
           			menuDict['multTest'].add_separator()
           		else:
           			menuDict['multTest'].add_command(label=text, \
           			command = lambda proc = text: self.multiple_comparision_correction(proc))

           ## Data
           menuDict['main'].add_command(label ="Data Transformation ..",state=tk.DISABLED,
           				foreground='darkgrey')#, image = self.calcAddColumn, compound = tk.LEFT)
           menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label='Smoothing', menu = menuDict['smoothing'])
           menuDict['smoothing'].add_cascade(label='Rolling', menu = menuDict['rolling'])
           for rolling in rollingOptions:
               menuDict['rolling'].add_command(label=rolling, command = lambda rolling = rolling: self.rolling_mod_data(rolling))
          
           for rolling in rollingOptions:
           		if rolling != 'quantile':
           			menuDict['aggRows'].add_command(label = rolling, command = lambda metric = rolling: self.aggregate_data(metric))
           
           menuDict['smoothing'].add_cascade(label='Aggregate n rows by ..', menu = menuDict['aggRows'])
           menuDict['smoothing'].add_command(label='IIR filter', command = self.iir_filter)

           #menuDict['main'].add_cascade(label='Module Intersection', command = self.create_module_intersection)

           menuDict['main'].add_cascade(label='Row & column calculations', menu = menuDict['rowColCalc'])
           menuDict['rowColCalc'].add_command(label='Summary Statistics', command = self.summarize)
           menuDict['rowColCalc'].add_cascade(label="Basic", menu = menuDict['basicCalc'])
           menuDict['rowColCalc'].add_cascade(label="Row Normalization", menu = menuDict['normalization'])
           menuDict['rowColCalc'].add_command(label="Quantile Normalization (col)", command = self.norm_quant_data)
           menuDict['rowColCalc'].add_command(label="Scale to mean and unit variance (col)", command = self.scale_quant_data)
           menuDict['rowColCalc'].add_command(label="Scale to mean (col)", command = lambda: self.scale_quant_data(withStd=False))
           
           
           
           menuDict['rowColCalc'].add_cascade(label="Logarithmic", menu = menuDict['logCalc'])
           
           for logType in ['log2','-log2','ln','log10','-log10']:
               menuDict['logCalc'].add_command(label=logType,
               				command = lambda logType = logType : self.transform_selected_columns(logType))
           menuDict['rowColCalc'].add_command(label='Z-Score [row]',
           		command = lambda transformation = 'Z-Score_row': self.transform_selected_columns(transformation))
           menuDict['rowColCalc'].add_command(label='Z-Score [columns]',
           		command = lambda transformation = 'Z-Score_col': self.transform_selected_columns(transformation))
           
           for metric in normOptions:
                    menuDict['normalization'].add_command(label=metric,
                    	command = lambda metric = metric: self.normalize_data(metric))
           for metric in rowColumnCalculations :
                    menuDict['basicCalc'].add_command(label=metric,
                    	command = lambda metric = metric: self.calculate_row_wise_metric(metric))
           menuDict['basicCalc'].add_cascade(label='Divide by ..', menu = menuDict['columnBasicDiv'])
           menuDict['basicCalc'].add_cascade(label='Subtract by ..', menu = menuDict['columnBasicSub'])
           n = 0 
           for subMenu in [menuDict['columnBasicDiv'],menuDict['columnBasicSub']]:
           		if n == 0:
           			operation = 'divide'
           		elif n == 1:
           			operation = 'subtract'
           		n += 1        		
           		subMenu.add_command(label='Column [row-wise]',command = lambda operation = operation:self.divide_or_subtract_columns(operation = operation))
           		subMenu.add_command(label='Column Median',command = lambda operation = operation: self.divide_or_subtract_columns(byMedian=True,operation = operation))
           		subMenu.add_command(label='Value',command = lambda operation = operation: self.divide_or_subtract_columns(byValue=True,operation = operation))        
        
           menuDict['rowColCalc'].add_command(label='Kernel Density Estimation [col]',
           										command = self.calculate_density)										
           menuDict['rowColCalc'].add_command(label='Compare groups (t-test,ANOVA ..)',
           									command = lambda : compare_groups.compareGroupsDialog(selectedColumns = self.DataTreeview.columnsSelected, dfClass = self.sourceData,
           										treeView = self.DataTreeview))					
           menuDict['rowColCalc'].add_cascade(label='Multiple Testing Correction', menu = menuDict['multTest'])
           menuDict['main'].add_command(label ="Filters ..",state=tk.DISABLED,foreground='darkgrey')
           					#image = self.filterIcon,compound="left")
           menuDict['main'].add_separator()
           menuDict['main'].add_command(label="Annotate Numeric Filter", accelerator = "{}+N".format(ctrlString),
           		command = self.numeric_filter_dialog)
           menuDict['main'].add_cascade(label="Categorical Filters", menu = menuDict['categories'],
           					)
           menuDict['categories'].add_command(label="Find Category & Annotate",
           	command = lambda : self.categorical_column_handler('Find category & annotate'))
           menuDict['categories'].add_command(label="Find String(s) & Annotate", accelerator = "{}+F".format(ctrlString),
           	command = lambda: self.categorical_column_handler('Search string & annotate'))
           menuDict['categories'].add_command(label = 'Custom Categorical Filter',
           	command = self.custom_filter)
           menuDict['categories'].add_command(label = 'Subset Data on Category',
           	command = lambda: self.categorical_column_handler('Subset data on unique category'))
        
           menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label ="Time series ..",menu = menuDict['time_series'])
           menuDict['time_series'].add_command(label = 'Base line correction', command = self.correct_baseline)
           menuDict['time_series'].add_command(label='Add as error' ,command = self.add_error)
           menuDict['time_series'].add_command(label='Adjust starting point', command = self.shift_data)

           #menuDict['main'].add_command(label ="Fit, Correlate and Predict..",state=tk.DISABLED,foreground='darkgrey')
           menuDict['main'].add_separator()
           #menuDict['main'].add_cascade(label='Correlation',menu = menuDict['correlation'])
           menuDict['correlation'].add_command(label="Correlate rows to ..." , command = self.calculate_correlations)
           #menuDict['correlation'].add_command(label="Display correlation analysis .." ,
           #	command = lambda: tk.messagebox.showinfo('Under construction','Under construction ..',parent=self))

          # for featureSel in ['Model','Variance']:#,'Recursive elimination']:
           menuDict['featureSelection'].add_command(label='Variance', command = lambda featureSel = 'Variance': self.select_features(featureSel))
           menuDict['featureSelection'].add_cascade(label='Model',menu = menuDict['modelFeatSel'])
           #for model in estimators.keys():
           	#menuDict['modelFeatSel'].add_command(label=model, command = lambda featureSel = model: self.select_features(featureSel))
           
           
           menuDict['main'].add_cascade(label='Feature selection by..', menu= menuDict['featureSelection'])
           menuDict['main'].add_separator()
           
       #    menuDict['featureSelection'].add_command(label='Drop cols with low variance ..', command = self.drop_cols_with_low_variance)           
           #menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label='Correlation',menu = menuDict['correlation'])
           menuDict['main'].add_cascade(label='Curve fit', menu= menuDict['curvefit'] )
           menuDict['curvefit'].add_command(label="Curve fit of rows to...", command = self.curve_fit)
           menuDict['curvefit'].add_command(label="Display curve fit(s)", command = self.display_curve_fits)
           menuDict['main'].add_cascade(label='Classification', menu = menuDict['classification'])
           menuDict['classification'].add_command(label='Cross Validation Based Grid Search', command = self.start_grid_search)
           menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label='Predictions', menu= menuDict['predict'] )
           menuDict['predict'].add_command(label = 'Predict Cluster', command = lambda: clustering.predictCluster(self.clusterCollection, self.sourceData, self.DataTreeview))
           menuDict['predict'].add_command(label = 'Predict Class', command = lambda: classification.predictClass(self.DataTreeview, self.sourceData,self.classificationCollection))
           menuDict['main'].add_separator()
           menuDict['main'].add_cascade(label='Transform by ..', menu= menuDict['transform'] )
           menuDict['transform'].add_command(label = 'Dimensional Reduction Model', command = self.apply_dimRed)
           menuDict['main'].add_separator()
           menuDict['main'].add_command(label='To clipboard', accelerator = "{}+C".format(ctrlString),
           		command = lambda: self.copy_file_to_clipboard(self.sourceData.get_current_data()))	
           self.menuCollection['main'] = menuDict['main']

     def build_pca_export_menu(self):
     	'''
     	'''
     	self.menuCollection['PCA'] = tk.Menu(self,**styleDict)
     	self.menuCollection['PCA'].add_command(label='Define groups ..',
     				command = self.define_groups_in_dimRed)
     	self.menuCollection['PCA'].add_command(label='Remove/show feature names',
     				command = lambda : self.plt.nonCategoricalPlotter.hide_show_feature_names())
     				
     	self.menuCollection['PCA'].add_command(label='Center plots',
     				command = lambda : self.plt.nonCategoricalPlotter.center_score_plot())
       	
     	self.menuCollection['PCA'].add_command(label='Add Hotellings T^2 CI (95%)',
     				command = lambda : self.plt.nonCategoricalPlotter.add_ci_to_dimRed())
     				   				     						
     	self.menuCollection['PCA'].add_command(label='Export Projections',
     				command = lambda :  self.export_dimRed_results('Export PCA Scores'))
     	self.menuCollection['PCA'].add_command(label='Export Loadings / Embedding',
     				command = lambda: self.export_dimRed_results('Export Loadings'))
     	self.menuCollection['PCA'].add_command(label='Add Loadings To Source Data',
     				command = lambda: self.export_dimRed_results('Add Loadings To Source Data'))

     def build_addit_file_menu(self):
     	''''''
     	self.menuCollection['addFiles'] = tk.Menu(self,**styleDict)
     	
     	self.menuCollection['addFiles'].add_command(label='Load Multiple Files', 
     		command = lambda: self.source_file_upload('',resetTreeEntries = len(self.sourceData.dfs) == 0, 	
     												   mergeMultipleFiles = True))
     	self.menuCollection['addFiles'].add_command(label='Load TDT tanks',
     		command = lambda: self.source_file_upload('',resetTreeEntries = len(self.sourceData.dfs) == 0, 	
     												   loadTDTTanks = True))	

     def build_hclust_menu(self):

         self.menuCollection['hclust'] =  tk.Menu(self,**styleDict)
         colorBarMenu = tk.Menu(self,**styleDict)
         
         self.menuCollection['hclust'].add_checkbutton(label='Circular row dendrogram',
         		variable = self.circulizeDendrogram,
         		command = self.plot_circulized_dendrogram) 
         for t in ['Raw data','Center 0','min = -1, max = 1','Custom values']:
         	colorBarMenu.add_command(label= t, command = lambda type = t: self.scale_colorbar(type))         

        
         self.menuCollection['hclust'].add_cascade(label='Colorbar',
         		menu = colorBarMenu)             
         
         self.menuCollection['hclust'].add_checkbutton(label='Show clusters',
         		variable = self.showCluster,
         		command = lambda: setattr(self.plt,'showCluster',self.showCluster.get()))          		         
         self.menuCollection['hclust'].add_checkbutton(label='Enforce row labels', 
         		variable = self.enforceLabel,
         		command = self.add_row_labels)         
         self.menuCollection['hclust'].add_command(label='Add cluster # to source file', 
         		command = self.add_cluster_to_source)
         self.menuCollection['hclust'].add_command(label='Find entry in hierarch. cluster', 
         		command = lambda: self.categorical_column_handler('Find entry in hierarch. cluster'))
         self.menuCollection['hclust'].add_command(label='Export Cluster to Excel', command = self.save_hclust_to_excel)	     		

	
    
     def scale_colorbar(self, type = 'Raw data'):
     	'''
     	'''
     	if type not in ['Raw data','Center 0','min = -1, max = 1','Custom values']:
     		return
     	
     	if type == 'Custom values':
     		xRange = [str(x) for x in np.arange(-10,10)]
     		input = simpleUserInputDialog(['Min','Max'],
     			self.plt.hclustLimitType if len(self.plt.hclustLimitType) == 2 and isinstance(self.plt.hclustLimitType,list) else ['-1','1'],
     			[xRange,xRange],
     			'Provide min and max',
     			'Set min and maximum for hierarchical clustering.')
     		if len(input.selectionOutput) != 0:
     			vals = []
     			for v in input.selectionOutput.values():
     				try:
     					vals.append(float(v))
     				except:
     					
     					tk.messagebox.showinfo('Error..',
     						'Transforming input to float failed. Aborting.')
     					return
     			
     			self.plt.hclustLimitType = vals
     		
     	else:
     	
     		self.plt.hclustLimitType = type
     	
     	if self.plt.currentPlotType in ['hclust','corrmatrix']:
     		
     		plotterInUse = self.plt.nonCategoricalPlotter
     		plotterInUse._hclustPlotter.update_colorMesh()
     

    
     def build_lineplot_menu(self):

     	self.menuCollection['line_plot'] = tk.Menu(self,**styleDict)
     	self.menuCollection['line_plot'].add_command(label='Find entry', command = lambda: self.categorical_column_handler('Find entry in line plot'))

     def build_receiverBox_menu(self):

     	self.menuCollection['receiverBox'] = tk.Menu(self,**styleDict)
     	self.menuCollection['receiverBox'].add_command( label= '               ', state = tk.DISABLED)
     	
     	self.menuCollection['receiverBox'].add_command(label='Sort', 
     		command = lambda : self.sort_source_data('Value',self.columnInReceiverBox,False,True))

     	self.menuCollection['receiverBox'].add_command(label='Color encode', 
     		command = lambda: self.update_color(columnNames = self.columnInReceiverBox))
     	
     	self.menuCollection['receiverBox'].add_command(label='Tooltip', 
     		command = lambda: self.add_tooltip_information(self.columnInReceiverBox))

     def build_merge_menu(self):

         self.menuCollection['data_frame'] = tk.Menu(self, **styleDict)
         exportCascade = tk.Menu(self, **styleDict)
         splitCascade = tk.Menu(self, **styleDict)
         self.menuCollection['data_frame'].add_command(label='Data frame menu', state = tk.DISABLED,foreground="darkgrey")
         self.menuCollection['data_frame'].add_separator()
         self.menuCollection['data_frame'].add_command(label = 'Collapse Tree', command = self.update_all_dfs_in_treeview)
         self.menuCollection['data_frame'].add_command(label='Re-Sort columns', command = self.custom_column_order)
         self.menuCollection['data_frame'].add_separator()
         self.menuCollection['data_frame'].add_command(label='Concatenate', command = lambda: self.join_data_frames('Concatenate'))
         self.menuCollection['data_frame'].add_command(label='Merge', command = lambda: self.join_data_frames('Merge'))
         self.menuCollection['data_frame'].add_command(label='Rename', command = self.rename_data_frame)
         self.menuCollection['data_frame'].add_separator()
         #self.menuCollection['data_frame'].add_cascade(label='Split..', menu = splitCascade)  
         
         #splitCascade.add_command(label = 'Shuffle Split') 
         #splitCascade.add_command(label = 'Stratified Shuffle Split') 
         #splitCascade.add_command(label = 'Stratified k-fold')
         #splitCascade.add_command(label = 'Time Series Split')      
         
         self.menuCollection['data_frame'].add_separator()
         self.menuCollection['data_frame'].add_command(label = "Delete", command = self.delete_data_frame_from_source)
         self.menuCollection['data_frame'].add_cascade(label='Export', menu= exportCascade)
         exportCascade.add_command(label='To .txt', command = lambda: self.export_data_to_file(data = None, format_type = 'txt', checkSelection = True))
         exportCascade.add_command(label='To .csv', command = lambda: self.export_data_to_file(data = None, format_type = 'csv', checkSelection = True))
   
         exportCascade.add_command(label='To Excel', command = lambda: self.export_data_to_file(data = None, checkSelection = True))
         exportCascade.add_command(label='To Clipboard', command = lambda: self.copy_file_to_clipboard(data = None))
         
           
     def build_scatter_menu(self):

         self.menuCollection['scatter'] = tk.Menu(self, **styleDict)
         self.menuCollection['scatter'].add_checkbutton(label='Binned Scatter', command = self.activate_binning_in_scatter)

     def build_analysis_menu(self):

     	self.menuCollection['analysis_treeview']  = tk.Menu(self, **styleDict)
     	resultMenu = tk.Menu(self, **styleDict)
     	self.menuCollection['analysis_treeview'] .add_cascade(label='Results', menu = resultMenu)
     	resultMenu.add_command(label='Compare two groups', command = lambda : self.show_statistical_test(True))
     	resultMenu.add_command(label='ANOVA', command = self.show_anova_results)
     	resultMenu.add_command(label='Area under curve', command = lambda : self.show_auc_calculations(True))


     def build_datatype_menu(self):

         self.menuCollection['data_type_menu']  = tk.Menu(self, **styleDict)
         self.menuCollection['data_type_menu'] .add_command(label='Sort and rename columns', 
         										state = tk.DISABLED, foreground = "darkgrey")
         self.menuCollection['data_type_menu'] .add_separator()
         self.menuCollection['data_type_menu'] .add_command(label = 'Custom column order', 
         									command  =  self.custom_column_order)#lambda: tk.messagebox.showinfo('Under revision','Currently under revision. Will be available in the next minor update.'))#self.design_popup(mode='Re-Sort'))
         self.menuCollection['data_type_menu'] .add_command(label='Sort columns alphabetically', 
         									command = self.re_sort_source_data_columns)
         self.menuCollection['data_type_menu'] .add_command(label='Colum names - Find and replace', 
         													command = lambda : findAndReplace.findAndReplaceDialog('ReplaceColumns',
         															self.sourceData,self.DataTreeview))
         self.menuCollection['data_type_menu'] .add_separator()
         self.menuCollection['data_type_menu'] .add_command(label = 'Collapse Tree', 
         									command = self.update_all_dfs_in_treeview)

     def build_selection_export_menu(self):

         self.menuCollection['selection_menu']  =  tk.Menu(self, **styleDict)
         self.menuCollection['selection_menu'] .add_command(label='Create sub-dataset',
         									 command = self.create_sub_data_frame_from_selection)
         self.menuCollection['selection_menu'] .add_command(label='Add annotation column', 			
         									 command = self.add_annotation_column_from_selection)
         self.menuCollection['selection_menu'] .add_command(label='Exclude from Dataset', 
         									command = lambda: self.drop_selection_from_df())
         self.menuCollection['selection_menu'] .add_command(label='Copy to clipboard', 
         									command = lambda: self.copy_file_to_clipboard(self.get_data_from_scatter_selection()))
         self.menuCollection['selection_menu'] .add_command(label='Export Selection [.txt]',
         									command = lambda: self.export_data_to_file(self.get_data_from_scatter_selection(),
         																			initial_file='Selection', format_type = 'txt'))
         self.menuCollection['selection_menu'] .add_command(label='Export Selection [.xlsx]',
         									command = lambda: self.export_data_to_file(self.get_data_from_scatter_selection(),
         																			initial_file='Selection',  format_type = 'Excel',sheet_name = 'SelectionExport'))
         self.menuCollection['selection_menu'] .add_command(label='Stop Selection', 
         									foreground = "red", command = self.stop_selection)
     
     def build_main_figure_menu(self):

         self.menuCollection['main_figure_menu']  = tk.Menu(self, **styleDict)
         self.menuCollection['main_figure_menu'] .add_command(label='Add in main figure to ..',
         															foreground='darkgrey')
         self.menuCollection['main_figure_menu'] .add_separator()


     def build_corrMatrix_menu(self):
     	'''
     	'''
     	self.variableDict = {'pearson':tk.BooleanVar(value=True),
     					'spearman':tk.BooleanVar(value=False),
     					'kendall':tk.BooleanVar(value=False)}
     	self.menuCollection['corrmatrix'] = tk.Menu(self, **styleDict)
     	
     	
     	self.menuCollection['corrmatrix'].add_checkbutton(label='Circularized row dendrogram', 
     									 variable = self.circulizeDendrogram,
     									 command = lambda: self.plot_circulized_dendrogram(plot_type = 'corrmatrix'))
     	
     	self.menuCollection['corrmatrix'].add_command(label = ' Corr. method ',
     									state = tk.DISABLED,foreground='darkgrey')
     	self.menuCollection['corrmatrix'].add_separator()
     	for method, variable in self.variableDict.items():
     		self.menuCollection['corrmatrix'].add_checkbutton(label = method,
     											variable = variable,
     											command = lambda method=method: self.update_corr_matrix_method(method))
     	self.menuCollection['corrmatrix'].add_separator()
     	self.menuCollection['corrmatrix'].add_command(label='Results', 
     											command = self.display_corrmatrix_results)

     def build_analysis_tree(self):
        '''
        Builds treeview in analysis label frame (statistical toolbox)
        '''
        seps_tests =   ['Model fitting',
                        'Compare two groups',
                        'Compare multiple groups',
                        'Two-W-ANOVA',
                        'Three-W-ANOVA',
                        'Cluster Analysis',
                        'Classification',
                        #'Feature Selection',
                        'Dimensional Reduction',
                        'Multi-Block Analysis',
                        'Correlation',
                        'Curve fitting',
                        'Miscellaneous',]

        self.options_for_test = dict({'Model fitting':['linear',
                                         # 'logarithmic',
                                          'lowess',
                                          #'expontential',
                                          ],
                        'Compare two groups':['t-test','Welch-test',
                                              'Wilcoxon [paired non-para]',
                                              'Whitney-Mann U [unpaired non-para]'
                                              ],
                        'Compare multiple groups':['1W-ANOVA','1W-ANOVA-RepMeas','Kruskal-Wallis'],
                        'Miscellaneous':['Pairwise Comparision','AUC'],#,'Density'],
                        'Classification':['CV based Grid Search'],#,'PLSA-DA'],#classification.availableMethods,
                      #  'Feature Selection':['Predicitve Model','Recursive Elimination'],
                        'Cluster Analysis':clustering.availableMethods,
                        'Two-W-ANOVA':['2W-ANOVA','2W-ANOVA-RepMeas(1fac)','2W-ANOVA-RepMeas(2fac)'],
                        'Three-W-ANOVA':['3W-ANOVA','3W-ANOVA-RepMeas(1fac)','3W-ANOVA-RepMeas(2fac)','3W-ANOVA-RepMeas(3fac)'],
                        'Dimensional Reduction':list(stats.dimensionalReductionMethods.keys()),
                        'Multi-Block Analysis': ['SGCCA'],#'PLS Canonical', 'PLS Regression','PLS-DA',
                        'Correlation': ['Correlate rows to ..'],#'Display correlations'
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
        self.stats_tree.bind(right_click, self.post_analysis_menu)
        self.stats_tree.column("#0",minwidth=800)

        for heads in seps_tests:
            main = self.stats_tree.insert('','end',str(heads), text=heads)
            for opt_test in self.options_for_test[heads]:

                    sub1 = self.stats_tree.insert(main, 'end', str(opt_test), text = opt_test)
                    if heads == 'Compare two groups':
                        if opt_test in  ['t-test','Welch-test']:
                            for sub_opt in opt_two_groups:
                                 sub2 = self.stats_tree.insert(sub1, 'end','%s_%s' % (opt_test,sub_opt), text=sub_opt)
                                 for direction in direct_test:
                                    self.stats_tree.insert(sub2, 'end', '%s_%s_%s' %  (opt_test,sub_opt,direction), text=direction)
                        else:
                                for direction in direct_test:
                                    self.stats_tree.insert(sub1, 'end', '%s_%s' % (str(opt_test),str(direction)), text=direction)
        #grid items
        sourceScroll = ttk.Scrollbar(self, orient = tk.HORIZONTAL, command = self.stats_tree.xview)
        sourceScroll2 = ttk.Scrollbar(self,orient = tk.VERTICAL, command = self.stats_tree.yview)
        self.stats_tree.configure(xscrollcommand = sourceScroll.set,
                                          yscrollcommand = sourceScroll2.set)
        sourceScroll2.pack(in_=self.analysis_sideframe, side = tk.LEFT, fill=tk.Y, anchor=tk.N)
        self.stats_tree.pack(in_=self.analysis_sideframe, padx=0, expand=True, fill=tk.BOTH)
        sourceScroll.pack(in_=self.analysis_sideframe, padx=0,anchor=tk.N, fill=tk.X)


     def post_menu(self, event = None, menu = None):
     	'''
     	Posts any given menu at the mouse x y coordinates
     	'''
     	x = self.winfo_pointerx()
     	y = self.winfo_pointery()
     	menu.bind("<FocusOut>", self.app_has_focus)
     	self.hide_tooltips()

     	menu.focus_set()
     	menu.post(x,y)
     	self.currentMenu = menu
     	
     def hide_tooltips(self):
     	
     	try:
     		self.numTool.hide()
     		self.catTool.hide()
     	except:
     		pass     

     def def_split_sub_menu(self):

         self.menuCollection['split_sub']  =  tk.Menu(self, **styleDict)
         self.plotCumulativeDist = tk.BooleanVar()

         self.menuCollection['split_sub'] .add_checkbutton(label='Split Categories',
         								variable = self.split_on_cats_for_plot ,
         								command = lambda: self.plt.set_split_settings(self.split_on_cats_for_plot.get()))

         self.menuCollection['split_sub'] .add_separator()
         self.menuCollection['split_sub'] .add_checkbutton(label='Use Cumulative Density Function', variable = self.plotCumulativeDist , command = lambda: self.plt.set_dist_settings(self.plotCumulativeDist.get()))

################# Functions to build menus done #################

     def activate_binning_in_scatter(self,value = None):
     	'''
     	Change from normal to binned scatter
     	'''
     	if self.plt.binnedScatter:
     		self.plt.binnedScatter = False
     	else:
     		self.plt.binnedScatter = True

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
     	'Categorical column ({}) has been added. Indicating if data of that row were in selection.'.format(columnName),\
     	parent = self)

     def add_error(self):
     	'''
     	Adds an error represented by a grey area around a time series signal.
     	'''
     	if self.plt.currentPlotType != 'time_series':
     		tk.messagebox.showinfo('Error..','Only useful for chart type : "time series".')
     		return

     	if self.DataTreeview.onlyNumericColumnsSelected:
     		currentDataFrameId = self.sourceData.currentDataFile
     		selectionIsFromSameData,selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     		if selectionIsFromSameData:

     			dataId = self.plt.get_dataID_used_for_last_chart()
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
     								 ' the plotted columns simply enter "None".', h = 100,
     								 )
     			selection = dialog.selectionOutput
     			if len(selection) != 0:
     				self.plt.nonCategoricalPlotter.timeSeriesHelper.add_error_to_lines(selection)
     				self.plt.redraw()

     	else:
     		tk.messagebox.showinfo('Error ..','Please select only numerical columns (floats, and integers)')

 
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


     def add_cluster_to_source(self):
     	'''
     	Cluster being identified in a hclust plot can be added to the source file.
     	'''
     	if self.plt.currentPlotType in ['hclust','corrmatrix']:
     		plotterInUse = self.plt.nonCategoricalPlotter
     		idData = plotterInUse._hclustPlotter.dataID
     		self.sourceData.set_current_data_by_id(idData)
     		columnName = plotterInUse._hclustPlotter.export_cluster_number_to_source()
     		if columnName is None:
     			tk.messagebox.showinfo('Error ..','No row clustering was performed ..',parent=self)
     			return

     		self.DataTreeview.add_list_of_columns_to_treeview(idData,
     													dataType = 'object',
     													columnList = [columnName])
     		tk.messagebox.showinfo('Done ..','Cluster numbers were added.', parent=self)
     		
     		self.workflow.add('calcColumn', 
     					self.sourceData.currentDataFile,
     					{'funcDataR':'delete_columns_by_label_list',
     					'argsDataR':{'columnLabelList':[columnName]},
     					'funcTreeR':'delete_entry_by_iid',
     					'argsTreeR':{'iid':'{}_{}'.format(self.sourceData.currentDataFile,columnName)},
     					'description':OrderedDict([('Activity:','Cluster # num added.'),
     					('Description:','A column has been added indicating the identified cluster (hclust chart).'),
     					('Column name:',columnName),
     					('Selected Columns (hclust):',get_elements_from_list_as_string(plotterInUse.numericColumns, maxStringLength = None)),
     					('Data ID:',self.sourceData.currentDataFile)])})

     def add_row_labels(self):
     	'''
     	Cluster being identified in a hclust plot can be added to the source file.
     	'''
     	if self.enforceLabel.get():
     		value = np.inf
     	else:
     		value = 55
     		
     	self.plt.numRows = value
     	if self.plt.currentPlotType in ['hclust','corrmatrix']:
     		
     		plotterInUse = self.plt.nonCategoricalPlotter
     		plotterInUse._hclustPlotter.numRows = value
     		plotterInUse._hclustPlotter.on_ylim_change()
     
     
     def add_new_dataframe(self,newDataFrame,fileName):
     	'''
     	Add new subset to source data collection and treeview
     	'''
     	id = self.sourceData.get_next_available_id()
     	self.sourceData.add_data_frame(newDataFrame, id=id, fileName=fileName)
     	dict_ = self.sourceData.dfsDataTypesAndColumnNames
     	file_names = self.sourceData.fileNameByID

     	self.DataTreeview.add_all_data_frame_columns_from_dict(dict_,file_names)
     	

     def add_swarm_to_figure(self):
     	'''
     	Helper function to trigger the addition of
     	swarm plot onto the underlying graph.
     	Please note that if the data get bigger stripplot instead of swarm
     	will be used (less computing time) the difference is that in
     	swarm plots you can estimate the distribution much better due to non
     	overlapping points.
     	'''
     	help = self.plt.get_active_helper()
     	help.add_swarm_to_plot()
     	self.plt.redraw() 

     def add_tooltip_information(self, columnNames = None):
     	'''
     	Add tooltip information to plot. 
     	It will determine the free hand selection if active.
     	Input 
     	=======
     	
     	columnNames - list. Column names in the selected data frame used for tooltip text.
     	'''
     	self.stop_selection(replot = False)           
     	if columnNames is None:
     		columnNames = self.selection_is_from_one_df()
     		if columnNames is None:
     			return  
     		
     	self.plt.add_tooltip_info(columnNames)		
     	
     	if platform == 'LINUX' and self.tooltipFirstTime:
     		self.tooltipFirstTime =  False
     		tk.messagebox.showinfo('Note..',
     			'There is currently an error in Linux for the very first added Tooltip. '+
     			'Please just drag & drop the desired column again on the tooltip icon.',
     			parent=self)
 
    
     def aggregate_data(self, metric = 'mean'):
     	'''
     	Aggregates n row using a user specific mean. 
     	'''
     	selectedColumns = self.selection_is_from_one_df(onlyNumeric = True)
     	if selectedColumns is not None:
     		
     		n = ts.askinteger('Defin n',
     			prompt = 'Please provide the number of rows to be used for aggregation.')
     		
     		if n is not None:
     	    	
     			df = self.sourceData.metric_over_n_rows(selectedColumns,n)
     			
     			if df is None:
     			
     				tk.messagebox.showinfo('Error ..',
     					'An error occured. No data frame added.',
     					parent=self)
     				return
     				
     			# add data frame
     			fileName = '(n{}_{})_{}'.format(n,metric,
     				self.sourceData.get_file_name_of_current_data())
     				
     			self.add_new_dataframe(df,fileName)
     			
     			tk.messagebox.showinfo('Done ..',
     				'Aggregation applied. Data frame has been added.', 
     				parent = self)

     def apply_dimRed(self):
     	'''
     	Dimensional reduction applied to "unseen" data.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:       	
     		dimRedDialog = dimRed_transform.transformDimRedDialog(self.dimensionReductionCollection,
     															self.sourceData, self.DataTreeview)
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.',parent=self)

     def app_has_focus(self,event):
     	'''
     	Check if the app has focus unless destroy menu (essential for linux)
     	'''
     	if self.currentMenu is not None:
     		self.currentMenu.unpost()
     		self.currentMenu = None      
     			
     def calculate_correlations(self):
     	'''
     	Calculates correlation of rows against given values
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     		
     		if self.DataTreeview.onlyNumericColumnsSelected == False:
     			tk.messagebox.showinfo('Error ..','Please select only numeric data.',parent=self)
     			return

     		corrDialog = correlations.correlationDialog(dfClass=self.sourceData,
     										selectedColumns = selectedColumns)
     		cor_data = corrDialog.get_correlations()
     		if cor_data.empty:
     			return

     		columnsAddedToDf = self.sourceData.join_df_to_currently_selected_df(cor_data, exportColumns = True)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,'float64',columnsAddedToDf)
     		tk.messagebox.showinfo('Done..','Calculations were performed and data were added to the source data treeview.',parent=self)

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
     	
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:

     		if metric in askFloatTitlePrompt:
     			title, prompt =  askFloatTitlePrompt[metric]
     			## changed to use askstring instread of asfloat because
     			## the function float() can also interprete entered strings
     			## such as 1/400
     			promptValue = ts.askstring(title,prompt,initialvalue='2')
     			try:
     				promptN = float(promptValue)
     			except:
     				promptNumbers = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", promptValue)
     				try:
     					promptN = float(promptNumbers[0])
     				except:
     					tk.messagebox.showinfo('Error ..',
     						'Could not convert input to number.',
     						parent=self)
     			if promptN is None:
     				return

     		newColumnNames = self.sourceData.calculate_row_wise_metric(metric,columns,promptN)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													dataType = 'float64',
     													columnList = newColumnNames)
     		tk.messagebox.showinfo('Done ..','Calculations performed. Columns added.')
     		
     		self.workflow.add('calcColumn', 
     					self.sourceData.currentDataFile,
     					{'funcDataR':'delete_columns_by_label_list',
     					'argsDataR':{'columnLabelList':newColumnNames},
     					'funcTreeR':'delete_entry_by_iid',
     					'argsTreeR':{'iid':['{}_{}'.format(self.sourceData.currentDataFile,col) for col in newColumnNames]},
     					'description':OrderedDict([('Activity:','Calculation of {}.'.format(metric)),
     					('Description:','A column has been added containing the results.'),
     					('Column name(s):',get_elements_from_list_as_string(newColumnNames, maxStringLength = None)),
     					('Selected Columns:',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     					('Data ID:',self.sourceData.currentDataFile)])})    		




     def calculate_density(self):
     	'''
     	Calculates kernel density estimate and adds new column to the datatreeview
     	'''
     	if self.DataTreeview.onlyNumericColumnsSelected == False:
     		tk.messagebox.showinfo('Error ..','Please select only numerical columns for this type of calculation.',
     								parent=self)
     		return

     	numericColumns = self.selection_is_from_one_df()
     	if numericColumns is not None:

     		densityColumnName = self.sourceData.calculate_kernel_density_estimate(numericColumns)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,dataType = 'float64',
     														columnList = [densityColumnName])
     		self.workflow.add('calcColumn', 
     					self.sourceData.currentDataFile,
     					{'funcDataR':'delete_columns_by_label_list',
     					'argsDataR':{'columnLabelList':[densityColumnName]},
     					'funcTreeR':'delete_entry_by_iid',
     					'argsTreeR':{'iid':'{}_{}'.format(self.sourceData.currentDataFile,densityColumnName)},
     					'description':OrderedDict([('Activity:','Cluster # num added.'),
     					('Description:','A column has been added containing kernel density values.'),
     					('Column name:',densityColumnName),
     					('Selected Columns (hclust):',get_elements_from_list_as_string(numericColumns, maxStringLength = None)),
     					('Data ID:',self.sourceData.currentDataFile)])})     		

     		tk.messagebox.showinfo('Done ..','Representation of a kernel-density estimated using '+
     								'Gaussian kernels done. Column has been added.', parent = self)


     def copy_file_to_clipboard(self, data= None, fromSelection = False):
         '''
         Copies data to clipboard
         '''
         if fromSelection:
         	columns = self.DataTreeview.columnsSelected
         	data = self.sourceData.get_current_data_by_column_list(columns)
         
         if data is None:
         	data = self.get_selected_data()
         	if data is None:
         			return
         try:
         	data.to_clipboard(excel=True, na_rep = "NaN",index=False, encoding='utf-8', sep='\t')
         except:
         	tk.messagebox.showinfo('Error ..','No copy/paste mechanism found for your system. Please export via txt files.')

     def count_valid_values(self):
     	'''
     	Count valid values
     	'''
     	if self.DataTreeview.onlyNumericColumnsSelected == False:
     		tk.messagebox.showinfo('Error ..',
     			'Please select only numerical columns for this type of calculation.')
     		return

     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     		columnName = self.sourceData.count_valid_values(selectedColumns)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													dataType = 'int64',
     													columnList = [columnName],
     													)
     		tk.messagebox.showinfo('Done ..','Counting done. New column added (integers).')
     		
     		self.workflow.add('calcColumn', 
     					self.sourceData.currentDataFile,
     					{'funcDataR':'delete_columns_by_label_list',
     					'argsDataR':{'columnLabelList':[columnName]},
     					'funcTreeR':'delete_entry_by_iid',
     					'argsTreeR':{'iid':'{}_{}'.format(self.sourceData.currentDataFile,columnName)},
     					'description':OrderedDict([('Activity:','Count valid values.'),
     					('Description:','A column has been added indicating the number of valid values (non NaN) in selected columns.'),
     					('Column name:',columnName),
     					('Selected Columns:',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     					('Data ID:',self.sourceData.currentDataFile)])})


     def create_categorical_module(self):
     	'''
     	Creates a cateogrial module.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:

     		dialog = simpleUserInputDialog(['Split String','Aggregate Method'],
     										[';','median'], 
     										[[';',',','//','_'],['mean','median','sum']],
     										'Annotation Module Settings',
     										'Tis method will find unique categories in selected column'+
     										' and aggregate data by these categories. If a row is annotated'+
     										' by multiple categories it is inlcuded multiple times within the'
     										' aggregated data frame.')
     										
     	
     		if dialog.selectionOutput is None:
     			return
     	
     		splitString = dialog.selectionOutput['Split String']
     		aggMethod = dialog.selectionOutput['Aggregate Method']     	
     		progressBar = Progressbar(title= 'Categorical modules')
     		dataID, fileName = self.sourceData.create_categorical_modules(aggregate = aggMethod, sepString = splitString,
     								categoricalColumn = selectedColumns, progressBar = progressBar)
     		if dataID is None: # aggreagation did not work
     			return
     		self.DataTreeview.add_new_data_frame(dataID,
     								fileName,
     								self.sourceData.dfsDataTypesAndColumnNames[dataID])

     def create_module_intersection(self):
      	'''
      	Intersection modules Create a new data frame.
      	'''
      	selectedColumns = self.selection_is_from_one_df()
      	if selectedColumns is not None: 
      	
      		unique = self.sourceData.get_current_data()['UniqueCats']
      		model = find_cat_overlap.findCategoricalIntersection(self.sourceData.get_current_data()[selectedColumns[0]],
      			unique,data=self.sourceData.get_current_data(),
      			numericColumns = self.sourceData.get_numeric_columns())
      			
      		df = model.fit()
      		self.add_new_dataframe(df,'ModuleSelection')

     def create_sub_data_frame_from_selection(self):
         '''
         When user has defined data selection by Lasso, a new data frame is created
         and added to the treeview
         '''
         sub_data = self.sourceData.df[self.sourceData.df.index.isin(self.data_selection.index)]
         currentFileName = self.sourceData.fileNameByID[self.plt.get_dataID_used_for_last_chart()]
         nameOfSubset = 'selection_{}'.format(currentFileName)
         self.add_new_dataframe(sub_data,nameOfSubset)

     def create_count_through_column(self):
     	'''
     	Counts through the data in current order.
     	'''
     	selectedColumns  = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     		columnName = self.sourceData.add_count_through_column()
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													dataType = 'int64',
     													columnList = [columnName],
     													startIndex = -1)
     		tk.messagebox.showinfo('Done ..','Index column was added to the treeview.')
     		self.workflow.add('calcColumn', 
     					self.sourceData.currentDataFile,
     					{'funcDataR':'delete_columns_by_label_list',
     					'argsDataR':{'columnLabelList':[columnName]},
     					'funcTreeR':'delete_entry_by_iid',
     					'argsTreeR':{'iid':'{}_{}'.format(self.sourceData.currentDataFile,columnName)},
     					'description':OrderedDict([('Activity:','Count through.'),
     					('Description:','A column has been added that enumerates over the data.'),
     					('Column name:',columnName),
     					('Data ID:',self.sourceData.currentDataFile)])})

     def check_input(self):
     	'''
     	Check if the data frame that was used for the last chart is the same
     	as the one that is newly dropped onto the receiver boxes. If not
     	the receiver boxed will be cleaned up.
     	'''
     	dataFrames = self.DataTreeview.dataFramesSelected
     	lastUsedDf = self.plt.get_dataID_used_for_last_chart()
     	if lastUsedDf is not None and len(dataFrames) != 0:
            	if lastUsedDf == dataFrames[0]:
            		pass
            	else:
            		self.clean_up_dropped_buttons()
     
     def change_pointSize_swarm(self):
     	'''
     	Allows adjustment of swarm points.
     	'''
    
    
     def add_swarm_to_figure(self):
     	'''
     	Helper function to trigger the addition of
     	swarm plot onto the underlying graph.
     	Please note that if the data get bigger stripplot instead of swarm
     	will be used (less computing time) the difference is that in
     	swarm plots you can estimate the distribution much better due to non
     	overlapping points.
     	'''
     	help = self.plt.get_active_helper()
     	help.add_swarm_to_plot()
     	self.plt.redraw() 
     	
     	
     
     def change_plot_style(self, plot_type = ''):
         '''
         Function that handles the event triggered by plot options.
         Very important step is to set the data selection back to the one
         that was used to generate the last chart. Otherwise you might
         experience difficulties that column headers are not present.
         '''
         
         if len(self.sourceData.dfs) == 0:
         	tk.messagebox.showinfo('No data ..','Please upload a file first.',parent = self)
         	return
	
         if len(self.plt.plotHistory) == 0:
             return

         dataID = self.plt.get_dataID_used_for_last_chart()
         self.sourceData.set_current_data_by_id(dataID)

         numericColumns = list(self.selectedNumericalColumns.keys())
         categoricalColumns = list(self.selectedCategories)

         underlying_plot = self.plt.currentPlotType

         if plot_type  not in ['boxplot','violinplot','barplot','add_swarm']:

             self.but_stored[10].configure(image = self.add_swarm_icon)
             self.swarm_but = 0
             self.plt.addSwarm = False

         if plot_type == 'add_swarm':
             if underlying_plot not in ['boxplot','violinplot','barplot']:

                 tk.messagebox.showinfo('Error..','Not useful to add swarm plot to this '+
                 						'type of chart. Possible chart types: Boxplot, '+
                 						'Violinplot and Barplot')
                 return

             if self.swarm_but == 0:
                 self.but_stored[10].configure(image= self.remove_swarm_icon)
                 self.add_swarm_to_figure()
                 self.swarm_but = 1

             else:
                 self.but_stored[10].configure(image = self.add_swarm_icon)
                 self.swarm_but = 0
                 self.plt.get_active_helper().remove_swarm()
                 self.plt.redraw()

         else:
         	if plot_type in ['hclust','corrmatrix'] and len(numericColumns) > 1\
         	and len(categoricalColumns) > 0:
         	## forces removable of categories upon selection
         		self.clean_up_dropped_buttons('cat',replot=False)
         		categoricalColumns = []
         	
         	if len(numericColumns) == 0 and len(categoricalColumns) != 0 \
         	and plot_type != 'countplot':
         		tk.messagebox.showinfo('Error ..',
         								'Need at least one numeric columns. Drag & Drop them from the source data '+
         								'treeview into the numerical receiver box.',
         								parent=self)
         		return	
         	
         	if len(numericColumns) % 2 != 0 and len(categoricalColumns) == 0\
         	and plot_type == 'scatter':
         		tk.messagebox.showinfo('Error ..',
         							'Need even number of numeric columns for a scatter plot.',
         							parent = self)
         		return

         	self.prepare_plot(colnames = numericColumns,
             				   catnames = categoricalColumns,
             				   plot_type = plot_type )


     def change_column_type(self, selectedColumns  = None, id = None, changeColumnTo = 'float64'):
     	'''
     	Changes the column type of the selected one.
     	'''
     	
     	if selectedColumns is None:
		
     		selectedColumns = self.selection_is_from_one_df()
     		
     	elif id is not None:
     		self.sourceData.set_current_data_by_id(id)
     	else:
     		return
			
     	if selectedColumns is not None:
     		oldDataTypes = self.sourceData.get_data_types_for_list_of_columns(selectedColumns)
     		status = self.sourceData.change_data_type_in_current_data(selectedColumns,changeColumnTo)
     		if status == 'worked':
     			if changeColumnTo == 'str':
     				changeColumnTo = 'object'
     			self.DataTreeview.change_data_type_by_iid(self.DataTreeview.columnsIidSelected,changeColumnTo)
     			tk.messagebox.showinfo('Done..','Column type changed.')
     		
     			
     			self.workflow.add('deleteRows',
     				self.sourceData.currentDataFile,
     				
     				{'funcDataR':'change_data_type_in_current_data',
     				'argsDataR':{'columnList':selectedColumns, 'newDataType': oldDataTypes},
     				'funcTreeR':'change_data_type_by_iid',
     				'argsTreeR': {'iidList':self.DataTreeview.columnsIidSelected, 'newDataType' : oldDataTypes},
     				'description':OrderedDict([('Activity:','Change column type to - ({})'.format(changeColumnTo)),
     				('Description:','Column type has been changed.'),
     				('Selected Columns',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     				('Data ID:',self.sourceData.currentDataFile)])})
     				     			
     		else:
     			if changeColumnTo == 'int64':
     				addToMsg = ' If the column contains NaN you cannot change it type to integer.Please remove NaN and try again.'
     			else:
     				addToMsg = ''
     			tk.messagebox.showinfo('Error..','An error occured trying to change the column type.' + addToMsg)


     def combine_selected_columns(self):
     	'''
     	Combines the content of selected columns.
     	'''
     	if len(self.DataTreeview.columnsSelected  ) < 2:
     		tk.messagebox.showinfo('Error..','Please select at least two columns')
     		return
     		
     	selectedColumns  = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     		combinedColumnName = self.sourceData.combine_columns_by_label(self.DataTreeview.columnsSelected  )
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													dataType = 'object',
     													columnList = [combinedColumnName])

     		tk.messagebox.showinfo('Done ..','Selected columns were combined in a newly added column.')
     		self.workflow.add('add_column',
     				self.sourceData.currentDataFile,
     				{'funcDataR':'delete_columns_by_label_list',
     				'argsDataR':{'columnLabelList':[combinedColumnName]},
     				'funcTreeR':'delete_entry_by_iid',
     				'argsTreeR':{'iid':'{}_{}'.format(self.sourceData.currentDataFile,combinedColumnName)}})
     	else:

      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
      		return


     def correct_baseline(self):
     	'''
     	Baseline Correction for time series data
     	'''
     	if self.plt.currentPlotType != 'time_series':
     		tk.messagebox.showinfo('Error..','Only useful for time series.')
     		return
     	if self.DataTreeview.onlyNumericColumnsSelected	== False:
     		tk.messagebox.showinfo('Error ..','Only numeric columns allowed.')
     		return
     	selectedColumns = self.selection_is_from_one_df()
     	selectionDataFrameId = self.sourceData.currentDataFile 
     	if selectedColumns is not None:
     	 	dataId = self.plt.get_dataID_used_for_last_chart()
     	 	if dataId != selectionDataFrameId:
     	 		tk.messagebox.showinfo('Error ..','Data frame of selected columns and the one used for plotting do not match!')
     	 		return     	 	
     	 	self.plt.nonCategoricalPlotter.timeSeriesHelper.activate_baselineCorr_or_aucCalc(columns = selectedColumns,
     	 																					DataTreeview = self.DataTreeview,
     	 																					workflow = self.workflow)
      		
     def custom_filter(self):
     	'''
     	Custom filter dialog.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:

     		customFilter = custom_filter.customFilterDialog(self.sourceData, self.DataTreeview.columnsSelected)

     		data, mode, match_annotation = customFilter.get_data()
     		del customFilter
     		currentFileName = self.sourceData.get_file_name_of_current_data()
     		if mode == 'remove':
     			#self.sourceData.update_data_frame(self.sourceData.currentDataFile, data)
     			dfIndex = self.sourceData.get_current_data().index.tolist()		
     			idxRemove = [i for i in  dfIndex if i not in data.index.tolist()]
     			self.sourceData.save_dropped_rows(self.sourceData.currentDataFile,idxRemove)
     			self.sourceData.update_data_frame(self.sourceData.currentDataFile, data)
     			self.workflow.add('deleteRows',     			
     				self.sourceData.currentDataFile,
     				{'funcDataR':'save_dropped_rows',
     				'argsDataR':{'id':self.sourceData.currentDataFile,
     				'reverse':True},
     				'description':OrderedDict([('Activity:','Drop Rows after Custom Filtering.'),
     				('Description:','Rows not matching the custom filtering are removed.'),
     				('Removed Rows:','{}'.format(len(data.index))),
     				('Selected Columns',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     				('Data ID:',self.sourceData.currentDataFile)])})
  	
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
     				
     				
        			matchedData = pd.Series(self.sourceData.get_current_data().index, index = self.sourceData.get_current_data().index)
        			
        			idx_ = matchedData.isin(data.index)
        			replace_dict = {True : "+",
                         	   False: self.sourceData.replaceObjectNan
                         	   }
                         	   
        			outputColumn = pd.Series(idx_.map(replace_dict), index = self.sourceData.get_current_data().index)
        			
        			self.sourceData.add_column_to_current_data(columnName,outputColumn)

     			else:

        			joinHelper = pd.DataFrame(data.values, columns = [columnName], index= data.index)
        			self.sourceData.join_df_to_currently_selected_df(joinHelper)
        			del joinHelper
     			self.sourceData.change_data_type_in_current_data(columnName,'object')
     			self.DataTreeview.add_list_of_columns_to_treeview(id = self.sourceData.currentDataFile,
        													dataType = 'object',
        													columnList = [columnName])
      			#self.DataTreeview.add_list_of_columns_to_treeview(id = self.sourceData.currentDataFile,
     			#self.sourceData.change_data_type_in_current_data(columnName,'object')
     			infoDict = {'funcDataR':'delete_columns_by_label_list',
       			
        				'argsDataR':{'columnLabelList':[columnName]},
     					
     					'funcTreeR':'delete_entry_by_iid',
     					'argsTreeR':{'iid':'{}_{}'.format(self.sourceData.currentDataFile,columnName)},
     					'description':OrderedDict([('Activity:','Custom Categorical Filter.'),
     					('Description:','Column has been added indicating by a "+" if the entered search string is present.'+
     					' Depending on the settings, matches might also be indicated by the search string itself.'),
     					('Column name:',columnName),
     					('Match Annotations',match_annotation),
     					('Selected Columns:',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     					('Data ID:',self.sourceData.currentDataFile)])}   	
        			
     			self.workflow.add('filter',        			
     						self.sourceData.currentDataFile,
     						infoDict)


     	if mode is not None:
        	tk.messagebox.showinfo('Done..',
        		'Custom filter was applied successfully. Well done.',
        		parent=self)

    
     def custom_sort_values(self):
     	'''
     	Opens a dialog window that enables the user to sort values in a specific
     	column by custom order.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:

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
     			tk.messagebox.showinfo('Done ..','Custom sorting done.',parent=self)

     		else:
     			pass

     		del dialog

     def custom_column_order(self):
     	'''
     	Custom sorting of columns.
     	'''
     	
     	selectedDataTypes  = self.selection_is_from_one_df('allItemsSelected')
     	selectionDataFrameId = self.sourceData.currentDataFile
     	if selectedDataTypes  is not None:

     		inputValues = OrderedDict()
     		dataTypeRels = self.sourceData.get_columns_data_type_relationship()
     		for item in selectedDataTypes:
     			dataType = item.split('{}_'.format(selectionDataFrameId))[-1]
     			if dataType not in dataTypeRels:
     				inputValues = dataTypeRels
     				break
     			else:
     				inputValues[dataType] = dataTypeRels[dataType]     		

     		dialog = custom_sort.customSortDialog(inputValues)

     		if dialog.resortedValues is not None:
     			columnOrder = []
     			for key,values in dialog.resortedValues.items():
     				columnOrder.extend(values)
     			df1 = self.sourceData.df[columnOrder]
     			dfOut = self.sourceData.join_missing_columns_to_other_df(df1,id=selectionDataFrameId)
     			self.sourceData.update_data_frame(selectionDataFrameId,dfOut)
     			self.update_all_dfs_in_treeview()
     			tk.messagebox.showinfo('Done ..','Custom sorting done.',parent=self)
     		else:
     			pass		
     
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

		
     def clean_up_dropped_buttons(self, mode = 'all', replot = True, clearFigure = True):
         
         if mode == 'all':
             for button in self.selectedNumericalColumns.values():
                      button.destroy()
             for button in self.selectedCategories.values():
                      button.destroy()

             
             self.interactiveWidgetHelper.clean_frame_up()
             self.selectedNumericalColumns.clear()
             self.selectedCategories.clear()
             self.update_tooltip_in_receiverBox('reset')
             if clearFigure:
             	self.plt.clean_up_figure()

         elif mode == 'num':
             for button in self.selectedNumericalColumns.values():
                      button.destroy()

             self.selectedNumericalColumns.clear()
             self.but_stored[10].configure(image= self.add_swarm_icon)
             self.plt.addSwarm =  False
             self.update_tooltip_in_receiverBox('numeric')

         elif mode == 'cat':
                for button in self.selectedCategories.values():
                      button.destroy()
                self.selectedCategories.clear()
                self.update_tooltip_in_receiverBox('categories')
         
         if replot:
            	plot_type = self.estimate_plot_type_for_default()
            	self.interactiveWidgetHelper.clean_frame_up()
            	self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),                
                                  catnames = list(self.selectedCategories.keys() ),
                                  plot_type = plot_type)
 
 

     def calculated_droped_stats_for_all_combs(self):
     	'''
     	Calculates the droped statitic.
     	'''
     	if hasattr(self,'groupedStatsData') == False:
     		tk.messagebox.showinfo('Error ..',
     								'You have to reopen the window by a drag & drop event onto the figure to apply another test. Aborting..',
     								parent=self)
     		return
     		
     	dataDict = OrderedDict([('id',[]),('Group 1',[]), ('Group 2',[])])
     	numericColumns = list(self.selectedNumericalColumns.keys())

     	if len(numericColumns) != 0:

     		columnsToAdd = []
     		columnsForTest = numericColumns + columnsToAdd

     		for column in columnsForTest:
     			dataDict['{} p-value'.format(column)] = []
     			dataDict['{} test statistic'.format(column)] = []

     		testSettings = {'paired':self.paired,
     						'test':self.test,
     						'mode':self.mode}     		
     		iterationObject = itertools.combinations(self.groupedStatsKeys,2)	     			
     		
     		for n,combination in enumerate(iterationObject):
     			if self.split_on_cats_for_plot.get():
     				valuesGroup1 = self.groupedStatsData.get_group(combination[0])
     				valuesGroup2 = self.groupedStatsData.get_group(combination[1])
     			## we do this again, because user could change the df and resort
     			## to redo seems easier than matching and the df is usually not that long
     			dataDict['id'].append(str(n+1))
     			dataDict['Group 1'].append(get_elements_from_list_as_string(combination[0]))
     			dataDict['Group 2'].append(get_elements_from_list_as_string(combination[1]))

     			for column in columnsForTest:
     				
     				
     				if self.split_on_cats_for_plot.get():
     					data1 = valuesGroup1[column].dropna().values
     					data2 = valuesGroup2[column].dropna().values
     				else:
     					data1 = self.groupedStatsData[column][combination[0]].dropna().values
     					data2 = self.groupedStatsData[column][combination[1]].dropna().values
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

     def check_button_handling(self, colormap):
         '''
         Change color map.
         '''
         self.cmap_in_use.set(colormap)

         if self.plt.plotCount != 0:
         	color_changer.colorChanger(self.plt,self.sourceData,
         		colormap, self.interactiveWidgetHelper)
     
     def categorical_column_handler(self,mode):
     	'''
     	Open categorical filter dialog. Please note that this is also used
     	to annotate scatter plot points which looks strange. But since the annotation
     	can also only be considered as a categorical value, the dialog window is used as well.

     	Parameter
     	==========
     	mode - Can only be one of :

     			- Find category & annotate
				- Search string & annotate
				- Subset data on unique category
				- Annotate scatter points
				- Find entries in hierarch clustering
				- Find entry in line plot
		Output
		==========
		None - But new data frames are entered automatically from within the dialog
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	
     	if selectedColumns is None:
     		return
     	
     	self.annot_label_scatter = False
     	if mode == 'Annotate scatter points' and len(selectedColumns) == 0:
     		filterColumn  = None
     		dataSubset = None
     	elif mode == 'Find entry in hierarch. cluster':
     		if self.plt.currentPlotType != 'hclust':
     			tk.messagebox.showinfo('Error ..','Please plot a hierarchical clustermap.')
     			return
     		else:
     			filterColumn  = None
     			dataSubset = self.plt.nonCategoricalPlotter._hclustPlotter.df
     	elif mode == 'Find entry in line plot':
      		if self.plt.currentPlotType != 'line_plot':
      			tk.messagebox.showinfo('Error ..','Please plot a profile plot first.')
      			return
      		else:
      			dataSubset = self.plt.nonCategoricalPlotter.linePlotHelper.get_data()
      			filterColumn = None

     	elif mode == 'Search string & annotate':

     		filterColumn = selectedColumns
     		dataSubset = None

     	else:

     		filterColumn = selectedColumns[0]
     		dataSubset = None

     	categorical_filter.categoricalFilter(self.sourceData,self.DataTreeview,
             										self.plt,operationType = mode,
             										columnForFilter = filterColumn,
             										dataSubset = dataSubset,
             										workflow = self.workflow)
     def clip_data(self):
      	'''
      	Let user define specific clipping mask to pot specific subsets.
      	'''
      	dialog = mask_filtering.clippingMaskFilter(self.plt,self.sourceData,self)
    	
   
     def get_data_from_scatter_selection(self):
     	'''
     	'''
     	idx = self.data_selection.index
     	dataID = self.plt.get_dataID_used_for_last_chart()
     	df = self.sourceData.get_data_by_id(dataID)
     	boolIdx = df.index.isin(idx)
     	return df.loc[boolIdx,:]
     	
   
	  	
     
     def configure_chart(self):
        '''
        Helper function to open Chart Configuration Dialog that allows
        easy adjustment of several chart properties. It also then
        upgrades import things in the plotter class to maintain changes like
        box around subplots or grid lines
        Output 
        ========
        None
        '''
        plot_type = self.plt.currentPlotType
        if plot_type in ['PCA','corrmatrix','hclust','cluster_analysis']:
            tk.messagebox.showinfo('Not supported..','Configuration of this plot type is currently not supported.')
            return
        chart_configurator = chart_configuration.ChartConfigurationPopup(self.plt,
        													self.global_chart_parameter)

        self.global_chart_parameter = chart_configurator.global_chart_parameter
        self.plt.showSubplotBox = chart_configurator.show_box
        self.plt.showGrid = chart_configurator.show_grid
    
     def change_default_color(self, button, event = None):
         '''
         Changing the default color means that this color is used if the
         hue is not reserved by a categorical data separation/grouping   
               
         Output 
         ========
         None
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
         self.interactiveWidgetHelper.clean_frame_up()
         #self.plt.set_scatter_point_properties(color=col_get)
         self.plt.redraw()

 
     def define_groups_in_dimRed(self):
     	'''
     	Dialog to define grouping in dimensional reduction procedure
     	'''
     	define_groups_dim_reduction.defineGroupsDialog(self.dimensionReductionCollection,
     												   self.plt,
     												   self.colorHelper)
     def display_corrmatrix_results(self):
     	'''
     	Show the correlation matrix results to user.
     	'''
     	if len(self.plt.plotProperties) == 0:
     		tk.messagebox.showinfo('Error ..',
     							   'Please plot a correlation matrix first.',
     							   parent=self)
     		return
     	
     	numColumns,_,plot_type,_ = self.plt.current_plot_settings
     	if self.plt.nonCategoricalPlotter._hclustPlotter is not None or \
     	plot_type != 'corrmatrix':

     		data = self.plt.nonCategoricalPlotter._hclustPlotter.export_data_of_corrmatrix()
     		dataDialog = display_data.dataDisplayDialog(data,showOptionsToAddDf=True)

     		if dataDialog.addDf:
     			nameOfDf = 'Corrmatrix Results {}'.format(get_elements_from_list_as_string(numColumns))
     			self.add_new_dataframe(dataDialog.data,nameOfDf)

     	else:
     		tk.messagebox.showinfo('Error ..',
     			'No data found. Perform clustering.',
     			parent=self)

     def display_curve_fits(self):
      	'''
      	Display curve fits that were made. The user can define a name for each curve
      	fit.
      	'''
      	selectFitAndGrid = curve_fitting.displayCurveFitting(self.sourceData,self.plt,self.curveFitCollection)
      	fitsToPlot = selectFitAndGrid.curve_fits_to_plot
      	categoricalColumns = self.curveFitCollection.get_columns_of_fitIds(fitsToPlot)
      	if len(categoricalColumns) > 0:
      		self.plt.set_selectedCurveFits(selectFitAndGrid.curve_fits_to_plot)
      		self.plt.initiate_chart(numericColumns = [], categoricalColumns = categoricalColumns ,
      								 selectedPlotType = 'curve_fit', colorMap = self.cmap_in_use.get())
      	else:
      		pass


     def drop_cols_with_low_variance(self):
     	'''
     	Drops rows with a low variance threshold. 
     	'''
     	selectedColumns = self.selection_is_from_one_df(onlyNumeric = True)
     	if selectedColumns is not None:
    
     	#	value = ts.askfloat('Variance threshold','Provide variance cutoff.\nColumns that show less variance are going to be removed.')
     		dialog = simpleUserInputDialog(['Threshold','Copy'],
     							   ['0','True'],
     							   [np.linspace(0,4,20).tolist(),['True','False']],
     							   title = 'Remove columns with low variance - Settings',
     							   infoText='If copy == True. Columns with higher variance will be copied'+
     							   '.\nIf False columns that have lower variance will be removed from the data.')
     		if dialog.selectionOutput is not None:
     			try:
     				thres = float(dialog.selectionOutput['Threshold'])
     				copy = dialog.selectionOutput['Copy'] == 'True'
     			except:
     				tk.messagebox.showinfo('Error ..',
     					'Converting your input raised an error. (Threshold must be a '+
     					'float, Copy must be True or False')
     					
     			newFeatureNames = self.sourceData.remove_columns_with_low_variance(selectedColumns,
     																			thres,copy) 
     			if newFeatureNames is None:
     				tk.messagebox.showinfo('Error..','No feature meets the given threshold.')
     				return	
     			elif newFeatureNames == 'Same':
     				tk.messagebox.showinfo('Error ..','Variance is equal over all columns.')
     				return													
     			
     			if copy:
     			
     				self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     																	'float64',newFeatureNames)		
     			else:
     				self.update_all_dfs_in_treeview()
					
     	
     def divide_or_subtract_columns(self, byValue = False, byMedian = False, operation = 'divide'):
     	'''
     	Divide / Substract row wise data.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     		if byValue or byMedian:
     			if byValue:
     				value = ts.askfloat('Provide Value','Enter value for division/subtraction:')
     				if value is not None and value != 0:
     					calcDict = OrderedDict([(column,value) for column in selectedColumns])
     					baseString = 'by_{}:'.format(value)
     				else:
     					if int(value) == 0:
     						tk.messagebox.showinfo('Error..','Division/Subtraction by zero.')
     					return
     						
     			elif byMedian:
     				values = self.sourceData.get_current_data()[selectedColumns].median()
     				calcDict = OrderedDict([(column,value) for column,value in zip(selectedColumns,values.tolist())])
     				baseString = 'by_median:' 
     			
     			if operation == 'divide':
     				baseString = 'div_' + baseString
     				newColumns = self.sourceData.divide_columns_by_column(calcDict,baseString)
     			elif operation == 'subtract':
     				baseString = 'sub_' + baseString
     				newColumns = self.sourceData.substract_columns_by_value(calcDict,baseString)
     		
     		else:
     			columns = self.sourceData.get_numeric_columns()
     			
     			dialog = simpleListboxSelection('Select columns to perform calculation',
     								   self.sourceData.get_columns_of_current_data())				
     			selection = dialog.selection
     			if len(selection) != 0:
     				if len(selection) != 1 and len(selection) != len(selectedColumns):
     					tk.messagebox.showinfo('Error..',
     						'Please select either one column or the exact same number of columns that were selected in the treeview.', 
     						parent = self)
     					return
     				elif len(selection) == 1 and len(selectedColumns) != 1:
     					
     					selection = selection * len(selectedColumns)
     				
     				if operation == 'divide':
     					newColumns = self.sourceData.divide_columns_by_column(selectedColumns,selection)
     				elif operation == 'subtract':
     					newColumns = self.sourceData.substract_columns_by_column(selectedColumns,selection)
     			else:
     				return
     		
     		self.DataTreeview.add_list_of_columns_to_treeview(id = self.sourceData.currentDataFile,
        													dataType = 'float64',
        													columnList = newColumns)
     		tk.messagebox.showinfo('Done..','Calculations performed. Columns added.')
     
     def drop_selection_from_df(self):
         '''
         Drops rows from data frame and reinitiates chart.
         '''
         self.sourceData.delete_rows_by_index(self.data_selection.index)
         self.plt.figure.canvas.mpl_disconnect(self.selection_press_event)
         self.plt.initiate_chart(*self.plt.current_plot_settings)
         self.select_data()
     
     def define_size_range(self):
     	'''
     	Defining range of size interval to change to edges of a scatter plot
     	where size is encoded. 
     	'''
     	if self.plt.plotCount == 0:
     		return
     	sizeDialog = size_configuration.sizeIntervalDialog(self.plt)

     def duplicate_column(self):
     	'''
     	Duplicates selected columns. Changes dataframe selection if needed. Eventually
     	it will change back to the previous selected dataframe.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:  
     		columnLabelListDuplicate  = self.sourceData.duplicate_columns(self.DataTreeview.columnsSelected  )
     		dataTypes = self.sourceData.get_data_types_for_list_of_columns(columnLabelListDuplicate)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													dataTypes,
     													columnLabelListDuplicate)
     		tk.messagebox.showinfo('Done ..','Selected column(s) were duplicated and added to the source data treeview.')
     		self.workflow.add('addColumn',
     				self.sourceData.currentDataFile,
     				{'funcDataR':'delete_columns_by_label_list',
     				'argsDataR':{'columnLabelList':columnLabelListDuplicate},
     				'funcTreeR':'delete_entry_by_iid',
     				'argsTreeR':{'iid':['{}_{}'.format(self.sourceData.currentDataFile,col) for col in columnLabelListDuplicate]},
     				'description':OrderedDict([('Activity:','Duplicate column.'),
     				('Description:','One or multiple columns were duplicated.'),
     				('Selected Columns:',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     				('Data ID:',self.sourceData.currentDataFile)])})

	
     def delete_column(self,event=None):
     	'''
     	Removes selected columns. Changes dataframe selection if needed. Eventually
     	it will change back to the previous selected dataframe.
     	'''
     	if event is not None:
     		if len(self.DataTreeview.columnsSelected) == 0:
     			return
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:    
     		dfForUndo =  self.sourceData.get_current_data_by_column_list(selectedColumns, ignore_clipping = True)
     		self.sourceData.delete_columns_by_label_list(self.DataTreeview.columnsSelected)
     		self.DataTreeview.delete_selected_entries()
     		tk.messagebox.showinfo('Done ..','Selected columns were removed.')
     		self.workflow.add('deleteColumn', 
     					self.sourceData.currentDataFile,
     					{'funcDataR':'join_df_to_df_by_id',
     					'argsDataR':{'dfToAdd':dfForUndo,'id':self.sourceData.currentDataFile},
     					'funcAnalyzeR':'update_all_dfs_in_treeview',
     					'argsAnalyzeR':{},
     					'description':OrderedDict([('Activity:','Delete column.'),
     					('Description:','One or multiple columns were deleted.'),
     					('Selected Columns:',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     					('Data ID:',self.sourceData.currentDataFile)])})
     
     def delete_data_frame_from_source(self, fileIid = None):
     	'''
     	Removes data frames from sourceDataCollection and from DataTreeview.
     	The data cannot be restored.
     	'''
     	if fileIid is None:
		
     		for fileIid in self.DataTreeview.dataFramesSelected:

     			fileName = self.sourceData.fileNameByID[fileIid]
     			dataFrameIid = '{}_{}'.format(fileIid,fileName)
     			self.DataTreeview.delete_entry_by_iid(dataFrameIid)
     			self.sourceData.delete_data_file_by_id(fileIid)
     			self.remove_savedCalculations(fileIid)
     			self.workflow.delete_branch(fileIid)

     			if len(self.sourceData.fileNameByID) == 0:
     				self.clean_up_dropped_buttons(mode = 'num')
     				self.clean_up_dropped_buttons(mode = 'cat')
     			if len(self.sourceData.dfs) == 0 or \
     				fileIid == self.plt.get_dataID_used_for_last_chart():
     			
     				self.clean_up_dropped_buttons(mode = 'num')
     				self.clean_up_dropped_buttons(mode = 'cat')
					
     		tk.messagebox.showinfo('Done..','Selected data frame(s) deleted.')
		
     	else:
		
     			fileName = self.sourceData.fileNameByID[fileIid]
     			dataFrameIid = '{}_{}'.format(fileIid,fileName)
     			self.DataTreeview.delete_entry_by_iid(dataFrameIid)
     			self.sourceData.delete_data_file_by_id(fileIid)
     			self.remove_savedCalculations(fileIid)
     			self.workflow.delete_branch(fileIid)
     			
     			if len(self.sourceData.dfs) == 0  or fileIid == self.plt.get_dataID_used_for_last_chart():
					
     				self.clean_up_dropped_buttons(mode = 'num')
     				self.clean_up_dropped_buttons(mode = 'cat')
    
     def delete_dragged_buttons(self, event, but_name, columns=False):
         '''
         Remove dragged buttons from receiver boxes.
         '''
         # set correct df
         dataID = self.plt.get_dataID_used_for_last_chart()
         self.sourceData.set_current_data_by_id(dataID)
		
         if columns:
             self.selectedNumericalColumns[but_name].destroy()
             del self.selectedNumericalColumns[but_name]
             self.update_tooltip_in_receiverBox('numeric')
             self.plt.addSwarm = False
             self.but_stored[10].configure(image = self.add_swarm_icon)
             self.swarm_but = 0
             	             	              	
         else:
             self.selectedCategories[but_name].destroy()
             del self.selectedCategories[but_name]
             self.update_tooltip_in_receiverBox('categories')		 

         if len(self.selectedCategories) == 0 and len(self.selectedNumericalColumns) == 0:
             self.plt.clean_up_figure()
             self.interactiveWidgetHelper.clean_frame_up()
             self.plt.redraw()
             return
         numericColumns = list(self.selectedNumericalColumns.keys())
         categoricalColumns = list(self.selectedCategories.keys())
                                 
         _,_, plot_type, cmap = self.plt.current_plot_settings
         if plot_type in ['hclust','corrmatrix'] and len(numericColumns) == 1:
             plot_type = 'boxplot'
         if columns == False and plot_type in ['scatter_matrix','hclust','corrmatrix']:
             return
         if plot_type == 'PCA':
             plot_type = 'boxplot'
		
         self.interactiveWidgetHelper.clean_frame_up()
         
         if len(numericColumns) == 0:
         	plot_type = 'countplot'
         elif len(numericColumns) == 1:
         	plot_type = 'boxplot' 
         	
         self.plt.initiate_chart(numericColumns,categoricalColumns,
         							plot_type, cmap)
     	     
     def export_selected_figure(self,event):
         '''
         Cast a menu to export subplots from the main window into a
         main figure template.
         '''
         if (event.dblclick or event.button > 1) and event.inaxes is None:
         	self.is_just_outside(event)
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
             self.post_menu(menu = self.menuCollection['main_figure_menu'] )
     		   

     def export_dimRed_results(self, which):
     	'''
     	Can be used to export results of a dimensional reduction either to file
     	or to be added to source treeview and data collection
     	'''
     	if 'Export PCA Scores' in which:

     		_,components,columns, dataID = \
     		self.dimensionReductionCollection.get_drivers_and_components(which='Components')
     		mainString = 'Scores'
     		data = components.T
     		data['Feature'] = data.index
     		if hasattr(self.plt.nonCategoricalPlotter,'dimRedGroups'):
     			data['Groups'] = self.plt.nonCategoricalPlotter.dimRedGroups

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

     def estimate_plot_type_for_default(self):
         '''
         Estimate default plot type.
         '''
         colnames = list(self.selectedNumericalColumns.keys())
         catnames = list(self.selectedCategories.keys())
         used_plot_style = self.plt.currentPlotType
         n_col = len(colnames)
         n_categories = len(catnames)
         if used_plot_style in ['hclust','corrmatrix'] and n_categories > 0:
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
     
     def export_data_to_file(self, data = None, format_type = 'Excel',
     			sheet_name = 'ExportInstantClue', initial_file = 'Untitled',
     			checkSelection = False):
         '''
         Export data frame to txt or excel.
         Parameters
         ==========
         
         data			- pandas data frame. Data to be saved
         format_type	- string. Can be 'Excel' or 'txt'. 
         sheet_name		- string. For Excel export.
         initial_file	- string. File name.	
         
         Output 
         ==========
         None
         
         '''
         if data is None:
         	if checkSelection:
         		data = self.get_selected_data()
         		if data is None:
         			return			
         	else:
         		data = self.sourceData.get_current_data()
         if isinstance(data, pd.DataFrame) == False:
         	#print('Data must be a pandas data frame.')
         	return
         if format_type not in ['Excel','txt','csv']:
         	#print('format_type must be "Excel" or "txt".')
         	return
         progressBar  = Progressbar(title = 'Saving')
         if format_type in ['txt','csv']:

             file_name_saving = tf.asksaveasfilename(title='Select name for saving file',
             		defaultextension = '.{}'.format(format_type) ,
             		initialfile=initial_file,
             		)
             if file_name_saving == '' or file_name_saving is None:
             	progressBar.close()
             	return
             try:
             	if format_type == 'txt':
             		data.to_csv(file_name_saving, index=None, na_rep ='NaN', sep='\t')
             	else:
             		data.to_csv(file_name_saving, index=None, na_rep ='NaN')
            		
             except PermissionError:
             	tk.messagebox.showinfo('Error ..','Permission denied.')
             	progressBar.close()
             	return
             except:
             	tk.messagebox.showinfo('Error ..','Unknown Error. Could not save file.')
             	return
        
         else:
              file_name_saving = tf.asksaveasfilename(title='Select name for saving file',
              		defaultextension='.xlsx',
              		initialfile=initial_file,
              		filetypes = [('Excel files', '.xlsx')])
              if file_name_saving == '' or file_name_saving is None:
              	progressBar.close()
              	return
             	
              try:
              	data.to_excel(file_name_saving, index=None, sheet_name = sheet_name, na_rep = 'NaN')
              except:
              	tk.messagebox.showinfo('Error ..',
              						   'File could not be saved. Might be due to denied permission.')
              	progressBar.close()
              	return
         progressBar.update_progressbar_and_label(100,'Done..')
         progressBar.close()
         tk.messagebox.showinfo('Done..',
         	'File has been saved!\nLocation - {}'.format(file_name_saving))
    
     def load_and_append_files(self):
     	'''
     	Load multiple files and append them to each other. 
     	'''
     	txt_file_importer.multipleTxtFileLoader()
     	return	
     		
     		
     def shift_data (self):
     	'''
     	Shifts data by given time points. 
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     	
     		shift_data.shiftTimeData(selectedColumns,self.sourceData,self.DataTreeview,self)

 
     def save_hclust_to_excel(self):
     	'''
     	'''
     	if hasattr(self.plt.nonCategoricalPlotter,'_hclustPlotter'):
     		if self.plt.nonCategoricalPlotter._hclustPlotter is not None:
     			self.plt.nonCategoricalPlotter._hclustPlotter.save_data_to_excel()
     			return
     	tk.messagebox.showinfo('No Cluster ..','Plot a hierarchical cluster first.')
     
     def plot_circulized_dendrogram(self, plot_type = 'hclust'):
     	'''
     	Plot a circulized dendrogram
     	'''
     	if self.circulizeDendrogram.get():
     	
     		self.plt.circulizeDendrogram = True 
     	
     	else:
     		self.plt.circulizeDendrogram = False
     		
     	self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),
                          catnames = list(self.selectedCategories.keys()),
                          plot_type = plot_type)

     def norm_quant_data(self):
     	'''
     	
     	'''
     	selectedColumns = self.selection_is_from_one_df(onlyNumeric = True)
     	if selectedColumns is not None:   
     		
     		df = self.sourceData.df[selectedColumns]
     		if len(df.dropna().index) != len(df.index):
     			tk.messagebox.showinfo('Careful..','Quantile normalization works only on complete data at the moment. Please remove NaN first.',parent=self)
     			return
     		data = quantileNormalize(df).values
     		newColumnNames = ['quantN_{}'.format(col) for col in selectedColumns]
     		dfToAdd = pd.DataFrame(data, columns = newColumnNames)
     		evalColumnNames = self.sourceData.join_df_to_currently_selected_df(dfToAdd, exportColumns = True)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													'float64',evalColumnNames)
     		tk.messagebox.showinfo('Done..','Calculations done. New columns added.')
     			
     def scale_quant_data(self, withMean = True, withStd = True):    
     	'''
     	
     	'''
     	from sklearn.preprocessing import scale
    		
     	selectedColumns = self.selection_is_from_one_df(onlyNumeric = True)
     	if selectedColumns is not None:   
     		
     		data = scale(self.sourceData.df[selectedColumns].values, 
     				with_mean = withMean,
     				with_std = withStd, axis = 0)
     		
     		newColumnNames = ['scaled_{}'.format(col) for col in selectedColumns]   
     		
     		dfToAdd = pd.DataFrame(data, columns = newColumnNames)
     		evalColumnNames = self.sourceData.join_df_to_currently_selected_df(dfToAdd, exportColumns = True)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													'float64',evalColumnNames)
     		tk.messagebox.showinfo('Done..','Calculations done. New columns added.')
     		
     
     
     def normalize_data(self,metric):
     	'''
     	
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     	
     		columnsSelected = self.DataTreeview.columnsSelected
     		if metric == '0->1':
     			scaler = dataNormalizer(metric,**{'feature_range':(0.01, 1)})
     		else:
     			scaler = dataNormalizer(metric)
     		
     		columnNames = self.sourceData.fit_transform(scaler,columnsSelected)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													'float64',columnNames)
     		tk.messagebox.showinfo('Done..','Calculations done. New columns added.')
    
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
     	
     def rename_data_frame(self):
     	'''
     	Reanames data frames
     	'''
     	dataFrames = self.DataTreeview.dataFramesSelected
     	fileNames = [self.sourceData.fileNameByID[fileIid] for fileIid in dataFrames]
     	optionList = [[fileNames[n]] for n in range(len(dataFrames))]
     	renameDialog = simpleUserInputDialog(fileNames,fileNames,
     										optionList,'Rename Data Frame',
     										'Enter a new name for selected data frames')
     	
     	newDataFrameNames = list(renameDialog.selectionOutput.values())
     	if len(newDataFrameNames) != len(dataFrames):
     		return
     	for n,dfIID in enumerate(dataFrames):
     	
     		dfName = newDataFrameNames[n]
     		self.sourceData.rename_data_frame(dfIID,dfName)
     	
     	self.update_all_dfs_in_treeview()
	

     def rename_columns(self, selectedColumns = None, event = None):
     	'''
     	Opens a dialog window that allows the user to change the column names.
     	Can also be triggered by double click.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is None or len(selectedColumns) == 0:
     		return
     	
     	if event is not None:
     		itemClicked = [self.DataTreeview.clicked_item(event)]
     		if itemClicked[0] == '': 
     			return	
     		     		
     	if selectedColumns is not None:
#<<<<<<< HEAD
     		renameDialog = change_columnName.ColumnNameConfigurationPopup(self.DataTreeview.columnsSelected,
     														self.sourceData, self.DataTreeview, self)
#=======
 #    		if byValue or byMedian:
  #   			if byValue:
   ##  				value = ts.askfloat('Provide Value','Enter value for division:')
     #				if value is not None:
     #					calcDict = OrderedDict([(column,value) for column in selectedColumns])
     ##					baseString = 'by_{}:'.format(value)
     	#		elif byMedian:
     	#			values = self.sourceData.df[selectedColumns].median()
     	#			calcDict = OrderedDict([(column,value) for column,value in zip(selectedColumns,values.tolist())])
     	#			baseString = 'by_median:'  
     	##		
     	#		if operation == 'divide':
     	#			baseString = 'div_' + baseString
     	#			newColumns = self.sourceData.divide_columns_by_value(calcDict,baseString)
     	#			
     	##		elif operation == 'substract':
     	#			baseString = 'sub_' + baseString
     	#			newColumns = self.sourceData.substract_columns_by_value(calcDict,baseString)
#>>>>>>> 4839eb2fc5d58e35d92c34afdab504beafb469d2
     		
     		if renameDialog.renamed: #indicates if any renaming was done (or closed)
     			
     			tk.messagebox.showinfo('Done..','Column names replaced.',parent=self)
     			
     			self.workflow.add('renameColumn', 
     							  self.sourceData.currentDataFile,
     							  renameDialog.reverseFuncs)
     	
     	
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

     def selection_is_from_one_df(self,itemSelection = 'columnsSelected', onlyNumeric = False):
     	'''
     	Check if user's selection is from one data set
     	'''
     	selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     	if selectionIsFromSameData is None:
     		return
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		if onlyNumeric and self.DataTreeview.onlyNumericColumnsSelected == False:
     			tk.messagebox.showinfo('Error ..','Please select only numeric data.',parent=self)
     			return
     		else:
     			return getattr(self.DataTreeview,itemSelection)
     		
     	else:
     		tk.messagebox.showinfo('Error ..','Please select only columns from one file.',parent=self)     		
     	

     def transpose_data(self):
     	'''
     	Transpose Data
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:

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
     									'by adding the value index?', parent = self)
     				if quest == 'yes':
     					columnValues = self.sourceData.df[columnForColumns].values
     					newColumns = ['{}_{}'.format(column,n) for n,column in enumerate(columnValues)]
     				else:
     					return
     			else:
     				newColumns = self.sourceData.df[columnForColumns].astype(str)
     		if len(selectedColumns) != len(self.sourceData.df.columns):
     			quest = tk.messagebox.askquestion('Column selection ..',
     						'You have selected a subset of column. Do you only want to use the selected ones (yes) or all columns (no)',parent=self)
     			if quest == 'yes':
     				useColumns = selectedColumns
     			else:
     				useColumns = self.sourceData.get_columns_of_current_data()
     		# transpose data
     		data = self.sourceData.get_current_data()[useColumns].transpose()
     		# add new column names (selected by user)
     		data.columns = newColumns
     		# add index as pure numbers
     		data.index = np.arange(0,len(data.index))
     		# inser a column with index holding old columns
     		indexName = self.sourceData.evaluate_column_name('Index',newColumns)
     		data.insert(0,indexName,useColumns)
			#show transposed data.
     		dataDialog = display_data.dataDisplayDialog(data,showOptionsToAddDf=True)

     		if dataDialog.addDf:
     			nameOfDf = 'Transpose - {}'.format(self.sourceData.get_file_name_of_current_data())
     			self.add_new_dataframe(data,nameOfDf)

     def pivot_data(self):
     	'''
     	Perform pivot Table
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     		pivotDialog = pivot_table.pivotDialog(self.sourceData,self.sourceData.currentDataFile)
     		data = pivotDialog.pivotedDf
     		if data.empty:
     			return
     		dataDialog = display_data.dataDisplayDialog(data,showOptionsToAddDf=True)

     		if dataDialog.addDf:
     			nameOfDf = 'Pivot - {}'.format(self.sourceData.get_file_name_of_current_data())
     			self.add_new_dataframe(data,nameOfDf)


     def perform_export(self, axExport,axisId,figureTemplate, exportId = None):
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
         plotExporter = self.plt.get_active_helper()
         
         if hasattr(plotExporter,'_hclustPlotter') and plotExporter._hclustPlotter is not None \
         and self.circulizeDendrogram.get():#plot and 
         
         	tk.messagebox.showinfo('Error ..','Sorry, this plot type cannot be exported to main figures yet.')
         	return
         if self.plt.currentPlotType == 'scatter' and self.plt.categoricalPlotter is not None:
         	tk.messagebox.showinfo('Error ..',
         		'This chart can currently not be expoted to main figure templates.')
         	return

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
     
     
     def stop_selection(self, replot = True, event = None):
     	'''
     	Stop selection of data.
     	'''
     	try:
             self.canvas.mpl_disconnect(self.selection_press_event)
             if replot:
             	self.plt.save_axis_limits()
             	self.plt.clean_up_figure()
             	self.plt.reinitiate_chart(updateData=True)
             self.selection_press_event = None
     	except:
             pass


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
                               lambda verts, ax = event.inaxes:
                               self.selection_callback(verts,ax))

                 self.lasso.line.set_linestyle('--')
                 self.lasso.line.set_linewidth(0.3)

             elif event.button in [2,3]:
                 self.post_menu(menu=self.menuCollection['selection_menu'] )


     def selection_callback(self, verts,ax):
             '''
             Get indices of selected data and marks them in current scatter
             plot.
             '''
             p = path.Path(verts)
             if p is None:
                 return
             if hasattr(self,'selectionColl'):
             	self.selectionColl.set_visible(False)
             
             scatterPlots = self.plt.get_scatter_plots()
             idxAx = self.plt.get_axes_of_figure().index(ax)
             for n,scatterPlot in enumerate(scatterPlots.values()):
             	if idxAx == n:
             		column1, column2 = scatterPlot.numericColumns   
             		self.slectionDataAsTuple = self.sourceData.get_data_as_list_of_tuples(scatterPlot.numericColumns,
     											data = scatterPlot.data)
             		indexList = p.contains_points(self.slectionDataAsTuple)
             		self.data_selection = scatterPlot.data.iloc[indexList]             		
             		self.selectionColl = self.plt.add_scatter_collection(scatterPlot.ax,self.data_selection[column1],
             								self.data_selection[column2],
             								color = 'red',
             								returnColl = True)
             		continue
             		
             del self.lasso
             self.plt.redraw()


     def factorize_column(self):
     	'''
     	Counts through the data in current order.
     	'''
     	selectedColumns  = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     		columnNames, dfLabels = self.sourceData.factorize_column(selectedColumns)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													dataType = 'int64',
     													columnList = columnNames,
     													)
     		tk.messagebox.showinfo('Done ..','Index column was added to the treeview (integer).')
     		display_data.dataDisplayDialog(dfLabels,			
     										showOptionsToAddDf=True,
     										analyzeClass = self,
     										dfOutputName = 'FactorizedCategories',
     										topmost=True)
     		self.workflow.add('calcColumn', 
     					self.sourceData.currentDataFile,
     					{'funcDataR':'delete_columns_by_label_list',
     					'argsDataR':{'columnLabelList':columnNames},
     					'funcTreeR':'delete_entry_by_iid',
     					'argsTreeR':{'iid':['{}_{}'.format(self.sourceData.currentDataFile,columnName) for columnName in columnNames]},
     					'description':OrderedDict([('Activity:','Count through.'),
     					('Description:','Values in the selected columns has been factorized.'),
     					('Column names:',columnNames),
     					('Data ID:',self.sourceData.currentDataFile)])})


     def multiple_comparision_correction(self,method,alpha = 0.05):
     	 '''
     	 Checks if column is numerical. And then computes the selected method.
     	 '''

     	
     	

     	 selectedColumns  = self.selection_is_from_one_df()
     	 
     	 
     	 if self.DataTreeview.onlyNumericColumnsSelected == False:
     	 	tk.messagebox.showinfo('Select float ..',
     	 						   'Please select a numerical column or change the data type.')
     	 	return
     	 	     	 
     	 if selectedColumns is not None:
     	 	method = multCorrAbbr[method]
     	 	if method == 'fdr_tsbky':
     	 		alpha = ts.askfloat(title = 'Set alpha..',
     	 						prompt='You have to provide an alpha for the two stage FDR\n'+
     	 						' calculations a priori. Note that the corrected p-values\n '+
     	 						'are not valid for other alphas. You have to compute them\n '+
     	 						'again when switchting to another alpha!',
     	 						initialvalue = 0.05, minvalue = 0, maxvalue = 1)
     	 		if alpha is None:
     	 			return
     	 	corrColumns = []
     	 	for col in selectedColumns:
     	 		if self.sourceData.df[col].min() < 0 or self.sourceData.df[col].max() > 1:
     	 			tk.messagebox.showinfo('Error..',
     	 			'You need to select an untransformed p-value column with data in [0,1].'+
     	 			'If you have -log10 transformed p-values please transform them first using 10^p.')
     	 			if col == selectedColumns[-1]:
     	 				return
     	 			else:
     	 				continue
     	 		data_ = self.sourceData.df.dropna(subset=[col])
     	 		if 'storey' not in method:
     	 			reject, corr_pvals,_,_ = multipletests(data_[col], alpha = alpha,
     	 									 method = method, is_sorted= False, returnsorted=False)
     	 		else:
     	 			corr_pvals, pi0 = stats.estimateQValue(data_[col].values)
     	 		if method =='fdr_tsbky':
     	 			newCol = 'alpha_'+str(alpha)+'_corr_pVal_'+col
     	 		elif 'storey' in method:
     	 			newCol = 'qValue_pi_'+str(pi0)+'_'+col
     	 		else:
     	 			newCol = 'corr_pVal_'+method+'_'+col

     	 		toBeJoined = pd.DataFrame(corr_pvals,
     	 							  columns=[newCol],
     	 							  index= data_.index)
     	 		evaluatedColNames = self.sourceData.join_df_to_currently_selected_df(toBeJoined, 
     	 														exportColumns = True)
     	 		corrColumns.append(evaluatedColNames)
     	 	self.DataTreeview.add_list_of_columns_to_treeview(id = self.sourceData.currentDataFile,
     	 												   dataType = ['float64'],
     	 												   columnList = corrColumns)
     	 	tk.messagebox.showinfo('Done ..','Calculations performed. Corrected p'
     	 									 '-values were added.')

     def iir_filter(self):
     	'''
     	Smoothing data by iir filter.
     	'''
     	if self.DataTreeview.onlyNumericColumnsSelected == False:
     		tk.messagebox.showinfo('Error ..',
     			'Please select only numerical columns for this type of calculation.',
     			parent = self.toplevel)
     		return

     	n = ts.askinteger('IIR Filter - N',
     		prompt='Provide number n for filtering.\nThe higher the number the smoother the outcome.',
     		initialvalue = 20, minvalue = 1, maxvalue = len(self.sourceData.df.index),
     		parent = self)
     	if n is None:
     		return
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:
			 
     		newColumnNames = self.sourceData.iir_filter(selectedColumns  ,n)

     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													dataType = 'float64',
     													columnList = newColumnNames)
     		tk.messagebox.showinfo('Done ..','IIR Filter calculated. New columns were added.')
     		self.workflow.add('calcColumn', 
     					self.sourceData.currentDataFile,
     					{'funcDataR':'delete_columns_by_label_list',
     					'argsDataR':{'columnLabelList':newColumnNames},
     					'funcTreeR':'delete_entry_by_iid',
     					'argsTreeR':{'iid':['{}_{}'.format(self.sourceData.currentDataFile,col) for col in newColumnNames]},
     					'description':OrderedDict([('Activity:','IIR Filter (N = {})'.format(n)),
     					('Description:','A column has been added containing the results.'),
     					('Column name(s):',get_elements_from_list_as_string(newColumnNames, maxStringLength = None)),
     					('Selected Column(s):',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     					('Data ID:',self.sourceData.currentDataFile)])})    		




     def rolling_mod_data(self, rollingMetric, columns = None, quantile=0.5):
     	'''
     	Rolling window metric calculation.
     	'''
     	if columns is None:
     		columns =  self.DataTreeview.columnsSelected
     	window = ts.askinteger('Set window size...',
     		prompt = 'Please set window for rolling.\nIf window=10, 10 following values are\nused to calculate the '+str(rollingMetric)+'.',
     		initialvalue = 10, parent = self)
     	if window is None:
     		return

     	if rollingMetric == 'quantile':
     		quantile = ts.askfloat('Set quantile...',prompt = 'Please set quantile\nMust be in (0,1):', initialvalue = 0.75, minvalue = 0.0001, maxvalue = 0.99999)

     		if quantile is None:
     			return

     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:

     		newColumnNames = self.sourceData.calculate_rolling_metric(columns,window,rollingMetric,quantile)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,'float64',newColumnNames)
     		tk.messagebox.showinfo('Done ..','Rollling performed. New columns were added.')
     		
     		self.workflow.add('calcColumn', 
     					self.sourceData.currentDataFile,
     					{'funcDataR':'delete_columns_by_label_list',
     					'argsDataR':{'columnLabelList':newColumnNames},
     					'funcTreeR':'delete_entry_by_iid',
     					'argsTreeR':{'iid':['{}_{}'.format(self.sourceData.currentDataFile,col) for col in newColumnNames]},
     					'description':OrderedDict([('Activity:','Rolling window calculation {}'.format(rollingMetric)),
     					('Description:','Column(s) has/have been added containing data from a rolling window calculation. The metric is calculated to the right side interval limit.'),
     					('Column name(s):',get_elements_from_list_as_string(newColumnNames, maxStringLength = None)),
     					('Selected Column(s):',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     					('Data ID:',self.sourceData.currentDataFile)])})    

     def resort_columns_in_receiver_box(self, mode  = None, 
     										  replot = True):
     	'''
     	User defined order of items that have been placed in a receiver box
     	'''
     	if len(self.selectedNumericalColumns) + len(self.selectedCategories) == 0:
     		return
     	elif len(self.selectedNumericalColumns) <= 1  and len(self.selectedCategories) == 0:
     		return 
     	elif len(self.selectedNumericalColumns) == 0  and len(self.selectedCategories) <= 1:
     		return      	
     		
     	inputValues  = OrderedDict() 
     	if len(self.selectedNumericalColumns)  > 0:
     		inputValues['Numeric Columns'] = list(self.selectedNumericalColumns.keys()) 
     	if len(self.selectedCategories) > 0:
     		inputValues['Categorical Columns'] = list(self.selectedCategories.keys()) 
     	
     	if mode == 'num':
     		parentOpen = ['Numeric Columns']
     		
     	else:
     		parentOpen = ['Categorical Columns']     	
     				
     	dialog = custom_sort.customSortDialog(inputValues,parentOpen = parentOpen, enableDeleting = True,
     											infoText = 'You may remove columns by DEL or BACKSPACE.')
     	
     	if dialog.resortedValues is not None:
     	    
     		for key,value in dialog.resortedValues.items():
     			storeButton = []
     			if key == 'Numeric Columns':
     				selectedItems = self.selectedNumericalColumns
     				cols_ = True
     				# replace text on buttons 
     			else:
     				selectedItems = self.selectedCategories
     				cols_ = 'Categories'
     				     					
     			for text_,button in zip(value,selectedItems.values()):
     					button.configure(text = text_)
     					button.bind(right_click, lambda event, column = text_:\
     					self.delete_dragged_buttons(event,column,columns=cols_))	
     					storeButton.append(button)
     				
     			if len(storeButton) < len(selectedItems):
     				delButton = [but for but in selectedItems.values() if but not in storeButton]
     				for button in delButton:
     						button.destroy() 		
     			
     			if key == 'Numeric Columns':
     					 
     				self.selectedNumericalColumns = OrderedDict(zip(value,storeButton))
     			
     			else:
     			
     				self.selectedCategories = OrderedDict(zip(value,storeButton))
    				     			     						 
     			if replot:
     				if len(storeButton) < len(selectedItems):
     					plotType  = self.estimate_plot_type_for_default()	
     				else:
     					plotType = self.plt.currentPlotType
     				self.interactiveWidgetHelper.clean_frame_up()
     				self.prepare_plot(colnames = list(self.selectedNumericalColumns.keys()),
                                             catnames = list(self.selectedCategories.keys()),
                                             plot_type = plotType)	
            	
     		self.update_tooltip_in_receiverBox('reset')

   		 



     def save_current_session(self):
     	'''
     	There are x main classes that we need to restart a session.
     	'''
     	tk.messagebox.showinfo('Note ..','Please note that style changes on axis labels/ticks in main'+
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
     						  'dimReductionTests':self.dimensionReductionCollection,
     						  'statCollection':self.statResultCollection,
     						  'aucCollection':self.aucResultCollection,
     						  'workflow':self.workflow}

     	try:
     		performed = save_and_load_sessions.save_session(saveCollectionDict)
     		if performed != False:
     			tk.messagebox.showinfo('Done..','Session has been saved.')
     	except Exception as e:
     		tk.messagebox.showinfo('Error ..','Session not saved.\nError {}'.format(e))



     def open_saved_session(self):
         '''
         Opens a saved session. You select a folder/dir. But this will probably be changed
         soon so that you can save files where ever you/the user want(s).
         '''
         savedSession = save_and_load_sessions.open_session()
         if savedSession is None:
         	return
         elif savedSession == 'Not pckl found':
         	tk.messagebox.showinfo('Not found ..','Could not find a saved session file in selected directory..')
         	return
         self.plt = savedSession['plotter']
         self.sourceData = savedSession['sourceData']
         self.mainFigureCollection = savedSession['mainFigureCollection']
         self.curveFitCollection = savedSession['curveFitCoellection']
         self.clusterCollection = savedSession['clusterAnalysis']
         self.classificationCollection = savedSession['classificationAnalysis']
         self.anovaTestCollection = savedSession['anovaTests']
         self.dimensionReductionCollection = savedSession['dimReductionTests']
         self.statResultCollection = savedSession['statCollection']
         self.aucResultCollection = savedSession['aucCollection']
         self.workflow = savedSession['workflow']
         
         self.workflow.inFront = tk.BooleanVar(value=False)
         self.workflow.get_images()
         self.workflow.add_handles(sourceData = self.sourceData, plotter = self.plt, 
						treeView =self.DataTreeview, analyzeData = self)
		 
		

         self.plt.define_new_figure(self.f1)
         self.plt.reinitiate_chart()
         if self.plt.nonCategoricalPlotter is not None:
         	if self.plt.nonCategoricalPlotter.createIntWidgets:
         		self.interactiveWidgetHelper.create_widgets(plotter=self.plt,analyzeData = self)

         dataTypeColumCorrelation = self.sourceData.dfsDataTypesAndColumnNames
         file_names = self.sourceData.fileNameByID
         self.DataTreeview.add_all_data_frame_columns_from_dict(dataTypeColumCorrelation,file_names)
         if self.plt.plotCount in self.plt.plotHistory:
         	numericColumns, categoricalColumns  = self.plt.get_active_helper().columns
         	self.place_buttons_in_receiverbox(numericColumns, dtype = 'numeric')
         	self.place_buttons_in_receiverbox(categoricalColumns, dtype = 'category')

         self.open_main_figures()
         tk.messagebox.showinfo('Done ..','Session loaded. Happy working :).', parent=self)

     def open_main_figures(self):
     	'''
     	Open main figures from saved sessions. Utilizes the function (unpack_exports) to
     	replot used charts.
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
     		'''
     		Unpack plotting details. Eg. read the last chart and main figure contents.
     		'''

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
     						'Could not load image. Moved file to another place?',
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
     					limits = limits,boxBool = boxBool, gridBool = gridBool,
     					plotCount = exportDetails['plotCount'])
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
     	Setup main figure. Opens a dialog window to collect figures on a main
     	figure template.
     	'''
     	main_figures.mainFigureTemplateDialog(self.mainFigureCollection)

     def melt_data_by_groups(self):
     	'''
     	Melts data using selected groups. 
     	
     	Result
     	=========
     	- new data frame will be added
     	
     	Return 
     	=========
     	None
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	
     	if selectedColumns is not None:
     		
     		defineGroups = compare_groups.compareGroupsDialog(selectedColumns,self.sourceData, treeView = None, 
     										   statTesting = False)
     		
     		#print(defineGroups.groups)
     		groups = defineGroups.groups
     		groupNames = list(groups.keys())
     		if any(len(groups[group]) != len(groups[groupNames[0]]) for group in groupNames):
     			tk.messagebox.showinfo('Error ..','Groups must be of equal lengths.')
     			return
     		
     		concatDf = self.sourceData.melt_data_by_groups(groups)
     		self.add_new_dataframe(concatDf,fileName = 'group_melt ({})'.format(self.sourceData.get_file_name_of_current_data()))
     		tk.messagebox.showinfo('Done..','Grouped melting done. New data frame has been added.')
     		
     def melt_data(self):
     	'''
     	Melts the data using selected columns. Enters new data columns into the source treeview.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	
     	if selectedColumns is not None:


     		fileID,fileName,columnNameDataTyperRelationship = \
     		self.sourceData.melt_data_by_column(self.DataTreeview.columnsSelected )

     		self.DataTreeview.add_new_data_frame(fileID,fileName,columnNameDataTyperRelationship)

     		tk.messagebox.showinfo('Done ..','Melting done. New data frame was added to the treeview.')

     		
     		
     def unstack_column(self, columnName = None, separator = ';'):
     	'''
     	Unstacks cells of a column. User provide a separator. 
     	'''
     	if columnName is None:
     		selectedColumns = self.selection_is_from_one_df()
     		
     		if selectedColumns is None:
     			return
     			
     		columnName = selectedColumns[0]
     		separator = ts.askstring('Separator?',
     								 'Please provide separator used for unstacking.',
     								 initialvalue=';')
     		if separator is None:
     			return
     		
     	if isinstance(columnName, list):
     		print('columnName must be string!')
     		return
     		
     	
     	if columnName is not None:		
     		
     		
     		fileID,fileName,columnNameDataTyperRelationship = \
     		self.sourceData.unstack_column(columnName, separator, True)
     		
     		self.DataTreeview.add_new_data_frame(fileID,fileName,columnNameDataTyperRelationship)
     		tk.messagebox.showinfo('Done ..','Column unstacked. New data frame added.')
	

     def get_all_combs(self):
     	'''
     	Shows all comparisions within the categorical values.
     	The user can drag & drop statistical tests from the analysis frame / treeview
     	onto the label called ("Drop Statistic here")
     	'''
     	if hasattr(self,'groupedStatsData'):
     		## user cannot open new window
     		return

     	categoricalColumns = list(self.selectedCategories.keys())
     	numericColumns = list(self.selectedNumericalColumns.keys())
     	
     	if len(categoricalColumns) == 0:
     		tk.messagebox.showinfo('No categories..',
     							'Please load categorical columns into the receiver box.')
     		return
     	
     	if self.split_on_cats_for_plot.get():
		
     		self.groupedStatsData = self.sourceData.get_groups_by_column_list(categoricalColumns)

     		self.groupedStatsKeys = self.groupedStatsData.groups.keys()


     	else:
     		self.groupedStatsData = OrderedDict()
     		
     		for numColumn in numericColumns:
     			self.groupedStatsData[numColumn] , self.groupedStatsKeys = self.sourceData.get_positive_subsets([numColumn],
																				   categoricalColumns,
																				   self.sourceData.get_current_data())

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


     def select_features(self,featureSel):
     	'''
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	dataID = self.sourceData.currentDataFile
     	nFeatures = None
     	if selectedColumns is not None:       		
     		data = self.sourceData.get_current_data()[selectedColumns]
     		if self.DataTreeview.onlyNumericColumnsSelected == False:
     			
     			tk.messagebox.showinfo('Error ..',
     				'Please select only features containing floats and/or integers.')
     			return
     		X = data[selectedColumns].dropna(axis=1)
     		featuresSelected = X.columns.values.tolist()
     		
     		settings = {}
     		
     		if featureSel == 'Variance':
     			Y = None
     			value = ts.askfloat('Variance threshold',
     				prompt='Please provide variance threshold',
     				parent = self, initialvalue = 0.0)
     			if value is None:
     				return
     			settings['threshold'] = value
     				
     		elif featureSel in estimators:
     			targetSelection = simpleListboxSelection('Select target column for feature selection.',
     			 			data = self.sourceData.get_categorical_columns_by_id(id=self.sourceData.currentDataFile),
     			 			title = 'Select target column')
     			if len(targetSelection.selection) > 0:
     				Y = self.sourceData.get_current_data()[targetSelection.selection[0]]
     			else:
     				return
     			
     			modelProps = estimatorSettings[featureSel].keys() 
     			
     			defaultSett = [x[0] for x in estimatorSettings[featureSel].values()]
     			optionSett = [x[1] for x in estimatorSettings[featureSel].values()]
     			
     			
     			modelSettingDialog = simpleUserInputDialog(modelProps,defaultSett, optionSett,
     				title = 'Select settings for model', 
     				infoText = 'Enter or select settings for model')
     				
     			modelSettings = modelSettingDialog.selectionOutput
     			settings = checkDataType(modelSettings)   
     			nFeatures = ts.askinteger('Number of features',  	
     				prompt='Provide the number of the most important features that should be selected.',
     				parent = self, initialvalue = 20)		

     			if nFeatures is None:
     				return
     		else:
     			return	
				
     		model = selectFeaturesFromModel(X.values,Y = Y ,
     			model=featureSel, 
     			max_features = nFeatures,
     			addSettings = settings)
     		featureImportance = model.featureImportance
     		mask = model.featureMask
     					
     		df = pd.DataFrame()
     		df['Selected Features'] = [featuresSelected[n] for n,bool in enumerate(mask) if bool]
     		if featureImportance is not None:
     			df['Feature Importance'] = featureImportance[mask]
     		
     		dataDialog = display_data.dataDisplayDialog(df,showOptionsToAddDf=True)     		
     		quest = tk.messagebox.askquestion('Subset..',
					'Would you like to subset the base data frame using the selected features?',
					parent = self)
     		
     		if quest == 'yes':
     		
     			dfColumns = self.sourceData.get_columns_of_df_by_id(dataID)
     			newColumns = [column for column in dfColumns if column in df['Selected Features'].values.tolist()]
     			oldColumns = [column for column in dfColumns if column not in selectedColumns]
     			combColumns = newColumns + oldColumns 
     			df = self.sourceData.get_data_by_id(dataID)[combColumns]
     			self.add_new_dataframe(df,'FeatureSelection ({})'.format(self.sourceData.get_file_name_of_current_data()))
					
     def replace_data_in_df(self,replaceOption):
     	'''
     	replaces NaNs or 0s with constant or metric
     	'''
     	if 'Constant' in replaceOption:
     		value = ts.askfloat('Constant ..',
     			prompt='Please provide constant to be used for NaN replacement',
     			parent = self)
     		if value is None:
     			return
     			
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:   
     		if self.DataTreeview.onlyNumericColumnsSelected == False:
     			tk.messagebox.showinfo('Error ..','Please select only columns containing floats and/or integers.')
     			return
     		numericColumns = self.DataTreeview.columnsSelected
     		if 'Gauss Distribution' in replaceOption:

     			valueDialog = simpleUserInputDialog(descriptionValues = ['Downshift (in stdev)','Rel. Stdev','Mode'],
     											initialValueList = ['1.8','0.4','Replace & add indicator'],
     											optionList = [np.round(np.linspace(-2,2,num=10,endpoint=True),decimals=1).tolist(),
     														  np.round(np.linspace(0.1,1.5,num=10,endpoint=True),decimals=1).tolist(),
     														  ['Replace','Create new columns','Replace & add indicator']],
     											title = 'Settings for NaN Replacement',
     											infoText = 'Replace NaN with data drawn from a Gaussian Distribution.')

	     		selection = valueDialog.selectionOutput
	     		if len(selection) != 0:
	     			newColumns = self.sourceData.fill_na_with_data_from_gauss_dist(numericColumns,
	     				float(selection['Downshift (in stdev)']),
	     				float(selection['Rel. Stdev']),
	     				selection['Mode'])
	     			if newColumns is not None:
	     				self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     										dataType = 'float64' if selection['Mode'] == 'Create new columns' else 'object',
     										columnList = newColumns,
     										)
	     		else:
      				return
      				
     		elif replaceOption == 'NaN -> Mean[row]':
     			self.sourceData.fill_na_in_columnList_by_rowMean(numericColumns)
      				
     		elif 'NaN -' in replaceOption:
     			if '0' in replaceOption:
     				replaceValue = 0
     			elif 'Constant' in replaceOption:
     				replaceValue = value ##from ask float above
     			elif 'Mean[col]' in replaceOption:
     				replaceValue = self.sourceData.df[numericColumns].mean(axis=0)
     			elif 'Median' in replaceoption:
     				replaceValue = self.sourceData.df[numericColumns].median(axis=0)
     			
     			self.sourceData.fill_na_in_columnList(numericColumns,replaceValue)
     		
     		else:
     			replaceDict = dict()
     			for numColumn in numericColumns:
     				replaceDict[numColumn] = {0:np.nan}
     			self.sourceData.replace_values_by_dict(replaceDict)
     		tk.messagebox.showinfo('Done ..','Calculations done.', parent = self)


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


     def remove_rows_with_nan(self, how = 'all == NaN', dropColumns = False):
     	'''
     	Drops rows from selected columns. Changes, if necessary, to the selected DataFrame.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:  
     	
     		thresh = None
     		if 'all' in how:
     			how = 'all'
     		elif 'any' in how:
     			how = 'any'
     		else:
     			thresh = ts.askinteger('Threshold ..',prompt = 'Please provide threshold for dropping rows/columns with nan.\n'+
     									' A threshold of two means that at least 2 non-nan\nvalues are required '+
     									'to keep a row.', parent = self,
     									initialvalue = int(float(len(self.DataTreeview.columnsSelected)/2)),
     									minvalue = 1,
     									maxvalue = len(self.DataTreeview.columnsSelected))
     			if thresh is None:
     				return
     		if dropColumns:
     			columnsRemained = self.sourceData.drop_columns_with_nan(self.DataTreeview.columnsSelected, how, thresh)
     			nanRemoved = len(self.DataTreeview.columnsSelected) - len(columnsRemained)
     			deleteColumns = [col for col in self.DataTreeview.columnsSelected if col not in columnsRemained]
     			if len(deleteColumns) == 0:
     				tk.messagebox.showinfo('Aborting..','No columns contain NaN matching the selected threshold/criteria')
     				return
     			quest = tk.messagebox.askquestion('Proceed ?','The following columns will be removed: {}'.format(str(deleteColumns).replace(',','\n')))
     			if quest == 'yes':
     				self.sourceData.delete_columns_by_label_list(deleteColumns)
     				tk.messagebox.showinfo('Done..','Columns containing NaN removed. In total {} columns were removed'.format(nanRemoved))     		
     				dict_ = self.sourceData.dfsDataTypesAndColumnNames
     				file_names = self.sourceData.fileNameByID
     				self.DataTreeview.add_all_data_frame_columns_from_dict(dict_,file_names)
     		else:
     			nanRemoved = self.sourceData.drop_rows_with_nan(self.DataTreeview.columnsSelected, how, thresh)
     		
     			tk.messagebox.showinfo('Done ..','NaN were removed in selected columns. In total {} rows.'.format(nanRemoved))
     		
     			self.workflow.add('deleteRows',
     				self.sourceData.currentDataFile,
     				{'funcDataR':'save_dropped_rows',
     				'argsDataR':{'id':self.sourceData.currentDataFile,
     				'reverse':True},
     				'description':OrderedDict([('Activity:','Drop Rows with NaN - ({})'.format(how)),
     				('Description:','Rows containing NaN were dropped. If a threshold of 3 is given rows are retained that contain at least 3 non-NaN values'),
     				('Removed Rows:','{}'.format(nanRemoved)),
     				('Selected Columns',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     				('Data ID:',self.sourceData.currentDataFile)])})
  	
     def transform_selected_columns(self, transformation):
     	'''
     	Transform data. Adds a new column to the data frame.
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:

     		transformedColumnName = self.sourceData.transform_data(self.DataTreeview.columnsSelected  ,transformation)
     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													dataType = 'float64',
     													columnList = transformedColumnName,
     													)
     		tk.messagebox.showinfo('Done ..','Calculations performed.')
     		
     		self.workflow.add('calcColumn',
     				self.sourceData.currentDataFile,
     				{'funcDataR':'delete_columns_by_label_list',
     				'argsDataR':{'columnLabelList':transformedColumnName},
     				'funcTreeR':'delete_entry_by_iid',
     				'argsTreeR':{'iid':['{}_{}'.format(self.sourceData.currentDataFile,col) for col in transformedColumnName]},
     				'description':OrderedDict([('Activity:','Data transformation {}'.format(transformation)),
     				('Description:','Column(s) has/have been transformed and a new column has been added.'),
     				('Column name(s):',get_elements_from_list_as_string(transformedColumnName, maxStringLength = None)),
     				('Selected Column(s):',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     				('Data ID:',self.sourceData.currentDataFile)])})    
     				


     def split_column_content_by_string(self,splitStringCommand = None):
     	'''
     	Splits the content of a column row-wise with given splitString.
     	For example: KO_10min would be split by '_' into two new columns KO , 10min
     	'''
     	if splitStringCommand is None or splitStringCommand  == "Custom String":
     		splitString = ts.askstring('Split strings','Please provide string for splitting')
     		if splitStringCommand == '' or splitStringCommand is None:
     			return
     	else:
     		splitString = splitStringCommand[-2]
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:

     		splitColumnName, indexStart = self.sourceData.split_columns_by_string(self.DataTreeview.columnsSelected  ,splitString)

     		if splitColumnName is None:
     			tk.messagebox.showinfo('Error..','Split string was not found in selected column.')
     			return

     		self.DataTreeview.add_list_of_columns_to_treeview(self.sourceData.currentDataFile,
     													dataType = 'object',
     													columnList = splitColumnName,
     													startIndex = indexStart)

     		tk.messagebox.showinfo('Done ..','Selected column(s) were split and added to the source data treeview.')
     		self.workflow.add('addColumn',
     				self.sourceData.currentDataFile,
     				{'funcDataR':'delete_columns_by_label_list',
     				'argsDataR':{'columnLabelList':splitColumnName},
     				'funcTreeR':'delete_entry_by_iid',
     				'argsTreeR':{'iid':['{}_{}'.format(self.sourceData.currentDataFile,col) for col in splitColumnName]},
     				'description':OrderedDict([('Activity:','Split string by {}'.format(splitStringCommand)),
     				('Description:','Column(s) has/have been added containing the split strings.'),
     				('Column name(s):',get_elements_from_list_as_string(splitColumnName, maxStringLength = None)),
     				('Selected Column(s):',get_elements_from_list_as_string(selectedColumns, maxStringLength = None)),
     				('Data ID:',self.sourceData.currentDataFile)])})



     def sort_source_data(self,sortType = 'Value', columnNames = None, 
     						   verbose = True, replot = False):
     	'''
     	Sort columns either according to the value or by string length. Value can handle mutliple
     	columns while string length is only able to sort for one column. Note that the
     	sort is ascending first, then upon second sort it will be descending.
     	'''
     	if columnNames is None:

     		selectionIsFromSameData, selectionDataFrameId = self.DataTreeview.check_if_selection_from_one_data_frame()
     		selectedColumns = self.DataTreeview.columnsSelected 
     	else:
     		if isinstance(columnNames,list):
     			selectedColumns = columnNames
     			# if this is true, function is called from receiver box item
     			selectionIsFromSameData = True
     			selectionDataFrameId = self.plt.get_dataID_used_for_last_chart()
     		
     	
     	if selectionIsFromSameData:
     		self.sourceData.set_current_data_by_id(selectionDataFrameId)
     		
     		if sortType == 'Value':
     			
     			ascending = self.sourceData.sort_columns_by_value(selectedColumns)
     		
     		elif sortType == 'String length':
     			if len(self.DataTreeview.columnsSelected  ) > 1:
     				tk.messagebox.showinfo('Note..','Please note that this sorting can handle only one column. The column: {} will be used'.format(self.DataTreeview.columnsSelected  [0]))
     			self.sourceData.sort_columns_by_string_length(self.DataTreeview.columnsSelected[0])
			     		
     		if verbose:
     		
     			if sortType == 'String length':
     				tk.messagebox.showinfo('Done ..','Data sorted by string length.')
     				return
     			
     			elif ascending:
     			
     				infoString = 'in ascending order. Sort again to get descending order.'
     			else:
     				infoString = 'in descending order.'

     			tk.messagebox.showinfo('Done ..','Selected column(s) were sorted {}'.format(infoString))
     		if replot:
     			self.plt.initiate_chart(*self.plt.current_plot_settings)
				
     	else:

      		tk.messagebox.showinfo('Error ..','Please select only columns from one file.')
      		return



     def re_sort_source_data_columns(self):
     	'''
     	Resorts column in currently selected data frame.
     	(Alphabetical order)
     	'''
     	self.sourceData.resort_columns_in_current_data()
     	dict_ = self.sourceData.dfsDataTypesAndColumnNames
     	file_names = self.sourceData.fileNameByID
     	self.DataTreeview.add_all_data_frame_columns_from_dict(dict_,file_names)
   	
     						
     def remove_savedCalculations(self, dataId):
     	'''
     	Removes made calculations that were associated with the
     	particular dataId.
     	'''
     	self.curveFitCollection.remove_fits_by_dataId(dataId)

     def post_analysis_menu(self, event):
     	'''
     	'''
     	self.post_menu(menu=self.menuCollection['analysis_treeview'] )



     def post_treeview_menu(self, event):
         '''
         Button-3 drop-down menu.
         '''
         if self.DataTreeview.onlyDataFramesSelected:
         	self.post_menu(menu = self.menuCollection['data_frame'])
         elif self.DataTreeview.onlyDataTypeSeparator:
         	self.post_menu(menu=self.menuCollection['data_type_menu'] )
         else:
         	self.post_menu(menu = self.menuCollection['main'])


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
             	self.frame.configure(bd=2,relief=tk.SOLID)
             else:
             	but_text = str(self.DataTreeview.columnsSelected)[1:-1]
             	if len(but_text) > 40 and len(self.DataTreeview.columnsSelected) > 1:
             		but_text = '#multiple'
             	elif len(but_text) > 40:
             		but_text = str(self.DataTreeview.columnsSelected)[0:15]
             		
             	self.indicate_drag_drop_areas(self.data_types_selected[0])

             self.mot_button = tk.Button(self, text=but_text, bd=1,
                                     		   fg="darkgrey", bg=MAC_GREY)

             if  len(self.mot_button_dict) != 0:

             	for item in self.mot_button_dict.values():
             		item.destroy()

             	self.mot_button_dict.clear()

             self.mot_button_dict[self.mot_button] = self.mot_button

         x = self.winfo_pointerx() - self.winfo_rootx()
         y = self.winfo_pointery() - self.winfo_rooty()

         self.mot_button.place( x= x-20 ,y = y-30) 
         ## offset because otherwise dropped widget will always be the same button

         if analysis:

             if self.widget == self.canvas.get_tk_widget():
                 self.mot_button.configure(fg = "blue")

             else:
                 self.mot_button.configure(fg = "darkgrey")

         else:
                 if len(self.data_types_selected) == 0:
                 	return
                 unique_dtypes_selected = self.data_types_selected[0]
                 if self.widget in self.dataTypeSpecWidgets[unique_dtypes_selected]:
                 	self.mot_button.configure(fg="blue")

                 elif self.widget == self.color_button_droped and unique_dtypes_selected  in ['object','int64']:
                 	self.mot_button.configure(fg="blue")

                 else:
                 	self.mot_button.configure(fg = "darkgrey")


     def indicate_slice_mark_buttons(self, which = 'all', frameRelief=tk.SOLID):
     	'''
     	Give Slice and Marks button a frame
     	'''
     	if isinstance(which,list):
     		pass
     	elif which == 'all':
     		which = ['color','size','label','filter','tooltip']

     	for key, frame in self.framesSliceMarks.items():
     			if key in which:
     				frame.config(relief =  frameRelief, bd = 1)
     	

     def indicate_drag_drop_areas(self, dataType, frameRelief=tk.SOLID):
     	'''
     	Change frames that accept are drag and drop areas
     	'''

     	plotTypeWithHue = self.plt.currentPlotType in ['hclust','PCA','scatter_matrix',\
     		'cluster_analysis','line_plot'] or (self.plt.currentPlotType == 'scatter'\
     		 and self.plt.binnedScatter == False)

     	if dataType == 'float64':
     		frames = [self.column_sideframe]

     	elif dataType == 'object':
     		frames =  [self.category_sideframe]

     	elif dataType == 'int64':
     		frames =  [self.column_sideframe,self.category_sideframe]

     	if plotTypeWithHue == False:
     			self.indicate_slice_mark_buttons(['filter'])
     	else:
     			self.indicate_slice_mark_buttons()

     	for frame in frames:

     		frame.config(relief=frameRelief)

     def remove_mpl_connection(self, plot_type = ''):

         try:
                              self.canvas.mpl_disconnect(self.pick_label)
                              self.canvas.mpl_disconnect(self.pick_freehand)

         except:
                                  pass
         mpl_connections = [
                            self.selection_press_event,
                            self.pick_label,
                            self.pick_freehand,
                            self.release_event,
                            ]
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



     def bind_events_to_button_in_receiverbox(self,columnName,button,numeric):
     	'''
     	'''
     	button.bind(right_click, lambda event, columnName = columnName: \
     		self.delete_dragged_buttons(event,columnName,columns=numeric))
     	button.configure(text = columnName, 
     		command = lambda columnName = columnName: self.receiver_button_menu(columnName))
     	

     def update_receiver_box(self,numericColumns,categoricalColumns):
     	'''
     	'''
     	self.clean_up_dropped_buttons(replot=False, clearFigure = False)
     	self.place_buttons_in_receiverbox(numericColumns,'numeric')
     	self.place_buttons_in_receiverbox(categoricalColumns,'category')

     def start_grid_search(self,features = [], targetColumn = None):
     	'''
     	Start Grid search from context menu.
     	'''
     	if self.DataTreeview.onlyNumericColumnsSelected == False:
     		tk.messagebox.showinfo('Error..',
     			'Please select only numeric columns. Note: You can change the data type using the context menu.',
     			parent = self)
     			
     	features = self.selection_is_from_one_df()
     	if features is not None:
     		
     		if targetColumn is None:
     		
     			 targetSelection = simpleListboxSelection('Select target column.',
     			 						data = self.sourceData.get_categorical_columns_by_id(id=self.sourceData.currentDataFile),
     			 						title = 'Select target column for grid search')
				
     			 if len(targetSelection.selection) > 0:
     			 	targetColumn = [targetSelection.selection[0]]
     			 	if len(targetSelection.selection) != 1:
     			 		tk.messagebox.showinfo('Note..',
     			 			'Multiple columns selected, only the first one will be used: {}'.format(targetSelection.selection[0]),
     			 			parent=self)
     			 else:
     			 	return
     	
     		classification.gridSearchClassifierOptimization(self.classificationCollection, self.sourceData,
                     											features,targetColumn,plotter=self.plt)

   
     def place_buttons_in_receiverbox(self,columnNames,dtype):
     	'''
     	Receiver boxes do receive drag & dropped items by the user.
     	We separate between numeric data and categorical data. This function allows
     	place button into these receiver boxes
     	'''
     #	self.check_input()
     	
     	for column in columnNames:

     		button = tk.Button(self.receiverFrame[dtype], 
     							text = column)
     		if dtype == 'numeric':
     			self.selectedNumericalColumns[column] = button
     			numeric = True
     		else:
     			self.selectedCategories[column] = button
     			numeric = False
     		self.update_tooltip_in_receiverBox(dtype)

     		button.pack(side=tk.LEFT,padx=2)
     		self.bind_events_to_button_in_receiverbox(column,button,numeric)
     		

     def receiver_button_menu(self, columnName):     
     	'''
     	'''
     	self.columnInReceiverBox = [columnName]
     	self.post_menu(menu=self.menuCollection['receiverBox']) 


     def update_tooltip_in_receiverBox(self,dtype):
     	'''
     	Adds tooltip information to receiver boxes. 
     	Updates upon any change.
     	'''
     	toolTipText = ''
     	
     	if dtype in ['numeric','reset']:
     	
     		for col in self.selectedNumericalColumns.keys():
     			 toolTipText = toolTipText+'{}\n'.format(col)
     		self.numTool.text = toolTipText  
     	
     	toolTipText = ''	  
     	if dtype != 'numeric':
     		 		
     		for col in self.selectedCategories.keys():
     			 toolTipText = toolTipText+'{}\n'.format(col)
     		
     		self.catTool.text = toolTipText       		
	 

     def release(self,event,analysis=''):
         '''
         Handles all release events by drag & drop.
         Currently too long and complex..
         '''

         if len(self.data_types_selected) == 0:
             return
        
         if event is None:
         	return
         		
         widget = self.winfo_containing(event.x_root, event.y_root)
         self.frame.configure(bd=2,relief=tk.FLAT)
         self.indicate_drag_drop_areas(self.data_types_selected[0],frameRelief=tk.GROOVE)
         self.indicate_slice_mark_buttons(frameRelief=tk.FLAT)


         if self.mot_button is not None:
         	self.mot_button.destroy()
         	self.mot_button = None

         dataFrames = self.DataTreeview.dataFramesSelected
         if len(dataFrames) == 0:
         	return
         
         self.sourceData.set_current_data_by_id(dataFrames[0])

         if analysis == '':
             if widget == self.source_treeview:
             	return
             try:
                 self.check_input()
             	 
                 self.cat_filtered = [col for col in self.DataTreeview.columnsSelected  \
                 if col not in self.selectedCategories and self.sourceData.df[col].dtype != np.float64]

                 self.col_filtered = [col for col in self.DataTreeview.columnsSelected   \
                 if col not in self.selectedNumericalColumns and \
                 (self.sourceData.df[col].dtype == np.float64 or \
                 self.sourceData.df[col].dtype == np.int64)]

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


             if self.plt.currentPlotType not in ['scatter','hclust','PCA']:
                 return
             if self.label_button_droped is not None:
                 self.label_button_droped.destroy()
                 self.label_button_droped = None

             self.anno_column = self.DataTreeview.columnsSelected
             if self.plt.currentPlotType in ['scatter','PCA']:
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
                     tk.messagebox.showinfo('Info..','Please note that only one column can be'+
                     						' used for labeling rows in a h-clust.\nHowever'+
                     						' you can also merge columns with the function:'+
                     						' Combine columns from the drop-down menu'
                                            )

                 self.plt.nonCategoricalPlotter._hclustPlotter.add_label_column(self.anno_column)

             s =  self.return_string_for_buttons(self.DataTreeview.columnsSelected[0])
             self.label_button_droped  = create_button(self.interactiveWidgetHelper.frame, text = s,
             											image= self.but_label_icon,
             											compound=tk.CENTER)

             if self.plt.currentPlotType != 'hclust':
             	self.label_button_droped.bind(right_click,
             					self.remove_annotations_from_current_plot)
             else:
             	self.label_button_droped.bind(right_click,
             					lambda event, label = self.label_button_droped:
             					self.plt.nonCategoricalPlotter._hclustPlotter.remove_labels(event,label))

             self.label_button_droped.grid(columnspan=2, padx=1, pady=1)

         elif widget == self.sliceMarkFrameButtons['filter']:

             if any(self.sourceData.df[col].dtype not in [np.float64,np.int64] for col  in self.DataTreeview.columnsSelected):

                     if len(self.DataTreeview.columnsSelected) == 1:

                     	self.categorical_column_handler('Find category & annotate')

                     else:

                     	self.custom_filter()

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

             s =  self.return_string_for_buttons(self.DataTreeview.columnsSelected[0])


             self.size_button_droped = create_button(self.interactiveWidgetHelper.frame, text = s,
             											image= self.but_size_icon,
             											compound=tk.CENTER,
             											command = self.define_size_range)

             self.size_button_droped.bind(right_click, self.remove_sizes_)
             self.size_button_droped.grid(columnspan=2, padx=1, pady=1)
             CreateToolTip(self.size_button_droped,
                 	text = get_elements_from_list_as_string(self.DataTreeview.columnsSelected))


         elif widget == self.sliceMarkFrameButtons['color']:

            if len(self.plt.plotHistory) == 0:
                return
            last_plot_type = self.plt.currentPlotType


            if self.color_button_droped is not None:
                 self.color_button_droped.destroy()
                 self.color_button_droped = None

            s =  self.return_string_for_buttons(self.DataTreeview.columnsSelected[0])

            if self.plt.binnedScatter and last_plot_type == 'scatter':
            	return

            self.color_button_droped = create_button(self.interactiveWidgetHelper.frame, text = s,
             											image= self.but_col_icon,
             											compound=tk.CENTER, 
             											command = lambda: legend_handler.legendDialog(self.plt))
            if last_plot_type != 'hclust':
                    self.color_button_droped.bind(right_click,
                    		self.remove_color_)
                    self.interactiveWidgetHelper.clean_color_frame_up()

            else:
                    self.color_button_droped.bind(right_click,
                    		lambda event, label = self.color_button_droped:
                    		self.plt.nonCategoricalPlotter._hclustPlotter.remove_color_column(event,label))

            self.color_button_droped.grid(columnspan=2, padx=1, pady=1)
            CreateToolTip(self.color_button_droped,
                 	text = get_elements_from_list_as_string(self.DataTreeview.columnsSelected))

            if last_plot_type == 'hclust':
                 self.plt.nonCategoricalPlotter._hclustPlotter.add_color_column(self.DataTreeview.columnsSelected)
            else:
                ret_ = self.update_color()
                if ret_ == False:
                 	return

         elif widget == self.color_button_droped:

             columnSelected = self.DataTreeview.columnsSelected
             if self.plt.nonCategoricalPlotter is not None:
             	#print(self.plt.nonCategoricalPlotter.dtypeColorColumn)
				
             	if 'change_color_by_categorical_column' == self.plt.nonCategoricalPlotter.dtypeColorColumn:
             		
             		alreadyUsedColors = self.plt.nonCategoricalPlotter.get_size_color_categorical_column()

             	elif 'change_color_by_numerical_column' == self.plt.nonCategoricalPlotter.dtypeColorColumn:

             		alreadyUsedColors = self.plt.nonCategoricalPlotter.get_size_color_categorical_column('change_color_by_numerical_column')

             	elif hasattr(self.plt.nonCategoricalPlotter, 'linePlotHelper') and \
             	'change_color_by_categorical_columns' in self.plt.nonCategoricalPlotter.linePlotHelper.sizeStatsAndColorChanges:

             		alreadyUsedColors = self.plt.nonCategoricalPlotter.linePlotHelper.sizeStatsAndColorChanges['change_color_by_categorical_columns']
             	else:
             		alreadyUsedColors = []
             else:
             	alreadyUsedColors = []

             self.DataTreeview.columnsSelected = [col for col  in alreadyUsedColors if col not in columnSelected] + columnSelected
             proceed = self.update_color()
             if proceed:
                 self.color_button_droped.configure(text = "Multiple")
                 CreateToolTip(self.color_button_droped,
                 	text = get_elements_from_list_as_string(self.DataTreeview.columnsSelected))





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
                     elif self.test in ['SGCCA']:
                     
                     	multi_block.sggcaDialog(self.sourceData, self.DataTreeview)
					 	

                     elif self.test in clustering.availableMethods:

                     	clustering.clusteringDialog(self.sourceData,self.plt, self.DataTreeview,
                     								self.clusterCollection , self.interactiveWidgetHelper,
                     								numericColumns = list(self.selectedNumericalColumns.keys()),
                     								initialMethod = self.test,
                     								cmap = self.cmap_in_use.get())


                     elif self.test == 'CV based Grid Search':

                     	tk.messagebox.showinfo('Note ..','This activity is still under construction '+
                     					'and should so far only be used to learn, not for real data analysis.',
                     					parent = self)
                     	if len(self.selectedNumericalColumns) < 2:
                     		tk.messagebox.showinfo('Error ..',
                     			'Need more features (numerical Columns) in receiver box.',
                     			parent=self)
                     		return
                     	
                     	
                     		
                     	if len(self.selectedCategories) == 0:
                     		tk.messagebox.showinfo('Error ..',
                     			'Need at  least one categorical column. (Target variable, class)',
                     			parent=self)
                     		return
						
                     	features = list(self.selectedNumericalColumns.keys())
                     	targetColumn = list(self.selectedCategories.keys())

                     	classification.gridSearchClassifierOptimization(self.classificationCollection, self.sourceData,
                     											features,targetColumn,plotter=self.plt)

                     elif 'ANOVA' in self.test:

                     	anova_calculations.anovaCalculationDialog(self.test,
                     											  list(self.selectedNumericalColumns.keys())[0],
                     											  list(self.selectedCategories.keys()),
                     											  self.sourceData,
                     											  self.anovaTestCollection)

                     	self.stat_button_droped.configure(command = lambda : \
                     	self.show_anova_results())

                     elif self.test =='Kruskal-Wallis':
                         self.perform_one_way_anova_or_kruskall()

                     ## check if test should be used to compare two groups
                     elif self.test =='Pairwise Comparision':
                     	self.get_all_combs()

                     elif self.test in self.options_for_test['Compare two groups']:
                     	## note that the statistic results are being saved in self.plt associated with the plot
                     	## count number

                        statTestInformation = {'test':self.test, 'paired':self.paired, 'mode':self.mode}
                        if self.twoGroupstatsClass is not None:
                        	## if it exists already -> just update new stat test settings
                        	self.twoGroupstatsClass.selectedTest = statTestInformation
                        else:
                        	self.twoGroupstatsClass = stats.interactiveStatistics(self.plt,
                        								self.sourceData,statTestInformation,
                        								self.statResultCollection)
                        self.stat_button_droped.configure(command = self.show_statistical_test)


                     elif self.test in  ['Correlate rows to ..']:

                     	self.calculate_correlations()

                     elif self.test == 'Display curve fit(s)':

                          self.display_curve_fits()

                     elif self.test == 'Curve fit ..':

                     	self.curve_fit(from_drop_down=False)

                     elif self.test in stats.dimensionalReductionMethods:

                         self.interactiveWidgetHelper.clean_frame_up()
                         self.dimReduction_button_droped = create_button(self.interactiveWidgetHelper.frame, text = s,
                         									image= self.but_stat_icon, compound=tk.CENTER,
                         									command = lambda: self.post_menu(menu=self.menuCollection['PCA']))#self.post_pca_menu)
                         self.dimReduction_button_droped.grid(columnspan=2, padx=0, pady=1)
                         self.perform_dimReduction_analysis()


     def show_anova_results(self):
     	'''
     	Display anova results.
     	'''
     	if len(self.anovaTestCollection.anovaResults) == 0:
     		tk.messagebox.showinfo('Error ..','No ANOVA test results found.')
     		return
     	anova_calculations.anovaResultDialog(self.anovaTestCollection)


     def show_statistical_test(self, showAllTests = False):
     	'''
     	Displays calculated statistics in a pandastable.
     	Allows the user to add these data to the data collection and
     	treeview. Which then can be could be used for plotting.
     	'''
     	if showAllTests:
     		data = self.statResultCollection.performedTests
     	else:
     		data = self.twoGroupstatsClass.performedTests
     	if len(data.index) == 0:
     		tk.messagebox.showinfo('Error ..','No test results found..')
     		return

     	dataDialog = display_data.dataDisplayDialog(data,showOptionsToAddDf=True)

     	if dataDialog.addDf:
     		nameOfDf = 'StatResults_plotID:{}'.format(self.plt.plotCount)
     		self.add_new_dataframe(dataDialog.data,nameOfDf)
     	del dataDialog


     def show_auc_calculations(self, showAllTests = False):
     	'''
     	Display determined AUC in pandastable.
     	Users can also add the data frame to the source data tree view
     	'''

     	if showAllTests:
     		df = self.aucResultCollection.performedCalculations
     	else:
     		df = self.plt.nonCategoricalPlotter.timeSeriesHelper.get_auc_data()

     	dataDialog = display_data.dataDisplayDialog(df,showOptionsToAddDf=True)
     	if dataDialog.addDf:
     		del dataDialog
     		nameOfDf = 'aucResults_plotID:{}'.format(self.plt.plotCount)
     		self.add_new_dataframe(df,nameOfDf)
     
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
     	
     	datToInspect = self.sourceData.get_data_by_id(currentDataSelection[0]).copy()  ## this is needed otherwise the self.df will be updated instantly
     	dataID = currentDataSelection[0]
     	dataDialogue = display_data.dataDisplayDialog(datToInspect,self.plt)
     	data_ = dataDialogue.get_data()
     	del dataDialogue
     	if data_.equals(self.sourceData.df):
     		pass
     	elif dataID == self.sourceData.currentDataFile:
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


     def remove_stat_(self,event):
     	'''
     	Deletes results of statistical tests.
     	'''
     	quest = tk.messagebox.askquestion('Confirm ..','This will remove all statistical'+
     									  ' test results from the current chart. Proceed?')
     	if quest == 'yes':
     		if self.twoGroupstatsClass is not None:

     			self.twoGroupstatsClass.delete_all_stats()
     			self.twoGroupstatsClass.disconnect_event()
     			
     			del self.twoGroupstatsClass
     			self.twoGroupstatsClass = None
     			self.plt.redraw()
     		
     		if self.plt.currentPlotType == "scatter":
     			scatterPlots = self.plt.get_scatter_plots()
     			for scatterPlot in scatterPlots.values():
     				scatterPlot.remove_stat_line()
     			self.plt.redraw()
				
					
				
     		
     		self.stat_button_droped.destroy()
     			

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
         ## no chart created
         if self.plt.plotCount == 0:
         	return

         _,catnames,plot_type,_ = self.plt.current_plot_settings
         n_categories = len(catnames)

         quest = tk.messagebox.askquestion('Confirm ..', 'This will remove all size changes.'+
         									' Proceed?')
         if quest == 'yes':

         	self.plt.remove_size_level()
         	self.size_button_droped.destroy()
         	self.size_button_droped = None
         	self.plt.redraw()


     def remove_tool_tip_active(self,event):
         '''
         Removes tooltip activity.
         '''
         self.plt.disconnect_tooltips()
         self.tooltip_button_droped.destroy()


     def remove_annotations_from_current_plot(self,event):
         '''
         Removes all annotations from a plot
         '''
         quest = tk.messagebox.askquestion('Deleting labels..','This step will remove all labels from your plot.\nPlease confirm..')
         if quest == 'yes':

          	self.plt.nonCategoricalPlotter.remove_all_annotations()
          
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
         '''
         Performs dimensional reduction on data.
         '''
         self.plt.set_data_frame_of_last_chart()
         numericColumns = list(self.selectedNumericalColumns.keys())
         numbNumericColumns = len(numericColumns)

         if len(numericColumns) < 3:
             tk.messagebox.showninfo('Error..','Please add columns to the numeric data receiver box. You need at least three numeric columns.')
             return

         nComps = self.dimensionReductionCollection.nComponents
         if nComps > numbNumericColumns:
         	nComps = numbNumericColumns

         if self.test == 'Linear Discriminant Analysis':
         	dialog = simpleListboxSelection('Select column that contains information about the class labels',
         		data = self.sourceData.get_categorical_columns_by_id(id=self.sourceData.currentDataFile))
         	
         	selection = dialog.selection
         	if len(selection) != 0:
         		outcomeVariable = self.sourceData.df[selection[0]].values
         	else:
         		return
     		
         else:
         	outcomeVariable = None

	
         dimRedResult = stats.get_dimensionalReduction_results(self.sourceData.df[numericColumns],
         													method = self.test,
         													nComps = nComps,
         													outcomeVariable = outcomeVariable)
         # save dim red. results (a pca can also be used to fit other unseen data)
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
         '''
         Return string
         '''
         string_length  = len(items_for_col)
         if string_length > lim:
                s = items_for_col[:lim-1]+'..'
         else:
                s = items_for_col
         return s


     def summarize(self):
     	'''
     	Summarize Table
     	'''
     	selectedColumns = self.selection_is_from_one_df()
     	if selectedColumns is not None:
     	
     		summarizedData = self.sourceData.get_current_data()[selectedColumns].describe(
     													percentiles = [.25, .5, .75],
     													include = 'all')
     		countNanValues = self.sourceData.get_current_data()[selectedColumns].isnull().sum()
     		summarizedData.loc['nan count',:] = countNanValues
     		summarizedData.insert(0,'Measure',summarizedData.index)
     		dataDialog = display_data.dataDisplayDialog(summarizedData,showOptionsToAddDf=True)

     		if dataDialog.addDf:
     			nameOfDf = 'Summary of {}'.format(get_elements_from_list_as_string(selectedColumns))
     			self.add_new_dataframe(dataDialog.data,nameOfDf)

     def update_size(self, selectedColumn = None):
         '''
         Change the size level in a scatter plot.
         '''

         if len(self.plt.plotHistory) == 0:
         	return False

         _, categoricalColumns, plotType, _  = self.plt.current_plot_settings
         n_categories = len(categoricalColumns)

         if plotType not in ['scatter','scatter_matrix','PCA']:
             tk.messagebox.showinfo('Error..','Only useful for scatter plots!')
             if self.size_button_droped is not None:
                 self.size_button_droped.destroy()
             return False

         if selectedColumn is None:

             selectedColumns = self.selection_is_from_one_df()
             if selectedColumns is None:
             	return
             else:
             	selectedColumn = selectedColumns[0]

         dtype = self.sourceData.df[selectedColumn].dtype
         if dtype == np.float64 or dtype == np.int64:
             if plotType == 'scatter_matrix':

             	self.plt.nonCategoricalPlotter._scatterMatrix.change_size_by_numeric_column(self.DataTreeview.columnsSelected[0])

             elif plotType in ['scatter','PCA']:
                 if self.plt.nonCategoricalPlotter is not None:
                 	self.plt.nonCategoricalPlotter.change_size_by_numerical_column(self.DataTreeview.columnsSelected,update=False)
                 else:
                 	self.plt.categoricalPlotter.scatterWithCategories.change_size_by_numerical_column(self.DataTreeview.columnsSelected[0])

             self.plt.redraw()
             return

         else:


             if plotType in ['scatter','PCA']:
                 if n_categories == 0:
                 	self.plt.nonCategoricalPlotter.change_size_by_categorical_column(self.DataTreeview.columnsSelected,update=False)
                 else:
                 	self.plt.categoricalPlotter.scatterWithCategories.change_size_by_categorical_columns(self.DataTreeview.columnsSelected[0])
                 self.plt.redraw()




     def update_color(self, columnNames = None, numericData = None):
         '''
         Change color. Adds a color level depending on the sued column data type.
         '''


         if columnNames is None:
         	 columnNames = self.selection_is_from_one_df()
         	 if columnNames is None:
         	 	return
         if 'color' in columnNames:
         	tk.messagebox.showinfo('Error..',
         		'Please do not used "color" and "size" as column names because they will be overwritte if you add a color/hue level.',
         		parent=self)
         	return        
         if numericData is None:
         	numericData = self.DataTreeview.onlyNumericColumnsSelected
                  
         if numericData:
         	if self.plt.currentPlotType in ['scatter','PCA']:
         		if self.plt.nonCategoricalPlotter is not None:
         			self.plt.nonCategoricalPlotter.change_color_by_numerical_column(columnNames, update= False)
         		else:
         			self.plt.categoricalPlotter.scatterWithCategories.change_color_by_numerical_column(columnNames, updateColor = False)
         	elif self.plt.currentPlotType == 'scatter_matrix':
         		self.plt.nonCategoricalPlotter._scatterMatrix.change_color_by_numeric_column(columnNames, updateColor = False)
         	elif self.plt.currentPlotType == 'cluster_analysis':
         		self.plt.nonCategoricalPlotter.clustPlot.change_color_by_numerical_column(columnNames)         	
         	elif self.plt.currentPlotType == 'line_plot':
         		self.plt.nonCategoricalPlotter.linePlotHelper.change_color_by_numerical_column(columnNames, updateColor=False)
         else:
         	if self.plt.currentPlotType == 'scatter_matrix':
         		self.plt.nonCategoricalPlotter._scatterMatrix.change_color_by_categorical_column(columnNames)
         	elif self.plt.currentPlotType == 'cluster_analysis':
         		self.plt.nonCategoricalPlotter.clustPlot.change_color_by_categorical_columns(columnNames)
         		self.interactiveWidgetHelper.clean_color_frame_up()
         		self.interactiveWidgetHelper.create_widgets(plotter = self.plt, analyzeData = self,
         													droppedButton = self.color_button_droped )

         	elif self.plt.currentPlotType == 'line_plot':
         		self.plt.nonCategoricalPlotter.linePlotHelper.change_color_by_categorical_columns(columnNames, updateColor=False)
         		self.interactiveWidgetHelper.clean_color_frame_up()
         		self.interactiveWidgetHelper.create_widgets(plotter = self.plt, analyzeData = self,
         									droppedButton = self.color_button_droped)
         	elif self.plt.currentPlotType in ['PCA','scatter']:
         		if self.plt.nonCategoricalPlotter is not None:
         			self.plt.nonCategoricalPlotter.change_color_by_categorical_columns(columnNames, updateColor=False)
         		else:
         			self.plt.categoricalPlotter.scatterWithCategories.change_color_by_categorical_columns(columnNames, updateColor=False)
         		
         		self.interactiveWidgetHelper.clean_color_frame_up()
         		self.interactiveWidgetHelper.create_widgets(plotter = self.plt, analyzeData = self,
         													droppedButton = self.color_button_droped )

         self.plt.redraw()
         return True

     def is_just_outside(self, event):
        '''
        Handle events that are close to the axis.
        Allows to center and log-scale specific axis.
     	'''
        fig = self.plt.figure
        if len(fig.axes) == 0:
        	return
        if len(self.original_vals) == 0:
                 self.original_vals = [(ax.get_ylim(),ax.get_xlim()) for ax in fig.axes]# [0].get_ylim(),fig.axes[0].get_xlim()]
                 self.center_x = False
                 self.center_y = False

        if len(fig.axes) == 0:
            return

        x,y = event.x, event.y
        for n,ax in enumerate(fig.axes):
            xAxes, yAxes =  ax.transAxes.inverted().transform([x, y])
            if (-0.20 < xAxes < -0.01):
                if event.dblclick and event.button == 1:

                	if self.center_y:
                		lim = [x[0] for x in self.original_vals]
                		self.center_y = False
                		self.plt.change_limits_for_axes(limits=lim,which='y')
                		
                	else:
                		ymin, ymax  = self.original_vals[n][0]
                		maxValue = max(abs(ymin),abs(ymax))
                		lim = (-maxValue, maxValue )
                		self.center_y = True
                		self.plt.center_axes('y',lim)
                	break

                elif event.button == 3:

                     self.plt.log_axes(which='y')
                
            elif  (-0.20 < yAxes < -0.01):
				
                if event.dblclick and event.button == 1:
                	if self.center_x:
                		lim = [x[1] for x in self.original_vals]
                		self.center_x = False
                		self.plt.change_limits_for_axes(limits=lim,which='x')
                	else:
                		xmin, xmax  = self.original_vals[n][1]
                		maxValue = max(abs(xmin),abs(xmax))
                		lim = (-maxValue ,maxValue )
                		self.center_x = True
                		self.plt.center_axes('x',lim)
                	break	
                	

                elif event.button == 3:
                     axes  = fig.axes
                     self.plt.log_axes(which='x')
        
        self.plt.redraw()
             


     def define_some_stuff_before_plot(self):


             self.center_x = False
             self.center_y = False
             self.log_x = False
             self.log_y = False

             self.label_button_droped = None
             self.tooltip_button_droped = None
             self.size_button_droped = None
             self.color_button_droped = None
             self.stat_button_droped = None


     def prepare_plot(self, colnames = [], catnames = [], plot_type = "" , cmap_ = ''):
         '''
         Prepare the plot using the plotter (self.plt)
         '''
         n_cols = len(colnames)
         n_categories = len(catnames)

         if self.twoGroupstatsClass is not None:
         	self.twoGroupstatsClass.disconnect_event()
         	del self.twoGroupstatsClass
         	self.twoGroupstatsClass = None

         if True:

             self.colormaps.clear()
             self.remove_mpl_connection(plot_type = plot_type)
             self.selection_press_event = None
             self.original_vals = []
             self.define_some_stuff_before_plot()
             gc.collect()

             if cmap_ == '':
                 cmap_ = self.cmap_in_use.get()


             if n_cols == 0  and n_categories == 0:

                 self.plt.clean_up_figure()
                 self.plt.redraw()
                 return


             if catnames != []:
                 if any(len(self.sourceData.get_unique_values(cat)) > 100 for cat in \
                 catnames if cat in self.sourceData.get_columns_of_current_data()) \
                 and plot_type not in ['hclust']:

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

         elif plot_type == 'scatter' and n_categories >= 0 and n_cols  > 0:
          	        	
          	self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
          	return

         elif plot_type == 'display_fit':


                 curve_fitting.displayCurveFitting(self.sourceData,
                 								   self.plt,self.curveFitCollection)
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


         elif plot_type in ['hclust','corrmatrix']:

             if n_cols == 1:
                 tk.messagebox.showinfo('Error..','You need at least 2 numeric columns for this plot type.')
                 return
             else:
             		self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
             		return

         elif plot_type == 'scatter_matrix' and n_cols > 1:


             self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)


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

                     self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                     return

                 elif n_cols > 0 and n_categories > 1:

                     self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                     return

                 else:
                     messagebox.showinfo('Error..','Not yet supported.')

         elif  plot_type == 'pointplot' :

             self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)


         else:

             if n_cols == 2:

                 self.plt.initiate_chart(colnames,catnames,plot_type,cmap_)
                 return    		
		  
		  
		 


     def make_labels_selectable(self):
		
         self.stop_selection(replot = False)

         if len(self.plt.plotHistory) == 0:
         	return
         plotExporter =  self.plt.get_active_helper()
         plotExporter.bind_label_event(self.anno_column)
    


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
             self.fire = 0
             self.artist_list = list()
             self.canvas = FigureCanvasTkAgg(f1,self.frame)
             f1.canvas.mpl_connect('button_press_event',lambda e: self.export_selected_figure(e))
             if hasattr(self.canvas,'show'):
             	self.canvas.show()
             elif hasattr(self.canvas,'draw'):
             	self.canvas.draw()
             	
             self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.frame)
             self.canvas.get_tk_widget().pack(in_=self.frame,
                                                 side="top",fill='both',expand=True)
             self.canvas._tkcanvas.pack(in_=self.frame,
                                                 side="top",fill='both',expand=True)


     
    
     def get_selected_data(self):
     	
         		
         	dfsSelected = self.DataTreeview.get_data_frames_from_selection()
         	if len(dfsSelected) == 1:
         			
         			data = self.sourceData.get_data_by_id(dfsSelected[0])
         			return data
         		
         	else:
         			tk.messagebox.showinfo('Error ..',
         				'More than one data frame selected. Please select only one.',
         				parent=self)
         			return
	





     def open_color_configuration(self):
     	'''
     	Opens a dialog window to change color/alpha settings.
     	'''
     	color_configuration.colorChooseDialog(self.colorHelper,self,
     			self.cmap_in_use.get(),float(self.alpha_selected.get()))


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

             size = self.settings_points['sizes'][0]
             size_handle = size_configuration.SizeConfigurationPopup(self.plt)
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

         center(popup,size=(w,h))
         if 'excel' in mode or 'add_error' in mode or 'choose subject column' in mode or 'choose subject and repeated measure column' in mode or 'Color' in mode:
              self.wait_window(popup)


     def open_label_window(self):
         '''
         Opens a popup dialog to annotate desired rows.
         '''
         if len(self.plt.plotHistory) > 0:
             if self.plt.currentPlotType not in ['scatter','PCA','cluster_analysis']:
                     return
             self.categorical_column_handler('Annotate scatter points')

     def reset_dicts_and_plots(self):
        '''
        Resetting dicts.
        '''
        self.source_treeview.delete(*self.source_treeview.get_children())
        self.clean_up_dropped_buttons(replot=False)
        self.but_stored[9].configure(image= self.add_swarm_icon)
        self.swarm_but = 0
        self.add_swarm_to_new_plot = False
        self.remove_mpl_connection()
        self.performed_stats.clear()
        self.plt.clean_up_figure()
        self.plt.redraw()


     def source_file_upload(self, pathUpload = None, resetTreeEntries = True, 
     						mergeMultipleFiles = False, loadTDTTanks = False):
          """Upload file, extract data types and insert columns names into the source data tree"""
          
          if resetTreeEntries and len(self.sourceData.dfs) != 0:
          		quest = tk.messagebox.askquestion('Note ..',
          						'This will remove all loaded data. Proceed? ',
          						icone = None)
          		if quest != 'yes':
          			return
          			
          if pathUpload is None:
          	
              pathUpload = tf.askopenfilename(initialdir=self.pathUpload,
                                                         title="Choose File")
             
              if pathUpload == '':
                  return
              else:
                   self.pathUpload = pathUpload            
          
          fileName = pathUpload.split('/')[-1]
          matplotlib.rcParams['savefig.directory'] = pathUpload
          
          if mergeMultipleFiles:
          	
          	fileImporter = txt_file_importer.multipleTxtFileLoader() 
          	
          	uploadedDataFrame, fileName , naString = fileImporter.get_results()
          	if len(uploadedDataFrame) == 0:
          		return

          elif loadTDTTanks:
          	
          	uploadedDataFrame, fileName ,naString = self.import_TDT()
          	if uploadedDataFrame is None:
          		return

          elif '.xsl' in fileName or '.xlsx' in fileName:

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

          
          elif '.xml' in fileName:
          
          	fileImporter = txt_file_importer.xmlImporter(pathUpload)
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
          	
          	self.workflow.clear()
          	
          	self.sourceData = data.DataCollection(self.workflow)
          	self.workflow.add_handles(self.sourceData)          	
          	self.plt.clean_up_figure()
          	self.plt.redraw()


          if isinstance(uploadedDataFrame,dict):
          		## if users selected all sheets in a Excel file.
          	for sheetName, dataFrame in uploadedDataFrame.items():

			
          		id = self.sourceData.get_next_available_id()
          		fileName = '{}_{}'.format(sheetName,fileName)
          		self.sourceData.add_data_frame(dataFrame, id = id, fileName = fileName)
          		self.sourceData.set_current_data_by_id(id)
          		self.update_all_dfs_in_treeview()
          		objectColumnList = self.sourceData.get_columns_data_type_relationship()['object']
          		self.sourceData.fill_na_in_columnList(objectColumnList,naString)
          	
          	self.update_all_dfs_in_treeview()
			
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
          	self.set_up_plotter_and_workflow()
			
     def set_up_plotter_and_workflow(self):
     	'''
     	'''
     	self.plt = plotter._Plotter(self.sourceData,self.f1, self.workflow)
     	self.workflow.add_handles(plotter = self.plt)
     	self.plt.set_auc_collection(self.aucResultCollection)
     	self.plt.set_scatter_point_properties(GREY,round(float(self.alpha_selected.get()),2),
          								int(float(self.size_selected.get())))
     	for boolVar in ['plotCumulativeDist','circulizeDendrogram','showCluster']:
     		bool = getattr(self,boolVar).get()
     		setattr(self.plt,boolVar,bool)
			
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




     def scale_icons_to_small(self,icon_ = "NORM"):

            global NORM_FONT

            if icon_ is not None:
            ## data / session load and save
                if icon_ == 'NORM':
                    #self.uploadFrameButtons['upload'].configure(image=self.open_file_icon_norm)
                    #self.uploadFrameButtons['saveSession'].configure(image=self.save_session_icon_norm)
                    #self.uploadFrameButtons['openSession'].configure(image=self.open_session_icon_norm )
                    #self.uploadFrameButtons['addData'].configure(image=self.add_data_icon_norm)

                    #self.sliceMarkFrameButtons['size'].configure(image = self.size_icon_norm)
                    #self.sliceMarkFrameButtons['filter'].configure(image = self.filter_icon_norm)
                    #self.sliceMarkFrameButtons['color'].configure(image = self.color_icon_norm)
                    #self.sliceMarkFrameButtons['label'].configure(image = self.label_icon_norm)
                    #self.sliceMarkFrameButtons['tooltip'].configure(image = self.tooltip_icon_norm)
                    #self.sliceMarkFrameButtons['selection'].configure(image = self.selection_icon_norm)
                    
                   # self.settingButton.configure(image = self.setting_icon_norm)
                   # self.workflowButton.configure(image=self.workflow_icon_norm)

                  #  self.but_col_icon = self.but_col_icon_norm
                  #  self.but_size_icon = self.but_size_icon_norm
                  #  self.but_tooltip_icon = self.but_tooltip_icon_norm
                  #  self.but_label_icon = self.but_label_icon_norm
                  #  self.but_stat_icon = self.but_stat_icon_norm


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


                    if platform in ['LINUX','WINDOWS']:
                        	NORM_FONT   = (defaultFont, 8)
                    else:
                        	NORM_FONT =  (defaultFont,11)

                    ###### LIST FOR PLOT OPTIONS
                    icon_list = [self.line_icon,self.point_plot_icon,self.scatter_icon,self.time_series_icon ,
                    			self.matrix_icon,self.dist_icon,self.barplot_icon,
                    			self.box_icon,self.violin_icon, self.swarm_icon ,self.add_swarm_icon,
                    			self.hclust_icon,self.corr_icon,self.chord_icon,self.config_plot_icon]
                                 
                                 
                    for i, icon in enumerate(icon_list):
                        self.but_stored[i].configure(image = icon)

                    self.main_fig.configure(image=self.main_figure_icon)
                    self.data_button.configure(image = self.streteched_data)
                    self.mark_sideframe .configure(pady=2,padx=1)


                    self.sideframe_upload.configure(pady=2)

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


     def get_images(self):
           '''
           Images are stored in base64 code in the module 'images'.
           Here we get all the button icons that are used in the analyze_data.py
           frame page.
           '''
          # self.size_icon, self.color_icon, self.label_icon, \
           #				self.filter_icon, self.selection_icon, self.tooltip_icon  = images.get_slice_and_mark_images()

           self.size_icon, self.color_icon, self.label_icon, \
           				self.filter_icon, self.selection_icon, self.tooltip_icon  = images.get_slice_and_mark_images_norm()

           #self.open_file_icon,self.save_session_icon,self.open_session_icon ,self.add_data_icon   = images.get_data_upload_and_session_images()

           self.back_icon, self.center_align,self.left_align,self.right_align, \
           					_, self.config_plot_icon = images.get_utility_icons()

           #self.box_icon,self.barplot_icon,self.scatter_icon,self.swarm_icon,self.time_series_icon\
           	#				,self.violin_icon,self.hclust_icon,self.corr_icon,self.point_plot_icon, \
           	#				self.matrix_icon, self.dist_icon, self.add_swarm_icon_,self.remove_swarm_icon_,\
           	#				self.setting_icon, self.line_icon  =   images.get_plot_options_icons()

           self.box_icon,self.barplot_icon ,self.scatter_icon ,self.swarm_icon ,self.time_series_icon \
           					,self.violin_icon ,self.hclust_icon ,self.corr_icon ,self.point_plot_icon , \
           					self.matrix_icon , self.dist_icon, self.add_swarm_icon , self.remove_swarm_icon, \
           					self.setting_icon, self.line_icon, self.chord_icon    = images.get_plot_options_icons_norm()

           self.open_file_icon,self.save_session_icon,self.open_session_icon,self.add_data_icon =  images.get_norm_data_upload_and_session_images()

           _ ,self.streteched_data  = images.get_data_images()

           self.delete_all_cols, self.delete_all_cols_norm, self.resortColumns = images.get_delete_cols_images()

           self.right, self.outer, self.left, self.inner = images.get_merge_images()
           self.mergeImages = {'right':self.right,'left':self.left,
						'outer':self.outer,'inner':self.inner}

           _, self.but_col_icon, _, \
        					self.but_label_icon,  _,self.but_size_icon, \
           					_, self.but_stat_icon, _, self.but_tooltip_icon =  images.get_drop_button_images()


           _, self.main_figure_icon = images.get_main_figure_button_images()
           _, self.workflow_icon = images.get_workflow_button_images()

           #self.but_col_icon = self.but_col_icon_
           #self.but_size_icon = self.but_size_icon_
           #self.but_tooltip_icon = self.but_tooltip_icon_
           #self.but_label_icon = self.but_label_icon_
           #self.but_stat_icon = self.but_stat_icon_
           #self.add_swarm_icon = self.add_swarm_icon_
           #self.remove_swarm_icon = self.remove_swarm_icon_
     
     def import_TDT(self,event = None):
     	
     	dir = tf.askdirectory()
     	if dir != '':
     		
     		importer = import_TDT.TDTTankToPandas(dir)
     	
     		data, columns, timeColumn, tankName = importer.get_raw_signal()
     	
     		df = pd.DataFrame(data, columns = columns)
     		df['time [sec]'] = np.linspace(0,timeColumn, num = len(df.index), endpoint = True)
     	
     		return df, 'TDTtank_.{}'.format(tankName) ,'-'
     	else:
     		return None, None, None
     	#self.add_new_dataframe(df,'TDTtank')
     
     def paste_file_to_clipboard(self,event = None):
     	'''
     	Reads clipboard and paste the data into Instant Clue.
     	'''
     	df = pd.read_clipboard()
     	if df.empty:
     		return
     	if len(self.sourceData.dfs) == 0:
     		self.set_up_plotter_and_workflow()
     	self.add_new_dataframe(df,'fromClipboard')
     	self.update_idletasks()
     	tk.messagebox.showinfo('Copied.',
     		'Data from clipboard copied and added.\nAt the moment only the dafault settings can be used.')
	
     def get_update_text(self):
     	'''
     	Read update data from text from instant clue webserver
     	'''
     	try:
     		text = 'None'
     		link = paperUrl
     		#try:
     		data = urlopen("http://www.instantclue.uni-koeln.de/updates.txt")
     		for n,line in enumerate(data):
     			#line = str(line).replace('\b','').replace('\n','').replace('\r','')
     			line = line.decode("utf-8")
     			if n == 0:
     				text = line.replace('\r','').replace('\n','')
     			elif n == 1:
     				link = line.replace('\r','').replace('\n','')
     			else:
     				break
     		return text, link
     	except:
     		return 'No internet connection' , 'http://www.instantclue.uni-koeln.de'

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
           
           text,link = self.get_update_text()
           
           labelUpdate = tk.Label(self, text = ':: Updates: {} ::'.format(text),
           						font=LARGE_FONT, fg = '#420d09', justify = tk.CENTER, bg = MAC_GREY)
           
           labelUpdate.grid(row=0,pady=5,padx=20,column=2,columnspan=2)#, sticky=tk.W)
           labelUpdate.bind('<Button-1>',lambda event, link = link: webbrowser.open(link))
           make_label_button_like(labelUpdate)
           

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
           						 ('addData',[self.add_data_icon,lambda: self.source_file_upload(resetTreeEntries = len(self.sourceData.dfs) == 0,)]),
           						 ('saveSession',[self.save_session_icon,self.save_current_session]),
           						 ('openSession',[self.open_session_icon,self.open_saved_session]),
           						 ])

           for key,values in imagesAndFunctionsUpload.items():
           		button = create_button(self.sideframe_upload, image = values[0], command = values[1])
           		self.uploadFrameButtons[key] = button
           
           self.uploadFrameButtons['upload'].bind(right_click, lambda event: self.post_menu(menu = self.menuCollection['addFiles']))

           ## mark/slice frame buttons
           self.sliceMarkFrameButtons = OrderedDict()
           self.framesSliceMarks = OrderedDict()
           imageAndFunctionsSliceMark = OrderedDict([('size',[self.size_icon,lambda: self.design_popup(mode = 'Size setting')]),
           											 ('color',[self.color_icon,self.open_color_configuration]),
           											 ('label',[self.label_icon,self.open_label_window]),
           											 ('tooltip',[self.tooltip_icon,'']),
           											 ('selection',[self.selection_icon,self.select_data]),
           											 ('filter',[self.filter_icon,self.clip_data])
           											])

           n=0
           for key,values in imageAndFunctionsSliceMark.items():
           		buttonFrame = tk.Frame(self.mark_sideframe, bd = 1, bg = MAC_GREY)
           		
           		button = create_button(buttonFrame, image = values[0], command = values[1])
           		button.grid()
           		CreateToolTip(button,text=sliceMarksTooltip[key])
           		n+=1
           		
           		# save buttons to change icons etc 
           		self.sliceMarkFrameButtons[key] = button
           		self.framesSliceMarks[key] = buttonFrame

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
           											             
           self.resort_numericColumns = create_button(self.column_sideframe,
           											  image = self.resortColumns,
           											  command = lambda: self.resort_columns_in_receiver_box(mode = 'num'))
           
           self.resort_categoricalColumns = create_button(self.category_sideframe,
           											  image = self.resortColumns,
           											  command = lambda: self.resort_columns_in_receiver_box(mode = 'cat'))                      

		   
           for but in [self.delete_all_button_cat,self.delete_all_button_num]:
           	CreateToolTip(but,text='Deletes all items from this receiver box. To delete single items, use right-click.')		   	
		   
           for but in [self.resort_numericColumns,self.resort_categoricalColumns]:
           	CreateToolTip(but,text='Opens a dialog to re-sort items in receiver boxes.')		   	
		   		   


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
           								command =  lambda: controller.show_frame())

           back_button.grid( in_=self,
                                         row=0,
                                         column = 5,
                                         rowspan=15,
                                         sticky=tk.N+tk.E,
                                         padx=4)


           chartTypes = ['line_plot','pointplot','scatter','time_series','scatter_matrix',
           		'density','barplot','boxplot','violinplot', 'swarm','add_swarm',
           		'hclust','corrmatrix','chord','configure']
           tooltip_info = tooltip_information_plotoptions
           # we are using the icon in desired order to create plot/chart options
           iconsForButtons = [
                                    self.line_icon,
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
                                    self.corr_icon,
                                    self.config_plot_icon]
           i = 0
           self.but_stored = []

           for n, buttonIcon in enumerate(iconsForButtons):

            	chartType = chartTypes[n]
            	if chartType in ['boxplot','hclust','configure',
            					'barplot','corrmatrix']:
            		pady = (5,1)
            	else:
            		pady = 1

            	if chartType == 'configure':

					
            		commandButton = self.configure_chart
            		ttk.Separator(self.plotoptions_sideframe,orient=tk.HORIZONTAL).grid(columnspan=2,sticky=tk.EW+tk.N,pady=(8,0))
            		i += 1
            	else:
            		commandButton = lambda plotChartType = chartType: self.change_plot_style(plot_type = plotChartType)

            	chartButton = create_button(self,
            								command = commandButton)

            	text, title = tooltip_info[n]

            	CreateToolTip(chartButton, text  = text, title_ = title,   wraplength=230, bg ="white")

            	self.but_stored.append(chartButton)
            	if chartType in ['density','barplot','violinplot','boxplot','swarm']:
            		chartButton.bind(right_click, 
            					lambda event: self.post_menu(event,self.menuCollection['split_sub'] ))

            	elif chartType in ['hclust','line_plot','corrmatrix','scatter']:
            		chartButton.bind(right_click, lambda event, chartType = chartType\
            		: self.post_menu(event,self.menuCollection[chartType]))

            	if (n & 1 and chartType not in ['configure','hclust']) or chartType == 'corrmatrix':
            		columnPos = 1
            	else:
            		columnPos = 0
            	if chartType == 'hclust':
            		i += 1
            	chartButton.grid(in_ = self.plotoptions_sideframe, row = i ,column = columnPos, pady=pady)
            	if columnPos == 1:
           			i += 1
           ttk.Separator(self.plotoptions_sideframe,orient=tk.HORIZONTAL).grid(columnspan=2,sticky=tk.EW+tk.N,pady=(7,0))
           self.main_fig.grid(in_=self.plotoptions_sideframe,pady=(9,0))
           CreateToolTip(self.main_fig,title_ = 'Main Figure Templates',
           								text = 'Felxible combination of multiple subplots.')
           								
           	
           self.workflowButton = create_button(self.plotoptions_sideframe,image = self.workflow_icon, command = lambda: self.workflow.show())
           self.workflowButton.grid(pady=(9,0))

           self.settingButton = create_button(self.general_settings_sideframe,
           		image = self.setting_icon,
           		command = lambda: settings.settingsDialog(self.plt, self.colorHelper,self.dimensionReductionCollection)) #tk.messagebox.showinfo('Under construction ..',
           			#'Under construction. Will allow to change settings on how charts are generated. (Error bar configuration etc)',
           			#parent=self))

           self.settingButton.grid(padx=2,pady=2)
           CreateToolTip(self.settingButton, 
           					title_ = 'Settings', 
           					text = 'Change General Settings.\n'+
           					'* Error bar calculations\n* Binned Scatter Plot Settings'+
           					'* Hierarchical Clustering (set metric to None to prevent clustering)\n'+
           					'* Dimensional Reduction Settings')
           					
           style_tree = ttk.Style(self)

           if platform in ['LINUX','WINDOWS']:
           
               style_tree.configure('source.Treeview', rowheight = 19, font = (defaultFont,8))
           else:
               style_tree.configure('source.Treeview', rowheight = 21, font = (defaultFont,11))

           self.source_treeview = ttk.Treeview(self.source_sideframe, height = "4", 
           										show='tree', style='source.Treeview')
           ## make the source treeview part of the sourceDataTreeview class that detects dataframes selected,
           ## data types selected, and handles adding new columns, as well as sorting
           self.DataTreeview = sourceDataTreeView.sourceDataTreeview(self.source_treeview)
           self.source_treeview.heading("#0", text="")
           self.source_treeview.column("#0",minwidth=800)
           
           self.source_treeview.bind("<B1-Motion>", self.on_motion)
           self.source_treeview.bind("<ButtonRelease-1>", self.release)
           self.source_treeview.bind("<BackSpace>", self.delete_column)
           self.source_treeview.bind("<Delete>", self.delete_column)
           self.source_treeview.bind("<Escape>", self.DataTreeview.deselect_all_items)
           self.source_treeview.bind("<Double-Button-1>", lambda event : self.rename_columns(event=event))
           

           if platform == 'MAC':

           		self.source_treeview.bind('<Command-c>', lambda event: self.copy_file_to_clipboard(fromSelection = True))
           		self.source_treeview.bind('<Command-v>', lambda event: self.paste_file_to_clipboard())
           		self.source_treeview.bind('<Command-f>', lambda event: self.categorical_column_handler('Search string & annotate'))
           		self.source_treeview.bind('<Command-r>', lambda event: findAndReplace.findAndReplaceDialog(dfClass = self.sourceData, dataTreeview = self.DataTreeview))
           		self.source_treeview.bind('<Command-n>', lambda event: self.numeric_filter_dialog())
           		self.source_treeview.bind('<Command-m>', lambda event: self.combine_selected_columns())
           		self.source_treeview.bind('<Command-z>', lambda event: self.workflow.undo(event))

           elif platform in ['LINUX','WINDOWS']:
           
           		self.source_treeview.bind('<Control-c>', lambda event: self.copy_file_to_clipboard(fromSelection = True))
           		self.source_treeview.bind('<Control-f>', lambda event: self.categorical_column_handler('Search string & annotate'))
           		self.source_treeview.bind('<Control-r>', lambda event: findAndReplace.findAndReplaceDialog(dfClass = self.sourceData, dataTreeview = self.DataTreeview))
           		self.source_treeview.bind('<Control-n>', lambda event: self.numeric_filter_dialog())
           		self.source_treeview.bind('<Control-m>', lambda event: self.combine_selected_columns())
           		self.source_treeview.bind('<Control-z>', lambda event: self.workflow.undo(event))
           		self.source_treeview.bind('<Control-v>', lambda event: self.paste_file_to_clipboard())

           self.source_treeview.bind(right_click, self.post_treeview_menu)


           sourceScroll = ttk.Scrollbar(self, orient = tk.HORIZONTAL, command = self.source_treeview.xview)
           sourceScroll2 = ttk.Scrollbar(self,orient = tk.VERTICAL, command = self.source_treeview.yview)
           self.source_treeview.configure(xscrollcommand = sourceScroll.set,
                                          yscrollcommand = sourceScroll2.set)

           self.build_analysis_tree()

           self.data_button = create_button(self.source_sideframe,
           						image = self.streteched_data,
           						command = self.show_data)#
           CreateToolTip(self.data_button,text='Opens a dialog window to inspect raw data. If multiple data frames were loaded into Instant Clue you will have to select the target data frame before by clicking on any item.')
			
           padDict = dict(padx=8,pady=1.5)
           

           self.framesSliceMarks['filter'].grid(in_=self.mark_sideframe, row=2, column = 1 , sticky=tk.W, **padDict)
           self.framesSliceMarks['selection'].grid(in_=self.mark_sideframe, row=2, column = 0, **padDict)
           self.framesSliceMarks['size'].grid(in_=self.mark_sideframe, row=5 , column=1, sticky=tk.W,**padDict)
           self.framesSliceMarks['color'].grid(in_=self.mark_sideframe, row=5 , column=0, **padDict)
           self.framesSliceMarks['label'].grid(in_=self.mark_sideframe, row=6 , column=0, **padDict)
           self.framesSliceMarks['tooltip'].grid(in_=self.mark_sideframe, row=6 , column=1, sticky=tk.W,**padDict)


           sep_marks1.grid(in_=self.mark_sideframe, row=1, column = 0 , columnspan=2, sticky = tk.EW)
           self.label_marks1.grid(in_=self.mark_sideframe, row=0, column = 0 , sticky = tk.W)
           sep_marks2.grid(in_=self.mark_sideframe, row=4, column = 0 , columnspan=2, sticky = tk.EW)
           self.label_marks2.grid(in_=self.mark_sideframe, row=3, column = 0 , sticky = tk.W)
           self.delete_all_button_num.pack(in_=self.column_sideframe,   anchor=tk.NE, side=tk.RIGHT)
           self.delete_all_button_cat.pack(in_=self.category_sideframe,   anchor=tk.NE, side=tk.RIGHT)
           self.resort_numericColumns.pack(in_=self.column_sideframe,anchor=tk.NW, side = tk.LEFT)
           self.resort_categoricalColumns.pack(in_=self.category_sideframe,anchor=tk.NW, side = tk.LEFT)
           
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

           for column,(key,button) in enumerate(self.uploadFrameButtons.items()):
           		if column == 2:
           			ttk.Separator(self.sideframe_upload,
           				orient=tk.VERTICAL).grid(row=1,column=2,sticky=tk.NS+tk.W,padx=6)
           			pad_ = (16,4)
           		else:
           			pad_ = 4
           		CreateToolTip(button,text=loadButtonTooltip[key])
           		button.grid(in_=self.sideframe_upload, row=1, column=column, padx=pad_)
           
           		
           self.data_button.pack(in_= self.source_sideframe,  pady=2, anchor=tk.W)
           sourceScroll2.pack(in_= self.source_sideframe, side = tk.LEFT, fill=tk.Y, anchor=tk.N)
           self.source_treeview.pack(in_= self.source_sideframe, padx=0, expand=True, fill=tk.BOTH, anchor = tk.NE)
           sourceScroll.pack(in_= self.source_sideframe, padx=0,anchor=tk.N, fill=tk.X)

           intWidgets = [self.tx_space,self.cat_space,self.color_button_droped,
           		self.column_sideframe,self.category_sideframe] + self.sliceMarkButtonsList

           floatWidgets = [self.tx_space,self.column_sideframe] + self.sliceMarkButtonsList

           objectWidgets = [self.cat_space,self.color_button_droped,self.category_sideframe] + self.sliceMarkButtonsList

           self.dataTypeSpecWidgets = {'float64':floatWidgets,
           							   'int64':intWidgets,
           							   'object':objectWidgets}


           self.frame.grid(in_=self,
                                     row=5,
                                     column =3,
                                     rowspan=25,
                                     pady=(90,20),
                                     sticky=tk.NSEW,
                                     padx=5)
