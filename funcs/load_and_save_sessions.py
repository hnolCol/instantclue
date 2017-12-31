# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:42:40 2017

@author: hnolte-101
"""
import os
import pandas as pd
import tkinter as tk
import tkinter.filedialog as tf
import tkinter.simpledialog as ts
import pickle
import time



def save_plotter(plotter):
	with open('chart_param','wb') as file:
         	pickle.dump(plotter, file)
	with open('chart_param','rb') as file:
		pull = pickle.load(file)
		print(pull)
	return pull
    
         
    
         	
         	




def save_session(source_data,path_file, dict_with_plots,associated_stats, sub_data, dict_with_file_names, marks_done_dict, dict_saving_main_figures_axes,dict_saving_axes_in_main_figures,global_chart_parameter, dict_for_fits,regression_in_plot,
					global_annotation_dict, annotations_dict):
         lines = ts.askstring('Provide session name',prompt = 'Session name: ', initialvalue= time.strftime("%d_%m_%Y"))
         session_path = os.path.join(path_file,'Data','stored_sessions',lines)
         if not os.path.exists(session_path):
             os.makedirs(session_path)
         else:
             quest = tk.messagebox.askquestion('Path exists already...','Session folder exists already. Overwrite content?')
             if quest == 'yes':
                 pass
             else:
                 lines = ts.askstring('Provide session name',prompt = 'Session name: ', initialvalue= time.strftime("%d_%m_%Y"))
                 session_path = os.path.join(path_file,'Data','stored_sessions',lines)
                   
         
         for idx,data in sub_data.items():
             #print(data)
             data[0].to_csv(os.path.join(session_path,idx+'.gzip'), compression = 'gzip', index = False, na_rep = 'NaN')
             
         #print(dict_with_file_names)
         #print(sub_data)
         print(dict_with_plots)
         #for key,fig in main_figures_created.items():
        
         with open(os.path.join(session_path,'chart_param'),'wb') as file:
             pickle.dump(global_chart_parameter, file)
         with open(os.path.join(session_path,'file_names'),'wb') as file:
             pickle.dump(dict_with_file_names, file)
         with open(os.path.join(session_path,'prepared_plots'),'wb') as file:
             pickle.dump(dict_with_plots, file)
         with open(os.path.join(session_path,'performed_stats'),'wb') as file:
             pickle.dump(associated_stats,file)
         with open(os.path.join(session_path,'marks_done'),'wb') as file:
             pickle.dump(marks_done_dict,file)
         with open(os.path.join(session_path,'main_figure_plots'),'wb') as file:
             pickle.dump(dict_saving_main_figures_axes,file)    
         with open(os.path.join(session_path,'main_figure_axes'),'wb') as file:
             pickle.dump(dict_saving_axes_in_main_figures ,file)        
         with open(os.path.join(session_path,'dic_for_fits'),'wb') as file:
             pickle.dump(dict_for_fits  ,file) 
         with open(os.path.join(session_path,'regressions'),'wb') as file:
             pickle.dump(regression_in_plot  ,file)        
         with open(os.path.join(session_path,'global_annotation_dict'),'wb') as file:
             pickle.dump(global_annotation_dict  ,file) 
         with open(os.path.join(session_path,'annotations_dict'),'wb') as file:
             pickle.dump(annotations_dict    ,file) 
          
            # self.regression_in_plot
           
def open_session(path_file):
         sub_data = dict() 
         directory = tf.askdirectory(initialdir = os.path.join(path_file,'Data','stored_sessions'), title ="Choose saved session")
         source_ = ''
         
                             
         
         
         
         with open(os.path.join(directory,'prepared_plots'),'rb') as file:
             pull = pickle.load(file)
         
         
         with open(os.path.join(directory,'performed_stats'),'rb') as file:
             pull1 = pickle.load(file)     
         with open(os.path.join(directory,'file_names'),'rb') as file:
             pull2 = pickle.load(file)     
         with open(os.path.join(directory,'marks_done'),'rb') as file:
             pull3 = pickle.load(file)         
         with open(os.path.join(directory,'main_figure_plots'),'rb') as file:
             pull4 = pickle.load(file) 
         with open(os.path.join(directory,'main_figure_axes'),'rb') as file:
             pull5 = pickle.load(file)
         with open(os.path.join(directory,'chart_param'),'rb') as file:
             pull6 = pickle.load(file)
         with open(os.path.join(directory,'dic_for_fits'),'rb') as file:
             pull7 = pickle.load(file) 
         with open(os.path.join(directory,'regressions'),'rb') as file:
             pull8 = pickle.load(file) 
         with open(os.path.join(directory,'global_annotation_dict'),'rb') as file:
             pull9 = pickle.load(file)          
         with open(os.path.join(directory,'annotations_dict'),'rb') as file:
             pull10 = pickle.load(file)     
         
          
         
         
         for idx,file_name in pull2.items():
             data_ =  pd.read_csv(os.path.join(directory,idx+'.gzip') , compression = 'gzip', chunksize=10000)
             chunks_listed = [] 
             
             for chunks in data_:
                            chunks_listed.append(chunks)
             data = pd.concat(chunks_listed)               
             sub_data[idx] = [data, data.columns.values.tolist()]
                     
         #print(pull2)
         print(pull)
         #print(sub_data)
         
         return sub_data, pull, pull1, pull2, pull3, pull4,pull5,pull6,pull7,pull8,pull9,pull10
     
