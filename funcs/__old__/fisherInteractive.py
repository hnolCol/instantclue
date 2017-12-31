def callback_fisher(self, verts):
        
           
        for key, item in self.sel_points_fisher.items():
    #                       
                item.remove() 
    #                        
        self.sel_points_fisher.clear() 
        
        
        p = path.Path(verts)
        ind = p.contains_points(self.data_as_tuple_for_fisher)
        if sum(ind) < 5:
            messagebox.showinfo('Error..','Less than 5 datapoints selected for fisher test.')
            del self.lasso_fisher
            self.canvas.draw_idle() 
            return
        
        
        self.data_to_test = self.source_file.iloc[ind]
        
        self.data_selection_rows.set(len(self.data_to_test.index))
        colnames = list(self.selected_columns.keys() ) 
        x_n_ = colnames[0]
        y_n_ = colnames[1]

        sc_sel = self.a[1].scatter(self.data_to_test[x_n_],self.data_to_test[y_n_], color=(255/255,42/255,36/255), s= self.scat.get_sizes()[0] ,alpha= round(float(self.alpha_selected.get()),2), edgecolor = "black", linewidth=0.3)  
        self.sel_points_fisher[sc_sel] = sc_sel
        self.canvas.draw_idle()
        del self.lasso_fisher 
        self.canvas.draw_idle()               
        


     def create_interactive_fisher_popup(self,event):
         if event.button == 3: #'button=3' in str(event):
            self.fisher_col = StringVar()
            popup = tk.Toplevel()
            popup.attributes('-topmost', True)
            self.popup_fisher_ = popup
            fisher_popup = self.create_frame(popup)
            fisher_popup.pack(expand=True, fill = tk.BOTH)
            fisher_popup.grid_columnconfigure(9, weight=1)
            fisher_popup.grid_rowconfigure(5, weight=1)
            w = 800
            h = 850
            lab = tk.Label(fisher_popup, text=''' Fisher's Exact Test ''', font =  LARGE_FONT,fg="#4C626F",
                           justify=LEFT, background = MAC_GREY)
            lab_col = tk.Label(fisher_popup, text= 'Categorical Colunn: ', background=MAC_GREY)
            lab_sep = tk.Label(fisher_popup, text='Enter Separator: ', background=MAC_GREY)
            lab_threads = tk.Label(fisher_popup, text = "Number of threads: ", background=MAC_GREY)
            lab_FDR = tk.Label(fisher_popup, text = "FDR (BH) cutoff: ", background=MAC_GREY)
            self.FDR_combo_box = ttk.Combobox(fisher_popup, exportselection = 0 , values = ['0.05','0.01','0.001'], width=6 )
            self.FDR_combo_box.set(self.fdr_fisher.get())
            sep_combo_box = ttk.Combobox(fisher_popup, exportselection = 0, values = [';',',',':','-'], width=6)
            sep_combo_box.set(';')
            row_sel = tk.Label(fisher_popup, text= "Selected Data: ", background=MAC_GREY)
            column_tree = ['Column','Cat. Term','Cat. Size','Cat. Size in Sel.','Enrichment','p-value','FDR']
            numbs_sel = ttk.Label(fisher_popup, textvariable = self.data_selection_rows)
            but = ttk.Button(fisher_popup, text="Calculate", command = lambda: self.calculate_fisher(self.fisher_col.get(), 
                                                                                                int(thread_combo.get()),
                                                                                                sep_combo_box.get(),
                                                                                                float(self.FDR_combo_box.get()), column_tree))
            obj_cols = [col for col in self.source_file_columns if self.source_file[col].dtype != np.float64 and self.source_file[col].dtype != np.int64]
            self.fisher_col.set(obj_cols[0])        
            om_menu_col = ttk.OptionMenu(fisher_popup,self.fisher_col,obj_cols[0], *obj_cols)
            thread_combo = ttk.Combobox(fisher_popup, exportselection = 0, values = list(str(x) for x in range(1,20)), width=6)
            thread_combo.set('6')  
            close_button = ttk.Button(fisher_popup, text = "Close", command = lambda: fisher_popup.destroy())
            
            lab.grid(sticky = E, padx=5,pady=2)            
            lab_col.grid(sticky = E, padx=5,pady=2)
            om_menu_col.grid(row=1, column = 1, sticky =E, padx=2,pady=2)
            numbs_sel.grid(row=0, column=4, padx=5, sticky=E)
            row_sel.grid(row=0, column=3, padx=5, sticky=E)
            lab_sep.grid(sticky = E, padx=5,pady=2)
            sep_combo_box.grid(row=2, column = 1, sticky =E, padx=2,pady=2)
            lab_threads.grid(sticky=E, padx=5,pady=2)
            thread_combo.grid(row=3, column = 1, sticky =E, padx=2,pady=2)
            lab_FDR.grid(sticky=E, padx=5,pady=2)
            self.FDR_combo_box.grid(row=4, column = 1, sticky =E, padx=2,pady=2)
            but.grid(row=4, column = 2,padx=5, pady=5,sticky=E) 
            close_button.grid(row=4, column = 3,padx=5, sticky=E, pady=5) 
            
            self.result_tree_view = ttk.Treeview(fisher_popup, 
                                                 column = column_tree,
                                                 show="headings",
                                                 height = '24')
            self.result_tree_view.bind(right_click, self.fisher_drop_down)
            self.result_tree_view.bind('<<TreeviewSelect>>', self.select_tree_fisher)
            for col in column_tree:
                
               self.result_tree_view.heading(col, text=col,
                                               command = lambda c=col: self.sortby(self.result_tree_view,
                                                                              c,
                                                                              0, 
                                                                              data_file = self.result_fish))
               if col != 'Cat. Term':
                   
                   col_w = tkFont.Font().measure(col)+15
                   
               else:
                   
                   #col_w = tk.Font.Font().measure()
                   col_w = 270
               self.result_tree_view.column(col, anchor=tk.W, width=col_w)
            if len(list(self.fisher_tests_made.keys())) > 0:              
            
                if str(self.data_selection_rows.get()) in str(list(self.fisher_tests_made.keys())[-1]):
                    self.filter_fisher_result(alpha = float(self.FDR_combo_box.get()), col_tree = column_tree)
                
                else:
                    pass
                
            self.result_tree_view.grid(columnspan=10, padx=10, pady=5, sticky=tk.NSEW, column = 0) 
                                      
            x = self.winfo_pointerx()
            y = self.winfo_pointery()
             
             
            popup.focus_force()
            popup.geometry('%dx%d+%d+%d' % (w, h, x, y)) 
                 
         else:
             pass
         
     def select_tree_fisher(self,event):
          sel_rows = self.result_tree_view.selection()
          sel_rows = [int(col[0]) for col in sel_rows]
          
          
          
          self.subset_fisher_selection = self.result_fish_filt.iloc[sel_rows]    
          
         
     def fisher_drop_down(self,event):
          
          x = self.winfo_pointerx()
          y = self.winfo_pointery()
          self.fisher_sub_menu.post(x,y)
          
          
          
     def build_fisher_export_menu(self):
         self.subset_fisher_selection = pd.DataFrame() 

         self.fisher_sub_menu =  Menu(self, tearoff=0, background = col_menu)
         
         self.fisher_sub_menu.add_command(label='Add annotation column', command = lambda: self.fisher_add_annotation_column(), foreground = "black")         
         self.fisher_sub_menu.add_command(label='Copy to clipboard', command = lambda: self.copy_file_to_clipboard(self.result_fish_filt), foreground = "black")
         self.fisher_sub_menu.add_command(label='Export Selection [.txt]', command = lambda: self.export_data_to_file(self.result_fish_filt,  format_type = 'txt'), foreground = "black")
         self.fisher_sub_menu.add_command(label='Export Selection [.xlsx]', command = lambda: self.export_data_to_file(self.result_fish_filt,  format_type = 'Excel',sheet_name = 'SelectionExport'), foreground = "black")
    
    
     def fisher_add_annotation_column(self, mode = 'Column'):
          if len(self.result_fish_filt.index) == 0:
              return 
          
          
          if len(self.subset_fisher_selection.index) == 0:
              quest = tk.messagebox.askquestion('Fisher annotation','No rows selected. Would you like to add for each category an annotation column?')
              if quest == 'yes':
                  self.subset_fisher_selection = self.result_fish_filt
              else:
                  return
                  
          col_to_search = self.subset_fisher_selection['Column'].iloc[0]   
          
          file_name = self.plots_plotted[self.count][-1]
          for key, file_ in self.idx_and_file_names.items():
             if file_name == file_:
                 self.set_source_file_based_on_index(key)
                 
          self.cat_selected = self.subset_fisher_selection['Categroy']   
          self.add_categorical_annotation(popup = None,  msg = False, source_col_provided = True, source_prov = self.source_file[col_to_search], col_name= col_to_search)
          tk.messagebox('Done..','Categorical annotation columns were added to the selected source data.', parent = self.popup_fisher_)
              


     
         
         
         
         
     def calculate_fisher(self,col,threads, sep,alpha, col_tree):
            
            ind_index = self.data_to_test.index   
            background = self.source_file
            background.loc[:,'idx_in_selection'] = background.index.isin(ind_index)
            
            
                
            data_split_for_all_cats = self.data_to_test[col].str.split(sep)
            self.cats_to_test = list(set(chain.from_iterable(data_split_for_all_cats)))    
           
            if self.cats_to_test not in list(self.fisher_tests_made.values()):     
                fish_test = fisher_on_mult_threads(col, background)
                pool = Pool(threads)
                self.end_res = pool.map(fish_test, self.cats_to_test)  
                pool.close()
                pool.join()     
                
                self.end_res =[x for x in self.end_res if x != None]
                self.result_fish = pd.DataFrame(self.end_res)
                
                self.end_res.clear() ## free up memory  
                
                self.result_fish.columns = ['Column','Categroy','Cat size','Cat size in sel.','Enrichment','p-value']
                self.result_fish[['Enrichment','p-value']] =  self.result_fish[['Enrichment','p-value']].astype('float')
                FDR = multipletests(self.result_fish['p-value'] ,method='b')
                self.result_fish['FDR'] = FDR[1]
                self.result_fish['FDR'] =  self.result_fish['FDR'].astype('float')
                self.fisher_tests_made[str(self.cats_to_test)+str(self.data_selection_rows.get())] = self.cats_to_test
            self.filter_fisher_result(alpha, col_tree) 
            del background                           
     def filter_fisher_result(self, alpha, col_tree):        
            try:
                self.result_tree_view.delete(*self.result_tree_view.get_children())
            except:
                pass
            self.result_fish_filt = self.result_fish[self.result_fish['FDR'] <= alpha]
            self.result_fish_filt.sort_values('p-value',inplace=True)
            for col in col_tree:
                self.result_tree_view.heading(col, text=col,
                                               command = lambda c=col: self.sortby(self.result_tree_view,
                                                                              c,
                                                                              0, 
                                                                              data_file = self.result_fish))
            for col_num in range(len(self.result_fish_filt.index)):
                        data_fill = tuple(self.result_fish_filt.iloc[col_num])
                        self.result_tree_view.insert('', 'end',str(col_num), values = data_fill)

            self.fdr_fisher.set(str(alpha))
            
            
     def onPressFisher(self,event):
         try:
             self.fisher_anno.remove()
             self.canvas.draw_idle() 
         except:
             pass                   
         if self.canvas.widgetlock.locked():
                 return
         if event.inaxes is None:
                 return
        
         self.lasso_fisher = Lasso(event.inaxes,
                            (event.xdata, event.ydata),
                            self.callback_fisher)