 elif mode == 'Create subject column':
             w = 800
             
             def return_occurance(row,start_,pract_combi):
                 
                 find_idx = pract_combi.index(row)
                 already_occured = self.start_count_through[find_idx]
                 self.start_count_through[find_idx] = already_occured + 1
                 return already_occured                        
                
                 
                 
             def count_through(listbox, ent_list,cat_columns):
                 sel_columns_idx = listbox.curselection()
                 sel_columns = [col for  v,col in enumerate(self.sourceData.df_columns) if v in sel_columns_idx]
                 collect_uniq_vals = []
                 for col in sel_columns:
                     uniq_vals = list(self.sourceData.df[col].unique())
                     collect_uniq_vals.append(uniq_vals)
                     
                 combinations = list(itertools.product(*collect_uniq_vals))
                 combi = self.sourceData.df[sel_columns].apply(tuple,axis=1)
                 uniq_combs_in_data = list(combi.unique())
                 pract_combi = [comb for comb in combinations if comb in uniq_combs_in_data]
                 
                 self.start_count_through = [1] * len(pract_combi)
                 
                 self.result_of_count_through = combi.apply(lambda row,start_ = self.start_count_through, pract_combi = pract_combi: return_occurance(row,start_,pract_combi)) 
                 
                 data_ = self.sourceData.df[cat_columns]
                 data_.loc[:,'Subject']  = self.result_of_count_through 
                
                 tree = self.save_tree_view_to_dict['subject_source']
                 tree.delete(*tree.get_children())    
                 self._fill_trees(col_sel = com_cols+['Subject col'], data = data_, tree_from_dict = True, dict_key  = 'subject_source'  , sort_active = False)    
                                
             def add_to_source(ent_list,popup):
                 already_ = [col for col in self.sourceData.df_columns if 'Subject_ic' in col]
                 occu = len(already_)
                 name = 'Subject_ic_'+str(occu)
                 
                 if self.result_of_count_through is not None:
                     self.sourceData.df.loc[:,name] = self.result_of_count_through
                 else:
                    collect_entries = []
                    for ent in ent_list:
                        collect_entries.append(int(ent.get()))
                    self.sourceData.df.loc[:,name] = collect_entries
                 self.sourceData.df_columns = self.sourceData.df.columns.values.tolist()                        
                 im_dat = self.int_icon
              
                 self.add_new_col_to_filt_source_file(name)
                 self.__add_information_of_col_to_dict(name,np.int64,self.current_data)
                 
                 self.source_treeview.insert(self.current_data+'int64',
                                          'end',
                                          self.current_data+name,
                                              text=name,
                                                      image = im_dat)
                 tk.messagebox.showinfo('Done..','Subect column: {} was added to the source file (Integer).'.format(name ),parent = popup)
                 

             colnames = list(self.selectedNumericalColumns.keys() )
             catnames = list(self.selectedCategories.keys())
             com_cols = colnames+catnames
             
             self.result_of_count_through = None         
             popup.attributes('-topmost', True)
             ##set_data_to_current_selection()
             cont = self.create_frame(popup)  
             cont.pack(fill='both', expand=True)
             #cont.grid_columnconfigure(len(com_cols)+1, weight=1)
             cont.grid_columnconfigure(0, weight=1,minsize=240)
             cont.grid_rowconfigure(4, weight=1)
             
             lab_text =  'Create new column indicating subject for repeated measurement analysis'
             
             lab_main = tk.Label(cont, text= lab_text, 
                                     font = LARGE_FONT, 
                                     fg="#4C626F", 
                                     justify=tk.LEFT, bg = MAC_GREY)
             
             lab_main.grid(padx=10, pady=15, columnspan=16, sticky=tk.W)
             
             lab_description = tk.Label(cont,text = 'Subject must be indicated by integers [0,1,2,3...]', bg = MAC_GREY)
             lab_description.grid(padx=10, pady=15, columnspan=6, sticky=tk.W)
             
             
             
             
             if len(com_cols) == 0:
                 com_cols = self.DataTreeview.columnsSelected  
             if len(com_cols) == 0:
                 com_cols = [col for col in self.sourceData.df_columns if self.sourceData.df[col].dtye != np.float64]
             cont1 = self.create_frame(cont)
             cont1.grid(row=3,rowspan=14,columnspan=4,padx=3,column=0, sticky=tk.NSEW)
                 
             self._initiate_tree_widget(cont1, col_sel = com_cols+['Subject col'], save_to_dict = True, dict_key = 'subject_source'  )  
             self._fill_trees(col_sel = com_cols+['Subject col'], data = self.sourceData.df[com_cols], tree_from_dict = True, dict_key  = 'subject_source'  , sort_active = False)    
             

             entry_list = []
             but_add = ttk.Button(cont, text= 'Add', command = lambda ent_list = entry_list, popup = popup: add_to_source(ent_list,popup))
             
             
             lab_count= tk.Label(cont, text = 'Selected columns to count\nthrough unique combinations:' , bg=MAC_GREY, justify = tk.LEFT)
             lab_count.grid(column = len(com_cols ), row=2, padx=20, sticky=tk.W, pady=5)
             
             scrollbar1 = ttk.Scrollbar(cont,
                                          orient=VERTICAL)
             scrollbar2 = ttk.Scrollbar(cont,
                                          orient=HORIZONTAL)
             lb_for_sel = Listbox(cont, width=50, height = 200,  xscrollcommand=scrollbar2.set,
                                      yscrollcommand=scrollbar1.set, selectmode = tk.MULTIPLE)
             #cont.columnconfigure(len(com_cols), weight = 1, minsize=250)
             lb_for_sel.grid(row=4, column=len(com_cols), columnspan=3, sticky=tk.E, padx=(20,0))
             
             scrollbar1.grid(row=4,column=len(com_cols )+3,sticky = 'ns'+'w')
             scrollbar2.grid(row=5,column =len(com_cols),columnspan=3, sticky = 'ew', padx=(20,0))
             
             scrollbar1.config(command=lb_for_sel.yview)
             scrollbar2.config(command=lb_for_sel.xview)
             fill_lb(lb_for_sel, data_to_enter = self.sourceData.df_columns)
             
             count_through_button = ttk.Button(cont, text ="Enumerate", command = lambda listbox = lb_for_sel, ent_list = entry_list, cat_columns = com_cols: count_through(listbox,ent_list,cat_columns))
             count_through_button.grid(row=3, column = len(com_cols), sticky=tk.W, pady=5, padx=20)
             but_add.grid(row=3, column = len(com_cols ),  padx=5, sticky=tk.E, pady=15)
             