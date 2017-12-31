


import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk

class TreeviewDataPresentation(object):
    def __init__(self, data_frame,popup):
        self.tree = None
        
        self.col_headers = data_frame.columns.values.tolist()
        self.data = data_frame
        self._initiate_widget(popup)
        self._fill_trees()
        
    def _initiate_widget(self,popup):

        cont = ttk.Frame(popup)
        cont.pack(fill="both", expand = True)

        self.data_tree = ttk.Treeview(cont,columns = self.col_headers,
                                      show="headings")
        self.data_tree.bind('<<TreeviewSelect>>', self.onSelect)
        scroll_vert = ttk.Scrollbar(cont,orient='vertical',
                                    command = self.data_tree.yview)
        scroll_hor = ttk.Scrollbar(cont,orient='horizontal',
                                   command = self.data_tree.xview)
        self.data_tree.configure(yscrollcommand = scroll_vert.set,
                                 xscrollcommand = scroll_hor.set)
        self.data_tree.grid(column = 0, row = 0,
                            sticky ='nsew', in_=cont)
        scroll_vert.grid(column=1, row=0, sticky = 'ns', in_=cont)
        scroll_hor.grid(column=0, row=1, sticky = 'ew', in_=cont)
        cont.grid_columnconfigure(0, weight=1)
        cont.grid_rowconfigure(0, weight=1)

    def _fill_trees(self):

        for col in self.col_headers:
            self.data_tree.heading(col, text=col,
                                   command = lambda c=col: sortby(self.data_tree,
                                                                  c,
                                                                  0))

        for col_num in range(len(self.data.index)):
            data_fill = tuple(self.data.iloc[col_num])
            self.data_tree.insert('', 'end',str(col_num), values = data_fill)
        
    def onSelect(self,event):
     
        print("selected items:")
        for item in self.data_tree.selection():
           print(item)


def sortby(tree, col, descending):
    """sort tree contents when a column header is clicked on"""
    # grab values to sort
    data = [(tree.set(child, col), child) \
        for child in tree.get_children('')]
    # if the data to be sorted is numeric change to float
    #data =  change_numeric(data)
    # now sort the data in place
    data.sort(reverse=descending)
    for ix, item in enumerate(data):
        tree.move(item[1], '', ix)
    # switch the heading so it will sort in the opposite direction
    tree.heading(col, command=lambda col=col: sortby(tree, col, \
        int(not descending)))        
        

