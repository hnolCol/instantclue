#License 


InstantClue was licensed under the GPL3 clause.



#Tutorials 


PLease visit https://www.instantclue.uni-koeln.de for video tutorials.

   
   
******************** IMPORTANT NOTE ********************

To make the treeview work you will have to change 
the function 

    def selection in the ttk.py file in the tkinter package folder:
	line 1392
    
    def selection(self, selop=None, items=None):
        """If selop is not specified, returns selected items."""
        if isinstance(items, (str, bytes)):
            items = (items,)
        return self.tk.splitlist(self.tk.call(self._w, "selection", selop, items))

#Dependencies (alphabetic order):
    
    -husl (color palettes)
    -fastcluster
    -matplotlib (https://matplotlib.org/users/license.html)
    -numpy (BSD 3 - https://docs.scipy.org/doc/numpy-1.10.0/license.html)
    -pandas (BSD 3)
    -pandastable (GPL v3 - note: copied in source code, changes are indicated)
    -scikit-learn (BSD 3)
    -scipy (‎BSD-new license - https://www.scipy.org/scipylib/license.html)
    -seaborn (BSD 3)
    -statsmodels (BSD 3 https://github.com/statsmodels/statsmodels/blob/master/
    -tslearn
    
"""



