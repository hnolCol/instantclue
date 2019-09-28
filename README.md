# Instant Clue

Python (>3.4) based application (GUI) based on Tkinter for data visualization and analysis.
Binary files are available at at the [website](http://www.instantclue.uni-koeln.de) for Mac OS and Windows.


## Getting Started 

Clone the repository, navigate to the folder and execute 

```
python instant_clue.py
```

The Graphical User Interface (GUI) will open. 


## Tutorials 


PLease visit https://www.instantclue.uni-koeln.de for video tutorials.

## Citation

Original article was published in [Scientific reports](https://www.nature.com/articles/s41598-018-31154-6)

Nolte et al. Instant Clue: A Software Suite for Interactive Data Visualization and Analysis, Scientific reports 8, 12648 (2018) 

## License 

InstantClue was licensed under the GPL3 clause.

## Versioning

We use [SemVer](http://semver.org/) for versioning.
   
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

## Dependencies:

* [husl](https://pypi.org/project/husl/)
* [fastcluster](https://pypi.org/project/fastcluster/)
* [matplotlib](https://matplotlib.org/users/license.html)
* [numpy](https://docs.scipy.org) 
* [pandas](https://pandas.pydata.org)
* [pandastable](https://github.com/dmnfarrell/pandastable)
* [scipy](https://docs.scipy.org) 
* [statsmodels](https://github.com/statsmodels/statsmodels/blob/master/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [seaborn](http://seaborn.pydata.org)
* [tslearn](https://github.com/rtavenar/tslearn)





