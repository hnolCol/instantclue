
import pandas as pd 
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import multiprocessing 
"""
Experimental. 
"""
def loessCorr(X,Y,name,frac = 0.6, it = 3, delta=0):
    validBoolIdx = np.isnan(X.values)
    xvals = X.loc[~validBoolIdx].sort_values()
    yvals = Y.loc[xvals.index]
    
    xdiff = xvals - yvals
    out = lowess(
            xdiff.values.flatten(),
            yvals.values.flatten(),
            frac=frac,
            it=it,
            delta=delta,
            is_sorted=False,
            return_sorted=False,)

    R = pd.DataFrame(xvals-out, index = xvals.index,columns=[name])
    R["out"] = out 
    R["xdiff"] = xdiff
    R["yfit"] = yvals
    R["raw"] = xvals
    
    return R



class LoessColumnNormalizator(object):
    
    def __init__(self,frac=0.8,*args,**kwargs):
        ""

        self.params = {
            "frac" : frac
        }


    def fit_transform(self,X):
        ""
        X = X.dropna()
        Y = X.mean(axis=1)
        print(Y)
        for n in range(3):
            Y = X.mean(axis=1)
            with multiprocessing.Pool(4) as p:
                r = p.starmap(loessCorr,[(X[colName],Y,colName) for colName in X.columns])
                print(r)
                R = pd.concat(r,axis=1)
                print(R)
                R.to_csv("data_out{}.txt".format(n),sep="\t")
            X = R[X.columns]
            print("it {}done".format(n) )
            

X = pd.read_csv("data.txt",sep="\t")
LCM = LoessColumnNormalizator().fit_transform(X[["CTRL_1","CTRL_2","CTRL_3","WT_1","WT_2","WT_3"]])







