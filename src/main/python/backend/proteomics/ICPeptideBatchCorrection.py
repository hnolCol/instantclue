from threading import local
import pandas as pd 
import numpy as np
from pandas.core.algorithms import isin 

from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt 
from loess.loess_1d import loess_1d 
from numba import jit 
import numpy as np
import time
import math
from skmisc.loess import loess

def loess_fit(x, y, span=0.75):
    """
    loess fit and confidence intervals
    """
    # setup
    lo = loess(x, y, span=span)
    # fit
    lo.fit()
    # Predict
    prediction = lo.predict(x, stderror=True)
    # Compute confidence intervals
    ci = prediction.confidence(0.05)
    # Since we are wrapping the functionality in a function,
    # we need to make new arrays that are not tied to the
    # loess objects
    yfit = np.array(prediction.values)
    ymin = np.array(ci.lower)
    ymax = np.array(ci.upper)
    return yfit, ymin, ymax

class ICPeptideBatchCorrection(object):

    def __init__(self, peptideSequenceColumn = "Stripped.Sequence", intensityColumns = []) -> None:
        ""
        self.peptideSequences = []
        self.longFormat = False 
        self.peptideSequenceColumn = peptideSequenceColumn
        self.intensityColumns = intensityColumns


    def setData(self, data, intensityColumns = None, longFormat=None, peptideSequences = None):
        ""
        if isinstance(peptideSequences,list) and len(peptideSequences) > 0:
            self.setPeptideSequences(peptideSequences)
        if isinstance(longFormat,bool):
            self.longFormat = longFormat
        if isinstance(intensityColumns,list):
            self.intensityColumns = [columnName for columnName in intensityColumns if columnName in data.columns]
            
        self.data = data 
        self.subsetDataForPeptides()

    def setPeptideSequences(self,peptideSequences):
        ""
        self.peptideSequences = peptideSequences 

    def subsetDataForPeptides(self):
        ""
        boolIdx = self.data[self.peptideSequenceColumn].isin(self.peptideSequences)
        self.peptideData = self.data.loc[boolIdx,:]

    def getPeptideSequences(self):
        ""
        return self.peptideSequences


    def fitLowess(self):
        ""
        x = np.arange(len(self.intensityColumns))
        Y = self.peptideData[self.intensityColumns].values
        f = plt.figure()
        ax = f.add_subplot(111)
        #o = fitLoess(x,Y)
        #print(o)
        for y in Y:
           # lowessFit = lowess(y,x, is_sorted=True, frac=0.5)
           # print(lowessFit)
            yfit,ymin,ymax = loess_fit(x,y,span=0.75)
           
            ax.plot(x,yfit)
            ax.fill_between(x, ymin, ymax, color='lightgrey', alpha=.3)
            #ax.scatter(lowessFit[:,0],lowessFit[:,1])
            ax.scatter(x,y)
            
        plt.show()

if __name__ == "__main__":
    batchCorrection = ICPeptideBatchCorrection()
    batchCorrection.setPeptideSequences(["ADVTPADFSEWSK"])#,"GAGSSEPVTGLDAK"
    data = pd.read_csv("testData.txt", sep="\t")
    intensityColumns = [col for col in data.columns if "Col" in col]
    #print(data.columns.values.tolist())
    batchCorrection.setData(data,intensityColumns=intensityColumns)
    batchCorrection.fitLowess()


    