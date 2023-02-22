import os 
import numpy as np 
import pandas as pd 

from collections import OrderedDict


class ICSmartReplace(object):
    def __init__(self, nNaNForDownshift = 4, minValidvalues = 4, grouping = {}, downshift = 1.8, scaledWidth = 0.3):

        self.params= {"nNaNForDownshift":nNaNForDownshift,
                      "minRequiredValues" : minValidvalues,
                      "grouping":grouping,
                      "downshift":downshift,
                      "scaledWidth":scaledWidth}

    def replaceFun(self, a, columnNames):
        ""
        arrayLength = a.size
        nanBool = np.isnan(a)
        nanCount = nanBool.sum()
        
        if nanCount < self.params["nNaNForDownshift"] and arrayLength - nanCount >= self.params["minRequiredValues"]:
            
            # if there are not enough missing values to assume that the quantitification is lacking due to detection limit,
            # then just replace by normal distribution without downshift. 
            mu = np.nanmean(a)
            sigma = 0.3 * np.nanstd(a)
            replaceData =  np.random.normal(loc = mu, scale = sigma, size = arrayLength )
            
        elif arrayLength == nanCount:

            # if all da are missing, replace by downshift gaussian
            replaceData = np.array([np.random.normal(
                                     loc = self.columnMeans.loc[columnNames[i]] - self.params["downshift"] * self.columnStds.loc[columnNames[i]], 
                                     scale = self.params["scaledWidth"] * self.columnStds.loc[columnNames[i]], 
                                     size = 1) for i in range(arrayLength)])
            replaceData = replaceData.flatten()
        else:
            return a
        a[nanBool] = replaceData[nanBool]
        return a


    def getColumnMeans(self):
        ""
        self.columnMeans = self.X.mean()
      

    def getColumnStds(self):
        ""
        self.columnStds = self.X.std()
       

    def fitTransform(self,X, columnNames = []):
        ""
        self.X = X
        self.getColumnMeans()
        self.getColumnStds()
        columnNames = []
        for groupColumnNames in  self.params["grouping"].values():
            columnNames.extend(groupColumnNames.values.tolist())
       
        outPut = pd.DataFrame(columns=columnNames, index = self.X.index) 
        
        for columns in self.params["grouping"].values():
            groupColumnNames = columns.values.tolist()
            X = self.X[groupColumnNames].values
            
            Xreplaced = np.apply_along_axis(func1d = self.replaceFun,axis=1, arr= X, columnNames = groupColumnNames)
            
            outPut[columns] = Xreplaced
        return outPut
