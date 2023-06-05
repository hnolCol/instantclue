


from collections import OrderedDict
from datetime import time
import pandas as pd 
import numpy as np 
from numpy.linalg import eig
import statsmodels.api as sm
import statsmodels.formula.api as smf
data = sm.datasets.get_rdataset("dietox", "geepack").data

data = pd.DataFrame(
    [x.split("\t") for x in """columnNames  treat  time  batch  sample  compIdx        pc
0     _23_1CON      0     0      1       1        0  0.296379
1     _23_2CON      0     7      1       2        0 -0.310216
2     _23_3NO0      1     0      1       3        0  0.386527
3     _23_4NO7      1     7      1       4        0 -0.147179
4     _26_1CON      0     0      2       5        0  0.253927
5     _26_2CON      0     7      2       6        0 -0.299453
6     _26_3NO0      1     0      2       7        0  0.380901
7     _26_4NO7      1     7      2       8        0  0.065552
8     CONTROL7      0     7      3       9        0 -0.445159
9     CONTROLZ      0     0      3      10        0  0.188493
10      NO_7_5      1     7      3      11        0 -0.290607
11     NO_ZERO      1     0      3      12        0  0.152242
""".split("\n")])


class PCVA(object):
    def __init__(self,*args,**kwargs) -> None:
        ""
        self.nComps = 3
        

    def scale(self,X):
        #center data
        X = X.subtract(X.mean(axis=1),axis="index")
        return X

    def fit(self,X,columnNames,groupings):
        ""
        X = self.scale(X)
        corrMatrixX = X.corr()
        values, vectors = eig(corrMatrixX)
        eigValueSum = np.sum(values)
        percPCs = values/eigValueSum

        R = pd.DataFrame(columnNames.values,columns=["columnNames"])
     
        for groupingName, grouping in groupings.items():
            R[groupingName] = [groupName for colName in R["columnNames"] for groupName, groupItems in grouping.items() if colName in groupItems]

        R = pd.concat([R]*self.nComps)
        R["compIdx"] = np.repeat(np.arange(self.nComps),columnNames.size)
        R["pc"] = vectors[:,:3].reshape(-1,1, order="F")
   
        #mixed linear model
        for compIdx,compData in R.groupby("compIdx"):

            vcf = {"treat": "0 + C(treat)", "time": "0 + C(time)", "batch": "0 + C(batch)"}
            md = smf.mixedlm("pc ~ 1",  
                data = compData, 
                vc_formula=vcf,
                groups=compData["compIdx"])
            mdf = md.fit()
          

    def fit_transform(self,X):
        ""

    
    def transform(self,*args,**kwargs):
        ""



X = pd.read_csv("data.txt",sep="\t",index_col="probe_set")
groups = pd.read_csv("groups.txt",sep="\t")




timeGroup = OrderedDict([(x,y["columnname"].values.tolist()) for x,y in groups[["Time","columnname"]].groupby("Time")])
treatGroup = OrderedDict([(x,y["columnname"].values.tolist()) for x,y in groups[["Treatment","columnname"]].groupby("Treatment")])
batchGroup = OrderedDict([(x,y["columnname"].values.tolist()) for x,y in groups[["Batch","columnname"]].groupby("Batch")])
sampleGroup = OrderedDict([(x,y["columnname"].values.tolist()) for x,y in groups[["sample","columnname"]].groupby("sample")])

grouping = {"treat":treatGroup,"time":timeGroup,"batch":batchGroup,"sample":sampleGroup}

print(PCVA().fit(X,groups["columnname"],grouping))
