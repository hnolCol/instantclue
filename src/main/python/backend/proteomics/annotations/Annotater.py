from multiprocessing import Pool
from numba.np.ufunc import parallel
import numpy as np
from numba import jit, prange, types
import pandas as pd 
import numba as nb


def findFirstIdxMatch(s,Y):
    for i in prange(Y.size):
        if s == Y[i]:
            return i
    return -1

def findAllMatches(ss,Y):
    ms = np.ones(shape=len(ss)) * (-1)
    for i in np.arange(Y.size):
        for m in np.arange(len(ss)):
            if ss[m] == Y[i]:
                ms[m] = i 
    return ms 


def idxMatch(X,Y,splitString = ";", chunkId = None):
    ""
    A = np.ones(shape=X.size) * (-1)
    for xi in np.arange(X.size):
        if splitString in X[xi]:
            ss = X[xi].split(splitString)
            
            allMatches = findAllMatches(ss,Y)
           
        else:
            
            A[xi] = findFirstIdxMatch(X[xi],Y)


    return A 


class Annotator(object):

    def __init__(self,sourceData, *args,**kwargs):
        ""
        
        self.sourceData = sourceData 
        self._loadInternal()
        
    def matchIDs(self,X):
        NProcesses= 5
        Y = self.mitoCarta["UniProt"].dropna().astype("str").values.flatten() 
        #if X.shape[0] > Y.shape[0]:
        chunks = np.array_split(X,NProcesses,axis=0)
	     
        with Pool(NProcesses) as p:
            rs = p.starmap(idxMatch, [(chunk,Y, ";",chunkIdx) for chunkIdx, chunk in enumerate(chunks)])
            #print(rs)
   

    def _loadInternal(self):
        ""

        
        self.mitoCarta = pd.read_csv("/Users/hnolte/Documents/GitHub/instantclue/src/main/python/annotations/intern/Human.MitoCarta3.0.txt",sep="\t")



if __name__ == "__main__":

    X = pd.read_csv("test.txt",sep="\t").dropna().astype("str").values.flatten() 
    Annotator(None).matchIDs(X)
    

    #X = np.array(["ABC","AHSDAH","BASC;22323"])
    #Y = np.array(["asdad","ABC","AHSDAH","22323","asdadada"])
    #print(X.size)
    #print(idxMatch(X,Y))