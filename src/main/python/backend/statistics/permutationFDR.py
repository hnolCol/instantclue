## permuations based statistics
from time import time
from numba import jit, prange
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd 
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from multiprocessing import Pool

@jit()
def mean(X):
    a = 0
    N = X.size
    for x in X:
        a += x 
    return a/N 

@jit()
def squareSum(X,meanValue):
    
    a = 0 
    for x in X:
        a += (x-meanValue)**2 

    return a 

@jit()
def sstat(X,Y,s0):
    J = X.size
    K = Y.size
    if J >= 2 and K >= 2:
        meanX = mean(X)
        meanY = mean(Y)
        SSX = squareSum(X,meanX)
        SSY = squareSum(Y,meanY)
        p = (1/J + 1/K) / (J + K - 2)
        s = (p * (SSX + SSY))**0.5
        d = (meanX - meanY) / (s + s0)
        return d
    return np.nan

jit()
def performTest(X,Y, s0 = 0.1, sort = True):
    ""
    s = np.zeros(shape=X.shape[0])
    for n in np.arange(s.size):
        boolX = np.isnan(X[n,:])
        boolY = np.isnan(Y[n,:])
       
        XX = X[n,~boolX]
        YY = Y[n,~boolY]
       
        s[n] = sstat(XX,YY,s0)
    if sort:
        return np.sort(s) 
    else: 
        return s
        

def permutedTest(p,PP,s0):
    xp = p 
    yp = [pi for pi in range(PP.shape[1]) if pi not in p]
    Xp = PP[:,xp]
    Yp = PP[:,yp]
    return pd.Series(performTest(Xp,Yp,s0=s0)).abs().sort_values(ascending=False)

def performPermutationTests(X,Y,PP,perms,s0):
    ""
   # permStats = np.zeros(shape=(PP.shape[0],len(perms)))
    #with Pool(5) as p:
    rs = [permutedTest(p,PP,s0) for p in perms]
    return pd.concat(rs,axis=1,ignore_index=True)

@jit() 
def calculateFP(PP,dcut):
    "Find the number of false positive at a certain d value."
    A = 0
    n, B = PP.shape
    for b in prange(B):
        for m in prange(n):
            if PP[m,b] >= dcut:
                A += 1
            else:
                break #since the data are sorted we can just stop here.          
    return A / B


def calculateQ(ss_unsorted,SS,PP):
    "Calculates the Q value for each stat value."
    Q = np.ones(shape=SS.size)
    sunAbs = np.abs(ss_unsorted)
    ssAbs = np.abs(SS)
    ppAbs = PP.abs().values
    for n in range(ss_unsorted.size):
        dcut = sunAbs[n]
        TP = np.sum(ssAbs >= dcut)
        FP = calculateFP(ppAbs,dcut)
 
        qValue = FP/TP
       
        Q[n] = qValue

    boolIdx = Q == np.inf
    Q[boolIdx] = np.min(Q)
    boolIdx = Q > 1 
    Q[boolIdx] = 1
    return Q

def calculatePermutationBasedFDR(X,Y,PP,P=200,s0=0.1):
    "Permutation based FDR."
    rng = default_rng()
    permutations = np.unique(np.array([np.sort(rng.choice(PP.shape[1], size = X.shape[1], replace=False)) for _ in range(P)]).reshape(P,X.shape[1]),axis=1)
    PP = performPermutationTests(X,Y,PP,perms=permutations,s0=s0)
    SS = performTest(X,Y,s0)
    ss_unsorted = performTest(X,Y,s0,sort=False) 
    Q = calculateQ(ss_unsorted,SS,PP)

    return Q , ss_unsorted











# ss_unsorted = performTest(X,Y,sort=False)
# Q = np.ones(shape=SS.size)
# for n,dcut in enumerate(ss_unsorted):
#     TP = np.sum(np.abs(SS) > abs(dcut))
#     FP = calculateFP(PP.abs().values,abs(dcut))
#     ##print(FP/TP)
#     #print("positive",TP)
#     qValue = FP/TP
    
#     Q[n] = qValue
#     boolIdx = Q == np.inf 
#     Q[boolIdx] = np.min(Q)
#     #print(qValue,FP,TP)

# log2FC = df[XNAMES].mean(axis=1) - df[YNAMES].mean(axis=1)
# t,pvalue =  ttest_ind(X,Y,axis=1)