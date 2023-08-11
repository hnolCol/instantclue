from numba import jit, prange
from .base import sum, squareSum, mean, countValuesBelowThresh
import numpy as np 

@jit(nopython=True)
def calculateQValue(pvaluesRealData : np.ndarray, pvaluesPerm : np.ndarray, permutations : int):
    N = pvaluesRealData.size
    Q = np.ones(shape=(N,1))
    for idx in prange(N):
        pvalue = pvaluesRealData[idx]
        FP = countValuesBelowThresh(pvaluesPerm,pvalue)
        TP = countValuesBelowThresh(pvaluesRealData,pvalue)
        if FP == 0 and TP > 0:
            Q[idx] = 0 
        elif TP == 0 and FP > 0:
            Q[idx] = 1.0
        else:
            q = FP / TP * (1/permutations)
            if q > 1:
                q = 1
            Q[idx] = q

    return Q 

@jit(nopython = True)
def sstat(X : np.ndarray,Y : np.ndarray, s0 : float):
    J = X.size
    K = Y.size
    if J == 0 or K == 0: return 0 
    meanX = mean(X)
    meanY = mean(Y)
    SSX = squareSum(X,meanX)
    SSY = squareSum(Y,meanY)
    p = (1/J + 1/K) / (J + K - 2)
    s = (p * (SSX + SSY))**0.5
    d = (meanX - meanY) / (s + s0)
    return d

@jit(nopython=True)
def fstat(data : np.ndarray, numberGroups : int, groupSize : int) -> float:
    """
    Returns the f-statistic of a dataarray
    """
    sampleNumber = numberGroups * groupSize
    overallMean = mean(data.flatten())
    groupMeans = [mean(d) for d in data]
    groupSS = [squareSum(d,groupMeans[n]) for n,d in enumerate(data)]
    SSE = sum(groupSS)

    SSR = sum([groupSize*(groupMeans[n] - overallMean)**2 for n in range(len(groupMeans))])
    #SST = SSE + SSR

    dfTreat = numberGroups - 1 
    dfError = sampleNumber - numberGroups
   # dfTotal = sampleNumber - 1 

    MSTreat = SSR / dfTreat
    MSError = SSE / dfError
    F = MSTreat / MSError
    return  F


@jit(parallel=True,nopython=True)
def performFTest(X : np.ndarray) -> np.ndarray:
    """
    Calculates the ANOVA F statistic

    :X ```np.ndarray``` 
    """
    numberGroups, N , nreps = X.shape
    F = np.zeros(shape=N)
    for idx in prange(N):
        rowData = X[:,idx,:]
        fvalue = fstat(rowData,numberGroups,nreps)
        F[idx] = fvalue

    return F

@jit(nopython = True,parallel=True)
def performSAMTest(X : np.ndarray,Y : np.ndarray, s0 : float, sort : bool = True, abs : bool = False) -> np.ndarray:
    """
    Performs the statistical sam test for each row.
    """
    D = np.zeros(shape=X.shape[0])
    boolX = np.isnan(X)
    boolY = np.isnan(Y)
    for n in prange(D.size):
        XX = X[n,~boolX[n]]
        YY = Y[n,~boolY[n]]
        D[n] = sstat(XX,YY,s0)
    if sort:
        if abs:
            return np.sort(np.abs(D))
        return np.sort(D) 
    else: 
        return D