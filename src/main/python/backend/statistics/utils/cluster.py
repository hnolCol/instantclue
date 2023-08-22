from numba import jit, prange 
import numpy as np 
from .base import mean, squareSum

@jit(nopython=True)
def euclidean(p : np.ndarray, q:np.ndarray, nanp : np.ndarray, nanq : np.ndarray) -> float:
    """
    """
    d = 0
    for i in range(p.size):
        if not nanp[i] and not nanq[i]:
            d += (p[i] - q[i])**2 
        
    return d**0.5 

@jit(nopython=True, parallel = True)
def nanEuclideanCluster(X : np.ndarray) -> np.ndarray:
    """
    """
    nan = np.isnan(X)
    i,ii = X.shape 
    e = np.zeros(shape=(i,i))
    
    for i in prange(i):
        for j in prange(i):
            if i != j:
                d = euclidean(X[i],X[j], nan[i], nan[j])
                e[i,j] = d
                e[j,i] = d 

    return e 

@jit(nopython=True)
def pearson(p : np.ndarray, q : np.ndarray, nanp : np.ndarray, nanq : np.ndarray) -> float:
    """
    Distance pearson implementation  (1-r). 
    """
    nan = nanp + nanq > 0   
    p = p[~nan]
    q = q[~nan]
    psize = p.size
    if p.size < 2: return 1.0
    pmean = mean(p)
    qmean = mean(q)
    
    SSP = squareSum(p,pmean)
    SSQ = squareSum(q,qmean)
    
    ri = 0
    for i in range(psize):
        ri+= (p[i]-pmean) * (q[i]-qmean)
    rd = (SSP * SSQ)**0.5
    r = 1 - ri / rd 
    return r


@jit(nopython=True,parallel = True)
def nanCorrelationCluster(X : np.ndarray) -> np.ndarray:
    """
    Wrapper function to apply pearson to an array X.
    """
    nan = np.isnan(X)
    i,ii = X.shape 
    e = np.zeros(shape=(i,i))
    
    for i in prange(i):
        for j in prange(i):
            if i != j:
                d = pearson(X[i],X[j], nan[i], nan[j])
                e[i,j] = d
                e[j,i] = d 
    return e 
