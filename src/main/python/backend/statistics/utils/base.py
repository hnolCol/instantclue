from numba import jit, prange 
import numpy as np 
from typing import List 

@jit(nopython = True)
def mean(X : np.ndarray|List[float]) -> float:
    "Calculates the mean of array x"
    N = X.size
    s = sum(X)
    return s/N 


@jit(nopython=True)
def sd(X : np.ndarray, mean : float , N : int) -> float:
    """Calculates the standard deviation with known average/mean and N"""
    return np.sqrt(squareSum(X,mean)/(N-1))


@jit(nopython=True)
def sum(X : np.ndarray|List[float]) ->float:
    "Calculates the sum"
    a = 0
    for x in X:
        a += x 
    return a

@jit(nopython = True)
def squareSum(X : np.ndarray|List[float], meanValue : float) -> float:
    a = 0 
    for x in X:
        a += (x-meanValue)**2 
    return a 

@jit(nopython=True)
def countValuesBelowThresh(X : np.ndarray, thresh : float) -> int:
    N = 0 
    XX = X.flatten()
    ii = XX.size
    for i in range(ii):
            if XX[i] <= thresh:
                N += 1 
    return N

@jit(nopython=True)
def countValuesAboveThresh(X : np.ndarray, thresh : float) -> int:
    N = 0 
    XX = X.flatten()
    ii = XX.size
    for i in range(ii):
            if XX[i] >= thresh:
                N += 1 
    return N