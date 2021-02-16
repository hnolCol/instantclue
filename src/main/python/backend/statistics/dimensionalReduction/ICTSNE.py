import numpy as np 
import pandas as pd 

from sklearn.decomposition import PCA

from collections import OrderedDict

from numpy import array
from numpy import mean
from numpy import cov


class ICPCA(object):
    ""
    def __init__(self,X, n_components = 2, scale = True):
        ""
        self.scale = scale
        self.n_components = n_components
        self.X = X
        

    def _fit_transform(self):
        #calc embedding values
        V = cov(self.X.T)
        # eigendecomposition of covariance matrix
        values, vectors = eig(V)
        # project data
        P = vectors.T.dot(self.X.T)
        return P.T, vectors

    def _scale(self):
        ""
        M = np.mean(self.X.T,axis=1)
        self.X = self.X - M 

    def run(self):
        ""
        if self.scale:
            self._scale()
        xEmbedding, eigV = self._fit_transform()
        if xEmbedding.shape[1] < self.n_components:
            return xEmbedding, eigV
        return xEmbedding[:,:self.n_components], eigV[:,:self.n_components]

