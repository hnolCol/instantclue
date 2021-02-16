import numpy as np 
import pandas as pd 

from sklearn.decomposition import PCA

from collections import OrderedDict



from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig


class PCA_REGISTRY(object):
    ""
    def __init__(self):
        self.transformID = 0
        self.pcas = OrderedDict() 

    def registerTSNE(self, TSNE):
        ""
        self.transformID += 1
        self.pcas[self.transformID] = TSNE
        return self.transformID


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
        return P.T, vectors, values

    def _scale(self):
        ""
        M = np.mean(self.X.T,axis=1)
        self.X = self.X - M 

    def run(self):
        ""
        if self.scale:
            self._scale()
        xEmbedding, eigVectors, eigValues = self._fit_transform()
        variance_explained = []
        for i in eigValues:
            variance_explained.append((i/sum(eigValues))*100)

        if xEmbedding.shape[1] < self.n_components:
            return xEmbedding, eigVectors, variance_explained
        return xEmbedding[:,:self.n_components], eigVectors[:,:self.n_components], variance_explained[:self.n_components]

