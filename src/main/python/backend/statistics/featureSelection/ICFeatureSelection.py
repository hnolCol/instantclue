
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectFpr, SelectFdr, VarianceThreshold, RFECV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#too much repetitive code here, fix! 

class ICFeatureSelection(object):
    ""

    def __init__(self):
        ""
        self.maxFeatures = 10
        self.alpha = 0.05

    def setMaxFeautres(self,k):
        ""
        if isinstance(k,int):
            self.maxFeatures = k 
        
    def setAlpha(self,a):
        ""
        if isinstance(a,float) and a <= 1 and a >= 0:
            self.alpha = a

    def selectFeaturesByFpr(self, X, Y, scale = True, **kwargs):
        ""
        if scale:
            X = self.scaleData(X)
        sfp = SelectFpr(alpha=self.alpha, **kwargs).fit(X,Y)
        return sfp.get_support()

    def selectFeaturesByFdr(self, X, Y, scale = True, **kwargs):
        ""
        if scale:
            X = self.scaleData(X)
        sfp = SelectFdr(alpha=self.alpha, **kwargs).fit(X,Y)
        return sfp.get_support()
    
    def selectFeaturesBySVM(self, X, Y, SVMwargs = {}, scale = True):
        "Random forest selection"

        svm = SVC(**SVMwargs) #random firest classifier
        sfm = SelectFromModel(svm, max_features = self.maxFeatures)
       # sfm = SelectFpr()
       # sfm = RFE(rfc,self.maxFeatures,step=2)
        if scale:
            X = self.scaleData(X)
        X_reduced = sfm.fit_transform(X,Y)
      #  print(X_reduced)
       # print(sfm.get_support())
        return sfm.get_support()

    def selectFeaturesByRF(self, X, Y, RFKwargs = {}, scale = True):
        "Random forest selection"
        rfc = RandomForestClassifier(**RFKwargs) #random firest classifier
        if scale:
            X = self.scaleData(X)
        rfc.fit(X,Y)
        sfm = SelectFromModel(rfc, max_features = self.maxFeatures, prefit = True)
        sfm.transform(X)
        return sfm.get_support(), rfc.feature_importances_

    def selectFeaturesByRFECV(self,X,Y,model = "Random Forest", ModelKwargs = {}, cv = 3, scale=True):
        ""
        if model == "Random Forest":
            modelEstimator = RandomForestClassifier(**ModelKwargs)
        else:
            modelEstimator = SVC(**ModelKwargs)

        sfm = RFECV(estimator = modelEstimator, min_features_to_select= self.maxFeatures, cv=cv)
        if scale:
            X = self.scaleData(X)
        sfm.fit(X,Y)
        return sfm.get_support()


    def scaleData(self,X):
        ""
        return StandardScaler().fit_transform(X)