import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def getTxtFilesFromDir(pathToDir):
    ""
    if os.path.exists(pathToDir):
        filesInDir = os.listdir(pathToDir)
        txtFiles = [file for file in filesInDir if file.endswith(".txt") and not file.startswith("._")]
        return txtFiles
    return []

def getKeyMatchInValuesFromDict(value,inputDict):
    ""
    if isinstance(inputDict,dict):
        for k,v in inputDict.items():
            if value in v:
                return k 

def replaceKeyInDict(toReplaced, inputDict, replaceValue):
    ""
    if toReplaced in inputDict:
        inputDict.update({toReplaced: replaceValue})

    return inputDict


def scaleBetween(data,feature_range=(0,1)):
    ""
    xmin, xmax = feature_range
    minV = np.nanmin(data)
    maxV = np.nanmax(data)
    return (data - minV) / (maxV - minV) * (xmax - xmin) + xmin
    #scaler = MinMaxScaler(feature_range=feature_range)
    #return scaler.fit_transform(data)
