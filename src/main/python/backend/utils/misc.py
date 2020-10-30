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


def replaceKeyInDict(toReplaced, inputDict, replaceValue):
    ""
    if toReplaced in inputDict:
	    inputDict.update({toReplaced: replaceValue})

    return inputDict


def scaleBetween(data,feature_range=(0,1)):
    ""
    minV = np.nanmin(data)
    maxV = np.nanmax(data)
    return (data - minV) / (maxV - minV)
    #scaler = MinMaxScaler(feature_range=feature_range)
    #return scaler.fit_transform(data)
