
import os, sys
import pandas as pd
import pickle 
from ..utils.stringOperations import getMessageProps

class ICSessionHandler(object):
    ""
    def __init__(self,mainController):
        self.mC = mainController

    def saveSession(self,sessionPath, overwrite = False):
        "Saves session"
        if not overwrite and os.path.exists(sessionPath):
            return False, "Session exists."
        #data frames
        dataFrames = self.mC.data.dfs
        #names of data farmes
        dataFrameNames = self.mC.data.fileNameByID
        if len(dataFrames) == 0:
            return False, "No data loaded."

        #get main figures
        mainFigures = self.mC.mainFrames["right"].mainFigureRegistry.getMainFigures()
        print(mainFigures)

        combinedObj = {"dfs":dataFrames,"dfsName":dataFrameNames,"mainFigures":mainFigures}
        print(combinedObj)
        with open(sessionPath,"wb") as icSession:
            pickle.dump(combinedObj,icSession)
        
        return getMessageProps("Done..","Session saved ..")
        

    def openSession(self,sessionPath):
        ""
        with open(sessionPath,"rb") as icSession:
            combinedObj = pickle.load(icSession)
        print(combinedObj)
        for dataID, dataFrame in combinedObj["dfs"].items():
            response = self.mC.data.addDataFrame(dataFrame,dataID,fileName = combinedObj["dfsName"][dataID])
        print(response)
        #open mainf figures
        response["mainFigures"] = combinedObj["mainFigures"] #cannot be done from WorkingThread
        return response