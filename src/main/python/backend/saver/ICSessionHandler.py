
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
        currentDataFrameID = self.mC.data.dataFrameId
        if len(dataFrames) == 0:
            return False, "No data loaded."

        #get main figures
        
       # print(mainFigures)
        mainFigureRegistry = self.mC.mainFrames["right"].mainFigureRegistry
        mainFigureRegistry.removeLabels()
        mainFigures = self.mC.mainFrames["right"].mainFigureRegistry.getMainFiguresByID()
       
        comboSettings = []#self.mC.mainFrames["right"].mainFigureRegistry.getMainFigureCurrentSettings() 
       # print(mainFigureRegistry.mainFigureTemplates)
        combinedObj = {
                    "dfs":dataFrames,
                    "dfID":currentDataFrameID,
                    "dfsName":dataFrameNames,
                    "mainFigures":mainFigures,
                    "mainFigureRegistry":mainFigureRegistry,
                    "mainFigureComboSettings":comboSettings}
       # print(combinedObj)
        with open(sessionPath,"wb") as icSession:
            pickle.dump(combinedObj,icSession)
        
        return getMessageProps("Done..","Session saved ..")
        

    def openSession(self,sessionPath):
        ""
        with open(sessionPath,"rb") as icSession:
            combinedObj = pickle.load(icSession)
      #  print(combinedObj)
        
        for dataID, dataFrame in combinedObj["dfs"].items():
            response = self.mC.data.addDataFrame(dataFrame,dataID,fileName = combinedObj["dfsName"][dataID])
      #  print(response)
        #open mainf figures
        self.mC.data.dataFrameId = combinedObj["dfID"]
        response["mainFigures"] = combinedObj["mainFigures"] #cannot be done from WorkingThread
        response["mainFigureRegistry"] = combinedObj["mainFigureRegistry"]
        response["mainFigureComboSettings"] = []#combinedObj["mainFigureComboSettings"]
        print("this is in saved",combinedObj["mainFigureRegistry"].mainFigureTemplates)
        print(combinedObj["mainFigures"])
        return response