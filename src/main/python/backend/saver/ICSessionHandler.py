
import os, sys
import re
import pandas as pd
import pickle 
from ..utils.stringOperations import getMessageProps

class ICSessionHandler(object):
    ""
    def __init__(self,mainController):
        self.mC = mainController

    def saveSession(self,sessionPath, overwrite = False):
        "Saves session"
        # if not overwrite and os.path.exists(sessionPath):
        #     return False, "Session exists."
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
       
        #get data from graph
        exists, graph = self.mC.getGraph()
        if exists:
                
                plotType = graph.plotType
                graphData = graph.data
                if graph.hasTooltip():
                    tooltipColumnNames = graph.getTooltipData()
                else:
                    tooltipColumnNames = []


        else:
            plotType = None
            graphData = {}
            tooltipColumnNames = []

        #get groupings
        groupingState = self.mC.grouping.getAllGroupings()

        #get current dataset index
        comboboxIndex = self.mC.mainFrames["data"].dataTreeView.getDfIndex()
        
        #receiverBoxState 
        receiverBoxItems = self.mC.mainFrames["middle"].getReceiverBoxItems()
        
        comboSettings = []#self.mC.mainFrames["right"].mainFigureRegistry.getMainFigureCurrentSettings() 
       # print(mainFigureRegistry.mainFigureTemplates)
        combinedObj = {
                    "dfs":dataFrames,
                    "dfID":currentDataFrameID,
                    "dfsName":dataFrameNames,
                    "graphData": graphData,
                    "currentPlotType" : plotType,
                    "mainFigures":mainFigures,
                    "mainFigureRegistry":mainFigureRegistry,
                    "mainFigureComboSettings":comboSettings,
                    "dataComboboxIndex" : comboboxIndex,
                    "receiverBoxItems":receiverBoxItems,
                    "groupingState": groupingState,
                    "tooltipColumnNames":tooltipColumnNames}
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

        response["dataComboboxIndex"] = combinedObj["dataComboboxIndex"]

        if "graphData" in combinedObj and "currentPlotType" in combinedObj and combinedObj["currentPlotType"] is not None:
            response["graphData"] = combinedObj["graphData"]
            response["plotType"] = combinedObj["currentPlotType"]

        if "receiverBoxItems" in combinedObj:
            response["receiverBoxItems"] = combinedObj["receiverBoxItems"]
            
        if "groupingState" in combinedObj:
            response["groupingState"] = combinedObj["groupingState"]

        if "tooltipColumnNames" in combinedObj:
            response["tooltipColumnNames"] = combinedObj["tooltipColumnNames"]
        response["sessionIsBeeingLoaded"] = True
        response["dataID"] = dataID
        return response