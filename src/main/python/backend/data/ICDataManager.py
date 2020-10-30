import numpy as np
import pandas as pd

from collections import OrderedDict

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from ..utils.stringOperations import getMessageProps

sepConverter = {"tab":"\t","space":"\s+"}
dTypeConv = {"float64":"Numeric Floats","int64":"Integers","object":"Categories"}

import time
class ICDataManger(QObject):
    ""
    loadDf = pyqtSignal(str,str,dict)
    runDimenReduction = pyqtSignal(str,list)

    sendMsg = pyqtSignal(dict)
    updateDataTreeview = pyqtSignal(dict)
    updateDfs = pyqtSignal(dict, bool)

    def __init__(self,*args,**kwargs):
        QObject.__init__(self,*args,**kwargs)

        self.dfs = OrderedDict() 
        self.fileNameByID = OrderedDict()
        self.dfsDataTypesAndColumnNames = OrderedDict() 
        self.dataFrameId = 0

        self.loadDf.connect(self.loadDataFrame)
        self.runDimenReduction.connect(self.runPCA)
        
    @pyqtSlot()
    def addDataFrame(self,
                    dataFrame, 
                    dataID = None, 
                    fileName = '', 
					cleanObjectColumns = False):
        """
        Adds new dataFrame to collection (dfs).
        """
        if dataID is None:
            dataID  = self.getNextDataID()
        print(" i am at sleep")
        time.sleep(5)
        self.dfs[dataID] = dataFrame
        self.extractDataTypeOfColumns(dataID)
        self.saveFileName(dataID,fileName)
        dtypeColumns = self.dfsDataTypesAndColumnNames[dataID]
        self.updateDataTreeview.emit(dtypeColumns)
        self.updateDfs.emit(self.fileNameByID, False)
    @pyqtSlot()
    def loadDataFrame(self,pathToFile,fileName,loadFileProps ):
        """
        """
        print(pathToFile)
        try:
            if loadFileProps is None:
                loadFileProps = {"sep":"tab","skiprows":0}
            loadFileProps = self.checkLoadProps(loadFileProps)
            df = pd.read_csv(pathToFile,**loadFileProps)
            self.addDataFrame(df,fileName=fileName)
            self.sendMsg.emit({"title":"Loaded","message":"blub"})
        except Exception as e:
            print(e)
    @pyqtSlot()
    def checkLoadProps(self,loadFileProps):
        """
        """
        if loadFileProps is None:
            return {"sep":"\t"}
        if loadFileProps["sep"] in ["tab","space"]:
            loadFileProps["sep"] = sepConverter[loadFileProps["sep"]]
        if "thousands" not in loadFileProps or loadFileProps["thousands"] == "None":
            loadFileProps["thousands"] = None
        try:
            skiprows = int(loadFileProps["skiprows"])
        except:
            skiprows = 0
        loadFileProps["skiprows"] = skiprows
        return loadFileProps
    @pyqtSlot()
    def extractDataTypeOfColumns(self,dataID):
        '''
        Saves the columns name per data type. In InstantClue there is no difference between
        objects and others non float, int-like columns.
        '''
        dataTypeColumnRelationship = dict() 
        for dataType in ['float64','int64','object']:
            try:
                if dataType != 'object':
                    dfWithSpecificDataType = self.dfs[dataID].select_dtypes(include=[dataType])
                else:
                    dfWithSpecificDataType = self.dfs[dataID].select_dtypes(exclude=['float64','int64'])
            except ValueError:
                dfWithSpecificDataType = pd.DataFrame() 		
            columnHeaders = dfWithSpecificDataType.columns.values.tolist()
            dataTypeColumnRelationship[dTypeConv[dataType]] = pd.Series(columnHeaders)
                
        self.dfsDataTypesAndColumnNames[dataID] = dataTypeColumnRelationship	
    @pyqtSlot()
    def getNextDataID(self):
        """
        To provide consistent labeling, use this function to get the id the new df should be added
        """
        self.dataFrameId += 1
        idForNextDataFrame = 'DataFrame: {}'.format(self.dataFrameId)
        
        return idForNextDataFrame
    @pyqtSlot()
    def saveFileName(self,dataID,fileName):
        '''
        '''
        self.fileNameByID[dataID] = fileName
    @pyqtSlot()
    def runPCA(self,dataID,columnNames):
        ""
        try:
            X = self.dfs[dataID][columnNames].dropna()
            M = np.mean(X.T,axis=1)
            X = X - M
            #calc embedding values
            V = np.cov(X.T)
            # eigendecomposition of covariance matrix
            values, vectors = np.linalg.eig(V)
            # project data
            P = vectors.T.dot(X.T)
            print(P)
        except Exception as e:
            print(e)