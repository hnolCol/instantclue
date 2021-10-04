


from operator import pos
from random import sample
from typing import OrderedDict
import numpy as np 
import pandas as pd 
import string

from scipy.linalg.special_matrices import convolution_matrix
from scipy.sparse import base
from ..utils.stringOperations import getMessageProps, getRandomString, mergeListToString
from datetime import datetime


class ICSampleListCreator(object):
    def __init__(self,numberRows = 8, numberColumns = 12, samplesInRows = True, sampleStartPositionIndex = 0, scramble = True, addDate = True, *args,**kwargs):
        ""
        self.samplesInRows = samplesInRows
        self.numberRows = numberRows
        self.numberColumns = numberColumns
        self.startIndex = sampleStartPositionIndex
        self.scramble = scramble
        self.addDate = addDate
        self.constants = OrderedDict()
        self.updateIndexedPositions()
        
    def setAddDate(self,newAddDate):
        ""
        self.addDate = newAddDate

    def setConstants(self, constantString):
        "constantString = paramName,paramValue;otherParamName,otherParamValue"
        #print(constantString)
        if len(constantString) == 0:
            self.constants= OrderedDict()  
        elif "," not in constantString:
            self.constants = OrderedDict()
        else:
            constants = constantString.split(";")
            constantValues = [x.split(",") for x in constants]
       
            self.constants = OrderedDict([(constName,constValue) for constName, constValue in constantValues])

    def setRowNumber(self,n):
        ""
        self.numberRows = n 

    def setColumnNumber(self,n):
        ""
        self.numberColumns = n

    def setStartIndex(self,startPos):
        ""
        if isinstance(startPos,int):
            self.startIndex = startPos

    def setScramble(self,scrambleList):
        ""
        if isinstance(scrambleList,bool):
            self.scramble = scrambleList

    def setSamplesInRows(self,samplesInRows):
        ""
        if isinstance(samplesInRows,bool):
            self.samplesInRows = samplesInRows
    
    def updateIndexedPositions(self, numberOfSamples = None):
        ""
        rowLetters = list(string.ascii_uppercase)[:self.numberRows]
        
        columnNumbers = np.arange(self.numberColumns) + 1
        if self.samplesInRows:
            positionsOnPlate = ["{}{}".format(rowLetter,columnNumber) for rowLetter in rowLetters for columnNumber in columnNumbers]
        else:
            positionsOnPlate = ["{}{}".format(rowLetter,columnNumber) for columnNumber in columnNumbers for rowLetter in rowLetters ]
        if numberOfSamples is None:
            self.positionsOnPlate = positionsOnPlate * 100
        else:
            self.positionsOnPlate = positionsOnPlate * int(numberOfSamples/len(positionsOnPlate)+10)

    def createSampleList(self, numberSamples=25, baseSampleName = ""):
        ""
         
        constantColumnNames = list(self.constants.keys())
        sampleNames = self.createSampleNames(numberSamples,baseSampleName)
        if sampleNames is None:
            return getMessageProps("Error..","Number of samples exceeds plate positions.")
        self.updateIndexedPositions(numberSamples)
        df = pd.DataFrame(columns=["Sample Name"]+constantColumnNames+["Position"])
        df["Sample Name"] = self.sampleNames
        for constantName,constantValue in self.constants.items():
            df.loc[:,constantName] = constantValue
        
        df["Position"] = self.positionsOnPlate[self.startIndex:self.startIndex+numberSamples]
        n = 1
        plateIdx = []
        for idx,pos in enumerate(df["Position"].values):
            if pos == "A1" and idx != 0:
                n += 1
            plateIdx.append("P"+str(n))
        df["Plate"] = plateIdx
     
        if self.scramble: 
            df = df.groupby("Plate").sample(frac=1)
        
        return df

    
        
    def createSampleNames(self,numberSamples,baseString=""):
        ""
        if self.addDate:
            baseString = datetime.now().strftime("%Y%m%d") + "_" + baseString
        # if numberSamples > self.numberColumns * self.numberRows:
        #     return 
        if self.numberColumns * self.numberRows > 99:
            sampleNames = ["{}_{:03d}".format(baseString,n+1)  for n in range(numberSamples)]
        else:
            sampleNames = ["{}_{:02d}".format(baseString,n+1)  for n in range(numberSamples)]

        self.sampleNames = sampleNames 
        return sampleNames 
        
    def getColumnNames(self):
        ""
        return ["Sample Name","Position","Plate"] + list(self.constants.keys())

    def getExampleName(self,baseString):
        ""
        example = self.createSampleList(1,baseString)[["Sample Name","Position"]].values.flatten()
        name, position = example
        return "{} : {}".format(name,position)

    def getPositionByIndex(self,index):
        ""
        if len(self.positionsOnPlate) - 1 < index:
            return "Index not valid on plate." 
        return self.positionsOnPlate[index]
    
if __name__ == "__main__":

    ex = ICSampleListCreator(samplesInRows=False)
    ex.setConstants("Inj Vol,3;Sample Type,Unknown")
    ex.createSampleList()

    