
from .data.params import DEFAULT_PARAMETER
from .paramter import Parameter

from collections import OrderedDict
import os 
import pickle
import matplotlib
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
class Config(object):
    ""
    def __init__(self, mainController):
        ""
        self.parameters = OrderedDict()
        self.mC = mainController
        self.loadDefaults()
        self.saveParameters()
        self.lastConfigGroup = None

    
    def clearSettings(self):
        "Clears parameters"
        self.parameters.clear()

    def loadDefaults(self):
        ""
        self.clearSettings()
        savedDefaultParam = self.loadParameters()
        if savedDefaultParam is None:
            savedDefaultParam = DEFAULT_PARAMETER
            
        for n,dictAttr in enumerate(savedDefaultParam):
            param = Parameter(paramID=n, updateParamInParent = self.updateParamInParent)
            param.readFromDict(dictAttr)
            self.parameters[param.getAttr("name")] = param
            param.updateAttrInParent()
    
    def loadParameters(self,settingName = "default"):
        "Load Parameters from pickled file"
        configFolder = os.path.abspath(os.path.join(self.mC.mainPath,"conf"))
        if not os.path.exists(configFolder):
            os.mkdir(configFolder)
        fileName = '{}.ic'.format(settingName)
        filePath = os.path.join(configFolder,fileName)
        if os.path.exists(filePath):
            with open(filePath, 'rb') as paramFle:
                l = pickle.load(paramFle)
                if isinstance(l,list):
                    return l

    def getParam(self,paramName):
        ""
        if paramName in self.parameters:
            return self.parameters[paramName].getAttr("value")
        
    def getParams(self, paramNames):
        ""
        if isinstance(paramNames,list):
            ps = []
            for paramName in paramNames:
                ps.append(self.getParam(paramName))
            return ps

    def getParamRange(self,paramName):
        ""
        if paramName in self.parameters:
            return self.parameters[paramName].getAttr("range")
        else:
            return []

    def getParentTypes(self):
        "Returns Parameters Parent Type"

        parentTypes = []
        for p in self.parameters.values():
            pType = p.getAttr("parentType")
            if pType not in parentTypes:
                parentTypes.append(pType)
        return parentTypes

    def getParametersByType(self,parentType):
        "Get Parameters of a specific type"
        return [p for p in self.parameters.values() if p.getAttr("parentType") == parentType]
    
    def saveParameters(self, settingName = "default", overwriteDefault = True):
        ""
        configFolder = os.path.abspath(os.path.join(self.mC.mainPath,"conf"))
        if not os.path.exists(configFolder):
            os.mkdir(configFolder)
        fileName = '{}.ic'.format(settingName)
        filePath = os.path.join(configFolder,fileName)
        if overwriteDefault or not os.path.exists(filePath):
            with open(filePath, 'wb') as paramFle:
                pickle.dump([p.params for p in self.parameters.values()], paramFle)


    def setParam(self,paramName,value):
        ""
        if paramName in self.parameters:
            self.parameters[paramName].setAttr("value",value)
            self.parameters[paramName].updateAttrInParent()

    def toggleParam(self,paramName):
        ""
        try:
            if paramName in self.parameters:
                prevValue = self.parameters[paramName].getAttr("value")
                if isinstance(prevValue,bool):
                    self.parameters[paramName].setAttr("value",not prevValue)
        except Exception as e:
            print(e)

    def updateParamInParent(self, objectName, paramName, paramValue):
        ""

        if hasattr(self.mC,objectName):
            obj = getattr(self.mC,objectName)
            setattr(obj,paramName,paramValue)
            
    
    def updateAllParamsInParent(self):
        ""
        for param in self.parameters.values():
            param.updateAttrInParent()
