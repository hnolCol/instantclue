import numpy as np
import re
import matplotlib 

REQUIRED_ATTR_NAMES = ["id","name","value","dtype","range","parent","parentType","description"]
OPTIONAL_ATTR_NAMES = ["regEx"]

ATTR_NAMES = REQUIRED_ATTR_NAMES + OPTIONAL_ATTR_NAMES

class Parameter(object):
    def __init__(self, paramID = None, value = np.nan, dtype = "any", updateParamInParent = None):
        """
        Instant Clue Parameter Class
        
        Attributes
        ============================

        """
        self.params = {
            "id"    :   paramID,
            "dtype" :   dtype,
            "value" :   value
        }

        self.updateParamInParent = updateParamInParent
   
    
    def checkDtype(self,attrName, value):
        ""
        
        if self.params["dtype"] == "any":
            return True
        else:
            return isinstance(value,self.params["dtype"])
    
    def checkRange(self, value=None):
        ""
        

        if value is None:
            value = self.params["value"]
        if "range" not in self.params:
            return False
        elif self.params["range"] == "any":
            return True
        elif self.params["range"] == "regExMatch":
            return re.search(self.params["regEx"], value)
        elif isinstance(self.params["range"],list) and self.params["dtype"] == str:
            return value in self.params["range"]
        elif self.params["dtype"] == bool:
            return isinstance(self.params["value"],bool)
        elif self.params["dtype"] in [float,int]:
            return self.params["range"][0] <= value <= self.params["range"][1]
        else:
            return True

    def isValid(self):
        "Returns bool if parameter is valid."
        return all(attrName in self.params for attrName in  REQUIRED_ATTR_NAMES)

    def getAttr(self,attrName):
        "Returns attribute"
        if attrName in self.params:
            return self.params[attrName]

    def readFromDict(self,dictAttr,ignoreRange=True):
        "Read paramater values from dict."
        #store old params, if values are not valid
        prevParams = self.params.copy() 
        if isinstance(dictAttr,dict):
            for k,v in dictAttr.items():
                self.setAttr(k,v,ignoreRange)
            if not self.checkRange() or not self.isValid():
                self.params = prevParams
                print("old params restored")

    def setAttr(self,attrName,value,ignoreRange = False):
        "Set the value of an attribute"
        if attrName in ATTR_NAMES:
            if ignoreRange or self.checkRange(value):
                self.params[attrName] = value
                

    def updateAttrInParent(self):
        "Update value of parameter in parent"
        if all(k in self.params for k in ["parent","value","name"]):
            if isinstance(self.params["parent"],str):
                if self.params["parent"] == "matplotlib":
                    if self.params["name"] in matplotlib.rcParams:
                        matplotlib.rcParams[self.params["name"]] = self.params["value"]
                else:
                    if self.updateParamInParent is not None and callable(self.updateParamInParent):

                        self.updateParamInParent(self.params["parent"],self.params["name"],self.params["value"])
                
            else:
                setattr(self.params["parent"],self.params["name"],self.params["value"])
