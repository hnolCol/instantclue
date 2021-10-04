from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..utils import createTitleLabel,createLabel, createLineEdit
from ..custom.warnMessage import WarningMessage
from ..custom.buttonDesigns import ICStandardButton
from collections import OrderedDict 
import numpy as np 


class ICDataInput(QDialog):

    def __init__(self, mainController, title = " ", valueNames = [], defaultValues = {}, valueTypes = {}, *args,**kwargs):
        super(ICDataInput,self).__init__(*args,**kwargs)

        self.title = title
        self.valueNames = valueNames
        self.valueTypes = valueTypes
        self.defaultValues = defaultValues

        self.mC = mainController
        
        self.lineEdits = OrderedDict()
        self.providedValues = OrderedDict()
        
        self.__conrols()
        self.__layout()

    
    def __conrols(self):
        ""
        self.titleLabel = createTitleLabel(self.title,fontSize=15)
        self.valueGrid = self.addData()

        self.okayButton = ICStandardButton(itemName="Okay")
        self.cancelButton = ICStandardButton(itemName="Cancel")

        self.okayButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)
        



    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.titleLabel)
        self.layout().addLayout(self.valueGrid)
        buttonBox = QHBoxLayout() 
        buttonBox.addWidget(self.okayButton)
        buttonBox.addWidget(self.cancelButton)

        self.layout().addLayout(buttonBox)

    
    def addData(self):
        ""
        grid = QGridLayout()
        for n,valueName in enumerate(self.valueNames):
            l = createLabel(valueName)
            edit = createLineEdit("Enter value",tooltipText=None)
            if valueName in self.defaultValues:
                edit.setText(str(self.defaultValues[valueName]))
            grid.addWidget(l,n,0)
            grid.addWidget(edit,n,1)
            self.lineEdits[valueName] = edit

        grid.setColumnStretch(1,1)
        grid.setColumnStretch(0,0)
        
        return grid

    def accept(self,event=None):
        "Overwrite Accept Fn. Checks for correct data input"
        if self.saveData(): 
            super().accept()
        

    def saveData(self):
        
        "Saves data if input can be converted to required data type"
        for valueName in self.valueNames:
            edit = self.lineEdits[valueName]
            value = self.getValueType(valueName,edit.text())
            if value is None:
                warn = WarningMessage(title="Warning",
                    iconDir = self.mC.mainPath,
                    infoText = "Value for parameter {} could not be converted to requested type {}.".format(valueName,self.valueTypes[valueName]))
                warn.exec_()
                return False
            self.providedValues[valueName] = value
        else:
            return True
        

    def getValueType(self,valueName,value):
        ""
        if valueName in self.valueTypes:
            dtype = self.valueTypes[valueName]
            if dtype == str:
                return str(value)
            elif dtype == float:
                try:
                    return float(value)
                except:
                    return None
            elif dtype == int:
                try:
                    return int(float(value))
                except:
                    return None



