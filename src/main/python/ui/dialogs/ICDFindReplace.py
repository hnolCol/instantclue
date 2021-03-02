from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from ..custom.buttonDesigns import ICStandardButton
from ..utils import createLabel, createLineEdit, createCombobox
from ..custom.warnMessage import WarningMessage
from ..custom.utils import QToggle
import csv


class FindReplaceDialog(QDialog):
    def __init__(self,mainController, *args, **kwargs):

        super(FindReplaceDialog,self).__init__(*args, **kwargs)

        self.mC = mainController
        self.dataID = self.mC.getDataID()
        self.selectedDataType = "Numeric Floats"
        self.selectedColumn = ""
        self.selectedColumnIndex = None

        self.__controls()
        self.__layout()
        self.__windowUpdate()
        self.__connectEvents()

    def __windowUpdate(self):

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowOpacity(0.95)
    
    def __controls(self):
        """Init widgets"""
        #combobox to choose mode (replacing on columnNames)
        self.modeCombo = createCombobox(self, items = ["Column names","Specific column"])
        self.modeCombo.setToolTip("Define the mode of the find & replace procedure.")
        self.modeCombo.currentTextChanged.connect(self.onModeChange)
        self.modeCombo.setEnabled(False if self.dataID is None else True)

        self.dataTypeCombo = createCombobox(self,items=["Categories","Integers","Numeric Floats"])
        self.dataTypeCombo.setToolTip("Select data type. If you select 'Current Column Selection' only columns from this type will be considered.\nThe input is transformed to the selected data type. If it fails, nan will be entered or the selected categorical value for missing values (default:'-')")
        self.dataTypeCombo.currentTextChanged.connect(self.onDatTypeChange)
        self.dataTypeCombo.setEnabled(False)

        self.columnCombo = createCombobox(self,items = ["Current Column Selection"] + self.mC.data.getPlainColumnNames(self.dataID).values.tolist())
        self.columnCombo.setEnabled(False)
        self.columnCombo.setToolTip("Selecting 'Current Column Selection' will perfrom replacement on current column selection.")
        #create line edits
        self.findLine = createLineEdit("Find string(s) ..",
                                   'For multiple strings use "str1","str2" ')
        

        self.replaceLine = createLineEdit("Replace with ..",
                                   'For multiple strings use either one "str1"\n or equal length of strings: "str1","str2".\nStrings will be replaced in given order.')
        
        self.mustMatchLabel = createLabel("Entire match:","If enabled, only cell that match the search string completely will be replaced.",fontSize=12)
        self.mustMatchCompleteCellButton = QToggle(self)
        # set up okay buttons
        self.okayButton = ICStandardButton("Okay")
        self.okayButton.setEnabled(False if self.dataID is None else True)
        #set up cancel button
        self.cancelButton = ICStandardButton("Cancel")
        
    def __layout(self):
        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(5,5,5,5)
        self.layout().setSpacing(1)
        self.layout().addWidget(self.modeCombo)
        self.layout().addWidget(self.columnCombo)
        self.layout().addWidget(self.dataTypeCombo)
        self.layout().addWidget(self.findLine)
        self.layout().addWidget(self.replaceLine)
        hbox = QHBoxLayout()
        hbox.addWidget(self.mustMatchLabel)
        hbox.addWidget(self.mustMatchCompleteCellButton)
        self.layout().addLayout(hbox)
        hboxButtons = QHBoxLayout()
        hboxButtons.addWidget(self.okayButton)
        hboxButtons.addWidget(self.cancelButton)
        self.layout().addLayout(hboxButtons)

    def __connectEvents(self):
        """Connect events to functions"""
        self.okayButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.close)
        #attach textChange listeners 
        self.findLine.textChanged.connect(self.onFindStringChange)
        self.replaceLine.textChanged.connect(self.onReplaceStringChange)

    def accept(self,event = None):
        ""
        if self.validateInput():
            if self.columnCombo.isEnabled():
                self.specificColumnSelected = True
                self.selectedColumn = self.columnCombo.currentText()
                self.selectedColumnIndex = self.columnCombo.currentIndex()
                self.mustMatchCompleteCell = self.mustMatchCompleteCellButton.isChecked()
            else:
                self.specificColumnSelected = False
            super().accept() 

    def keyPressEvent(self,e):
        """Handle key press event"""
        if e.key() == Qt.Key_Escape:
            self.reject()

    def onDatTypeChange(self, currentText):
        ""
        self.selectedDataType = currentText

    def onFindStringChange(self,currentText):
        ""
        self.findString = currentText

    def onReplaceStringChange(self,currentText):
        ""
        self.replaceString = currentText

    def onModeChange(self,currentText):
        ""
        self.columnCombo.setEnabled(currentText == "Specific column")
        self.dataTypeCombo.setEnabled(currentText == "Specific column")

    def validateInput(self):
        ""
        if not hasattr(self,"replaceString"):
            self.replaceString = ""
        if hasattr(self,"findString"):
            if len(self.findString) == 0:
                w = WarningMessage(infoText = "Please enter a search string.",iconDir = self.mC.mainPath)
                w.exec_()
                return False
            else:
                self.findStrings, self.replaceStrings = self.getStringLists()
                
                if len(self.replaceStrings) == 1 and len(self.findStrings) >= 1:
                    return True
                elif len(self.replaceStrings) > 1 and len(self.replaceStrings) == len(self.findStrings):
                    return True
                else:
                    w = WarningMessage(infoText = "Please enter either a replace string or a matching number of strings (find vs replace).")
                    w.exec_()
                    return False
        
        else:
            w = WarningMessage(infoText = "Please enter strings to find.")
            w.exec_()
            return False
       

    def getStringLists(self):
        """
            Returns strings as list. IF user enters "String1","String2. 
            It is split. If no replace string was give, '' will be set.
        """
        splitFindString = [row for row in csv.reader([self.findString], 
                                        delimiter=',', quotechar='\"')][0]
        if self.replaceString == "":
            splitReplaceString = [""]
        else:
            splitReplaceString = [row for row in csv.reader([self.replaceString], 
                                        delimiter=',', quotechar='\"')][0]

        return splitFindString, splitReplaceString