
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from ...utils import createLabel, createTitleLabel
from ..Selections.ICSelectionDialog import SelectionDialog

class ICGroupingSelection(SelectionDialog):
    def __init__(self, selectionNames, selectionOptions, selectionDefaultIndex, title="Selection", selectionEditable=[], previewString = "", *args, **kwargs):
        super().__init__(selectionNames, selectionOptions, selectionDefaultIndex, title, selectionEditable, *args, **kwargs)
        self.__connectEventsForPreview()
        self.previewString = previewString 

        if previewString != "":
            self.previewTitle = createLabel("Preview for {}:".format(previewString))
            self.previewLabel = createTitleLabel(previewString,fontSize=15,colorString="#d13316")
            self.gridLayout.addWidget(self.previewTitle)
            self.gridLayout.addWidget(self.previewLabel)
            self.splitPreviewString()

    def __connectEventsForPreview(self):
        """Connect events to functions"""
       
        for selectionName, ws in self.selectionCombos.items():
            
            ws["cb"].currentTextChanged.connect(lambda changedValue, selectionName=selectionName: self.updatePreview(selectionName,changedValue))
            

    def updatePreview(self,selectionName, textChanged):
        "Function to called upon even (e.g. value change)"
        self.splitPreviewString()

    def splitPreviewString(self):
        "Handles the split preview"
        if self.previewString != "":
            try:
                splitString = self.selectionCombos["splitString"]["cb"].currentText()
                if splitString == "space":
                    splitString = " "
                index = int(float(self.selectionCombos["index"]["cb"].currentText()))
                maxSplit = self.selectionCombos["maxSplit"]["cb"].currentText()
                if maxSplit == "inf": maxSplit = -1 
                else: maxSplit = int(self.selectionCombos["maxSplit"]["cb"].currentText())
                
                rsplit = self.selectionCombos["splitFrom"]["cb"].currentText() == "right"
                removeN = int(float(self.selectionCombos["remove N from right"]["cb"].currentText()))
                if rsplit:
                    prevLabel = self.previewString.rsplit(splitString,maxsplit=maxSplit)[index][:len(self.previewString)-removeN]
                else:
                    prevLabel = self.previewString.split(splitString,maxsplit=maxSplit)[index][:len(self.previewString)-removeN]

                self.previewLabel.setText(prevLabel)
            except:
                self.previewLabel.setText("Error in extracting params for preview.")

    