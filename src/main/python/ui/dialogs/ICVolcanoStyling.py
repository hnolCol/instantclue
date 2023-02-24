from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from ..utils import createTitleLabel, createLabel, createLineEdit, createMenu
from ..custom.utils import LabelLikeCombo
from .ICDSelectItems import ICDSelectItems
from ..custom.buttonDesigns import ICStandardButton, LabelLikeButton
import pandas as pd 
import  numpy as np 
from collections import OrderedDict
from typing import List 

class ICVolcanoPlotStyling(QDialog):
    def __init__(self, mainController, dataID, numericColumns, categoricalColumns, *args, **kwargs):
        super(ICVolcanoPlotStyling, self).__init__(*args, **kwargs)
        self.setMaximumWidth(600)
        self.mC = mainController 
        self.dataID = dataID
        self.numericColumns = numericColumns
        self.nColumns = len(numericColumns)
        self.volcanos = self.nColumns/2
        self.categoricalColumns = categoricalColumns
        self.selectedSignificanceColumns = pd.Series()
        self.colorColumns = pd.Series() 
        self.sigStr = self.mC.config.getParam("scatter.volcano.significance.str")
        self.__controls() 
        self.__layout()
        self.__connectEvents()

    def __controls(self) -> None:
        ""
        self.headerLabel = createTitleLabel("Volcano Plot Styling")
        self.infoLabel = createLabel(f"Specify the settings for the volcano plot. {self.nColumns} were detected which were used for plotting.")
        self.infoLabel2 = createLabel(f"In total there are {self.volcanos} column pairs. Therefore please select {self.volcanos} significant columns for each volcano. The significance columns should contain the specified significant str {self.sigStr}. You can alter this in the scatter properties.")
        self.sigHeader = createTitleLabel("Select significant columns", fontSize=12)
        self.sigColumnSelection = ICStandardButton(itemName="Significance Columns",itemBorder=25)
        self.infoLabel3 = createLabel("No columns selected ...")
        self.warningLabel = createTitleLabel("",fontSize=12,colorString="#d03a00")

        self.extraColorHeader = createTitleLabel("Select columns for extra color coding", fontSize=12)
        self.colorColumnSelection = ICStandardButton(itemName="Color Columns",itemBorder=25)
        self.infoLabel4 = createLabel("Extra columns are combined with the significant columns. The colors will differentiate between up and down regulated.")
        self.okButton = ICStandardButton(itemName="Okay")
        self.cancelButton = ICStandardButton(itemName = "Cancel")

    def __layout(self) -> None:
        "" 
        self.setLayout(QVBoxLayout())
        vboxMainLayout = self.layout()
        vboxMainLayout.addWidget(self.headerLabel)
        vboxMainLayout.addWidget(self.infoLabel) 
        vboxMainLayout.addWidget(self.infoLabel2)
        vboxMainLayout.addWidget(self.sigColumnSelection)
        vboxMainLayout.addWidget(self.infoLabel3)
        vboxMainLayout.addWidget(self.warningLabel )
        vboxMainLayout.addWidget(self.extraColorHeader)
        vboxMainLayout.addWidget(self.infoLabel4)
        vboxMainLayout.addWidget(self.colorColumnSelection)
        #add the okay, cancel buttons
        hbox = QHBoxLayout()
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.cancelButton)

        vboxMainLayout.addLayout(hbox)

    def __connectEvents(self):
        ""
        self.sigColumnSelection.clicked.connect(self.openSigColumnSelection)
        self.colorColumnSelection.clicked.connect(self.openColorColumnSelection)
        self.okButton.clicked.connect(self.checkInputAndClose)
        self.cancelButton.clicked.connect(self.close)
    
    def checkInputAndClose(self) -> None:
        if self.selectedSignificanceColumns.size > 0:
            self.accept()

        self.close()

    def getColorColumns(self) -> pd.Series:
        ""
        return self.colorColumns

    def getSignificantColumns(self) -> pd.Series:
        ""
        return self.selectedSignificanceColumns

    def openColorColumnSelection(self,*args,**kwargs) -> None:
        ""
        selectionDialog = ICDSelectItems(data = pd.DataFrame(self.categoricalColumns),title="Significance Column Selection", selectAll=False)
        if selectionDialog.exec_():
            selectedItems = selectionDialog.getSelection()
            self.colorColumns = pd.Series(selectedItems.values.flatten())


    def openSigColumnSelection(self,*args,**kwargs) -> None:
        ""
        #print(self.categoricalColumns)

        selectionDialog = ICDSelectItems(data = pd.DataFrame(self.categoricalColumns),title="Significance Column Selection", selectAll=False)
        if selectionDialog.exec_():
            selectedItems = selectionDialog.getSelection()
            nSelected = selectedItems.size
            if self.volcanos == nSelected:
                self.selectedSignificanceColumns = pd.Series(selectedItems.values.flatten())
                data = self.mC.data.getDataByColumnNames(self.dataID,self.selectedSignificanceColumns)["fnKwargs"]["data"]
                foundAnySignifiance = np.any(data.values == self.sigStr)

                if not foundAnySignifiance:
                    self.warningLabel.setText(f"Warning: The significance string {self.sigStr} was not detected in selected columns.")
                else:
                    self.warningLabel.setText("Significance string found. Looks good.")
                
                columnText = " ".join([str(x) for x in selectedItems.values.flatten()]) #get values from pandas series
                self.infoLabel3.setText(f"Selected column(s): {columnText}")
            else:
                self.mC.sendToWarningDialog(infoText=f"The number of significance columns must either be {self.volcanos}")
                self.warningLabel.setText("The number of columns does not match.")  
                self.infoLabel3.setText("")
                self.selectedSignificanceColumns = pd.Series()