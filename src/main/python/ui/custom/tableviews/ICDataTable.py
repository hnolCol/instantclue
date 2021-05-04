from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

#ui utils
from ...utils import TABLE_ODD_ROW_COLOR, WIDGET_HOVER_COLOR, HOVER_COLOR, createTitleLabel, getMessageProps, createLabel
from .ICVSelectableTable import PandaTable, PandaModel
from ..warnMessage import AskQuestionMessage

#external imports
import pandas as pd 
import numpy as np
from collections import OrderedDict
#to do
#move to dialogs

contextMenuData = OrderedDict([
            ("deleteRows",{"label":"Delete Row(s)","fn":"deleteRows"}),
            ("copyRows",{"label":"Copy Row(s)","fn":"copyRows"}),
            ("copyData",{"label":"Copy Data Frame","fn":"copyDf"})
        ])


class PandaTableDialog(QDialog):

    def __init__(self, mainController, df, headerLabel = "Source Data", addToMainDataOption = True, ignoreChanges =  False, *args, **kwargs):
        super(PandaTableDialog,self).__init__(*args, **kwargs)
        
        self.headerLabelText = "{} ({} rows x {} columns)".format(headerLabel,*df.shape)
        self.addToMainDataOption = addToMainDataOption
        self.ignoreChanges = ignoreChanges
        self.df = df
        self.mC = mainController

        self.__control()
        self.__layout()
        self.__connectEvents()
        self.addData(df)

    def sizeHint(self):
        ""
        return QSize(650,600)

    def __control(self):
        ""
        self.headerLabel = createTitleLabel(self.headerLabelText, fontSize=12)
        self.selectionLabel = createLabel("0 row(s) selected")
        self.table = PandaTable(parent= self, mainController = self.mC) 
        self.model = PandaModel()#PandaModel()
        self.table.setModel(self.model)

        if self.addToMainDataOption:

            self.addDataToMain = QPushButton("Save")
            self.closeButton = QPushButton("Close")

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        hbox = QHBoxLayout()
        hbox.addWidget(self.headerLabel)
        hbox.addStretch(1)
        hbox.addWidget(self.selectionLabel)
        self.layout().addLayout(hbox)
        self.layout().addWidget(self.table)

        if hasattr(self,"addDataToMain"):
            hbox = QHBoxLayout()
            hbox.addWidget(self.addDataToMain)
            hbox.addWidget(self.closeButton)
            self.layout().addLayout(hbox)

    def __connectEvents(self):
        ""
        if hasattr(self,"closeButton"):
            self.closeButton.clicked.connect(self.close)

    def sendMessage(self, messageProps):
        ""

        self.parent().mC.sendMessageRequest(messageProps) 

    def sendToThread(self,funcProps):

        self.parent().mC.sendRequestToThread(funcProps)

    def setSelectedRowsLabel(self, nRows = 0):
        ""
        
        if isinstance(nRows,int):
            self.selectionLabel.setText("{} row(s) selected".format(nRows))

    def addData(self,X = pd.DataFrame()):
        ""
       
        if isinstance(X,pd.DataFrame):
            self.table.model().layoutAboutToBeChanged.emit()
            self.table.model().updateDataFrame(X)
            self.table.model().layoutChanged.emit()

    def closeEvent(self,e=None):
        if self.ignoreChanges:
            e.accept()
        elif self.df.equals(self.table.model().df):
            e.accept()
        else:
            quest = AskQuestionMessage(title = "Question", infoText = "Data have changed. Update data?")
            quest.exec_()
            if quest.state:
                funcProps = dict() 
                funcProps["key"] = "data::updateData"
                funcProps["kwargs"] = {"dataID":self.parent().getDataID(),"data":self.model.df}
                self.sendToThread(funcProps)
                
            self.close()
    
    