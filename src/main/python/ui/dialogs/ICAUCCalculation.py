from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from ..utils import createCombobox, createTitleLabel, createLabel

from ..custom.buttonDesigns import ICStandardButton


class ICDAUCDialog(QDialog):
    def __init__(self, mainController, dataID, columnPairs, data, *args, **kwargs):
        super(ICDAUCDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Area under curve.")
        self.mC = mainController
        self.dataID = dataID
        self.columnPairs = columnPairs
        self.chartData = data
        
        self.__controls()
        self.__layout()
        self.__connectEvents()

        #set size policy of dialog
        self.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Expanding)
       

    def __controls(self):
        """Init widgets"""
        
        self.headerLabel = createTitleLabel("Calculate area under curve", fontSize=14)
        self.infoLabel = createLabel("Uses the data of the xyplot to calculate the AUC. You can specify a column below that contains information about replicates. ")
        self.replicateColumn = createCombobox(self,items = ["None"] + self.mC.data.getNonFloatColumns(self.dataID).values.tolist())
    
        self.resultView = createCombobox(self,items = ["Add as data frame","Just display"])
        self.resultView.setToolTip("Choose how the results should be displayed. 'Just display' opens a window with the results, you can copy and export the data but not directly add them to the InstantClue data frame list.")
        self.okButton = ICStandardButton(itemName="Okay")
        self.cancelButton = ICStandardButton(itemName = "Cancel")

    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.headerLabel)
        self.layout().addWidget(self.infoLabel)
        self.layout().addWidget(self.replicateColumn)
        self.layout().addWidget(self.resultView)
        hbox = QHBoxLayout()
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.cancelButton)

        self.layout().addLayout(hbox)
       
    def __connectEvents(self):
        """Connect events to functions"""
        self.cancelButton.clicked.connect(self.close)
        self.okButton.clicked.connect(self.calculate)

    def calculate(self,e=None):
        "Sends a request to thread in order to calculate the AUC from the graph data. "
        funcKey =  "statistic::calculateAUC"
        funcKwargs = {
                "numericColumnPairs" : self.columnPairs,
                "dataID":self.dataID,
                "chartData":self.chartData,
                "replicateColumn": self.replicateColumn.currentText(),
                "addAsDataFrame" : self.resultView.currentText() == "Add as data frame",
                }
        self.mC.sendRequestToThread({"key":funcKey,"kwargs":funcKwargs})