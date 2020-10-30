from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from ..custom.buttonDesigns import ICStandardButton
from ..utils import createTitleLabel, createLabel, createLineEdit, createCombobox




class ICCompareGroups(QDialog):
    def __init__(self, test, mainController, *args, **kwargs):
        super(ICCompareGroups, self).__init__(*args, **kwargs)

        self.test = test
        self.mC = mainController

        self.__controls()
        self.__layout()
        self.__connectEvents()
    
    def __controls(self):
        """Init widgets"""
        
        self.headerLabel = createTitleLabel("Compare groups using: {}".format(self.test),fontSize=15)
        self.grouplabel = createLabel("Grouping:","Set grouping to compare. By default the currently selected grouping is shown.")
        self.groupCombo = createCombobox(self,self.mC.grouping.getGroupings())
        self.groupCombo.setCurrentText(self.mC.grouping.getCurrentGroupingName())

        self.refLabel = createLabel("Reference:","Groups will be compared against this reference only. Set None if you want to have all combinations.")
        self.referenceGroupCombo =  createCombobox(self,["None"] + self.mC.grouping.getCurrentGroupNames())
        

        
        self.okayButton = ICStandardButton("Okay")
        self.cancelButton = ICStandardButton("Cancel")


    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.headerLabel)
        groupGrid = QGridLayout()
        groupGrid.addWidget(self.grouplabel,0,0, Qt.AlignRight)
        groupGrid.addWidget(self.groupCombo,0,1)
        groupGrid.addWidget(self.refLabel,2,0, Qt.AlignRight)
        groupGrid.addWidget(self.referenceGroupCombo,2,1)

        groupGrid.setColumnStretch(0,0)
        groupGrid.setColumnStretch(1,1)
        groupGrid.setHorizontalSpacing(2)

        hbox = QHBoxLayout() 
        hbox.addWidget(self.okayButton)
        hbox.addWidget(self.cancelButton)

        self.layout().addLayout(groupGrid)
        self.layout().addLayout(hbox)

        self.layout().addStretch()
        
       
    def __connectEvents(self):
        """Connect events to functions"""
        self.cancelButton.clicked.connect(self.close)
        self.okayButton.clicked.connect(self.startCalculations)
        

    def startCalculations(self,event=None):
        """Start calculation by sending a request to thread."""
        self.mC.grouping.setCurrentGrouping(self.groupCombo.currentText())

        funcProps = {"key":"stats::compareGroups",
                      "kwargs":
                      {
                      "dataID":self.mC.getDataID(),
                      "grouping":self.mC.grouping.getCurrentGrouping(),
                      "test":self.test,
                      "referenceGroup":None if self.referenceGroupCombo.currentIndex() == 0 else self.referenceGroupCombo.currentText()}}
        
        self.mC.sendRequestToThread(funcProps)