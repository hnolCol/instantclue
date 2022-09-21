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
        self.grouplabel = createLabel("1. Grouping:","Set grouping to compare. By default the currently selected grouping is shown.")
        self.groupCombo = createCombobox(self,self.mC.grouping.getGroupings())
        self.groupCombo.setCurrentText(self.mC.grouping.getCurrentGroupingName())

        # if self.test == "2W-ANOVA":
        #     self.grouplabel2 = createLabel("2. Grouping:","Set grouping to compare. By default the currently selected grouping is shown.")
        #     self.groupCombo2 = createCombobox(self,self.mC.grouping.getGroupings())
        #     self.groupCombo2.setCurrentText("Select ..")

        if self.test not in ["1W-ANOVA"]:
            self.refLabel = createLabel("Reference:","Groups will be compared against this reference only. Set None if you want to have all combinations.")
            self.referenceGroupCombo =  createCombobox(self,["None"] + self.mC.grouping.getCurrentGroupNames())
        
        self.logPValuesCB = QCheckBox("-logp10 p-value")
        self.logPValuesCB.setTristate(False)
        self.logPValuesCB.setCheckState(True)
        self.logPValuesCB.setChecked(True)
        
        self.okayButton = ICStandardButton("Okay")
        self.cancelButton = ICStandardButton("Cancel")


    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.headerLabel)
        groupGrid = QGridLayout()
        groupGrid.addWidget(self.grouplabel,0,0, Qt.AlignRight)
        groupGrid.addWidget(self.groupCombo,0,1)
        if self.test not in ["2W-ANOVA"]:
            groupGrid.addWidget(self.refLabel,2,0, Qt.AlignRight)
            groupGrid.addWidget(self.referenceGroupCombo,2,1)
        else:
            groupGrid.addWidget(self.grouplabel2,2,0, Qt.AlignRight)
            groupGrid.addWidget(self.groupCombo2,2,1)

        groupGrid.addWidget(self.logPValuesCB)
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
        self.groupCombo.currentIndexChanged.connect(self.groupingChanged)
        self.cancelButton.clicked.connect(self.close)
        self.okayButton.clicked.connect(self.startCalculations)
        

    def groupingChanged(self,newComboIndex):
        ""
        grouping = self.mC.grouping.getGroupings()[newComboIndex]
        
        groupNames = self.mC.grouping.getGroupNames(grouping)
        self.referenceGroupCombo.clear()
        self.referenceGroupCombo.addItems(["None"] + groupNames)
        

    def startCalculations(self,event=None):
        """Start calculation by sending a request to thread."""
        self.mC.grouping.setCurrentGrouping(self.groupCombo.currentText())

        referenceGroup = None
        if hasattr(self,"referenceGroupCombo"):
            if self.referenceGroupCombo.currentIndex() != 0:
                referenceGroup = self.referenceGroupCombo.currentText()
        if self.test == "2W-ANOVA":
            groupName1 =  self.groupCombo.currentText()
            groupName2 =  self.groupCombo2.currentText()
            grouping = {"betweenGroupings"  :   [{"name":groupName1, "values": self.mC.grouping.getGrouping(groupName1)},
                                                 {"name": groupName2, "values": self.mC.grouping.getGrouping(groupName2)}],
                        "withinGroupings"    :   []}
        else:
            groupingName = self.groupCombo.currentText()

        funcProps = {"key":"stats::compareGroups",
                      "kwargs":
                      {
                      "dataID":self.mC.getDataID(),
                      "grouping":groupingName,
                      "test":self.test,
                      "referenceGroup": referenceGroup,
                      "logPValues" : self.logPValuesCB.isChecked()
                      }
                    }
        
        self.mC.sendRequestToThread(funcProps)