from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from ...custom.Widgets.ICButtonDesgins import ICStandardButton
from ...utils import createTitleLabel, createLabel, createLineEdit, createCombobox, getBoolFromCheckState, getCheckStateFromBool, createValueLineEdit
import numpy as np 



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

        if self.test == "SAM":

            fdr = self.mC.config.getParam("sam.statistic.fdr")
            s0 = self.mC.config.getParam("sam.statistic.s0")
            validInGroup = self.mC.config.getParam("sam.min.valid.in.group")
            
            
            self.validInGroupLabel = createLabel("#valid in each group",tooltipText="The number of values are valid (e.g. not NaN in each group).")
            self.validInGroupEdit = createValueLineEdit("#valid","The number of valid values in each group. Must be at least two.",2,np.inf)
            self.validInGroupEdit.setText(str(validInGroup))

            self.s0Label = createLabel("s0",tooltipText="Fudge factor (s0) t = (fc) / (s0 + s). If s0 = 0.0 it is a normal t-test statistic.")
            self.s0Edit = createValueLineEdit("s0","The smaller the s0 level, the less important the fold change becomes in calculating the FDR/test statistics.",0,np.inf)
            
            self.s0Edit.setText(str(s0))
            self.FDRLabel = createLabel("FDR",tooltipText="The False discovery rate. The Q-value will be returned and allows setting it after the test.")
            self.FDREdit = createValueLineEdit("FDR level",
                                               "Set the FDR (True positiveis / False positives) level to assign the significant columns, however the q-values will also be reported and allow for filtering on a different FDR level. The max is 0.5 /50%.",
                                               0.000000001,0.5)
            self.FDREdit.setText(str(fdr))
            
        else: #remove and log all?
            self.logPValuesCB = QCheckBox("-log10 p-value")
            self.logPValuesCB.setTristate(False)
            self.logPValuesCB.setCheckState(getCheckStateFromBool(True))
            self.logPValuesCB.setChecked(True)

        if self.test not in ["1W-ANOVA"]:
            self.refLabel = createLabel("Reference:","Groups will be compared against this reference only. Set None if you want to have all combinations.")
            self.referenceGroupCombo =  createCombobox(self,["None"] + self.mC.grouping.getCurrentGroupNames())
        
        
        
        self.okayButton = ICStandardButton("Okay")
        self.cancelButton = ICStandardButton("Cancel")


    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.headerLabel)
        groupGrid = QGridLayout()
        groupGrid.addWidget(self.grouplabel,0,0, Qt.AlignmentFlag.AlignRight)
        groupGrid.addWidget(self.groupCombo,0,1)
        if self.test not in ["2W-ANOVA"]:
            groupGrid.addWidget(self.refLabel,2,0, Qt.AlignmentFlag.AlignRight)
            groupGrid.addWidget(self.referenceGroupCombo,2,1)
        else:
            groupGrid.addWidget(self.grouplabel2,2,0, Qt.AlignmentFlag.AlignRight)
            groupGrid.addWidget(self.groupCombo2,2,1)

        if self.test == "SAM":

            groupGrid.addWidget(self.validInGroupLabel,3,0, Qt.AlignmentFlag.AlignRight)
            groupGrid.addWidget(self.validInGroupEdit,3,1)

            groupGrid.addWidget(self.FDRLabel,4,0, Qt.AlignmentFlag.AlignRight)
            groupGrid.addWidget(self.FDREdit,4,1)
            groupGrid.addWidget(self.s0Label,5,0, Qt.AlignmentFlag.AlignRight)
            groupGrid.addWidget(self.s0Edit,5,1)


        else:
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
                      "groupingName":groupingName,
                      "test":self.test,
                      "referenceGroup": referenceGroup,
                      "statParams" : {
                                    "fdr" : float(self.FDREdit.text()), 
                                    "s0" : float(self.s0Edit.text()), 
                                    "minValidInGroup" : int(float(self.validInGroupEdit.text()))} if self.test == "SAM" else None,
                      "logPValues" : self.logPValuesCB.isChecked() if hasattr(self,"logPValuesCB") else False
                      }
                    }
        
        self.mC.sendRequestToThread(funcProps)