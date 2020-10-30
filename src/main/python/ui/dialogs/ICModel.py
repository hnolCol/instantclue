
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from ..custom.buttonDesigns import ICStandardButton
from backend.color.colorHelper import ColorHelper
from ..utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR, createCombobox


class ICModelBase(QDialog):
    ""
    def __init__(self, mainController, title = "First Order Kinetic", *args,**kwargs):
        ""
        super(ICModelBase, self).__init__(*args,**kwargs)
        self.title = title
        self.mC = mainController
        self.__controls()
        self.__layout()
        self.__connectEvents()

    def __controls(self):
        ""
        self.title = createTitleLabel(self.title)

        self.dataTypeLabel = createLabel("Input type:")
        self.dataTypeCombo = createCombobox(self,["log2 transformed","remaining fraction","ln(remaining fraction)","raw data"])

        self.timeGroupLabel = createLabel("Time Grouping:","Grouping to indicate time. Groupnames should either be intergers (2,4,6...) \nor given as integer plus unit separted with a space (1 min)")
        self.timeGroupCombo = createCombobox(self,self.mC.grouping.getGroupings())


        self.compGroupLabel = createLabel("Comp. Grouping:","Grouping to comparision of kintetics such as genotypes or treatments.")
        self.compGroupCombo = createCombobox(self,self.mC.grouping.getGroupings())

        self.replicateLabel = createLabel("Replicates:","In column order: \nReplicates are ordered within the group for example: WT_01, WT_02 vs KO_01, KO_02")
        self.replicateCombo = createCombobox(self,["In column order","No replicates"])

        self.applyButton = ICStandardButton("Okay")
        self.closeButton = ICStandardButton("Close")

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.title)
        grid = QGridLayout()
        grid.addWidget(self.dataTypeLabel,0,0)
        grid.addWidget(self.dataTypeCombo,0,1)

        grid.addWidget(self.timeGroupLabel,1,0)
        grid.addWidget(self.timeGroupCombo,1,1)
        
        grid.addWidget(self.compGroupLabel,2,0)
        grid.addWidget(self.compGroupCombo,2,1)

        grid.addWidget(self.replicateLabel,3,0)
        grid.addWidget(self.replicateCombo,3,1)

        self.layout().addLayout(grid)

        hbox = QHBoxLayout()
        hbox.addWidget(self.applyButton)
        hbox.addWidget(self.closeButton)

        self.layout().addLayout(hbox)

    def __connectEvents(self):
        ""
        self.closeButton.clicked.connect(self.close)

    def closeEvent(self,event=None):
        "Overwrite close event"
        event.accept()




