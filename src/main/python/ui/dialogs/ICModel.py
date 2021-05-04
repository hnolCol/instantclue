
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from ..custom.buttonDesigns import ICStandardButton
from backend.color.colorHelper import ColorHelper
from ..utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR, createCombobox


class ICModelBase(QDialog):
    ""
    def __init__(self, mainController, title = "Model fitting", *args,**kwargs):
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

        self.dataTypeLabel = createLabel("Value Transformation:","First order kinetic requires calculation of remaining fraction based on division to basal (0h). If you have log2 data, use the 2^x funtion.")
        self.dataTypeCombo = createCombobox(self,["2^x","None","ln","SILAC -> ln(rf)","Int. -> ln(rf)","log2 Int. -> ln(rf)"])

        self.normalizaionLabel = createLabel("Value Normalization:","Normalization strategy if rf is calculated from intensity values.")
        self.normalizationCombo = createCombobox(self,["None","Divide by median of first timepoint"])

        self.timeGroupLabel = createLabel("Time Grouping:","Grouping to indicate time. Groupnames should either be intergers (2,4,6...) \nor given as integer plus unit separted with a space (1 min)")
        self.timeGroupCombo = createCombobox(self,self.mC.grouping.getGroupings())

        self.compGroupLabel = createLabel("Comp. Grouping:","Grouping to comparision of kintetics such as genotypes or treatments.")
        self.compGroupCombo = createCombobox(self,["None"] + self.mC.grouping.getGroupings())

        self.replicateLabel = createLabel("Replicate Grouping:","If 'no replicates', a single fit is performed not distinguishing between replicates.")
        self.replicateCombo = createCombobox(self,["None"] + self.mC.grouping.getGroupings())

        self.AUCCalcLabel = createLabel("Calculate AUC","Calculate area under curve (AUC) either on raw data or on the resulting fit.")
        self.AUCCalcCombo = createCombobox(self,["Both Types","Data AUC (trapz)","Fit AUC","None"])

        # self.modelLabel = createLabel("Model to fit")
        # self.modelCombo = createCombobox(self,["linear-model"])

        self.applyButton = ICStandardButton("Okay")
        self.closeButton = ICStandardButton("Close")

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.title)
        grid = QGridLayout()
        grid.addWidget(self.dataTypeLabel,0,0,1,1,Qt.AlignRight)
        grid.addWidget(self.dataTypeCombo,0,1)

        grid.addWidget(self.normalizaionLabel,1,0,1,1,Qt.AlignRight)
        grid.addWidget(self.normalizationCombo,1,1)

        grid.addWidget(self.timeGroupLabel,2,0,1,1,Qt.AlignRight)
        grid.addWidget(self.timeGroupCombo,2,1)
        
        grid.addWidget(self.compGroupLabel,3,0,1,1,Qt.AlignRight)
        grid.addWidget(self.compGroupCombo,3,1)

        grid.addWidget(self.replicateLabel,4,0,1,1,Qt.AlignRight)
        grid.addWidget(self.replicateCombo,4,1)


        # grid.addWidget(self.modelLabel,5,0,1,1,Qt.AlignRight)
        # grid.addWidget(self.modelCombo,5,1)

        grid.addWidget(self.AUCCalcLabel,6,0,1,1,Qt.AlignRight)
        grid.addWidget(self.AUCCalcCombo,6,1)

        self.layout().addLayout(grid)

        hbox = QHBoxLayout()
        hbox.addWidget(self.applyButton)
        hbox.addWidget(self.closeButton)

        self.layout().addLayout(hbox)

        grid.setColumnStretch(0,0)
        grid.setColumnStretch(1,1)
        
        grid.setRowStretch(7,2)

    def __connectEvents(self):
        ""
        self.closeButton.clicked.connect(self.close)
        self.applyButton.clicked.connect(self.performCalculations)

    def closeEvent(self,event=None):
        "Overwrite close event"
        event.accept()
    

    def performCalculations(self):
        ""
        #print("performing calculations")
        funcKey = "stats::fitModel"
        normalization = self.normalizationCombo.currentText()
        timeGrouping = self.timeGroupCombo.currentText()
        compGrouping = self.compGroupCombo.currentText()
        repGrouping = self.replicateCombo.currentText()
        transformation = self.dataTypeCombo.currentText()
        aucType = self.AUCCalcCombo.currentText() 
        addDataAUC = aucType in ["Both Types","Data AUC (trapz)"]
        addFitAUC =  aucType in ["Both Types","Fit AUC"]


        kwargs = {
            "dataID" : self.mC.getDataID(),
            "normalization" : normalization,
            "addFitAUC" : addFitAUC,
            "addDataAUC" : addDataAUC,
            "transformation" : transformation,
            "timeGrouping" : self.mC.grouping.getGrouping(timeGrouping), 
            "compGrouping" : self.mC.grouping.getGrouping(compGrouping),
            "replicateGrouping" : self.mC.grouping.getGrouping(repGrouping),
            "columnNames" : self.mC.grouping.getColumnNamesFromGroup(timeGrouping).values.tolist() + self.mC.grouping.getColumnNamesFromGroup(compGrouping).values.tolist()
            }
        funcProps = {"key":funcKey,"kwargs":kwargs}
        self.mC.sendRequestToThread(funcProps)




