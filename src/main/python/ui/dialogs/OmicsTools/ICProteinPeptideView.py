from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.pyplot import title 
from ...custom.Widgets.ICButtonDesgins import ICStandardButton, LabelLikeButton 
from ...custom.warnMessage import AskStringMessage
from ...custom.utils import LabelLikeCombo
from ..Selections.ICDSelectItems import ICDSelectItems
from ...utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR, createCombobox

import pandas as pd 

class ICProteinProteinView(QDialog):
    ""
    def __init__(self, mainController, title = "Protein-Peptive View.", *args,**kwargs):
        ""
        super(ICProteinProteinView, self).__init__(*args,**kwargs)
        self.title = title
        self.mC = mainController
        self.peptideDataID = None
       
        self.__controls()
        self.__layout()
        self.__connectEvents()

    def __controls(self):
        ""
        self.title = createTitleLabel(self.title)
        self.infoLabel = createLabel("Please note that it is assumed that the protein table is currently selected.\nIf you just want to explore the peptides without comparison to protein level, please use the Peptive View feature.")
        self.peptideDataLabel = createTitleLabel("Select Peptide Data:",fontSize=13)
        self.peptideDataIDCombo = LabelLikeCombo(parent = self, 
                                        items = self.mC.data.fileNameByID, 
                                        text = "Peptide Level Data Frame", 
                                        tooltipStr="Set Data Frame that contains peptide information.", 
                                        itemBorder=5)
        self.peptideColumnLabel = createTitleLabel("Select Peptide Intensity Columns:",fontSize=13)
        self.peptideIntensityColumns = LabelLikeButton(
                                        parent = self, 
                                        text = "Select peptide intensity column(s)", 
                                        tooltipStr="Select columns that should be used to visualize peptide profiles.", 
                                        itemBorder=5)                     

        self.closeButton = ICStandardButton("Close")

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.title)
        self.layout().addWidget(self.infoLabel)
        grid = QGridLayout()
        self.layout().addLayout(grid)
        
        grid.addWidget(self.peptideDataLabel)
        grid.addWidget(self.peptideDataIDCombo)
        grid.addWidget(self.peptideColumnLabel)
        grid.addWidget(self.peptideIntensityColumns)

        hbox = QHBoxLayout()
        hbox.addWidget(self.closeButton)

        self.layout().addLayout(hbox)

        grid.setColumnStretch(0,1)
        grid.setRowStretch(7,2)

    def __connectEvents(self):
        ""
        self.closeButton.clicked.connect(self.close)
        #choose specific column
        self.peptideIntensityColumns.clicked.connect(self.choosePeptideIntensityColumns)
        self.peptideDataIDCombo.selectionChanged.connect(self.peptideDataIDChanged)

    def closeEvent(self,event=None):
        "Overwrite close event"
        event.accept()
        
                        
    def choosePeptideIntensityColumns(self,*args,**kwargs):
        ""

        if self.peptideDataID is None:
            self.mC.sendToWarningDialog(infoText="Please select a data frame that contains peptide information first.",parent=self)
            return
        selectableColumns = pd.DataFrame(self.mC.data.getNumericColumns(self.mC.getDataID()))
        dlg = ICDSelectItems(data = selectableColumns, selectAll=False, singleSelection=False)
        # handle position and geomettry
        senderGeom = self.sender().geometry()
        bottomRight = self.mapToGlobal(senderGeom.bottomRight())
        h = dlg.getApparentHeight()
        dlg.setGeometry(bottomRight.x() + 15, bottomRight.y()-int(h/2), 185, h)
        #handle result
        if dlg.exec():
            selectedColumns = dlg.getSelection()
            self.peptideIntensityColumns = selectedColumns.values.flatten().tolist()
            numColumnsSelected = len(self.peptideIntensityColumns)
            if hasattr(self.sender(),"setText"):
                if numColumnsSelected > 1:
                    self.sender().setText("{} columns selected".format(numColumnsSelected))
                else:
                    self.sender().setText(self.peptideIntensityColumns[0][:20])
    
    def peptideDataIDChanged(self,item):
        ""

        dataID, fileName = item
        self.peptideDataID = dataID
        
        