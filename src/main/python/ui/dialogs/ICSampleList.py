from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pandas.core.algorithms import isin 
from backend.proteomics.ICSampleList import ICSampleListCreator
from ..custom.buttonDesigns import ICStandardButton, ResetButton, LabelLikeButton
from ..custom.resortableTable import ResortableTable
from ..custom.utils import QToggle
from ..utils import createTitleLabel, createLabel, createLineEdit, createCombobox, getMessageProps, WIDGET_HOVER_COLOR
import numpy as np 
import string 
from datetime import datetime


class ICSampleListCreater(QDialog):
    ""
    def __init__(self, mainController, title = "Sample List Creater", *args,**kwargs):
        ""
        super(ICSampleListCreater,self).__init__(*args,**kwargs)
        
        self.mC = mainController
        self.title = title 
        self.numSamples = None
        self.currentNumRow = 7
        self.currentNumColumns = 12
        self.defaultStartIndex = 0
        self.multiInject = 1 
        self.scrambleDefault = False
        self.sampleInRowsDefault = True
        self.addDateDefault = True
        self.validPos = True
        self.customColumnOrder = None
        
        fnKwargs = {
            "scramble" : self.scrambleDefault ,
            "numberRows" : self.currentNumRow+1,
            "multiInject" : self.multiInject,
            "numberColumns" : self.currentNumColumns,
            "samplesInRows" : self.sampleInRowsDefault,
            "addDate" : self.addDateDefault
        }
        self.sampleListCreator = ICSampleListCreator(**fnKwargs)
        self.__controls()
        self.__layout()
        self.__connectEvents()
        self.baseStringEdit.textChanged.emit(self.baseStringEdit.text())
        

    def __controls(self):
        ""
        
        self.titleLabel = createTitleLabel(self.title)

        self.plateLabel = createTitleLabel("Plate design",12)


        self.rowEditLabel = createLabel("Number of rows (A-{})".format(string.ascii_uppercase[self.currentNumRow] if self.currentNumRow is not None else  "..."))
        self.rowEdit = createLineEdit("Enter number of rows..")
        self.rowEdit.setValidator(QIntValidator())
        if self.currentNumRow is not None:
            self.rowEdit.setText(str(self.currentNumRow+1))

        self.columnEditLabel = createLabel("Number of columns (1-{})".format(self.currentNumColumns if self.currentNumColumns is not None else  "..."))
        self.columnEdit = createLineEdit("Enter number of columns..")
        self.columnEdit.setValidator(QIntValidator())
        if self.currentNumColumns is not None:
            self.columnEdit.setText(str(self.currentNumColumns))


        self.sampleNumberLabel = createLabel("Number of samples")
        self.sampleNumberEdit = createLineEdit("Enter number of samples..")
        self.sampleNumberEdit.setValidator(QIntValidator())

        self.startIndexLabel = createLabel("Start positions (examples: A1, 12, B8)")
        self.startIndexEdit = createLineEdit("Enter number of samples..")
        self.startIndexEdit.setText(self.sampleListCreator.getPositionByIndex(self.defaultStartIndex))

        self.scrambleLabel = createTitleLabel("Scramble sample list:",12)
        self.scrambleSwitch = QToggle()
        self.scrambleSwitch.setChecked(self.scrambleDefault )

        self.multipleInjectionLabel = createLabel("Set number of injections per sample")
        self.multipleInjectionEdit = createLineEdit("Enter number of samples..")
        self.multipleInjectionEdit.setText("1")
        self.multipleInjectionEdit.setValidator(QIntValidator())

        self.samplesInRowsLabel = createTitleLabel("Samples in rows:",12)
        self.samplesInRowsSwitch = QToggle()
        self.samplesInRowsSwitch.setChecked(self.sampleInRowsDefault)
        self.samplesInRowsSwitch.setToolTip("If set to True (samples in rows), the sample list will be created by enumerating through row wells (e.g. A1, A2, A3.. If set to False, sample list will be enumerated through columns: A1, B1, C1.")


        self.constants = createLineEdit("Add constants.. (constantName1,constantValue1;constantName2,constantValue2)","Constant values such as injection volumne can be added here. The syntax is: constantName1,constantValue1;constantName2,constantValue2")

        self.fileNameSettings = createTitleLabel("File name configration",12)

        self.columnOrder = LabelLikeButton(text="Define column order")

        self.addDateLabel = createTitleLabel("Add date label ({})".format(datetime.now().strftime("%Y%m%d")),12)
        self.addDateSwitch = QToggle()
        self.addDateSwitch.setChecked(self.addDateDefault)

        self.baseStringEdit = createLineEdit("Enter base file name","Base file name that will be added to every sample name.")

        self.baseStringEdit.setText("HN_0102_HeNo")
        self.fileNamePreviewLabel = createTitleLabel("Sample name preview (n=1)",12)
        self.exampleFileNameLabel = createTitleLabel("",14,colorString=WIDGET_HOVER_COLOR)

        self.createButton = ICStandardButton(itemName="Create")
        self.cancelButton = ICStandardButton(itemName="Cancel")
        

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        hbox = QHBoxLayout()
        self.layout().addWidget(self.titleLabel)
        self.layout().addWidget(self.plateLabel)
        self.layout().addWidget(self.rowEditLabel)
        self.layout().addWidget(self.rowEdit)

        self.layout().addWidget(self.columnEditLabel)
        self.layout().addWidget(self.columnEdit)

        self.layout().addWidget(self.sampleNumberLabel)
        self.layout().addWidget(self.sampleNumberEdit)

        self.layout().addWidget(self.startIndexLabel)
        self.layout().addWidget(self.startIndexEdit)

        self.layout().addWidget(self.multipleInjectionLabel)
        self.layout().addWidget(self.multipleInjectionEdit)

        hboxScramble = QHBoxLayout()
        hboxScramble.addWidget(self.scrambleLabel)
        hboxScramble.addWidget(self.scrambleSwitch)

        hboxSampleOrder = QHBoxLayout() 
        hboxSampleOrder.addWidget(self.samplesInRowsLabel)
        hboxSampleOrder.addWidget(self.samplesInRowsSwitch)

        hboxAddDate = QHBoxLayout()
        hboxAddDate.addWidget(self.addDateLabel)
        hboxAddDate.addWidget(self.addDateSwitch)
        

        self.layout().addLayout(hboxScramble)
        self.layout().addLayout(hboxSampleOrder)
        
        self.layout().addWidget(self.constants)

        self.layout().addWidget(self.fileNameSettings)
        self.layout().addWidget(self.columnOrder)

        self.layout().addLayout(hboxAddDate)
        self.layout().addWidget(self.baseStringEdit)
        self.layout().addWidget(self.fileNamePreviewLabel)
        self.layout().addWidget(self.exampleFileNameLabel)
        self.layout().addStretch(1)
        hbox.addWidget(self.createButton)
        hbox.addWidget(self.cancelButton)
        self.layout().addLayout(hbox)
        
        
    def __connectEvents(self):
        ""

        self.cancelButton.clicked.connect(self.reject)
        self.createButton.clicked.connect(self.createSampleList)
        self.rowEdit.textChanged.connect(self.setRowNumber)
        self.columnEdit.textChanged.connect(self.setColumnNumber)
        self.baseStringEdit.textChanged.connect(self.displaySampleFileName)
        self.sampleNumberEdit.textChanged.connect(self.setSampleNumber)
        self.startIndexEdit.textChanged.connect(self.validatePosition)
        self.multipleInjectionEdit.textChanged.connect(self.setMultiInject)
        self.addDateSwitch.clicked.connect(self.addDateChanged)
        self.scrambleSwitch.clicked.connect(self.scrambleChanged)
        self.samplesInRowsSwitch.clicked.connect(self.samplesInRowsChanged)
        self.columnOrder.clicked.connect(self.defineColumnOrder)

        
    def createSampleList(self,e=None):
        ""
        if self.currentNumColumns is None or self.currentNumRow is None:
            self.mC.sendToWarningDialog(infoText="Number of rows or columns is missing",parent=self)
            return
        if self.numSamples is None:
            self.mC.sendToWarningDialog(infoText="Number of samples is missing",parent=self)
            return
        self.sampleListCreator.setConstants(self.constants.text())
        df = self.sampleListCreator.createSampleList(numberSamples=self.numSamples,baseSampleName=self.baseStringEdit.text())
        if self.customColumnOrder is not None and isinstance(self.customColumnOrder,list):
            if all(x in self.customColumnOrder for x in df.columns) and len(self.customColumnOrder) == df.columns.size:
                df = df[self.customColumnOrder]
            elif all(x in self.customColumnOrder for x in df.columns) and len(self.customColumnOrder) != df.columns.size:
                customColumnOrderSubset = [x for x in self.customColumnOrder if x in df.columns]
                df = df[customColumnOrderSubset]
            else:
                customColumnOrderSubset = [x for x in self.customColumnOrder if x in df.columns] 
                otherColumns = [x for x in df.columns if x not in customColumnOrderSubset] 
                df = df[customColumnOrderSubset + otherColumns]
        self.mC.mainFrames["data"].openDataFrameinDialog(df, 
                                    ignoreChanges=True, 
                                    headerLabel="Sample List.", 
                                    tableKwargs={"forwardSelectionToGraph":False})

    def displaySampleFileName(self,newbaseString):
        ""
        if self.validPos:
            self.exampleFileNameLabel.setText(self.sampleListCreator.getExampleName(baseString=newbaseString))

    def defineColumnOrder(self):
        ""
        self.sampleListCreator.setConstants(self.constants.text())
        currentColumnNames = self.sampleListCreator.getColumnNames()


        sortDialog = ResortableTable(inputLabels=currentColumnNames)
        if sortDialog.exec_():
            if sortDialog.savedData is not None:
                self.customColumnOrder = sortDialog.savedData.values.tolist()

    def setRowNumber(self,rowNum):
        ""
        try:
            self.currentNumRow  = int(float(rowNum)) - 1
            self.rowEditLabel.setText("Number of rows (A-{})".format(string.ascii_uppercase[self.currentNumRow]))
        except:
            self.currentNumRow  = None
            self.rowEditLabel.setText("Number of rows (A-...)")

    def setColumnNumber(self,colNum):
        ""
        try:
            self.currentNumColumns  = int(float(colNum)) 
            self.columnEditLabel.setText("Number of columns (1-{})".format(self.currentNumColumns))
        except:
            self.currentNumColumns  = None
            self.columnEditLabel.setText("Number of columns (1-...)")

    def setSampleNumber(self,numSamples):
        ""
        if numSamples == "":
            self.numSamples = None
        else:
            self.numSamples = int(float(numSamples))
    
    def setMultiInject(self,multiInject):
        ""
        self.multiInject = int(float(multiInject))
        self.sampleListCreator.setMultiInject(multiInject)

    def validatePosition(self,newPosition):
        ""
        if newPosition in self.sampleListCreator.positionsOnPlate:
            self.validPos = True
            self.startIndex = self.sampleListCreator.positionsOnPlate.index(newPosition)
        else:
            try:
                intPos = int(float(newPosition))
                if intPos > len(self.sampleListCreator.positionsOnPlate) - 1:
                    self.validPos = False
                    self.startIndex = None
                else:
                    self.validPos = True
                    self.startIndex = intPos
            except:
                self.validPos = False
                self.startIndex = None
        if self.validPos:

            
            self.sampleListCreator.setStartIndex(self.startIndex)
            self.baseStringEdit.textChanged.emit(self.baseStringEdit.text())

    def addDateChanged(self,e=None):
        ""
        addDate = self.addDateSwitch.isChecked()
        self.sampleListCreator.setAddDate(addDate)
        self.baseStringEdit.textChanged.emit(self.baseStringEdit.text())

    def scrambleChanged(self,e=None):
        ""
        scramble = self.scrambleSwitch.isChecked() 
        self.sampleListCreator.setScramble(scramble)
    
    def samplesInRowsChanged(self,e=None):
        ""
        samplesInRows = self.samplesInRowsSwitch.isChecked() 
        self.sampleListCreator.setSamplesInRows(samplesInRows)