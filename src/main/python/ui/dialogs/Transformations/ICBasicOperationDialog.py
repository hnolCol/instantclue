from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from ...custom.ICTextSelectWidget import ICTextSelectWidget
from ...utils import createTitleLabel, createLabel, createCombobox
from ...custom.tableviews.ICVSelectableTable import PandaModel, PandaTable
from ...custom.Widgets.ICButtonDesgins import ICStandardButton
from ...custom.warnMessage import WarningMessage
from backend.utils.stringOperations import mergeListToString

from collections import OrderedDict
import pandas as pd 
import numpy as np 

slectableTypes = [
    ("Specific Column","Add specific column for operation"),
    ("Mean", "Select multiple columns to calculate the mean, row means are then used for the operation"),
    ("Median", "Select multiple columns to calculate the median, row means are then used for the operation"),
    ("Sum", "Select multiple columns to calculate the sum, row means are then used for the operation")]

operationOptions = ["subtract","addition","divide","multiply"]


class BasicOperationDialog(QDialog):

    def __init__(self, mainController, dataID, selectedColumns, *args,**kwargs):
        super(BasicOperationDialog,self).__init__(*args,**kwargs)   
        self.mC = mainController

        self.calculationProps = OrderedDict([(colName,{}) for colName in selectedColumns])

        self.selectedColumns = selectedColumns
        self.selectableColumns = self.mC.mainFrames["data"].dataTreeView.getColumns("Numeric Floats")["Numeric Floats"]
        self.setWindowTitle("Row-wise operations")
        self.setWindowIcon(self.mC.getWindowIcon())
        self.__controls()
        self.__layout()
        self.__connectEvents()

        
    
    def __controls(self):
        ""

        self.titleLabel = createTitleLabel("Basic Column operations", fontSize=16)
        self.infoLabel = createLabel("For selected columns, choose a metric to perform the selected operation (mean, sum, median).\nFor the metrices choose the columns that should be used for calculation.\nUse the '+' sign to assign metrices to selected columns.", fontSize = 12)
        self.operationLabel = createLabel("Operation: ")
        self.operationCombo = createCombobox(self,items=operationOptions)
        
        self.selectableTypes = OrderedDict()

        for selType,tooltip in slectableTypes:
            self.selectableTypes[selType] = ICTextSelectWidget(
                                parent = self,
                                descriptionText=selType, 
                                toolTipText=tooltip,
                                targetColumns= self.selectedColumns, 
                                selectableItems = self.selectableColumns,
                                reportBackSelection = self.updateMetricForColumn)   

        #set up model table
        self.table = PandaTable(self, cornerButton = False) 
        self.modelData = pd.DataFrame(self.selectedColumns, columns = ["Selected Columns"])
        self.modelData["Selected Metric/Column"] = ["" for _ in range(self.selectedColumns.size)]
        self.model = PandaModel(parent= self.table, df = self.modelData)
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch) 
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch) 

        self.okButton = ICStandardButton(itemName="Apply")
        self.cancelButton = ICStandardButton(itemName = "Cancel")

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.titleLabel)
        self.layout().addWidget(self.infoLabel)
        hboxTop = QHBoxLayout()
        hboxTop.addWidget(self.operationLabel)
        hboxTop.addWidget(self.operationCombo)
        self.layout().addLayout(hboxTop)

        hbox = QHBoxLayout()
        for selectWidet in self.selectableTypes.values():
            hbox.addWidget(selectWidet)
        hbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hbox.setSpacing(3)
        hbox.addStretch(1)

        self.layout().addLayout(hbox)
        self.layout().addWidget(self.table)

        hboxBottom = QHBoxLayout() 
        hboxBottom.addWidget(self.okButton)
        hboxBottom.addStretch(1)
        hboxBottom.addWidget(self.cancelButton)
        self.layout().addLayout(hboxBottom)

    def __connectEvents(self):
        ""
        self.cancelButton.clicked.connect(self.close)
        self.okButton.clicked.connect(self.apply)

    def apply(self,event=None):

        operation = self.operationCombo.currentText() 
        #get only usefull columns
        calculationProps = OrderedDict([(k,v) for k,v in self.calculationProps.items() if len(v) != 0])
        if len(calculationProps) == 0:
            w = WarningMessage(infoText="Use the '+' sings next to the metrices to add them to specific columns.")
            w.exec()
            return

        funcProps = {"key":"data::rowWiseCalculations",
                    "kwargs":{"calculationProps":calculationProps, "operation":operation,"dataID":self.mC.getDataID()}}
        self.mC.sendRequestToThread(funcProps)
        self.accept()

    def updateMetricForColumn(self,column,metricName,metricParams):
        ""

        df = self.model.df
        metricDescription = "{}:({})".format(metricName,mergeListToString(metricParams.values.flatten(),","))
        if column == "Use for all":
            if metricName == "Specific Column":
                if metricParams.values.size == df.index.size:
                    df["Selected Metric/Column"] = metricParams.values.flatten()
                elif metricParams.values.size == 1:
                    df["Selected Metric/Column"] = np.full(df.index.size,metricParams.values.flatten()[0])
                elif df.index.size % metricParams.values.size == 0:
                    df["Selected Metric/Column"] = np.tile(metricParams.values.flatten(),int(df.index.size/metricParams.values.size))
                else:
                    w = WarningMessage(infoText="Slected metric column size could not be used to fill selected columns.")
                    w.exec()
                    return

                for column,metricParam in df[["Selected Columns","Selected Metric/Column"]].values:

                    self.calculationProps[column] = {"metric":metricName,"columns":metricParam}
            else:
                df["Selected Metric/Column"] = metricDescription

                for column in df["Selected Columns"].values:
                    self.calculationProps[column] = {"metric":metricName,"columns":metricParams.values.flatten()}
        else:
            columnBool = df["Selected Columns"] == column
            if metricName != "Specific Column":
                df.loc[columnBool, "Selected Metric/Column"] = metricDescription
                self.calculationProps[column] = {"metric":metricName,"columns":metricParams.values.flatten()}
            else:
                metricColumn = metricParams.values.flatten()
                df.loc[columnBool, "Selected Metric/Column"] = metricColumn[0]
                self.calculationProps[column] = {"metric":metricName,"columns":metricColumn[0]}

        self.model.initData(df)
        self.model.completeDataChanged()

