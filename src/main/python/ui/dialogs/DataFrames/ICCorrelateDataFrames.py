from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from ...utils import createTitleLabel, createLabel, createLineEdit, createMenu
from ...custom.utils import LabelLikeCombo
from ..Selections.ICDSelectItems import ICDSelectItems
from ...custom.Widgets.ICButtonDesgins import ICStandardButton, LabelLikeButton
from ...custom.warnMessage import WarningMessage
import pandas as pd 
from collections import OrderedDict

class ICCorrelateDataFrames(QDialog):
    def __init__(self, mainController, *args, **kwargs):
        super(ICCorrelateDataFrames, self).__init__(*args, **kwargs)
       
        self.mC = mainController
        self.setWindowTitle("Correlate data frames.")
        self.setWindowIcon(self.mC.getWindowIcon())
        self.corrParams = dict()
        self.dfWidgets = dict()
        #dict to collect merge parameters
        self.corrParams["left"] = dict()
        self.corrParams["right"] = dict()

        
        self.__controls()
        self.__layout()
        self.__connectEvents()

        #set size policy of dialog
        self.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Expanding)
        self.setFixedHeight(200)
        self.setMaximumHeight(200)

    def __controls(self, header = "Correlate Two Data Frames"):
        """Init widgets"""
        
        self.headerLabel = createTitleLabel(header, fontSize=14)
        ### add menu button
        self.parameterGrid = self.addParameters()
        self.hbox1 = self.addDataFrame("left")
        self.hbox2 = self.addDataFrame("right")

        self.okButton = ICStandardButton(itemName="Correlate")
        self.cancelButton = ICStandardButton(itemName = "Cancel")
        
    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QGridLayout())
        self.layout().addWidget(self.headerLabel)
        self.layout().addLayout(self.parameterGrid,2,0,1,4)
        self.layout().addLayout(self.hbox1,3,0,1,4)
        self.layout().addLayout(self.hbox2,4,0,1,4)
        self.layout().addWidget(self.okButton,6,0)
        self.layout().addWidget(self.cancelButton,6,3)
        self.layout().setColumnStretch(2,1)
        self.layout().setRowStretch(5,1)
        
       
    def __connectEvents(self):
        """Connect events to functions"""
        self.cancelButton.clicked.connect(self.close)
        self.okButton.clicked.connect(self.correlate)
        
    def addDataFrame(self, dfID = "left"):
        """
        Add DataFrame related widgets to the QDialog. 
        Three options: 
        a) Selected the dataframe name
        b) Select Key columns to be used for merging
        c) Select columns to transfer to the merged data frame (if nothing selected, all columns will be attached.)
        """
        gridBox = QGridLayout()
        #data frame selection widgets
        dataFrameLabel = LabelLikeCombo(parent = self, 
                                        items = self.mC.data.fileNameByID, 
                                        text = "Data Frame ({})".format(dfID), 
                                        tooltipStr="Set Data Frame", 
                                        itemBorder=5)
        dataFrameLabel.selectionChanged.connect(self.dfSelected)

        columnsButton = ICStandardButton(itemName = "...", tooltipStr="Select columns to keep for merging (default: keep all numeric floats).")
        columnsButton.setFixedSize(15,15)    
        columnsButton.clicked.connect(lambda _,paramID = "selectedColumns": self.openColumnSelection(paramID=paramID))

        #add widgets
        gridBox.addWidget(dataFrameLabel,0,0,Qt.AlignmentFlag.AlignLeft)
        gridBox.addWidget(columnsButton,0,2,Qt.AlignmentFlag.AlignRight)

        #handle column stretch
        gridBox.setColumnStretch(0,2)
        gridBox.setColumnStretch(1,2)
        gridBox.setColumnStretch(2,0)

        self.dfWidgets[dfID] = [dataFrameLabel,columnsButton]
        return gridBox

    def addParameters(self):
        ""
        grid = QGridLayout() 
        self.label = createLabel("Correlate two data frames (row by row or column by column)\nIf you want to correlate all columns/rows with all other columns. Please use to Correlate features method.")
        self.label.setWordWrap(True)
        self.methodCombo = LabelLikeCombo(parent = self, items = dict([("pearson","Pearson"),("spearman","Spearman"),("kendall","Kendall")]), text = "Correlation Method", tooltipStr="Correlation method. Defaults to Pearson.", itemBorder=5)            
        self.axisCombo = LabelLikeCombo(parent = self, items = dict([(1,"Rows"),(0,"Columns")]), text = "Select axis.", tooltipStr="Axis which should be used for correlation ", itemBorder=5)       
        self.ignoreIndex = QCheckBox(parent=self)
        self.ignoreIndex.setText("Ignore index")
        self.ignoreIndex.setToolTip("If disabled, correlation will be performed between matching indicies (e.g. column names) of the two data frames.\nIf you have the same header names but different order, this should be disabled.")
        self.ignoreIndex.setCheckState(True)
        grid.addWidget(self.label)
        grid.addWidget(self.methodCombo)
        grid.addWidget(self.axisCombo)
        grid.addWidget(self.ignoreIndex)
        return grid

    def dfSelected(self,item):
        ""
        dataID, fileName = item
        
        dfID = self.getDfID(self.sender())
        self.setDfParams(dfID,dataID)

    def setDfParams(self,dfID,dataID):

        self.corrParams[dfID]["dataID"] = dataID
        self.corrParams[dfID]["columnNames"] = self.mC.data.getPlainColumnNames(dataID)
        self.corrParams[dfID]["selectedColumns"] = pd.Series()


    def getDfID(self,widget):
        ""
        if hasattr(self,"dfWidgets") and isinstance(self.dfWidgets,dict):
            if widget in self.dfWidgets["left"]:
                dfID = "left"
            else:
                dfID = "right"
            return dfID

    def correlate(self,e=None):
        ""
        if "dataID" in self.corrParams["left"] and "dataID" in self.corrParams["right"]:
            method = self.methodCombo.getItemID()
            axis = self.axisCombo.getItemID()
            if method == "Correlation Method":
                method = "pearson"
            if axis == "Select axis":
                w = WarningMessage(self,infoText="Please select the axis to which the correlation should be performed.",iconDir= self.mC.mainPath)
                w.exec_() 
                return

            corrParams = {  
                            "dataID1" : self.corrParams["left"]["dataID"],
                            "columnNames1" : self.corrParams["left"]["selectedColumns"],
                            "dataID2" : self.corrParams["right"]["dataID"],
                            "columnNames2" : self.corrParams["right"]["selectedColumns"],
                            "axis" : axis,
                            "method":method,
                            "ignoreIndex" : self.ignoreIndex.checkState()
                            }
            
            funcProps = {"key":"data::correlateDataFrames","kwargs":
                                {"corrParams":corrParams}
                        }
            self.mC.sendRequestToThread(funcProps)
        else:
            w = WarningMessage(self,infoText="Please select data frames.",iconDir= self.mC.mainPath)
            w.exec_() 
            return
    

    def openColumnSelection(self,event=None, paramID = None):
        ""
        try:
            dfID = self.getDfID(self.sender())
            if dfID in self.corrParams:
                dfProps = self.corrParams[dfID]
                if "columnNames" not in dfProps:
                    w = WarningMessage(title = "No data frame.", infoText = "Please select a dataframe first.",iconDir = self.mC.mainPath)
                    w.exec_()
                    return
                selectableColumns = pd.DataFrame(dfProps["columnNames"])
                preSelectionIdx = dfProps[paramID].index
            
                dlg = ICDSelectItems(data = selectableColumns)
                dlg.model.setCheckStateByDataIndex(preSelectionIdx)

                # handle position and geomettry
                senderGeom = self.sender().geometry()
                bottomRight = self.mapToGlobal(senderGeom.bottomRight())
                h = dlg.getApparentHeight()
                dlg.setGeometry(bottomRight.x() + 15,bottomRight.y()-int(h/2),185,h)

                #handle result
                if dlg.exec_():
                    selectedColumns = dlg.getSelection()
                    self.corrParams[dfID][paramID] = pd.Series(selectedColumns.values[:,0],index = selectedColumns.index)
                    self.sender().setToolTip("{} columns selected.".format(selectableColumns.size))
                    
           
                
        except Exception as e:
            print(e)


class ICCorrelateFeatures(ICCorrelateDataFrames):
    def __init__(self,mainController,*args,**kwargs):
        self.mC = mainController
        super(ICCorrelateDataFrames,self).__init__(*args,**kwargs)
        self.corrParams = dict()
        self.dfWidgets = dict()
        #dict to collect merge parameters
        self.corrParams["left"] = dict()
        self.corrParams["right"] = dict()

        self.__controls()
        self.__layout()
        self.__connectEvents()

    def __controls(self, header = "Correlate Features of two Data Frames"):
        """Init widgets"""
        self.headerLabel = createTitleLabel(header, fontSize=14)
        ### add menu button
        self.parameterGrid = self.addParameters()
        self.hbox1 = self.addDataFrame("left")
        self.hbox2 = self.addDataFrame("right")

        self.okButton = ICStandardButton(itemName="Correlate")
        self.cancelButton = ICStandardButton(itemName = "Cancel")
        
    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QGridLayout())
        self.layout().addWidget(self.headerLabel)
        self.layout().addLayout(self.parameterGrid,2,0,1,4)
        self.layout().addLayout(self.hbox1,3,0,1,4)
        self.layout().addLayout(self.hbox2,4,0,1,4)
        self.layout().addWidget(self.okButton,6,0)
        self.layout().addWidget(self.cancelButton,6,3)
        self.layout().setColumnStretch(2,1)
        self.layout().setRowStretch(5,1)
              
    def __connectEvents(self):
        """Connect events to functions"""
        self.cancelButton.clicked.connect(self.close)
        self.okButton.clicked.connect(self.correlate) 

    def addParameters(self):
        ""
        grid = QGridLayout() 
        self.label = createLabel("Correlate selected columns to all other selected columns in the other df. The number of rows must match between the data frames.")
        self.label.setWordWrap(True)

    def correlate(self):
        ""
        if "dataID" in self.corrParams["left"] and "dataID" in self.corrParams["right"]:
            corrParams = {  
                            "dataID1" : self.corrParams["left"]["dataID"],
                            "columnNames1" : self.corrParams["left"]["selectedColumns"],
                            "dataID2" : self.corrParams["right"]["dataID"],
                            "columnNames2" : self.corrParams["right"]["selectedColumns"],
                            }
            
            funcProps = {"key":"data::correlateFeaturesOfDataFrames","kwargs":
                                {"corrParams":corrParams}
                        }
            self.mC.sendRequestToThread(funcProps)