
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 
from backend.transformations.transformer import summarizeMetric
from ...utils import createLabel, createTitleLabel, createMenu, WIDGET_HOVER_COLOR, INSTANT_CLUE_BLUE, createCombobox, getCheckStateFromBool, getBoolFromCheckState, createValueLineEdit
from ...custom.Widgets.ICButtonDesgins import  ResetButton, BigPlusButton, LabelLikeButton, ICStandardButton, HelpButton
from ...custom.warnMessage import WarningMessage
from ..Selections.ICDSelectItems import ICDSelectItems
from ...custom.utils import  ICSCrollArea
#external imports
import pandas as pd
import numpy as np 
from collections import OrderedDict 
import webbrowser

LINE_EDIT_STATUS = {"Greater than":(True,False),
                    "Greater Equal than":(True,False),
                    "Between":(False,False),
                    "Not between":(False,False),
                    "Smaller than":(False,True),
                    "Smaller Equal than":(False,True),
                    "n largest":(True,False),
                    "n smallest":(False,True)} 

CB_OPTIONS  = ["Annotate Matches","Subset Matches","Set NaN","Set NaN in spec. columns"]
CB_TOOLTIPS = ["Create a new column indicating by '+' if numeric filter matched.",
               "Creates a new data frame with rows where numeric filter matches.",
               "Values that fulfill the condition are replaced with NaN",
               "Based on the numeric filtering in given columns, set nan in other numeric floats columns."]
filterTypes = ["Greater than","Greater Equal than","Smaller than","Smaller Equal than","Between", "Not between","n largest","n smallest"]


class ICNumericFilterForSelection(QDialog):
    """
    QDialog - ICNumericFilterForSelection
    -----------------------------------------------

    This dialog handles numeric filter data input. 
    The numeric filtering is applied to the selected 
    columns. 
    Users can choose if they want to apply the filter
    on selected column individually, or calculate
    a metric such as mean/max etc before the filter is
    applied on the calculated metric. A new column containing 
    the metric is added.

    """
    def __init__(self,mainController,selectedNumericColumns,*args,**kwargs):
        super(ICNumericFilterForSelection,self).__init__(*args,**kwargs)
        self.mC = mainController
        self.columnNames = selectedNumericColumns
        self.CBFilterOptions = dict()
        self.__controls()
        self.__layout()
        self.__connectEvents()

        self.onFilterTypeChange(filterTypes[0])

    def __controls(self):
        ""
        self.setWindowTitle("Numeric Filter For Selection")
        self.setWindowIcon(self.mC.getWindowIcon())
        self.titleLabel = createTitleLabel("Numeric Filter for Column Selection")
       
        self.operatorLabel = createLabel("Operator: ", fontSize = 12)
        self.operatorCombo = createCombobox(self,items = self.mC.numericFilter.getOperatorOptions())
        self.operatorCombo.setCurrentText(self.mC.numericFilter.getOperator())
        
        self.helpButton = HelpButton(
            buttonSize=(30,30), 
            tooltipStr="Opens the specific GitHub Wiki Page for selected Settings (requires Internet connection).")
    
        self.scrollFrame = QFrame(parent=self) 

        self.selectionlabel = createLabel("Selection:")
        self.selectedColumnlabel = LabelLikeButton("{} Columns selected.".format(self.columnNames.size),fontSize=14,txtColor=WIDGET_HOVER_COLOR)
        self.selectedColumnlabel.setToolTip("Click to change selection of column names.")
        
        
        self.filterModeLabel = createLabel("Filter mode:")
        
        self.filterMode = createCombobox(self.scrollFrame,items=["On individual columns","On columns metric"])
        self.filterMode.setToolTip("Choose 'on individual columns' if you want to filter based on the individual values in each of the selected columns.\nYou can also first summarize the columns by calculation a specific metric (e.g. mean, sum etc) per row and the apply the filtering. ")
        
        

        self.filterMetricLabel = createLabel("Filter metric:","Enabled only if 'On columns metric' is selected as mode.\nAllows to summarize selected columns first and then apply the filter.")
        self.filterMetric = createCombobox(self.scrollFrame,items=list(summarizeMetric.keys()))
        self.filterMetric.setEnabled(False) 

        self.filterTypeLabel = createLabel("Filter Type:")
        self.filterTypeCombo = createCombobox(self.scrollFrame,items=filterTypes)
        

        self.filterRangeLabel = createLabel("Filter range:","Filter range defined by min and max value. \nFor some filters the edit will be read only.")
        self.minValue = createValueLineEdit("Min Value..","Set min value",-np.inf,np.inf)
        self.maxValue = createValueLineEdit("Max Value..","Set max value",-np.inf,np.inf)

        self.applyButton = ICStandardButton(itemName="Apply")
        self.closeButton = ICStandardButton(itemName="Close")

       
        for n,filtOption in enumerate(CB_OPTIONS[0:3]):
            cb = QCheckBox(filtOption, toolTip = CB_TOOLTIPS[n])
            cb.setTristate(False)
            if n == 0:
                cb.setCheckState(getCheckStateFromBool(True))
            cb.clicked.connect(self.setCBCheckStates)
            self.CBFilterOptions[filtOption] = cb
      
        
        
    def __layout(self):
        ""
        labelArgs = [1,1,Qt.AlignmentFlag.AlignRight]
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.MinimumExpanding)
        
        self.setLayout(QGridLayout()) 
        
        self.layout().addWidget(self.titleLabel,0,0,2,1)
        self.layout().addWidget(self.helpButton,0,4,1,1)
        self.scrollFrame.setLayout(QGridLayout())
        scrollFrameGrid = self.scrollFrame.layout() 
        scrollFrameGrid.addWidget(self.selectionlabel,0,0,*labelArgs)
        scrollFrameGrid.addWidget(self.selectedColumnlabel,0,1,1,2,)
        scrollFrameGrid.addWidget(self.filterModeLabel,1,0,*labelArgs)
        scrollFrameGrid.addWidget(self.filterMode,1,1,1,2)

        scrollFrameGrid.addWidget(self.filterMetricLabel,2,0,*labelArgs)
        scrollFrameGrid.addWidget(self.filterMetric,2,1,1,2)
        scrollFrameGrid.addWidget(self.operatorLabel,3,0,*labelArgs)
        scrollFrameGrid.addWidget(self.operatorCombo,3,1,1,2)

        scrollFrameGrid.addWidget(self.filterTypeLabel,4,0,*labelArgs)
        scrollFrameGrid.addWidget(self.filterTypeCombo,4,1,1,2)

        scrollFrameGrid.addWidget(self.filterRangeLabel,5,0,*labelArgs)
        scrollFrameGrid.addWidget(self.minValue,5,1,1,1)
        scrollFrameGrid.addWidget(self.maxValue,5,2,1,1)
        scrollFrameGrid.setSpacing(10)
        self.layout().addWidget(self.scrollFrame,3,0,1,4)


        #add selection cbs
        hboxCB = QHBoxLayout()
        for cb in self.CBFilterOptions.values():
            hboxCB.addWidget(cb)
        self.layout().addLayout(hboxCB,6,0,1,2)

        hboxB = QHBoxLayout() 
        hboxB.addWidget(self.applyButton)
        hboxB.addWidget(self.closeButton)
        
        self.layout().addLayout(hboxB,7,3,1,1)
        self.layout().setAlignment(Qt.AlignmentFlag.AlignTop)


    def __connectEvents(self):
        ""
        self.operatorCombo.currentTextChanged.connect(self.setOperator)
        self.filterTypeCombo.currentTextChanged.connect(self.onFilterTypeChange)
        self.filterMode.currentTextChanged.connect(self.onFilterModeChange)
        self.selectedColumnlabel.clicked.connect(self.columnSelection)
        self.closeButton.clicked.connect(self.close)
        self.applyButton.clicked.connect(self.applyFilter)
        self.helpButton.clicked.connect(lambda: webbrowser.open("https://github.com/hnolCol/instantclue/wiki/Filtering#data-filtering"))

    def columnSelection(self,*args,**kwargs):
        "Update Column Selection"
        selectedColumns = self.mC.askForItemSelection(self.mC.data.getNumericColumns(self.mC.getDataID()))
        if selectedColumns is not None and selectedColumns.size > 0:

            self.selectedColumnlabel.setText("{} Columns selected.".format(selectedColumns.size))
            self.columnNames = selectedColumns

    def onFilterModeChange(self,filterMode):
        ""
        self.filterMetric.setEnabled(not filterMode == "On individual columns")

    def onFilterTypeChange(self,filterType):
        ""
        minReadOnly,maxReadOnly = LINE_EDIT_STATUS[filterType]
        self.minValue.setReadOnly(minReadOnly)
        self.maxValue.setReadOnly(maxReadOnly)

    def setOperator(self,newOperator):
        ""
        self.mC.numericFilter.setOperator(newOperator)
    def getFilterProps(self):
        ""
        return {"min":float(self.minValue.text()) if self.minValue.text() != "" else -np.inf,
                "max":float(self.maxValue.text()) if self.maxValue.text() != "" else np.inf,
                "filterType" : self.filterTypeCombo.currentText()}

    def applyFilter(self):
        ""
        #check user input
        try:
            filterProps = self.getFilterProps()
        except:
            self.mC.sendToWarningDialog(infoText = "Could not convert min/max value to floating number.",parent=self)
            return
        if filterProps["min"] == -np.inf and filterProps["max"] == np.inf:
            self.mC.sendToWarningDialog(infoText = "Please enter limits/range for filtering.",parent=self)
            return
        filterType = self.filterTypeCombo.currentText()
        minReadOnly,maxReadOnly = LINE_EDIT_STATUS[filterType]

        if not minReadOnly and  filterProps["min"] == -np.inf:
            self.mC.sendToWarningDialog(infoText = "Please enter min value.",parent=self)
            return
        if not maxReadOnly and filterProps["max"] == np.inf:
            self.mC.sendToWarningDialog(infoText = "Please enter max value.",parent=self)
            return
       
        #send to Thread
        funcProps = {
            "key" : "filter::selectionNumericFilter",
            "kwargs" : {
                "dataID" : self.mC.getDataID(),
                "columnNames" : self.columnNames,
                "metric" : self.filterMetric.currentText(),
                "filterMode" : self.filterMode.currentText(),
                "filterProps" : filterProps,
                "setNonMatchNan" : getBoolFromCheckState(self.CBFilterOptions["Set NaN"].checkState()),
                "subsetData" : getBoolFromCheckState(self.CBFilterOptions["Subset Matches"].checkState())
            }
        }
        self.mC.sendRequestToThread(funcProps)
        self.close()

    def setCBCheckStates(self,event=None):
        ""
        for _,cb in self.CBFilterOptions.items():
            if cb != self.sender():
                cb.setCheckState(getCheckStateFromBool(False))
            else:
                cb.setCheckState(getCheckStateFromBool(True))

class NumericFilter(QDialog):

    def __init__(self,mainController, selectedNumericColumn = [], *args, **kwargs):
        super(NumericFilter,self).__init__(*args, **kwargs)
        
        self.mC = mainController
        self.dataID = self.mC.mainFrames["data"].getDataID()
        self.numericColumns = pd.concat([self.mC.data.getNumericColumns(self.dataID), self.mC.data.getIntegerColumns(self.dataID)])
        
        self.selectedNumericColumn = selectedNumericColumn
        
        self.filterProps = OrderedDict() 
        self.CBFilterOptions = OrderedDict()

        self.__controls()
        self.__layout()
        self.__connectEvents()

        self.addDragFilters()
        

    def __controls(self):
        ""
        self.setWindowTitle("Numeric Filter")
        self.setWindowIcon(self.mC.getWindowIcon())
        self.titleLabel = createTitleLabel("Numeric Filter")
        self.filterLabel = createLabel("Filter on: ", fontSize = 12)
        self.columnNameCombo = createCombobox(self, items = self.numericColumns.values.tolist())
        self.operatorLabel = createLabel("Operator: ", fontSize = 12)
        self.operatorCombo = createCombobox(self,items = self.mC.numericFilter.getOperatorOptions())
        self.operatorCombo.setCurrentText(self.mC.numericFilter.getOperator())
        self.operatorCombo.currentTextChanged.connect(self.setOperator)
        self.scrollArea = ICSCrollArea(parent=self,getUpdatabelWidgets=self.sendUpdatabelWidgets)
        
        self.scrollFrame = QFrame(parent=self.scrollArea) 
        self.scrollArea.setWidget(self.scrollFrame)
        self.scrollArea.setWidgetResizable(True)
        

        self.addFilterIcon = BigPlusButton(buttonSize=(30,30))

        self.applyButton = ICStandardButton(itemName="Apply")
        self.closeButton = ICStandardButton(itemName="Close")

       
        for n,filtOption in enumerate(CB_OPTIONS):
            cb = QCheckBox(filtOption, toolTip = CB_TOOLTIPS[n])
            cb.setTristate(False)
            if n == 0:
                cb.setCheckState(getCheckStateFromBool(True))
            cb.clicked.connect(self.setCBCheckStates)
            self.CBFilterOptions[filtOption] = cb
      
        
        
    def __layout(self):
        ""
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.MinimumExpanding)
        
        self.setLayout(QGridLayout()) 
        self.scrollFrame.setLayout(QVBoxLayout())
        
        self.layout().addWidget(self.titleLabel,0,0)
        self.layout().addWidget(self.filterLabel ,1,0)
        self.layout().addWidget(self.columnNameCombo,1,1,1,3)

        self.layout().addWidget(self.operatorLabel,2,0)
        self.layout().addWidget(self.operatorCombo,2,1,1,3)
        self.layout().setColumnStretch(2,1)
        self.layout().addWidget(self.scrollArea,3,0,1,4)
        self.layout().addWidget(self.addFilterIcon,4,0)
        
        #add selection cbs
        hboxCB = QHBoxLayout()
        for cb in self.CBFilterOptions.values():
            hboxCB.addWidget(cb)
        self.layout().addLayout(hboxCB,6,0,1,2)

        hboxB = QHBoxLayout() 
        hboxB.addWidget(self.applyButton)
        hboxB.addWidget(self.closeButton)
        
        self.layout().addLayout(hboxB,7,3,1,1)
        self.layout().setAlignment(Qt.AlignmentFlag.AlignTop)


    def __connectEvents(self):
        ""
        self.addFilterIcon.clicked.connect(self.addFilter)
        self.closeButton.clicked.connect(self.close)
        self.applyButton.clicked.connect(self.applyFilter)

    def addDragFilters(self,event=None):
        ""
        for numericColumn in self.selectedNumericColumn.values:
            if numericColumn in self.numericColumns.values:
                self.addFilter(filterName=numericColumn)
                self.updateLineEdits(numericColumn, "Greater Equal than")

    def addFilter(self,event=None, filterName = None):
        ""
        if filterName is None:
            filterName = self.columnNameCombo.currentText()
            if filterName == "Choose additional column":
                w = WarningMessage(infoText="Choose column from drop down menu that you would like to use the filter on.")
                w.exec() 
                return
        if filterName not in self.filterProps:
            self.filterProps[filterName] = dict()
            self.filterProps[filterName]["frame"] = self.createFilterInfoWidgetLayout(filterName)
            self.scrollFrame.layout().addWidget(self.filterProps[filterName]["frame"])
            self.updateComboBox()

    def chooseType(self,event=None,filterName = None):
        ""
        menu = createMenu(parent=self)

        for filterType in filterTypes:
            menu.addAction(filterType)
        senderGeom = self.sender().geometry()
        topLeft = self.filterProps[filterName]["frame"].mapToGlobal(senderGeom.bottomLeft())
        #set sender status 
        self.sender().mouseOver = False
        #cast menu
        action = menu.exec(topLeft)
        if action:
            self.filterProps[filterName]["filterType"] = str(action.text())
            
            self.sender().setText(action.text())
            self.updateLineEdits(filterName, action.text())
            self.resetValidator(filterName, topN = "n " in action.text())


    def createFilterInfoWidgetLayout(self, columnName, filterType = "Greater Equal than"):
        ""
        outerFrame = QFrame(self)
        outerFrame.setStyleSheet("background:grey")

        frame = QFrame(outerFrame) 
        
        frame.setStyleSheet("background: #f7f7f7")
        frame.setLayout(QHBoxLayout())
        frame.layout().setContentsMargins(10,10,10,10)
        hbox = frame.layout()
        
        columnLabel = LabelLikeButton(parent = frame, text = columnName, tooltipStr="Selected column to apply filter on. Only one filter per column is allowed.", itemBorder=5)
        filterLabel = LabelLikeButton(parent = frame, text = filterType, tooltipStr="Filter Type: Between, Greater, Smaller ...", itemBorder=5)
        
        minValue, maxValue = self.mC.data.getMinMax(self.dataID,columnName)
        nValues = self.mC.data.getNumValidValues(self.dataID,columnName)

        minEditValue  = self.createValueLineEdit("Set value",
                                                "Set minimum value\nmin value: {}\nvalid values: {}".format(minValue,nValues),
                                                minValue,
                                                maxValue,
                                                columnName)

        maxEditValue = self.createValueLineEdit("Set value",
                                                "Set maximum value\nmax value: {}\nvalid values: {}".format(maxValue,nValues),
                                                minValue,
                                                maxValue,
                                                columnName)  

        specColumnLabel = LabelLikeButton(parent = frame, text = "Select column(s)", tooltipStr="If set NaN in spec column is selected specific column.\nIf a column is selected multiple times, the nan replacements will be performed in order of listed filter...", itemBorder=5)                     
        
        resetButton = ResetButton()


        #FilterTypeMenu
        filterLabel.clicked.connect(lambda _,filterName = columnName :  self.chooseType(filterName = filterName))

        #delete filter by clicking reset button
        resetButton.clicked.connect(lambda _,filterName = columnName : self.deleteFilter(filterName = filterName))

        #choose specific column
        specColumnLabel.clicked.connect(lambda _,filterName = columnName :self.chooseSpecColumn(filterName=filterName))

        #add items to hbox
        hbox.addWidget(columnLabel)
        hbox.addStretch(1)
        hbox.addWidget(filterLabel)
        hbox.addWidget(minEditValue)
        hbox.addWidget(maxEditValue)
        hbox.addWidget(specColumnLabel)
        hbox.addWidget(resetButton)
        hbox.setSpacing(4)
        #set outer frame layout
        outerFrame.setLayout(QVBoxLayout())
        outerFrame.layout().addWidget(frame)
        outerFrame.layout().setContentsMargins(1,1,1,1)

        #save filter props
        self.filterProps[columnName]["min"] = minEditValue
        self.filterProps[columnName]["max"] = maxEditValue
        self.filterProps[columnName]["minValue"] = minValue
        self.filterProps[columnName]["maxValue"] = maxValue
        self.filterProps[columnName]["filterType"] = filterType
        self.filterProps[columnName]["widgetsToUpdate"] = [columnLabel,specColumnLabel,filterLabel,resetButton]
        self.filterProps[columnName]["N"] = nValues
        self.filterProps[columnName]["specColumns"] = []
        self.updateLineEdits(columnName, filterType)

        return outerFrame

    def sendUpdatabelWidgets(self):
        ""
        return self.filterProps
    
    def chooseSpecColumn(self,filterName):
        ""
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
            self.filterProps[filterName]["specColumns"] = selectedColumns.values.flatten().tolist()
            numSelectedColumns = len(self.filterProps[filterName]["specColumns"])
            if hasattr(self.sender(),"setText"):
                if numSelectedColumns > 1:
                    self.sender().setText("{} columns selected".format(numSelectedColumns))
                else:
                    self.sender().setText(self.filterProps[filterName]["specColumns"][0][:20])
        else:
            if hasattr(self.sender(),"setText"):
                self.sender().setText("Choose Match column")
            self.filterProps[filterName]["specColumns"] = []

    def createValueLineEdit(self, placeholderText = "", tooltipStr = "", minValue = -np.inf, maxValue = np.inf, columnName = ""):
        
        valueEdit = createValueLineEdit(placeholderText,tooltipStr, minValue, maxValue)
        valueEdit.textChanged.connect(lambda _,filterName = columnName : self.lineEditChanged(filterName = filterName))
       
        return valueEdit
        

    def deleteFilter(self,event=None,filterName = None):
        "Removes widgets."    
        if filterName in self.filterProps:
            self.scrollFrame.layout().removeWidget(self.filterProps[filterName]["frame"])
            self.filterProps[filterName]["frame"].deleteLater()
            del self.filterProps[filterName]
            self.updateComboBox()

    def applyFilter(self,event=None):
        "Collect data and apply filter."
        funcProps = OrderedDict()
        for columnName, filtProps in self.filterProps.items():
            if filtProps["min"].text() != "" or filtProps["max"].text() != "":
                try:
                    funcProps[columnName] = {"min":float(filtProps["min"].text()) if filtProps["min"].text() != "" else -np.inf,
                                    "max":float(filtProps["max"].text()) if filtProps["max"].text() != "" else np.inf,
                                    "filterType":filtProps["filterType"]}
                except:
                    self.mC.sendToWarningDialog(infoText = "Entered values could not be converted to floats. Please use . instead of , for decimal numbers.",parent=self)
                    return
        if len(funcProps) == 0:
            self.mC.sendToWarningDialog(infoText = "Please enter values to specifiy the numeric filter.",parent=self)
            return

        funcProps = {"key":"filter::numericFilter",
            "kwargs":{
                "dataID":self.dataID,
                "filterProps":funcProps,
                "setNonMatchNan":getBoolFromCheckState(self.CBFilterOptions["Set NaN"].checkState()) or getBoolFromCheckState(self.CBFilterOptions["Set NaN in spec. columns"].checkState())}}
        if getBoolFromCheckState(self.CBFilterOptions["Set NaN in spec. columns"].checkState()):
            funcProps["kwargs"]["selectedColumns"] =  OrderedDict([(k,v["specColumns"]) for k,v in self.filterProps.items()])
        elif getBoolFromCheckState(self.CBFilterOptions["Subset Matches"].checkState()):
            funcProps["kwargs"]["subsetData"] = True
            funcProps["key"] = "filter::subsetNumericFilter"
       
        self.mC.sendRequestToThread(funcProps)
        self.close()
        

    def updateComboBox(self):
        ""
        numericColumns = [col for col in self.numericColumns if col not in self.filterProps] 
        self.columnNameCombo.clear()
        self.columnNameCombo.addItems(["Choose additional column"] + numericColumns )   
        self.columnNameCombo.model().item(0).setEnabled(False)

    def lineEditChanged(self,event=None,filterName = None):
        "Handles line edit changes"
        self.evaluateEditInput(self.sender())

        if self.sender().hasAcceptableInput() and "n " not in self.filterProps[filterName]["filterType"]:
           
            if self.sender() == self.filterProps[filterName]["min"]:
                
                validator = self.filterProps[filterName]["max"].validator()
                validator.setBottom(float(self.sender().text()))
                self.filterProps[filterName]["max"].setValidator(validator)
                self.evaluateEditInput(self.filterProps[filterName]["max"])

            elif self.sender() == self.filterProps[filterName]["max"]:
                
                validator = self.filterProps[filterName]["min"].validator()
                validator.setTop(float(self.sender().text()))
                self.filterProps[filterName]["min"].setValidator(validator) 
                self.evaluateEditInput(self.filterProps[filterName]["min"])  

        elif self.sender().text() == "":
            self.resetValidator(filterName, topN = "n " in self.filterProps[filterName]["filterType"])


    def resetValidator(self, filterName, topN = False):
        "Reset the Validator"
        for lineEdit in ["min","max"]:
            validator = self.filterProps[filterName][lineEdit].validator()
            if not topN:
                validator.setRange(self.filterProps[filterName]["minValue"], self.filterProps[filterName]["maxValue"])
                validator.setDecimals(20)
            else:
                validator.setRange(1, self.filterProps[filterName]["N"])
                validator.setDecimals(0)

            self.filterProps[filterName][lineEdit].setValidator(validator) 

    def evaluateEditInput(self,lineEdit):
        ""
        if not lineEdit.hasAcceptableInput():
            lineEdit.setStyleSheet("color : {}; background: white".format("black"))
        else:
            lineEdit.setStyleSheet("color : {}; background: white".format(INSTANT_CLUE_BLUE))


    def updateLineEdits(self,columnName, filterType):
        ""
        
        if filterType in LINE_EDIT_STATUS:
            
            minReadOnly , maxReadOnly = LINE_EDIT_STATUS[filterType]
           
            self.filterProps[columnName]["min"].setReadOnly(minReadOnly)
            self.filterProps[columnName]["max"].setReadOnly(maxReadOnly)

            self.setLineEditStyle(columnName, minReadOnly,maxReadOnly)

    def setLineEditStyle(self,columnName, minReadOnly,maxReadOnly):

        if minReadOnly:
            self.filterProps[columnName]["min"].setText("")
            self.filterProps[columnName]["min"].setPlaceholderText("...")
        else:
            self.filterProps[columnName]["min"].setPlaceholderText("Enter value")

        if maxReadOnly:
            self.filterProps[columnName]["max"].setText("")
            self.filterProps[columnName]["max"].setPlaceholderText("...")
        else:
            self.filterProps[columnName]["max"].setPlaceholderText("Enter value")
        
    def setOperator(self,newOperator):
        ""
        self.mC.numericFilter.setOperator(newOperator)
    
    def setCBCheckStates(self,event=None):
        ""
        for cbKey,cb in self.CBFilterOptions.items():
            if cb != self.sender():
                cb.setCheckState(getCheckStateFromBool(False))
            else:
                cb.setCheckState(getCheckStateFromBool(True))