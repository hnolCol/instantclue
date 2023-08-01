from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 


#internal imports
from ...custom.tableviews.ICVSelectableTable import SelectablePandaModel, PandaTable, MultiColumnSelectablePandaModel
from ...custom.utils import LabelLikeCombo
from ...custom.warnMessage import WarningMessage
from ...utils import createLabel, createLineEdit, getMessageProps, createCombobox, getCheckStateFromBool, toggleCheckState, getBoolFromCheckState
from ...custom.Widgets.ICButtonDesgins import AcceptButton, RefreshButton
from backend.utils.stringOperations import mergeListToString

#external imports
import pandas as pd
from collections import OrderedDict 

CB_OPTIONS= ["Annotate Selection","Subset Selection"]
CB_TOOLTIPS = ["Create a new column indicating by '+' if selected category was found in row",
               "Creates a new data frame with rows matching selected categories."]
FIND_STRING_OPTIONS = ["Annotate matches by search string","Case sensitive","Input is regular expression"]
FIND_STRING_TOOLTIPS = ["Inseatd of '+' use search string to indicate matches.","Enable case sensitive searches","Enable regular expression as input."]


CUSTOM_CB_OPTIONS = ["Annotate Selection","Subset Selection"]#"Annotate Selection by Category",
CUSTOM_CB_TOOLTIPS = ["Create a new column indicating by '+' if selected category was found in row",
                    #  "Annotate matches by selected category",
                      "Creates a new data frame with rows matching selected categories."]

class FilterBase(QDialog):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(FilterBase, self).__init__(*args,**kwargs)
        self.setSizeGripEnabled(True)
        self.setMinimumWidth(450)
        self.setMinimumHeight(700)
        
    def forceSearch(self,event=None):
        "Forcing the search"
        self.lineEditChanged(searchString=self.searchLine.text(), forceSearch=True)

    def lineEditChanged(self,searchString,**kwargs):
        "Line edit changed"
        reqKwargs = {"searchString":searchString,"updatedData":self.model.df}
        kwargs = {**reqKwargs,**kwargs}
        self.mC.sendRequest({"key":"filter::liveStringSearch","kwargs":kwargs})

    def updateModelDataByBool(self,boolIndicator, resetData=False):
        "Update model by bool series"
        
        self.table.model().layoutAboutToBeChanged.emit()
        self.table.model().updateDataByBool(boolIndicator,resetData)
        self.table.model().layoutChanged.emit() 

    def closeEvent(self,event=None):
        "Overwrite close event"
        self.mC.categoricalFilter.stopLiveFilter()
        event.accept()
    
    def keyPressEvent(self,event=None):
        ""
        if event.key() == Qt.Key.Key_Enter:
            return
        elif event.key() == Qt.Key.Key_Escape:
            self.close() 
    
        

class CustomCategoricalFilter(FilterBase):

    def __init__(self,mainController,categoricalColumns,*args,**kwargs):
        super(CustomCategoricalFilter,self).__init__(*args, **kwargs)

        self.mC = mainController
        self.categoricalColumns = categoricalColumns

        self.setWindowTitle("Categorical Filter ({}).".format(mergeListToString(categoricalColumns,",")))
        self.setWindowIcon(self.mC.getWindowIcon())
        self.__controls()
        self.__layout()
        self.__connectEvents()

    def __controls(self):
        ""
        self.searchLine = createLineEdit("Enter search strings .. ",
            """Multiple search strings must be separated: "String1","String2".\nPress enter to update view.""")
        self.searchLine.textChanged.connect(self.lineEditChanged)
        self.searchLine.returnPressed.connect(self.forceSearch)

        self.splitStringLine = createLineEdit("SplitString","Split string to find unique categories.")
        self.splitStringLine.setText(self.mC.config.getParam("splitString"))


        self.operatorLabel = createLabel("Operator:",fontSize = 10)
        self.operatorCombo = createCombobox(parent=self, items=["and","or"])
        self.checkButton = AcceptButton()
        self.updateButton = RefreshButton()

        self.CBFilterOptions = OrderedDict() 
        for n,filtOption in enumerate(CUSTOM_CB_OPTIONS ):
            cb = QCheckBox(filtOption, toolTip = CUSTOM_CB_TOOLTIPS[n])
            cb.setTristate(False)
            if n == 0:
                cb.setCheckState(getCheckStateFromBool(True))
            cb.clicked.connect(self.setCBCheckStates)
            self.CBFilterOptions[filtOption] = cb

        self.table = PandaTable(self, cornerButton = False, hideMenu = True) 
        self.model = MultiColumnSelectablePandaModel(parent= self.table, df = self.mC.categoricalFilter.liveSearchData)
        self.table.setModel(self.model)
        #set columns stretch
        for nColumn in range(self.mC.categoricalFilter.liveSearchData.columns.size):
            self.table.horizontalHeader().setSectionResizeMode(nColumn, QHeaderView.ResizeMode.Stretch)

    def __layout(self):

        self.setLayout(QGridLayout()) 

        self.layout().addWidget(self.searchLine,0,0,1,2)
        self.layout().addWidget(self.checkButton,0,2,1,1)
        self.layout().addWidget(self.splitStringLine,1,0,1,2)
        self.layout().addWidget(self.updateButton,1,2,1,1)
        hbox = QHBoxLayout()
        hbox.addWidget(self.operatorLabel)
        hbox.addStretch(1)
        hbox.addWidget(self.operatorCombo)
        self.layout().addLayout(hbox,2,0,1,2)
        # self.layout().addWidget(self.operatorLabel,2,0,1,1)
        # self.layout().addWidget(self.operatorCombo,2,1,1,1)
        for n,cb in enumerate(self.CBFilterOptions.values()):
             self.layout().addWidget(cb,3,n,1,1)
        self.layout().addWidget(self.table,4,0,1,3)
        
    def __connectEvents(self):

        self.checkButton.clicked.connect(self.applyFilter)
        self.updateButton.clicked.connect(self.updateData)

    def applyFilter(self,event=None):
        ""
        checkedData = self.table.model().getCheckedData()
        
        #check if any data were selected
        if len(checkedData) == 0 or all(len(v) == 0 for v in checkedData.values()):
            w = WarningMessage(infoText = "No category selected.")
            w.exec()
            return
        if not all(columnName in checkedData for columnName in self.categoricalColumns.values):
            w = WarningMessage(infoText = "No category selected for at least one column.")
            w.exec()
            return
        elif getBoolFromCheckState(self.CBFilterOptions["Annotate Selection"].checkState()):

            
            funcProps = {"key":"filter::applyLiveFilter","kwargs":{
                                            "searchString":checkedData,
                                            "operator" : self.operatorCombo.currentText()
                                            }}
            
            self.mC.sendRequest(funcProps)

        elif getBoolFromCheckState(self.CBFilterOptions["Subset Selection"].checkState()):
            funcProps = {"key":"filter::subsetData","kwargs":{"searchString":checkedData,
                                                              "dataID":self.mC.getDataID(),
                                                              "filterType":"multiColumnCategory",
                                                              "operator" : self.operatorCombo.currentText(),
                                                              "columnName":self.categoricalColumns.values}}
            self.mC.sendRequestToThread(funcProps)
       
        self.close() 

    def updateData(self,event=None):
        "Update data when update button is pressed"
        if hasattr(self.mC.data.categoricalFilter, "filterProps"):
            #get split string
            newSplitString = self.splitStringLine.text()
            #get used split string
            currentSplitString = self.mC.data.categoricalFilter.filterProps["splitString"]
            if newSplitString != currentSplitString:
                self.mC.data.categoricalFilter.setSplitString(newSplitString)
                dataID = self.mC.mainFrames["data"].getDataID()
                self.mC.data.categoricalFilter.setupLiveStringFilter(dataID,self.categoricalColumns,updateData=True)
                self.table.model().updateDataFrame(self.mC.categoricalFilter.liveSearchData)
                #save split string
                self.mC.config.setParam("splitString",newSplitString)


    def setCBCheckStates(self,event=None):
        ""
        for cb in self.CBFilterOptions.values():
            cbCheckState = cb.checkState()
            if cb != self.sender():
                newCheckState, _ = toggleCheckState(cb.checkState())
                cb.setCheckState(newCheckState)
            else:
                cb.setCheckState(cbCheckState)

class CategoricalFilter(FilterBase):

    def __init__(self,mainController, categoricalColumns = '', *args, **kwargs):

        super(CategoricalFilter,self).__init__(*args, **kwargs)
        self.mC = mainController
        self.categoricalColumns = categoricalColumns
        self.setWindowTitle("Categorical Filter ({}).".format(mergeListToString(categoricalColumns,",")))
        self.setWindowIcon(self.mC.getWindowIcon())
        self.__controls()
        self.__layout()
        self.__connectEvents()
    
    def __controls(self):

        self.searchLine = createLineEdit("Enter search strings .. ",
            """Multiple search strings must be separated: "String1","String2".\nPress enter to update view.""")
        self.searchLine.textChanged.connect(self.lineEditChanged)
        self.searchLine.returnPressed.connect(self.forceSearch)

        self.splitStringLine = createLineEdit("SplitString","Split string to find unique categories.")
        self.splitStringLine.setText(self.mC.config.getParam("splitString"))

        self.checkButton = AcceptButton()
        self.updateButton = RefreshButton()

        self.CBFilterOptions = OrderedDict() 
        for n,filtOption in enumerate(CB_OPTIONS):
            cb = QCheckBox(filtOption, toolTip = CB_TOOLTIPS[n])
            cb.setTristate(False)
            if n == 0:
                cb.setCheckState(getCheckStateFromBool(True))
            cb.clicked.connect(self.setCBCheckStates)
            self.CBFilterOptions[filtOption] = cb

        self.table = PandaTable(self, cornerButton = False, hideMenu = True) 
        self.model = SelectablePandaModel(parent= self.table, df = self.mC.categoricalFilter.liveSearchData)
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
       
    def __layout(self):

        self.setLayout(QGridLayout()) 

        self.layout().addWidget(self.searchLine,0,0,1,2)
        self.layout().addWidget(self.checkButton,0,2,1,1)
        self.layout().addWidget(self.splitStringLine,1,0,1,2)
        self.layout().addWidget(self.updateButton,1,2,1,1)
        for n,cb in enumerate(self.CBFilterOptions.values()):
            self.layout().addWidget(cb,2,n,1,1)
        self.layout().addWidget(self.table,3,0,1,3)
        
    def __connectEvents(self):

        self.checkButton.clicked.connect(self.applyFilter)
        self.updateButton.clicked.connect(self.updateData)

    def applyFilter(self,event=None):
        ""
        checkedData = self.table.model().getCheckedData()
        #check if any data were selected
        selectedCategories = checkedData.values[:,0].tolist()
        if checkedData.empty:
            self.mC.sendMessageRequest(getMessageProps("Error ..","No category selected."))
            return

        elif getBoolFromCheckState(self.CBFilterOptions["Annotate Selection"].checkState()):

            
            funcProps = {"key":"filter::applyLiveFilter","kwargs":{
                                            "searchString":selectedCategories}}
            self.mC.sendRequest(funcProps)

        elif getBoolFromCheckState(self.CBFilterOptions["Subset Selection"].checkState()):
            funcProps = {"key":"filter::subsetData","kwargs":{"searchString":selectedCategories,
                                                              "dataID":self.mC.mainFrames["data"].getDataID(),
                                                              "columnName":self.categoricalColumns.values[0]}}
            
            self.mC.sendRequestToThread(funcProps)
        #update column names
        self.close()

    def setCBCheckStates(self,event=None):
        ""
        for cb in self.CBFilterOptions.values():
            cbCheckState = cb.checkState()
            if cb != self.sender():
                newCheckState, _ = toggleCheckState(cb.checkState())
                cb.setCheckState(newCheckState)
            else:
                cb.setCheckState(cbCheckState)

    def updateData(self,event=None):
        "Update data when update button is pressed"
        if hasattr(self.mC.data.categoricalFilter, "filterProps"):
            #get split string
            newSplitString = self.splitStringLine.text()
            #get used split string
            currentSplitString = self.mC.data.categoricalFilter.filterProps["splitString"]
            if newSplitString != currentSplitString:
                self.mC.data.categoricalFilter.setSplitString(newSplitString)
                dataID = self.mC.mainFrames["data"].getDataID()
                self.mC.data.categoricalFilter.setupLiveStringFilter(dataID,self.categoricalColumns,updateData=True)
                self.table.model().updateDataFrame(self.mC.categoricalFilter.liveSearchData)
                #save split string
                self.mC.config.setParam("splitString",newSplitString)
        

class FindStrings(FilterBase):

    def __init__(self,mainController, categoricalColumns = '', *args, **kwargs):

        super(FindStrings,self).__init__(*args, **kwargs)
        self.mC = mainController
        self.categoricalColumns = categoricalColumns
        self.lastSearch = ""
        self.setWindowTitle("Find strings ({}).".format(mergeListToString(categoricalColumns,",")))
        self.setWindowIcon(self.mC.getWindowIcon())
        self.__controls()
        self.__layout()
        self.__connectEvents()

    def __controls(self):

        self.searchLine = createLineEdit("Enter search strings .. ",
            """Multiple search strings must be separated: "String1","String2".\nPress enter to update view.""")
        self.searchLine.textChanged.connect(self.checkLineEditText)
        self.searchLine.returnPressed.connect(self.forceSearch)

        self.checkButton = AcceptButton()
    
        self.CBFilterOptions = OrderedDict() 
        for n,filtOption in enumerate(FIND_STRING_OPTIONS):
            cb = QCheckBox(filtOption)
            cb.setTristate(False)
            if n == 0:
                cb.setCheckState(getCheckStateFromBool(True))
            cb.setToolTip(FIND_STRING_TOOLTIPS[n])
            self.CBFilterOptions[filtOption] = cb

        self.table = PandaTable(self, cornerButton = False, hideMenu = True) 
        self.model = SelectablePandaModel(parent= self.table, df = self.mC.categoricalFilter.liveSearchData)
        self.table.setModel(self.model)
        for nColumn in range(self.mC.categoricalFilter.liveSearchData.columns.size):
            self.table.horizontalHeader().setSectionResizeMode(nColumn, QHeaderView.ResizeMode.Stretch)
       
    def __layout(self):

        self.setLayout(QGridLayout()) 

        self.layout().addWidget(self.searchLine,0,0,1,2)
        self.layout().addWidget(self.checkButton,0,2,1,1)
        
        for n,cb in enumerate(self.CBFilterOptions.values()):
            self.layout().addWidget(cb,2,n,1,1)
        self.layout().addWidget(self.table,3,0,1,3)
        
    def __connectEvents(self):

        self.checkButton.clicked.connect(self.applyFilter)

    def checkLineEditText(self,searchString):
        ""
        splitNewLine = searchString.split("\n")
        if len(splitNewLine) > 1:
            s = ','.join('"{0}"'.format(w.replace("\r","")) for w in splitNewLine)
            searchString = s
        if self.lastSearch != searchString:
            self.lastSearch = searchString
            self.searchLine.setText(searchString)
            self.lineEditChanged(searchString, 
                            caseSensitive= getBoolFromCheckState(self.CBFilterOptions["Case sensitive"].checkState()),
                            inputIsRegEx = getBoolFromCheckState(self.CBFilterOptions["Input is regular expression"].checkState()))
        

    def applyFilter(self,event=None):
        ""
        
        #check if any data were selected
        if self.searchLine.text() == "":
            self.mC.sendMessageRequest(getMessageProps("Error ..","No string entered."))
            return
        
        funcProps = {"key":"filter::applyLiveFilter",
                    "kwargs":{"searchString":self.searchLine.text(),
                            "caseSensitive":getBoolFromCheckState(self.CBFilterOptions["Case sensitive"].checkState()),
                            "inputIsRegEx":getBoolFromCheckState(self.CBFilterOptions["Input is regular expression"].checkState()),
                            "annotateSearchString":getBoolFromCheckState(self.CBFilterOptions["Annotate matches by search string"].checkState())}}

        self.mC.sendRequest(funcProps)
       
        #update column names
        self.close()
        
