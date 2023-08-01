from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from collections import OrderedDict

from backend.config.data.params import encodingsCommonInPython, commonSepartor, decimalForFloats,thoursandsString, nanReplaceString

from ...custom.tableviews.ICVSelectableTable import SelectablePandaModel, PandaTable
from ...custom.Widgets.ICButtonDesgins import ICStandardButton
from ...utils import createLabel, createCombobox, createTitleLabel

import pandas as pd 
import openpyxl


compressionsForSourceFile = ['infer','gzip', 'bz2', 'zip', 'xz']


comboboxLabelToParam = dict([('Encoding:',"load.file.encoding"),
                            ('Column Separator:',"load.file.column.separator"),
                            ('Decimal Point String:',"load.file.float.decimal"),
                            ('Thousand Separator:',"load.file.float.thousands"),
                            ('Replace NaN in Object Columns:',"Object Replace String"),
                            ('skip Rows:',"load.file.skiprows"),
                            ("Add. na values","load.file.na.values")])

#ParamToComboLabel = dict([(v,k) for k,v in comboboxLabelToParam.items()])

comboBoxToGetInputFromUser = OrderedDict([('Encoding:',encodingsCommonInPython),
											('Column Separator:',commonSepartor),
											('Decimal Point String:',decimalForFloats),
											('Thousand Separator:',thoursandsString),
											('Decompression:',compressionsForSourceFile),
											('Skip Rows:',[str(x) for x in range(0,20)]),
											('Replace NaN in Object Columns:',nanReplaceString),
                                            ("Add. na values",["None","#VALUE!","#WERT"])])


excelParamFileSettings = dict([
                            ('Decimal Point String:',"load.file.float.decimal"),
                            ('thousands.separator',"load.file.float.thousands"),
                            ("skip.rows","load.file.skiprows"),
                            ("additional.na.values","load.file.na.values")])

comboboxExcelFile = OrderedDict([
                                 ("skip.rows",[str(x) for x in range(0,20)]),
                                 ('thousands.separator',thoursandsString),
                                 ("additional.na.values",["#VALUE!","#WERT","None"]),
                                 ("skip.footer",[str(x) for x in range(0,20)]),
                                 ("instant.clue.export",["False","True"])])


excelDataTypes = {
    "skip.rows": int,
    "skip.footer" : int,
    "thousands.separator" : str,
    "additional.na.values" : str,
    "instant.clue.export" : bool}

ICToPandasForExcelFiles = {
            "skip.rows" : "skiprows",
            "thousands.separator" : "thousands",
            "skip.footer" : "skipfooter",
            "additional.na.values": "na_values"
            }

pandasInstantClueTranslate = {'Encoding:':'encoding',
						  'Column Separator:':'sep',
						  'Decimal Point String:':'decimal',
						  'Thousand Separator:':'thousands',
						  'Decompression:':'compression',
						  'Skip Rows:':'skiprows',
                          "Excel sheets:":"sheet_name",
                          "Add. na values":"na_values"
						  }

tooltips = {"instant.clue.export":"If this option is set to True, the file uploader will check if a Grouping exists. The Grouping will then be loaded as well and remved from the headers.","Excel sheets:":"Provide names of excel sheets to load as: Sheet1;Sheet2.\nIf None - all excel sheets will be loaded.\nNote that a comma in the sheet name will result in unexpected file loading.",
            "Add. na values":"Add additional value to recognize as nan. If you want to provide multiple additional nan values separate them by a semicolon ;"}

class ImporterBase(QDialog):
    def __init__(self,mainController,*args,**kwargs):
        ""
        super(ImporterBase,self).__init__(*args,**kwargs)
        self.mC = mainController
    
    def _collectSettings(self):
        """Collects selection by user"""
        self.loadFileProps = dict() 
        for label,propCombo in self.widgetControl:
            propLabel = label.text()
            if propLabel in pandasInstantClueTranslate:
                pandasKwarg = pandasInstantClueTranslate[propLabel]
                comboText = propCombo.currentText()
                if propLabel in comboboxLabelToParam:
                    configName = comboboxLabelToParam[propLabel]
                    self.mC.config.setParam(configName,comboText)
                self.loadFileProps[pandasKwarg] = comboText if comboText != "None" else None
            elif propLabel == 'Replace NaN in Object Columns:':
                self.replaceObjectNan = propCombo.currentText() 
        self.accept()

    def getSettings(self):
        "Return Settings"
        if hasattr(self,"loadFileProps"):
            return self.loadFileProps     


class ExcelImporter(QDialog):
    def __init__(self,mainController,selectedFiles,*args,**kwargs):
        super(QDialog,self).__init__(*args,**kwargs)
        self.mC = mainController
        self.selectedFiles = selectedFiles
        self.isLoading = True
        self.excelSheetsInFiles = pd.DataFrame()
        self.__setupWindow()
        
        self.__control()
        self.__layout() 

        self.__bindEvents()
        self.sendFilesToThread()
       

    def __control(self):
        ""
        
        self.label = createTitleLabel("Excel is loading. Please wait..")
        self.table = PandaTable(parent=self)
       
        self.model = SelectablePandaModel(parent=self.table, df = self.excelSheetsInFiles, singleSelection=False)
        self.table.setModel(self.model)
        
       # self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)


        self.widgetControl = {}
        for label, options in comboboxExcelFile.items():
        
            propLabel = createLabel(label)
            propLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            propCombo = createCombobox(items=options)
            propCombo.setEditable(True)
            
            if label in tooltips:
                propLabel.setToolTip(tooltips[label])

            if label in excelParamFileSettings:
                
                defaultValue = self.mC.config.getParam(excelParamFileSettings[label])
                
                propCombo.setCurrentText(str(defaultValue))
            
            self.widgetControl[label] = (propLabel,propCombo)
    
        self.infoLabel = createLabel("For Instant Clue Excel file exports it is important\nthat sheet names have not been renamed\nand that a sheet called 'Software Info' exists (extracts - groupings info)")
        self.infoLabel.setWordWrap(True)
        self.loadAndCloseButton = ICStandardButton(itemName="Load & Close")
        self.loadAndCloseButton.setToolTip("Sheets will be loaded in the background and then added to the data treeview.")
        self.loadAndCloseButton.setEnabled(False)
        self.loadButton = ICStandardButton(itemName="Load Sheet(s)")
        self.loadButton.setEnabled(False)
        self.closeButton = ICStandardButton(itemName="Cancel")
        
    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.table)

        propGrid = QGridLayout()
        for n,(label,propCombo) in enumerate(self.widgetControl.values()):
            propGrid.addWidget(label,n,0)
            propGrid.addWidget(propCombo,n,1)

        propGrid.addWidget(self.infoLabel,n+1,0,1,2)
        
        propGrid.setColumnStretch(0,0)
        propGrid.setColumnStretch(1,2)

        self.layout().addLayout(propGrid)

        hbox = QHBoxLayout()
        hbox.addWidget(self.loadAndCloseButton)
        hbox.addWidget(self.loadButton)
        hbox.addWidget(self.closeButton)
        self.layout().addLayout(hbox)

    def __setupWindow(self):
        ""
        self.setWindowTitle("Load Excel (.xlsx) file.") 
        self.setWindowIcon(self.mC.getWindowIcon())
    
    def __bindEvents(self):
        ""
        self.loadAndCloseButton.clicked.connect(self.loadFileAndClose)
        self.loadButton.clicked.connect(self.loadFile)
        self.closeButton.clicked.connect(self.close)

    def loadFileAndClose(self):
        ""
        if self.loadFile():
            self.accept()

    def getLoadingProps(self):
        ""
        loadProps = {}
        instantClueImport = False
        
        for k, v in comboboxExcelFile.items():
            if k in self.widgetControl:
                value = self.widgetControl[k][1].currentText()
                try:
                    if excelDataTypes[k] == int:
                        value = int(float(value))
                    elif excelDataTypes[k] == str:
                        value = str(value) #always redundant? 
                    elif excelDataTypes[k] == bool:
                        value = value == "True"
                    
                    if k == "instant.clue.export":
                        instantClueImport = value
                    else:
                        #check if we want to translate the param 
                        if k in ICToPandasForExcelFiles:
                            k = ICToPandasForExcelFiles[k] # louzy overwrite
                        loadProps[k] = value
                except:
                    if k in excelDataTypes:
                        self.mC.sendToWarningDialog(infoText="There was an error when convertig {} to {}".format(k,excelDataTypes[k]),parent=self)
                    else:
                        self.mC.sendToWarningDialog(infoText="There was an error handling input for {}.".format(k),parent=self)
                    return None, None
        return loadProps,  instantClueImport


    def loadFile(self):
        ""
       
        selectedRows = self.model.getCheckedData()
        if selectedRows.size == 0:
            self.mC.sendToWarningDialog(infoText="Please select at least on sheet",parent=self)
            return False

        self.label.setText("Loading sheets .. Please Wait")

        ioAndSheets = {}
        for idx in selectedRows.index:
            fileName = selectedRows.loc[idx,"File"]
            sheetName = selectedRows.loc[idx,"Sheet"]
            
            dfName = "{}_{}".format(fileName,sheetName)
            ioAndSheets[dfName] = {"io":fileName,
                                    "sheet_name" : sheetName}
        if len(ioAndSheets) > 0:
            self.sendSheetForLoadingToThread(ioAndSheets)
            return True
        else:
            self.mC.sendToWarningDialog(infoText="Could not identify io and sheets..",parent=self)
            return False
    
    def sendSheetForLoadingToThread(self, ioAndSheets):
        ""
        loadFileProps, instantClueImport = self.getLoadingProps()
        if loadFileProps is None:
            return
        funcKey = "data::readExcelSheetsFromFile"
        kwargs = {
            "ioAndSheets":ioAndSheets,
            "props":loadFileProps,
            "instantClueImport":instantClueImport,
            }

        self.mC.sendRequestToThread({"key":funcKey,"kwargs":kwargs})
        

    def sendFilesToThread(self):
        ""
        funcKey = "data::readExcelFile"
        kwargs = {"pathToFiles":self.selectedFiles}
        self.mC.sendRequestToThread({"key":funcKey,"kwargs":kwargs})

    def setIsLoading(self,isLoading):
        ""
        setattr(self,"isLoading",isLoading)
        if not isLoading:
            self.label.setText("Files loaded. Select sheets to load.")
            self.loadButton.setEnabled(True)
            self.loadAndCloseButton.setEnabled(True)
        else:
            self.loadButton.setEnabled(False)
            self.loadAndCloseButton.setEnabled(False)

    def setExcelFiles(self,excelFiles):
        ""
        self.excelFiles = excelFiles
        
    def setSheetsToSelectTable(self,excelSheets):
        ""
        self.model.updateDataFrame(excelSheets)
        self.model.completeDataChanged()
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch)
        self.update()


class PlainTextImporter(ImporterBase):
    def __init__(self,*args, **kwargs):
        super(PlainTextImporter,self).__init__(*args, **kwargs)
        #self.setAttribute(Qt.WA_DeleteOnClose)
        try:
            self.setWindowTitle("Load plain text file.")
            self.setWindowIcon(self.mC.getWindowIcon())
            self.__controls()
            self.__layout()
            self.__bindEvents()

            self.replaceObjectNan = "-"
        except Exception as e:
            print(e)
    

    def __controls(self):

        self.loadButton = ICStandardButton(itemName="Load")
        self.closeButton = ICStandardButton(itemName="Cancel")
        

        self.widgetControl = []
        for label, options in comboBoxToGetInputFromUser.items():
            propLabel = createLabel(label)
            propLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            propCombo = createCombobox(items=options)
            propCombo.setEditable(True)
            if label in comboboxLabelToParam:
                
                defaultValue = self.mC.config.getParam(comboboxLabelToParam[label])
                if defaultValue not in options:
                    propCombo.addItem(defaultValue)
                propCombo.setCurrentText(defaultValue)
            
            self.widgetControl.append((propLabel,propCombo))

    def __layout(self):

        self.setLayout(QVBoxLayout())
        propGrid = QGridLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.loadButton)
        hbox.addWidget(self.closeButton)

        for n,(label,propCombo) in enumerate(self.widgetControl):
            propGrid.addWidget(label,n,0)
            propGrid.addWidget(propCombo,n,1)
        
        propGrid.setColumnStretch(0,0)
        propGrid.setColumnStretch(1,2)
        self.layout().addLayout(propGrid)
        self.layout().addLayout(hbox)

    def __bindEvents(self):
        ""
        self.loadButton.clicked.connect(self._collectSettings)
        self.closeButton.clicked.connect(self.close)

    


    





