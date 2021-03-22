from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from backend.config.data.params import encodingsCommonInPython, commonSepartor, decimalForFloats,thoursandsString, nanReplaceString
from ..custom.buttonDesigns import ICStandardButton
from ..utils import createLabel, createCombobox
from collections import OrderedDict

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


comboboxExcelFile = OrderedDict([("Excel sheets:",["None"]),
                                 ('Skip Rows:',[str(x) for x in range(0,20)]),
                                 ('Thousand Separator:',thoursandsString),
                                 ("Add. na values",["#VALUE!","#WERT","None"])])


pandasInstantClueTranslate = {'Encoding:':'encoding',
						  'Column Separator:':'sep',
						  'Decimal Point String:':'decimal',
						  'Thousand Separator:':'thousands',
						  'Decompression:':'compression',
						  'Skip Rows:':'skiprows',
                          "Excel sheets:":"sheet_name",
                          "Add. na values":"na_values"
						  }

tooltips = {"Excel sheets:":"Provide names of excel sheets to load as: Sheet1;Sheet2.\nIf None - all excel sheets will be loaded.\nNote that a comma in the sheet name will result in unexpected file loading.",
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


class ExcelImporter(ImporterBase):

    def __init__(self,*args, **kwargs):
        super(ExcelImporter,self).__init__(*args, **kwargs)
        #self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Load Excel (.xlsx) file.")
        self.__controls()
        self.__layout()
        self.__bindEvents()

    def __controls(self):

        self.loadButton = ICStandardButton(itemName="Load")
        self.closeButton = ICStandardButton(itemName="Cancel")

        self.widgetControl = []
        for label, options in comboboxExcelFile.items():
            
            propLabel = createLabel(label)
            propLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            propCombo = createCombobox(items=options)
            propCombo.setEditable(True)

            if label in tooltips:
                propLabel.setToolTip(tooltips[label])

            if label in comboboxLabelToParam:
                
                defaultValue = self.mC.config.getParam(comboboxLabelToParam[label])
                
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



class PlainTextImporter(ImporterBase):
    def __init__(self,*args, **kwargs):
        super(PlainTextImporter,self).__init__(*args, **kwargs)
        #self.setAttribute(Qt.WA_DeleteOnClose)
        try:
            self.setWindowTitle("Load plain text file.")
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
            propLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            propCombo = createCombobox(items=options)
            propCombo.setEditable(True)
            if label in comboboxLabelToParam:
                
                defaultValue = self.mC.config.getParam(comboboxLabelToParam[label])
                
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

    


    





