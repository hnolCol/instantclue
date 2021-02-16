from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from collections import OrderedDict

encodingsCommonInPython = ['utf-8','ascii','ISO-8859-1','iso8859_15','cp037','cp1252','big5','euc_jp']
commonSepartor = ['tab',',','space',';','/','&','|','^','+','-']
decimalForFloats = ['.',','] 
compressionsForSourceFile = ['infer','gzip', 'bz2', 'zip', 'xz']
nanReplaceString = ['-','None', 'nan','  ']
thoursandsString = ['None',',','.']
comboBoxToGetInputFromUser = OrderedDict([('Encoding:',encodingsCommonInPython),
											('Column Separator:',commonSepartor),
											('Decimal Point String:',decimalForFloats),
											('Thousand Separator:',thoursandsString),
											('Decompression:',compressionsForSourceFile),
											('Skip Rows:',[str(x) for x in range(0,20)]),
											('Replace NaN in Object Columns:',nanReplaceString)])


comboboxExcelFile = OrderedDict([("Excel sheets:",["None"]),
                                 ('Skip Rows:',[str(x) for x in range(0,20)]),
                                 ('Thousand Separator:',thoursandsString)])


pandasInstantClueTranslate = {'Encoding:':'encoding',
						  'Column Separator:':'sep',
						  'Decimal Point String:':'decimal',
						  'Thousand Separator:':'thousands',
						  'Decompression:':'compression',
						  'Skip Rows:':'skiprows',
                          "Excel sheets:":"sheet_name"
						  }
tooltips = {"Excel sheets:":"Provide names of excel sheets to load as: Sheet1;Sheet2.\nIf None - all excel sheets will be loaded.\nNote that a comma in the sheet name will result in unexpected file loading."}

class ImporterBase(QDialog):
    def __init__(self,*args,**kwargs):
        ""
        super(ImporterBase,self).__init__(*args,**kwargs)
    
    def _collectSettings(self):
        """Collects selection by user"""
        self.loadFileProps = dict() 
        for label,propCombo in self.widgetControl:
            propLabel = label.text()
            if propLabel in pandasInstantClueTranslate:
                pandasKwarg = pandasInstantClueTranslate[propLabel]
                comboText = propCombo.currentText()
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

        self.loadButton = QPushButton("Load")
        self.closeButton = QPushButton("Cancel")

        self.widgetControl = []
        for label, options in comboboxExcelFile.items():
            
            propLabel = QLabel()
            propLabel.setText(label)
            propLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            if label in tooltips:
                propLabel.setToolTip(tooltips[label])

            propCombo = QComboBox()
            
            propCombo.setEditable(True)
            propCombo.addItems(options)
            
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

        self.loadButton = QPushButton("Load")
        self.closeButton = QPushButton("Cancel")
        

        self.widgetControl = []
        for label, options in comboBoxToGetInputFromUser.items():
            propLabel = QLabel()
            propLabel.setText(label)
            propLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            propCombo = QComboBox()
            
            propCombo.setEditable(True)
            propCombo.addItems(options)
            
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

    


    





