from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from ..utils import createTitleLabel, createLabel, createLineEdit
from ..custom.tableviews.ICVSelectableTable import SelectablePandaModel, PandaTable
from ..custom.buttonDesigns import ResetButton, AcceptButton, CheckButton
import pandas as pd

class ICDSelectItems(QDialog):
    def __init__(self, data = pd.DataFrame(), stretch = True, *args, **kwargs):
        super(ICDSelectItems, self).__init__(*args, **kwargs)
    
        self.data = data
        self.stretch = stretch
        self.__windowUpdate()
        self.__controls()
        self.__layout()
        self.__connectEvents()
    
    def __controls(self):
        """Init widgets"""
        
        self.headerLabel = createTitleLabel("Compare groups using",fontSize=12)
        self.table = PandaTable(parent=self)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.model = SelectablePandaModel(parent=self.table, df = self.data)
        self.table.setModel(self.model)
        if self.stretch:
            self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch) 
       
        self.selectCB = QCheckBox("Select all")
        self.selectCB.setTristate(False)
        self.selectCB.stateChanged.connect(self.manageSelection)

        self.okButton = AcceptButton()
        self.okButton.setFixedSize(QSize(15,15))
        self.cancelButton = ResetButton()

    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QGridLayout())
        self.layout().setContentsMargins(3,3,3,3)
        self.layout().addWidget(self.table,0,0,1,3)
        self.layout().addWidget(self.selectCB,1,0)
        self.layout().addWidget(self.okButton,1,1)
        self.layout().addWidget(self.cancelButton,1,2)
        

       
    def __connectEvents(self):
        """Connect events to functions"""
        self.okButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)
        

    def __windowUpdate(self):
        ""
        self.setWindowFlags( Qt.WindowStaysOnTopHint)# Qt.FramelessWindowHint |
       # self.setWindowOpacity(0.95)


    def getApparentHeight(self):
        #15 is fixed height in used table
        
        height = int(15 * (self.data.index.size + 10 ) + 3 + 3 + 1) #+1 for margin, 3 top, 3 ottom + 1 spacing
        if height > 400:
            return 400
        return height

    def getSelection(self):
        ""
        return self.model.getCheckedData()
    
    def manageSelection(self,newSate):
        ""
        print(newSate)
        if newSate == 0:
            self.sender().setText("Select all")

        else:
            self.sender().setText("Deselect all")
        try:
            self.model.setAllCheckStates(newSate != 0)
            self.model.completeDataChanged()
            
        except Exception as e:
            print(e)