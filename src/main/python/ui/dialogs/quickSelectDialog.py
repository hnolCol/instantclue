from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * #works for pyqt5
from ..utils import createTitleLabel, createLabel, createLineEdit


class QuickSelectDialog(QDialog):
    def __init__(self,mainController,*args, **kwargs):
        super(QuickSelectDialog,self).__init__(*args, **kwargs)

        self.mC = mainController

        self.props = {"mode":"raw","sep":";"}

        self.__controls()
        self.__layout()
        self.__windowUpdate()
        self.__connectEvents()

    def __windowUpdate(self):

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowOpacity(0.85)

    
    def __controls(self):
        """Init widgets"""
        
        self.headerLabel = createTitleLabel("Quick Select",fontSize=12)
        self.infoLabel = createLabel("Select how you want to load the data.")
        self.rawValues = QCheckBox ("Raw values")
        self.rawValues.setCheckState(True)
        self.rawValues.setTristate(False)
        self.uniqueValues = QCheckBox("Unique values \nwith split string:")
        self.uniqueValues.setTristate(False)

        # set up line edit for separator
        self.separatorEdit = createLineEdit("Split String",tooltipText = "Separator to find unique values.") 
        self.separatorEdit.setFixedWidth(20)
        self.separatorEdit.setText(self.mC.config.getParam("quick.select.separator"))
        
        # set up okay buttons
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        

        
    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QGridLayout())

        self.layout().addWidget(self.headerLabel,0,0,1,3)
        self.layout().addWidget(self.infoLabel,1,0,1,3)

        self.layout().addWidget(self.rawValues,2,0,1,1)
        self.layout().addWidget(self.uniqueValues,2,1,1,1)
        self.layout().addWidget(self.separatorEdit,2,2,1,1)
        
        self.layout().addWidget(self.buttonBox,3,0,1,3)
        #self.layout().addWidget(self.cancelButton,3,1,1,2)


    def __connectEvents(self):

        """Connect events to functions"""
        self.separatorEdit.textChanged.connect(self.updateSep)
        self.rawValues.clicked.connect(lambda:self.changeProps(mode="raw"))
        self.uniqueValues.clicked.connect(lambda:self.changeProps(mode="unique"))

    def updateSep(self,newSep):
        ""
        self.props["sep"] = newSep

    def changeProps(self, mode = "unique"):
        ""
        #toogle checkbox state
        if self.sender() == self.rawValues:
            self.uniqueValues.setCheckState(not self.rawValues.checkState())
        elif self.sender() == self.uniqueValues:
            self.rawValues.setCheckState(not self.uniqueValues.checkState())

        self.props["mode"] = mode
        if mode == "unique":
            self.props["sep"] = self.separatorEdit.text()

    def getProps(self):
        ""
        return self.props

    def keyPressEvent(self,e):
        """Handle key press event"""
        if e.key() == Qt.Key_Escape:
            self.reject()
        
        