from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from backend.color.colorHelper import ColorHelper
from ..utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR
from ..custom.utils import PropertyChooser

import seaborn as sns



class ConfigDialog(QDialog):
    def __init__(self,mainController,*args, **kwargs):
        super(ConfigDialog,self).__init__(*args, **kwargs)
        self.setMinimumSize(QSize(350,420))
        self.mC = mainController

        self.__controls()
        self.__layout()
        
        self.__connectEvents()
        
    
    def __controls(self):
        """Init widgets"""
        propItems = sorted(self.mC.config.getParentTypes())
        self.titleLabel = createTitleLabel("Configurations")
        self.infoLabel = createLabel("Changed parameters are automatically saved.", fontSize=11)
        self.propHolder = PropertyChooser()
        
        self._setupScrollarea()
        self.propCombo = QComboBox()
        
        self.propCombo.addItems(propItems)
        if self.mC.config.lastConfigGroup is not None:
            self.updatePropHolder(self.mC.config.lastConfigGroup)
            self.propCombo.setCurrentText(self.mC.config.lastConfigGroup)
        else:
            if len(propItems) > 0:
                self.updatePropHolder(propItems[0])

        self.propCombo.currentTextChanged.connect(self.updatePropHolder)
        
    def __layout(self):
        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(10,10,10,10)
        self.layout().setSpacing(4)
        self.layout().addWidget(self.titleLabel)
        self.layout().addWidget(self.infoLabel)
        self.layout().addWidget(self.propCombo)
        self.layout().addWidget(self.itemFrame)

    def __connectEvents(self):
        """Connect events to functions"""
       
    def _setupScrollarea(self):
        ""
        self.itemFrame = QScrollArea()
        self.itemFrame.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
      #  self.itemFrame.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.itemFrame.setWidgetResizable(True)
        
        #self.itemFrame.setStyleSheet("background-color:red;");
        self.itemFrame.setFrameShape(QFrame.NoFrame)
        self.itemFrame.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.itemFrame.setWidget(self.propHolder)
        
    def closeEvent(self,event=None):
        ""
        #save current parameter group
        setattr(self.mC.config,"lastConfigGroup",self.propCombo.currentText())
        self.updateParams()
        self.mC.mainFrames["right"].updateTypeSpecMenus()
        event.accept()
    
    def updateParams(self,event=None):
        ""
        self.propHolder.updateParams()

    def updatePropHolder(self,parentType,**kwargs):
        ""
        parameters = self.mC.config.getParametersByType(parentType)
        self.propHolder.addProperties(parameters,**kwargs)