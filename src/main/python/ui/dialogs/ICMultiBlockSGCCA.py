from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 
from ..custom.buttonDesigns import ICStandardButton
from ..utils import createTitleLabel, createLabel, createLineEdit, createCombobox



# scale data
#c1
#feature names
#n components



class ICMultiBlockSGCCA(QDialog):
    def __init__(self, mainController, *args, **kwargs):
        super(ICMultiBlockSGCCA, self).__init__(*args, **kwargs)
        self.mC = mainController

        self.__controls()
        self.__layout()
        self.__connectEvents()
    
    def __controls(self):
        """Init widgets"""
        
        self.headerLabel = createTitleLabel("Variable Selection For Generalized Canonical Correlation Analysis ",fontSize=15)
       
        self.schemeCombobox = createCombobox(items=['horst','factorial','centroid'])
        
        self.okayButton = ICStandardButton("Okay")
        self.cancelButton = ICStandardButton("Cancel")


    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        
        self.layout().addWidget( self.schemeCombobox)

        hbox = QHBoxLayout() 
        hbox.addWidget(self.okayButton)
        hbox.addWidget(self.cancelButton)

        self.layout().addLayout(hbox)

        self.layout().addStretch()
        
       
    def __connectEvents(self):
        """Connect events to functions"""
        self.cancelButton.clicked.connect(self.close)
        self.okayButton.clicked.connect(self.startCalculations)
        

    def startCalculations(self,event=None):
        """Start calculation by sending a request to thread."""
        