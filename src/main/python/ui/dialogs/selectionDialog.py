from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * #works for pyqt5

from ..utils import createLabel, createLineEdit, createTitleLabel, createCombobox
from ..custom.buttonDesigns import ICStandardButton

from collections import OrderedDict 

class SelectionDialog(QDialog):
    def __init__(self,selectionNames, selectionOptions, selectionDefaultIndex, title ="Selection", selectionEditable = [], *args, **kwargs):
        super(SelectionDialog,self).__init__(*args, **kwargs)
        self.title = title
        self.savedSelection = OrderedDict()
        self.selectionCombos = OrderedDict()

        self.selectionNames = selectionNames
        self.selectionOptions = selectionOptions
        self.selectionDefaultIndex = selectionDefaultIndex

        self.selectionEditable = selectionEditable

        self.__controls()
        self.__layout()
        self.__windowUpdate()
        self.__connectEvents()

    def __windowUpdate(self):

       # self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowOpacity(0.95)
        self.setWindowTitle("Selection Dialog")
       
    
    def __controls(self):
        """Init widgets"""
        self.titleLabel =  createTitleLabel(self.title,fontSize=12)
        # set up okay buttons
        for selectionName in self.selectionNames:
            self.selectionCombos[selectionName] = dict()
            label = createLabel("{} :".format(selectionName))
            label.setAlignment(Qt.AlignRight | Qt.AlignCenter)
            selectionOptions = self.selectionOptions[selectionName]
            cb = createCombobox(self,selectionOptions)
            if selectionName in self.selectionDefaultIndex:
                
                cb.setCurrentText(self.selectionDefaultIndex[selectionName])
            if selectionName in self.selectionEditable:
                cb.setEditable(True)
                
            self.selectionCombos[selectionName]["label"] = label
            self.selectionCombos[selectionName]["cb"] = cb


        self.okayButton = ICStandardButton(itemName="Okay")
        self.closeButton = ICStandardButton(itemName="Close")
        
        
    def __layout(self):
        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.titleLabel)

        gridLayout = QGridLayout()
        for n,selection in enumerate(self.selectionCombos.values()):
            
            gridLayout.addWidget(selection["label"],n,0)
            gridLayout.addWidget(selection["cb"],n,1)
            gridLayout.setColumnStretch(1,1)
            gridLayout.setAlignment
            
        self.layout().addLayout(gridLayout)

        hbox = QHBoxLayout()
        hbox.addWidget(self.okayButton)
        hbox.addWidget(self.closeButton)

        self.layout().addLayout(hbox)
        self.layout().setAlignment(Qt.AlignTop)

    def __connectEvents(self):
        """Connect events to functions"""
        self.closeButton.clicked.connect(self.close)
        self.okayButton.clicked.connect(self.saveAndClose)


    def keyPressEvent(self,e):
        """Handle key press event"""
        if e.key() == Qt.Key_Escape:
            self.reject()
        
    def saveAndClose(self,event=None):
        ""
        for selectionName, selectionWidgets  in self.selectionCombos.items():
            self.savedSelection[selectionName] = selectionWidgets["cb"].currentText()
        self.accept()
        self.close()
        