from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from backend.config.data.params import WIKI_LINKGS
from ..utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR, createMenu, createCombobox
from ..custom.utils import PropertyChooser
from ..custom.buttonDesigns import BigArrowButton, HelpButton

import seaborn as sns
import os
import webbrowser 


class ConfigDialog(QDialog):
    def __init__(self,mainController,specificSettingsTab=None,*args, **kwargs):
        super(ConfigDialog,self).__init__(*args, **kwargs)
        self.setMinimumSize(QSize(390,440))
        
        self.mC = mainController
        self.specificSettingsTab = specificSettingsTab
        self.setWindowTitle("Settings.")
        self.setWindowIcon(self.mC.getWindowIcon())
        self.__controls()
        self.__layout()
        
        self.__connectEvents()
        
    
    def __controls(self):
        """Init widgets"""
       
        self.titleLabel = createTitleLabel("Settings")
        self.infoLabel = createLabel("Changed parameters are automatically saved. After changing the parameter, just close the dialog.")
        self.propHolder = PropertyChooser(mainController=self.mC,parent=self)

        self.helpButton = HelpButton(buttonSize=(30,30), tooltipStr="Opens the specific GitHub Wiki Page for selected Settings (requires Internet connection).")
        self.saveButton = BigArrowButton(tooltipStr = "Save Setting Profile", buttonSize=(30,30))
        self.loadButton = BigArrowButton(direction="up", tooltipStr = "Load Setting Profile", buttonSize=(30,30))
        
        self._setupScrollarea()
        propItems = sorted(self.mC.config.getParentTypes())
        self.propCombo = createCombobox(self,propItems)
        if self.specificSettingsTab is not None and self.specificSettingsTab in propItems:
            self.updatePropHolder(self.specificSettingsTab)
            self.propCombo.setCurrentText(self.specificSettingsTab)
        elif self.mC.config.lastConfigGroup is not None:
            self.updatePropHolder(self.mC.config.lastConfigGroup)
            self.propCombo.setCurrentText(self.mC.config.lastConfigGroup)
        else:
            if len(propItems) > 0:
                self.updatePropHolder(propItems[0])
        
    def __layout(self):
        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(10,10,10,10)
        self.layout().setSpacing(4)

        gridBox = QGridLayout() 
        gridBox.addWidget(self.titleLabel,0,0)
        gridBox.addWidget(self.infoLabel,1,0)
        #gridBox.setRowStretch(1,1)
        gridBox.addWidget(self.helpButton,0,1,2,1,Qt.AlignmentFlag.AlignTop)
        gridBox.addWidget(self.saveButton,0,2,2,1,Qt.AlignmentFlag.AlignTop)
        gridBox.addWidget(self.loadButton,0,3,2,1,Qt.AlignmentFlag.AlignTop)
        gridBox.setColumnStretch(2,0)
        gridBox.setColumnStretch(3,0)
        gridBox.setContentsMargins(2,2,2,2)
        
        gridBox.addWidget(self.propCombo,4,0,1,4)
        gridBox.addWidget(self.itemFrame,5,0,1,4)
        
        self.layout().addLayout(gridBox)

    def __connectEvents(self):
        """Connect events to functions"""
        self.propCombo.currentTextChanged.connect(self.updatePropHolder)
        self.saveButton.clicked.connect(self.saveProfile)
        self.loadButton.clicked.connect(self.loadProfile)
        self.helpButton.clicked.connect(self.openWiki)

    def _setupScrollarea(self):
        ""
        self.itemFrame = QScrollArea()
        self.itemFrame.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.itemFrame.setWidgetResizable(True)
        self.itemFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.itemFrame.setMinimumHeight(400)
        self.itemFrame.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self.itemFrame.setWidget(self.propHolder)
        
    def closeEvent(self,event=None):
        ""
        #save current parameter group
        setattr(self.mC.config,"lastConfigGroup",self.propCombo.currentText())
        self.handleParamChange()
        event.accept()
    
    def openWiki(self,event=None,ulrExt=""):
        ""
        self.helpButton.mouseLostFocus()
        currentText = self.propCombo.currentText() 
        if currentText in WIKI_LINKGS:
            ulrExt = WIKI_LINKGS[currentText]
        
        webbrowser.open("https://github.com/hnolCol/instantclue/wiki"+"/"+ulrExt)


    def handleParamChange(self):
        ""
        self.updateParams()
        self.mC.mainFrames["right"].updateTypeSpecMenus()

    def loadProfile(self,event=None):
        ""
        menu = createMenu(parent=self)
        action = menu.addAction("Reset default")
        action.triggered.connect(self.resetDefaultParameter)

        for profileName in self.mC.config.getSavedProfiles():
            if profileName != "current":
                action = menu.addAction(profileName)
                action.triggered.connect(lambda _, profileName = profileName : self.loadProfileParamater(profileName))

        senderGeom = self.sender().geometry()
        bottomLeft = self.mapToGlobal(senderGeom.bottomLeft())
        menu.popup(bottomLeft)
        self.sender().mouseLostFocus()

    def loadProfileParamater(self,profileName = None):
        ""
        self.mC.config.loadProfile(profileName)
        self.updatePropHolder(self.propCombo.currentText())
        self.propCombo.setCurrentText(self.propCombo.currentText())
        self.handleParamChange()
    
    def resetDefaultParameter(self):
        ""
        self.mC.config.resetFactoryDefaults()
        self.updatePropHolder(self.propCombo.currentText())
        self.propCombo.setCurrentText(self.propCombo.currentText())
        self.handleParamChange()

    def updateParams(self,event=None):
        ""
        self.propHolder.updateParams()

    def updatePropHolder(self,parentType,**kwargs):
        ""
        parameters = self.mC.config.getParametersByType(parentType)
        self.propHolder.addProperties(parameters,**kwargs)
        self.helpButton.selectColor(updateButton=True)
        

    def saveProfile(self,event=None):
        ""
        text, ok = QInputDialog.getText(self, 'Save Profile', 'Enter name of setting profile:')
        if ok:
            self.handleParamChange()
            self.mC.config.saveProfile(text)
