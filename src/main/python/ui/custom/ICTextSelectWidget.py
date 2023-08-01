from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from ..dialogs.Selections.ICDSelectItems import ICDSelectItems

from ..utils import createLabel, createTitleLabel, createSubMenu, createMenu
from .utils import LabelLikeCombo, LabelLikeButton
from .Widgets.ICButtonDesgins import ICStandardButton, ResetButton, BigPlusButton

import pandas as pd 

class ICTextSelectWidget(QWidget):
    ""
    def __init__(self,
                parent = None, 
                descriptionText = "Specific Column",
                toolTipText = "",
                targetColumns = pd.Series(dtype="object"),
                selectableItems = pd.Series(dtype="object"),
                reportBackSelection = None,  *args, **kwargs):
        super(ICTextSelectWidget, self).__init__(parent)

        self.descriptionText = descriptionText
        self.selectableItems = selectableItems
        self.targetColumns = targetColumns
        self.reportBackSelection = reportBackSelection
        self.currentSelection = []
        self.setMouseTracking(True)
        self.setToolTip(toolTipText)
        self.setStyleSheet(" .QFrame { background-color : white;border: 0.5px solid black } ")
        self.setFixedWidth(180)
        self.__controls()
        self.__layout()
        self.__connectEvents()
        

    def __controls(self):
        ""
        self.bigFrame = QFrame(self)
        
        self.label = createTitleLabel(self.descriptionText,fontSize=12)
        
        self.openItemSelectDialog =  LabelLikeButton(
                        parent = self.bigFrame,
                        text= "Select ..."
                        )

        

        self.resetButton = ResetButton(buttonSize=(15,15),parent=self.bigFrame, tooltipStr = "Reset selection")
        self.addButton = BigPlusButton(buttonSize=(15,15),parent=self.bigFrame,strokeWidth=1.5)

        
    
    def __layout(self):
        ""
        self.bigFrame.setLayout(QGridLayout())
        self.bigFrame.layout().setContentsMargins(5,5,5,5)
        self.bigFrame.layout().addWidget(self.label,0,0)
        self.bigFrame.layout().addWidget(self.openItemSelectDialog,1,0)
        self.bigFrame.layout().addWidget(self.addButton,1,1)
        self.bigFrame.layout().addWidget(self.resetButton,1,2)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(2,2,2,2)
        self.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout().addWidget(self.bigFrame)

    def __connectEvents(self):
        ""
        self.openItemSelectDialog.clicked.connect(self.openSelection)
        self.addButton.clicked.connect(self.showMenu)

    def openSelection(self):
        ""

        dlg = ICDSelectItems(data = pd.DataFrame(self.selectableItems))
                
        # handle position and geomettry
        senderGeom = self.sender().geometry()
        bottomRight = self.mapToGlobal(senderGeom.bottomRight())
        h = dlg.getApparentHeight()
        dlg.setGeometry(bottomRight.x() + 15,bottomRight.y()-int(h/2),185,h)
        if dlg.exec():
            selectedColumns = dlg.getSelection()
            self.openItemSelectDialog.setToolTip("\n".join([str(x) for x in selectedColumns.values.flatten()]))
            self.openItemSelectDialog.setText("{} columns selected".format(selectedColumns.size))
            self.currentSelection = selectedColumns

    def showMenu(self,event=None):
        ""
        if self.reportBackSelection is not None:
            if len(self.currentSelection) > 0:
                sender = self.sender()
                if hasattr(sender,"mouseLostFocus"):
                    sender.mouseLostFocus()

                menu = createMenu(parent=self)
                action = menu.addAction("Use for all")
                action.triggered.connect(lambda _,columnName = "Use for all": self.reportBackSelection(columnName,
                                                                                                self.descriptionText,
                                                                                                self.currentSelection)) 
                menu.addSeparator()
                for tColumn in self.targetColumns:
                    action = menu.addAction(tColumn)
                    action.triggered.connect(lambda _,columnName = tColumn: self.reportBackSelection(columnName,
                                                                                                self.descriptionText,
                                                                                                self.currentSelection)) 

                senderGeom = self.sender().geometry()
                bottomLeft = self.bigFrame.mapToGlobal(senderGeom.bottomLeft())
                menu.exec(bottomLeft)