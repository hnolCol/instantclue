from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from ..utils import createLineEdit


from .buttonDesigns import ResetButton
from .ICReceiverBox import BoxItem
from ..utils import getRandomString

import numpy as np 

class Tag(QWidget):
    ""
    def __init__(self, tagID, tagName,backgroundColor = "#f6f6f6",resetCallback = None,*args,**kwargs):
        super(Tag,self).__init__(*args,**kwargs)
        self.tagName = tagName
        self.backgroundColor = backgroundColor
        self.resetCallBack = resetCallback
        self.tagID = tagID
        self.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)

        self.__control()
        self.__initStyle()
        self.__layout()

    def __initStyle(self):
        ""
        self.outerFrame.setStyleSheet('background-color: {}; border: 0.5px solid black'.format(self.backgroundColor))  

    def __control(self):
        ""
        self.outerFrame = QFrame(self)
        self.boxitem = BoxItem(itemName=self.tagName,
                                parent=self.outerFrame, 
                                drawFrame= False)
        self.boxitem.setToolTip(None)
        self.deleteMe = ResetButton(parent=self.outerFrame, 
                                    buttonSize=(10,10), 
                                    strokeWidth=1.5, 
                                    drawFrame=False)
        if self.resetCallBack is not None:
            
            self.deleteMe.clicked.connect(lambda _, tagID = self.tagID : self.resetCallBack(tagID))

    def __layout(self):
        ""
        self.outerFrame.setLayout(QHBoxLayout())
        self.outerFrame.layout().addWidget(self.boxitem)
        self.outerFrame.layout().addWidget(self.deleteMe)
        self.outerFrame.layout().setSpacing(0)
        self.outerFrame.layout().setContentsMargins(0,0,0,0)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.outerFrame)


class ICSearchWithTags(QWidget):
    ""
    def __init__(self,onTextChanged,onEnterEvent,onTagDelete,*args,**kwargs):
        ""
        super(ICSearchWithTags, self).__init__(*args,**kwargs)
        self.onTextChanged = onTextChanged
        self.onEnterEvent = onEnterEvent
        self.onTagDelete = onTagDelete
        self.tags = dict()
        self.__control() 
        self.__layout() 
        self.__connectEvents()

    def __control(self):
        ""
        self.lineEdit = createLineEdit("Search..")

        validator = QRegularExpressionValidator(QRegularExpression("[+-]?([0-9]*[.])?[0-9]+")) #match any floating number
        self.lineEditMin = createLineEdit("min")
        self.lineEditMax = createLineEdit("max")
        self.lineEditMin.setValidator(validator)
        self.lineEditMax.setValidator(validator)

        self.cb = QCheckBox("Exact match")
        self.cb.setTristate(False)


    def __layout(self):
        ""
        self.tagLayout = QHBoxLayout()
        self.tagLayout.setSpacing(1)
        self.tagLayout.setContentsMargins(0,0,0,0)
        self.tagLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setLayout(QVBoxLayout())
        hbox = QHBoxLayout() 
        hbox.addWidget(self.lineEdit)
        hbox.addWidget(self.lineEditMin)
        hbox.addWidget(self.lineEditMax)
        hbox.addWidget(self.cb)
        self.layout().addLayout(hbox)
        self.layout().addLayout(self.tagLayout)
        self.layout().setContentsMargins(0,0,0,0)
        self.layout().setSpacing(0)
    
    def __connectEvents(self):
        self.lineEdit.textChanged.connect(self.onTextChanged)
        self.lineEdit.returnPressed.connect(self.handleEnterEvent)
        self.lineEditMin.returnPressed.connect(self.handleEnterEvent)
        self.lineEditMax.returnPressed.connect(self.handleEnterEvent)

    def handleEnterEvent(self):
        ""
        if self.lineEdit.isVisible():
            self.onEnterEvent(exactMatch = self.cb.isChecked())
        else:
            if self.lineEditMin.text() in ["","."]:
                minValue = -np.inf
            else:
                minValue = float(self.lineEditMin.text())
            if self.lineEditMax.text() in ["","."]:
                maxValue = np.inf
            else:
                maxValue = float(self.lineEditMax.text())

            self.onEnterEvent(minValue = minValue,maxValue=maxValue,filterType="numeric")
            

    def setNumericFilter(self):
        ""
        self.hideLineEdit()
        self.showNumericFilter()

    def hideLineEdit(self):
        ""
        self.lineEdit.hide() 
        self.cb.hide()
    
    def showLineEdit(self):
        ""
        self.lineEdit.show() 
        self.cb.show()
    
    def showNumericFilter(self, columnName = " "):
        ""
        self.lineEditMax.show()
        self.lineEditMax.setPlaceholderText("max ({})".format(columnName))
        self.lineEditMin.show()
        self.lineEditMin.setPlaceholderText("min ({})".format(columnName))
    
    def hideNumericFilter(self):
        ""
        self.lineEditMax.hide()
        self.lineEditMin.hide()

    def setPlaceHolderText(self,placeHolderText):
        ""
        if isinstance(placeHolderText,str):
            self.lineEdit.setPlaceholderText(placeHolderText)

    def resetLineEditText(self):
        ""
        self.lineEdit.setText("")

    def addTag(self,tagName,backgroundColor,toolTipStr = ""):
        ""
        tagID = getRandomString(5)
        tagWidget = Tag(tagID,tagName,backgroundColor=backgroundColor,resetCallback=self.onTagDelete)
        self.tagLayout.addWidget(tagWidget)
        self.tags[tagID] = tagWidget
        self.setToolTip("{}\nID: {}\n{}".format(tagName,tagID,toolTipStr))
        return tagID
    
    def removeTag(self,tagID):
        ""
        self.tags[tagID].deleteLater() 

    def setFocusToLineEdit(self):
        ""
        self.lineEdit.setFocus()
    
    def setFocusToMin(self):
        ""
        self.lineEditMin.setFocus() 
    
    def setFocusToMax(self):
        ""
        self.lineEditMax.setFocus()         

