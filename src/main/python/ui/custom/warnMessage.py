
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
import os

from matplotlib.pyplot import text 
from ..utils import createTitleLabel, createLabel, createLineEdit
from .buttonDesigns import ICStandardButton

class MessageBase(QDialog):
    ""
    def __init__(self,parent=None,iconName = "warnIcon.svg", title = "Warning", infoText = "", iconDir = ".", textIsSelectable=False, *args,**kwargs):
        super(MessageBase,self).__init__(parent,*args,**kwargs)
        
        self.iconName = iconName
        self.title = title
        self.infoText = infoText
        self.iconDir = iconDir
        self.state = None
        self.textIsSelectable = textIsSelectable
        self.setMinimumWidth(360)
        self.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)

        self.setWindowTitle("Message")
        self.__controls()
        self.__layout()

    def __controls(self):

        self.mainFrame = QFrame()
        #set up title label
        self.tLabel = createTitleLabel(self.title)
        self.tLabel.setAlignment(Qt.AlignHCenter)
        #set up logo label
        self.logoLabel = QLabel() 
         #find instant clue logo
        pathToSVG = os.path.join(self.iconDir,"icons","base",self.iconName)
        if os.path.exists(pathToSVG):
            pixmap = QPixmap(pathToSVG)
            self.logoLabel.setPixmap(pixmap)
        else:
            self.logoLabel.setText("...")
        self.logoLabel.setAlignment(Qt.AlignVCenter)

        #setup info label
        self.infoLabel = createLabel(self.infoText)
        if self.textIsSelectable:
            self.infoLabel.setTextInteractionFlags(Qt.TextSelectableByMouse) 
        self.infoLabel.setWordWrap(True)
        self.infoLabel.setAlignment(Qt.AlignRight  | Qt.AlignVCenter)
    
    def __layout(self):
        ""
        self.mainFrame.setLayout(QGridLayout())
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0,0,0,0)
        self.layout().addWidget(self.mainFrame)
        mainFLayout = self.mainFrame.layout()
        mainFLayout.addWidget(self.logoLabel,1,1,2,1)
        
        mainFLayout.addWidget(self.tLabel,1,2,1,2)
        mainFLayout.addWidget(self.infoLabel,2,2,1,2)
        
       # mainFLayout.addWidget(self.okButton,3,3)
        mainFLayout.setColumnStretch(1,0)
        mainFLayout.setColumnStretch(2,1)
        mainFLayout.setColumnStretch(3,0)
        mainFLayout.setRowStretch(1,0)
        mainFLayout.setRowStretch(2,1)
        mainFLayout.setRowStretch(3,0)


    def acknowledge(self):
        ""
        self.close()

    def setState(self,newState):
        ""
        self.state = newState

    def resetState(self, close=True):
        ""
        self.state = None
        if close:
            self.close()


class WarningMessage(MessageBase):

    def __init__(self,parent=None,*args,**kwargs):
        ""
        super(WarningMessage,self).__init__(parent,*args,**kwargs)
        self.setWindowTitle("Warning message!")
        self.__controls()
        self.__layout()  

    def __controls(self):
        ""
        
        #setup ok button
        self.okButton = ICStandardButton("Ok")
        self.okButton.clicked.connect(self.acknowledge)       

    def __layout(self):
        ""
        
        mainFLayout = self.mainFrame.layout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.okButton)

        mainFLayout.addLayout(hbox,3,0,1,3,Qt.AlignRight)
  
class AskStringMessage(MessageBase):

    def __init__(self,q ="",defaultText="",parent=None,passwordMode = False, *args,**kwargs):

        super(AskStringMessage,self).__init__(parent, infoText = q, title = "Question",*args,**kwargs)
       
        self.text = defaultText
        self.pwMode = passwordMode
        self.setWindowTitle("Ask for string")
        self.__controls()
        self.__layout()  
        self.__connectEvents()
    
    def __controls(self):
        ""
        
        self.lineEdit = createLineEdit()
        if self.pwMode:
            self.lineEdit.setEchoMode(QLineEdit.Password)
        self.okButton = ICStandardButton("Ok")

    def __layout(self):
        ""
        hbox = QVBoxLayout()
        
        hbox.addWidget(self.lineEdit)
        hbox.addWidget(self.okButton,Qt.AlignRight)
        mainFLayout = self.mainFrame.layout()
        mainFLayout.addLayout(hbox,3,1,1,2,Qt.AlignRight)

    def __connectEvents(self):
        ""
        self.okButton.clicked.connect(self.accept)
        self.lineEdit.textChanged.connect(self.lineEditting)

    def lineEditting(self, lineEditText):
        ""
        self.text = lineEditText
        self.state = lineEditText

class AskForFile(MessageBase):
    ""

    def __init__(self,placeHolderEdit = "Select file.",parent=None,*args,**kwargs):
        super(AskForFile,self).__init__(parent,title="Select file.",*args,**kwargs)
        
        self.placeHolderEdit = placeHolderEdit
        
        self.__controls()
        self.__layout()

    def __controls(self):
        ""

        self.fileLineEdit = createLineEdit(self.placeHolderEdit)
        self.selectButton = ICStandardButton("...")
        self.selectButton.clicked.connect(self.openFileDialog)
        self.yesButton = ICStandardButton("Okay")
        self.yesButton.clicked.connect(self.changeState)  

        self.noButton = ICStandardButton("Cancel")    
        self.noButton.clicked.connect(self.close)
    
    def __layout(self):
        ""

        mainFLayout = self.mainFrame.layout()
        hboxFile = QHBoxLayout()
        hboxFile.addWidget(self.fileLineEdit)
        hboxFile.addWidget(self.selectButton)
        hbox = QHBoxLayout()
        hbox.addWidget(self.yesButton)
        hbox.addWidget(self.noButton)
        mainFLayout.addLayout(hboxFile,3,0,1,3,Qt.AlignLeft)
        mainFLayout.addLayout(hbox,4,0,1,3,Qt.AlignRight)


    def changeState(self,event=None):
        ""
        currentPath = self.fileLineEdit.text()
        if not os.path.exists(currentPath):
            w = WarningMessage(infoText = "Path does not exist. Changed after Selection?")
            w.exec_()
        else:
            self.setState(currentPath)
            self.accept() 

    def openFileDialog(self):
        ""
        filePath, _ = QFileDialog.getOpenFileName(self,
                                "QFileDialog.getOpenFileName()", 
                                "",
                                "Fasta Files (*.fasta)")
        if filePath:
            self.fileLineEdit.setText(filePath)


class AskQuestionMessage(MessageBase):

    def __init__(self,yesCallback = None,parent=None,*args,**kwargs):
        ""
        super(AskQuestionMessage,self).__init__(parent,*args,**kwargs)
        
        self.yesCallback = yesCallback
        self.__controls()
        self.__layout()  

    def __controls(self):
        ""
        
        #setup ok button
        self.yesButton = ICStandardButton("Yes")
        self.yesButton.clicked.connect(self.changeState)  

        self.noButton = ICStandardButton("No")    
        self.noButton.clicked.connect(lambda _: self.resetState(close=True))

    def __layout(self):
        ""
        
        mainFLayout = self.mainFrame.layout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.yesButton)
        hbox.addWidget(self.noButton)
        mainFLayout.addLayout(hbox,3,0,1,3,Qt.AlignRight)
      
  
    def changeState(self,event = None, newState = True):
        ""
        if self.yesCallback is not None:
            self.setState(newState)
            self.yesCallback() 
            self.close() 
        else:
            self.setState(newState)
            self.close()
