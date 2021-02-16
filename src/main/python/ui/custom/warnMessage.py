
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
import os 
from ..utils import createTitleLabel


class MessageBase(QDialog):
    ""
    def __init__(self,parent=None,iconName = "warnIcon.svg", title = "Warning", infoText = "", iconDir = ".", *args,**kwargs):
        super(MessageBase,self).__init__(parent,*args,**kwargs)
        
        self.iconName = iconName
        self.title = title
        self.infoText = infoText
        self.iconDir = iconDir
        self.state = None
        self.setMinimumWidth(350)
        self.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)

        self.__controls()
        self.__layout()

    def __controls(self):

        self.mainFrame = QFrame()
        self.mainFrame.setStyleSheet("background-color: #f6f6f6}")
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
            self.logoLabel.setText("NoIconFound")
        self.logoLabel.setAlignment(Qt.AlignVCenter)

        #setup info label
        self.infoLabel = QLabel(self.infoText)
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
        
        self.__controls()
        self.__layout()  

    def __controls(self):
        ""
        
        #setup ok button
        self.okButton = QPushButton("Ok")
        self.okButton.clicked.connect(self.acknowledge)       

    def __layout(self):
        ""
        
        mainFLayout = self.mainFrame.layout()
        mainFLayout.addWidget(self.okButton,3,3)
  


class AskQuestionMessage(MessageBase):

    def __init__(self,parent=None,*args,**kwargs):
        ""
        super(AskQuestionMessage,self).__init__(parent,*args,**kwargs)
        
        self.__controls()
        self.__layout()  

    def __controls(self):
        ""
        
        #setup ok button
        self.yesButton = QPushButton("Yes")
        self.yesButton.clicked.connect(self.changeState)  

        self.noButton = QPushButton("No")    
        self.noButton.clicked.connect(lambda _: self.resetState(close=True))

    def __layout(self):
        ""
        
        mainFLayout = self.mainFrame.layout()
        
        mainFLayout.addWidget(self.yesButton,3,2)
        mainFLayout.addWidget(self.noButton,3,3)
  
    def changeState(self,event = None, newState = True):
        ""
        self.setState(newState)
        self.close()
