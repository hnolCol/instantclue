from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..utils import getExtraLightFont

counterText = {
    2 : "instant.",
    4 : "instant. clue",
    6 : "instant. clue."
}


class ICWelcomeScreen(QWidget):

    def __init__(self,*args,**kwargs):
        super(ICWelcomeScreen,self).__init__(*args,**kwargs)

        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        
        self.counter = 0
        
        
        self.__controls()
        self.__layout()
        self.__startTimer()
        
        
    def __controls(self):

        #set label
        self.label = QLabel("instant")
        self.label.setAlignment(Qt.AlignLeft)
        self.label.setFont(getExtraLightFont(fontSize=25, font="Courier New"))

    def __layout(self):

        self.setLayout(QVBoxLayout())
        #self.layout().setAlignment(Qt.AlignCenter)
        self.layout().addStretch(1)
        self.layout().addWidget(self.label)
        self.layout().addStretch(1)
        self.layout().setContentsMargins(200,0,0,0)
    
    def __startTimer(self):
        ""
        self.timer = QTimer()
        self.timer.timeout.connect(self.changeAppearance)
        self.timer.setInterval(400)
        self.timer.start()

    def changeAppearance(self,event=None):
        ""
       
        if self.counter in counterText:
            self.label.setText(counterText[self.counter])
        if self.counter == 8:
            self.parent().welcomeScreenDone()
        self.counter += 1