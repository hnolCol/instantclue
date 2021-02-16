from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * #works for pyqt5
import sys
import datetime

#.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)

class Message(QFrame):
    def __init__(self, title, message, timeoutDuration = 3, parent=None):
        QFrame.__init__(self, parent)
        self.parent = parent
        self.timeoutDuration = timeoutDuration

        self.effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.effect)
        self.animation = QPropertyAnimation(self.effect, b"opacity")
        
        self.setLayout(QVBoxLayout())

        self.titleLabel = QLabel(title, self)
        self.titleLabel.setStyleSheet(
            "font-family: Arial; font-size: 12px; font-weight: bold; padding: 0; color:#1E5A8F")

        self.messageLabel = QLabel(message, self)
        self.messageLabel.setStyleSheet(
            "font-family: Arial; font-size: 11px; font-weight: normal; padding-left: 3;")

        self.setStyleSheet("""background:#F2F2F2""")
        
        self.messageLabel.setFixedWidth(250)
        self.messageLabel.setWordWrap(True)

        self.layout().setContentsMargins(5,4,5,4)
        self.layout().setSpacing(5)
        self.layout().addWidget(self.titleLabel)
        self.layout().addWidget(self.messageLabel)
        self.layout().addStretch(1)
        self.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)

        self.startTimer()

    def startTimer(self):
        ""
        self.displayTimer = QTimer(self)
        self.displayTimer.timeout.connect(self.timerTimeout)
        self.displayTimer.start(1000)

    def timerTimeout(self):
        self.timeoutDuration -= 1
        if self.timeoutDuration == 0:
            self.displayTimer.stop()
            self.animation.setDuration(1000)
            self.animation.setStartValue(1)
            self.animation.setEndValue(0)
            self.animation.start()
            self.animation.finished.connect(lambda : self.parent.clearMessage(self))



class Notification(QWidget):
    signNotifyClose = pyqtSignal(str)
    def __init__(self, parent = None, padding = {"right":50,"top":50}):

        super(QWidget, self).__init__(parent)

        self.padding = padding
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        #get sreenWidth
        resolution = QDesktopWidget().screenGeometry(-1)
        self.screenWidth = resolution.width()

        self.nMessages = 0
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setContentsMargins(2,0,2,0)
        self.mainLayout.setSpacing(5)
 
        
    def setLocation(self,sizeHint):
        ""
        self.move(
                    self.screenWidth-sizeHint.width()-self.padding["right"],
                    self.padding["top"]
                )

    def setNotify(self, title, message):
        ""
        
        m = Message(title, message, parent = self)
        self.mainLayout.addWidget(m)
        self.setLocation(m.sizeHint())
        self.nMessages += 1
        self.adjustSize()
        self.show()

    def onClicked(self):
        ""
        messageWidget = self.sender().parent()
        self.clearMessage(messageWidget)

    def clearMessage(self,message):
        ""
        self.mainLayout.removeWidget(message)
        message.deleteLater()
        self.nMessages -= 1
        self.adjustSize()
        if self.nMessages == 0:
            self.close()
