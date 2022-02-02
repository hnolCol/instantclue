import time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..utils import getExtraLightFont
import requests

GITHUB_URL = "https://api.github.com/repos/hnolcol/instantclue/releases"

counterText = {
    2 : "instant.",
    4 : "instant. clue",
    6 : "instant. clue."
}


class ICWelcomeScreen(QWidget):

    def __init__(self,version,*args,**kwargs):
        super(ICWelcomeScreen,self).__init__(*args,**kwargs)

        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        
        self.counter = 0
        self.versionCheckedDone = False
        
        self.version = version
        self.__controls()
        self.__layout()
        self.__startTimer()
    
        
        
    def __controls(self):

        #set label
        self.label = QLabel("instant")
        self.label.setAlignment(Qt.AlignLeft)
        self.label.setFont(getExtraLightFont(fontSize=25, font="Courier New"))

        self.versionLabel = QLabel("v. {}".format(self.version))

    def __layout(self):

        self.setLayout(QVBoxLayout())
        #self.layout().setAlignment(Qt.AlignCenter)
        self.layout().addStretch(1)
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.versionLabel)
        self.layout().addStretch(1)
        self.layout().setContentsMargins(200,0,0,0)
    
    def __startTimer(self):
        ""
        self.timer = QTimer()
        self.timer.timeout.connect(self.changeAppearance)
        self.timer.setInterval(350)
        self.timer.start()

    def checkVersion(self):
        "Check for a new version using the GitHub release api"
        self.versionLabel.setText("v. {} .. checking for new version".format(self.version))
        try:
            response = requests.get(GITHUB_URL, timeout=2)
        except (requests.ConnectionError, requests.Timeout) as exception:
            self.versionLabel.setText("v. {} .. error - connection to GitHub failed.".format(self.version))
            self.versionCheckedDone = True
            return
        if response.status_code == 200:
            try:
                data = response.json()
                if "tag_name" in data[0] and "html_url" in data[0]:
                    tagName = data[0]["tag_name"]
                    releaseURL = data[0]["html_url"]
                    self.versionLabel.setText("{} .. new version found".format(self.version))
                    if tagName != self.version:
                        self.versionCheckedDone = True
                        self.parent().showMessageForNewVersion(releaseURL)
            except:
                self.versionCheckedDone = True
        else:
            self.versionLabel.setText("v. {} .. error while connection to GitHub.".format(self.version))
            self.versionCheckedDone = True

    def changeAppearance(self,event=None):
        "Changes the visual text to the user based on a simple counter."
        if self.counter == 0:
            self.checkVersion()
        if self.counter in counterText:
            self.label.setText(counterText[self.counter])
        if self.counter >= 8 and self.versionCheckedDone:
            self.parent().welcomeScreenDone()
        self.counter += 1