from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 
from ..custom.Widgets.ICButtonDesgins import ICStandardButton
from ..custom.warnMessage import WarningMessage

from ..utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR, createCombobox
import requests 
import os 

class ICValidateEmail(QDialog):
    ""
    def __init__(self, mainController, title = "Validate App by Email.", *args,**kwargs):
        ""
        super(ICValidateEmail, self).__init__(*args,**kwargs)
        self.title = title
        self.mC = mainController
        self.__controls()
        self.__layout()
        self.__connectEvents()

    def __controls(self) -> None:
        ""
        self.title = createTitleLabel(self.title)
        
        self.infoText = createLabel("Note: Your email will be saved decrypted on the server.\nA verification code will be send to the email.")

        self.emailEdit = createLineEdit("Verification Email","Provide a valid email adress. Your email adress will not be stored. All shared graphs will be stored under the generated application ID.")
        self.sendCodeButton = ICStandardButton(itemName="Send code")
        self.verificationCodeEdit = createLineEdit("Verification Code","Enter verification code received by email.")
        self.verificationCodeEdit.setEnabled(False)

        
        self.logoLabel = QLabel() 
         #find instant clue logo
        pathToSVG = os.path.join(self.mC.mainPath,"icons","base","ICMainLogo.svg")
        if os.path.exists(pathToSVG):
            pixmap = QPixmap(pathToSVG)
            self.logoLabel.setPixmap(pixmap)
    

        self.statusLabel = createLabel("Enter email.")

        self.applyButton = ICStandardButton("Validate")
        self.applyButton.setEnabled(False) #only gets enables once email send button was pressed and 10 chars were entered
        self.closeButton = ICStandardButton("Close")

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.title)
        self.layout().addWidget(self.infoText)
        hboxLogoGrid = QHBoxLayout()

        grid = QGridLayout()

        grid.addWidget(self.emailEdit,0,0)
        grid.addWidget(self.sendCodeButton,0,1)
        grid.addWidget(self.statusLabel,1,0,1,2)
        grid.addWidget(self.verificationCodeEdit,2,0,1,2)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.applyButton)
        hbox.addWidget(self.closeButton)

        hboxLogoGrid.addWidget(self.logoLabel)
        hboxLogoGrid.addLayout(grid)
        self.layout().addLayout(hboxLogoGrid)
        self.layout().addLayout(hbox)

        grid.setColumnStretch(0,1)
        grid.setRowStretch(7,2)

    def __connectEvents(self):
        ""
        self.closeButton.clicked.connect(self.close)
        self.sendCodeButton.clicked.connect(self.sendEmail)
        self.verificationCodeEdit.textChanged.connect(self.onVerificationCodeTextChange)
        

    def closeEvent(self,event=None):
        "Overwrite close event"
        event.accept()

    def sendEmail(self):
        ""
        currentText = self.emailEdit.text()
        if len(currentText) > 3 and "@" in currentText:
            URL = "http://127.0.0.1:5000/api/v1/app/validate"
            email = self.mC.webAppComm.encryptStringWithPublicKey(currentText.encode('utf-8'))
            appID = self.mC.webAppComm.getAppID()
            r = requests.put(URL,json={"app-id":appID,"email":email})
            print(email,appID)
            self.statusLabel.setText("Request sent. Please check email.")
            if r.status_code == 200:
                self.verificationCodeEdit.setEnabled(True)
            else:
                self.statusLabel.setText("Error returned by webapp api. Internet?")
        else:
            self.statusLabel.setText("Please enter valid email.")

    def onVerificationCodeTextChange(self,newText,*args,**kwargs):
        "Handles changes in the verification code line edit."
        if len(newText) != 10:
            self.applyButton.setEnabled(False)
            
        else:
            self.applyButton.setEnabled(True)
            URL = "http://127.0.0.1:5000/api/v1/app/validate"
            appID = self.mC.getAppID()
            r = requests.post(URL,json={"app-id":appID,"verification":newText})