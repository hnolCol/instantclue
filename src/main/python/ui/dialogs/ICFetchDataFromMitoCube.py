from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from ..custom.buttonDesigns import ICStandardButton
from ..custom.warnMessage import WarningMessage

from ..utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR, createCombobox
import requests 
import os 
import pandas as pd 
import numpy as np 

class ICFetchDataFromMitoCube(QDialog):
    ""
    def __init__(self, mainController, title = "Fetch data from MitoCube.", *args,**kwargs):
        ""
        super(ICFetchDataFromMitoCube, self).__init__(*args,**kwargs)
        self.title = title
        self.mC = mainController

        self.APIUrlConfig = self.mC.config.getParam("mitocube.api.url")
        self.shareTokenConfig = self.mC.config.getParam("mitocube.share.token")
        self.__controls()
        self.__layout()
        self.__connectEvents()

    def __controls(self) -> None:
        ""
        self.title = createTitleLabel(self.title)
        self.infoText = createLabel("Connect to the MitoCube API and fetch data.\nYou require a share-token and a password as well as the dataID.")
        self.dataID = createLineEdit("dataID","Provide a valid dataID.")
        self.APIUrl = createLineEdit("API-URL","Provide the API Url.")
        self.websitePassword = createLineEdit("Password","Provide the website password.")
        self.websitePassword.setEchoMode(QLineEdit.Password)
        self.shareToken = createLineEdit("Share Token","Enter the share token provided by the MitoCube admin.")
        if self.APIUrlConfig is not None:
            self.APIUrl.setText(self.APIUrlConfig)
        if self.shareTokenConfig is not None:
            self.shareToken.setText(self.shareTokenConfig)
        self.infoTextFromAPI = createLabel("")
        self.fetchDataButton = ICStandardButton(itemName="Fetch & Load")
        self.closeButton = ICStandardButton(itemName="Close")

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        l = self.layout()
        l.addWidget(self.title)
        l.addWidget(self.infoText)
        l.addWidget(self.dataID)
        l.addWidget(self.APIUrl)
        l.addWidget(self.websitePassword)
        l.addWidget(self.infoTextFromAPI)
        hbox = QHBoxLayout()
        hbox.addWidget(self.fetchDataButton)
        hbox.addWidget(self.closeButton)
        l.addLayout(hbox)
        
    def __connectEvents(self):
        ""
        self.closeButton.clicked.connect(self.close)
        self.fetchDataButton.clicked.connect(self.fetchData)
        

    def closeEvent(self,event=None):
        "Overwrite close event"
        event.accept()


    def fetchData(self): 
        ""
        self.infoTextFromAPI.setText("Loading...")
        self.infoTextFromAPI.repaint()
        QApplication.processEvents()
        if any(x.text() == "" for x in [self.dataID,self.APIUrl,self.websitePassword]):
            self.mC.sendToWarningDialog(infoText="Missing information for dataID or URL.")
            return
        url = f"{self.APIUrl.text()}/api/dataset/instantclue"
        pw = self.websitePassword.text()
        dataID = self.dataID.text()

        try:
            r = requests.post(url=url,json={"dataID":dataID,"pw":pw})
        except ConnectionError:
            self.mC.sendToWarningDialog(infoText="Connection error raised. Please try again later and make sure that you have entered the correct url.")
            return
        except Exception:
            self.mC.sendToWarningDialog(infoText="Request returned an unexpecteed Exception.")
            return
        if r.status_code == 200:
            responseData = r.json()
            if "success" in responseData and not responseData["success"]:
               pass
            else:
                if "data" in responseData and "params" in responseData:
                    params = responseData["params"]
                    df = pd.DataFrame().from_dict(responseData["data"]).replace("NaN",np.nan)
                    if "groupings" in params:
                        for groupingName, grouping in params["groupings"].items():
                            grouping = dict([(groupName,pd.Series(groupItems)) for groupName, groupItems in grouping.items()])
                            self.mC.grouping.addGrouping(groupingName,grouping )
                    funcProps = {
                        "key" : "data::addDataFrame",
                        "kwargs" : {"dataFrame" : df, "fileName":f"{dataID} - MC"}
                    }
                    self.mC.sendRequestToThread(funcProps)

            if "msg" in responseData:
                    self.infoTextFromAPI.setText(responseData["msg"])
            else:
                    self.infoTextFromAPI.setText("Unknown msg response..")
        else:
            self.mC.sendToWarningDialog(infoText=f"API returned an error. Status code: {r.status_code}")

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