
from ..utils.stringOperations import getRandomString
import pandas as pd
import os  
import requests 
import json
import base64
import base64
import json 

from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP


class ICAppValidator(object):
    ""
    def __init__(self, mainController):

        self.mC = mainController

        self.validateApp()


    def checkAppIDForCollision(self,b64EncAppID):
        "Checks in InstantClue Webapp API if id exists alread. Since we create random strings as an id, a collision is possible and should be avoided. The chances are however vereeery looow :) "
        URL = "http://127.0.0.1:5000/api/v1/app/id/exists"
        
        r = requests.get(URL,params={"app-id":b64EncAppID})
        if r.status_code == 200:
            validID = json.loads(r.json())["valid"] == "True"
            return validID, False
        else:
            return False,True

    def copyAppIDToClipboard(self):
        ""
        appID = self.getAppID()
        if appID is not None:
            pd.DataFrame([appID]).to_clipboard(index=False,header=False, excel=False)
            self.mC.sendMessageRequest({"title":"Copied","message":"Encrypted App ID has been copied."})


    def encryptStringWithPublicKey(self, byteToEntrypt):
        ""
        publickKeyPath = os.path.join(self.mC.mainPath,"conf","key","receiver.pem")
        if os.path.exists(publickKeyPath):
            privateKeyString = RSA.import_key(open(publickKeyPath).read())
            encryptor = PKCS1_OAEP.new(privateKeyString)
            encrypted = encryptor.encrypt(byteToEntrypt)
            b64EncStr = base64.b64encode(encrypted).decode("utf-8")
            return b64EncStr 

    def validateApp(self):
        ""
        appIDPath, validPath = self.appIDFound()
       
        if not validPath:
            appIDValid = False
            while not appIDValid:
                #create app ID
                appID = getRandomString(20).encode("utf-8")
                #encrypt appid for sending
                b64EncAppID = self.encryptStringWithPublicKey(appID)
                #check for collision
                appIDValid, httpRequestFailed = self.checkAppIDForCollision(b64EncAppID)
                if httpRequestFailed:
                    self.mC.sendMessageRequest({"title":"Error..","message":"HTTP Request failed. App could not be validated."})
                    return
            self.saveAppID(appIDPath,b64EncAppID)

    def appIDFound(self):
        ""
        appIDPath = self.getAppIDPath()

        return appIDPath, os.path.exists(appIDPath)

    def saveAppID(self,appIDPath, b64EncStr):
        ""
        with open(appIDPath,"w") as f:
            f.write(b64EncStr)

    def getAppIDPath(self):
        "Returnst the file path in which the app-id is stored."
        return os.path.join(self.mC.mainPath,"conf","key","app_id")


    def getAppID(self):
        "Returns the app-id. The app-id is stored in a file."
        appIDPath,idFileFound = self.appIDFound()
        if idFileFound:
            with open(appIDPath,"r") as f:
                b64EncStr = f.read()
                return b64EncStr
        
    def getChartTitles(self):
        try:
            df = self.getChartsByAppID()
            if not df.empty:
                return df["title"].values.tolist()
            else:
                return []
        except:
            return []

    def getChartTextInfo(self, chartDetails, isProtected, password = None):
        ""
        appID = self.getAppID()
        shortUrl = chartDetails.loc["short-url"] 
        if appID is not None:
            URL = "http://127.0.0.1:5000/api/v1/graph/text"
            if isProtected:
                encryptPwd = self.encryptStringWithPublicKey(password)
                r = requests.post(URL,json={"url":str(shortUrl),"pwd":encryptPwd})
            else:
                r = requests.get(URL,params={"url":shortUrl})
            rJson = r.json()
            if "success" in rJson and rJson["success"]:
                print(rJson["success"])
                print(type(rJson["success"]))
            elif "success" in rJson and not rJson["success"]:
                self.mC.sendToWarningDialog(infoText="The password was not correct.")
                
            else:
                self.mC.sendToWarningDialog(infoText="There was an error connecting to the web app and retrieving the data.")


    def isChartProtected(self,graphURL):
        ""
        URL = "http://127.0.0.1:5000/api/v1/graph/protected"
        r = requests.get(URL,params={"url":graphURL})
        if r.status_code == 200:
            graphIsProtected = not r.json()["protected"] == "false"
            return graphIsProtected

    def getChartsByAppID(self):
        "Returns the charts that were uplaoded and are associated to the app-id"
        appID = self.getAppID() 
        if appID is not None:
            URL = "http://127.0.0.1:5000/api/v1/app/graphs"
            r = requests.post(URL,json={"app-id":appID})
            df = pd.read_json(r.json(),orient="records")
            
            return df
        else:
            return  pd.DataFrame()

    def displaySharedCharts(self):
        ""
        df = self.getChartsByAppID()
        if not df.empty:
            self.mC.mainFrames["data"].openDataFrameinDialog(df,ignoreChanges=True, 
                                                            headerLabel="Count plot data.", 
                                                            tableKwargs={"forwardSelectionToGraph":False})