import json
import time 
from collections import OrderedDict

class ICLogger(object):

    def __init__(self, config, version):
        ""
        self.loggerPath = config.getParam("logger.path")
        self.version = version
        self.createFile()


    def createFile(self):
        "" 
        generalInformation = OrderedDict([("Software","InstantClue"),("Version",self.version),("Licence","GPL 3.0")])
        logHeader = {"General Information" : generalInformation}
        #print(json.dumps(logHeader,indent=4))
        with open('my_log.txt', 'w') as f:
            f.write(json.dumps(logHeader,indent=4))


    def add(self,fnKey,kwargs):
        ""
        with open('my_log.txt', 'a') as f:
            
            f.write("\n"+json.dumps({fnKey:kwargs}, indent=4))