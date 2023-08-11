
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


from pynndescent import NNDescent, PyNNDescentTransformer #try to remove error got with pyinstaller.

from ui.mainFrames.ICDataHandleFrame import DataHandleFrame
from ui.mainFrames.ICPlotOptionsFrame import PlotOptionFrame
from ui.mainFrames.ICSliceMarksFrame import SliceMarksFrame
from ui.custom.warnMessage import AskStringMessage
from ui.dialogs.Selections.ICDSelectItems import ICDSelectItems
from ui.dialogs.ICWorkfowBuilder import ICWorkflowBuilder
from ui.utils import removeFileExtension, areFilesSuitableToLoad, isWindows, standardFontSize, getHashedUrl, createMenu
from ui.mainFrames.ICFigureReceiverBoxFrame import MatplotlibFigure
from ui.custom.warnMessage import AskQuestionMessage, WarningMessage
from ui.dialogs.ICAppValidation import ICValidateEmail

from backend.utils.worker import Worker
from backend.data.data import DataCollection
from backend.update.Update import UpdateChecker
from backend.data.ICGrouping import ICGrouping
from backend.utils.funcControl import funcPropControl
from backend.utils.misc import getTxtFilesFromDir
from backend.utils.stringOperations import getRandomString
from backend.utils.Logger import ICLogger
from backend.config.config import Config
from backend.saver.ICSessionHandler import ICSessionHandler
from backend.webapp.ICAppValidator import ICAppValidator
from backend.plotting.plotterCalculations import PlotterBrain

import sys, os
import numpy as np
import pandas as pd
import time
from datetime import datetime
import webbrowser
import requests
import warnings
import multiprocessing
import importlib
import numpy as np
import copy



#ignore some warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

__VERSION__ = "v0.12.2"

filePath = os.path.dirname(sys.argv[0])
exampleDir = os.path.join(filePath,"examples")
exampleFuncs = []



for fileName in getTxtFilesFromDir(exampleDir):
    addExample =  {
        "subM":"Load Examples",
        "name":"name",
        "fn" : {"obj":"self","fn":"sendRequestToThread",
                "kwargs":{"funcProps":{"key":"data::addDataFrameFromTxtFile",
                          "kwargs":{"pathToFile":os.path.join(filePath,"examples","name.txt"),"fileName":"name.txt"}}}
        }}
    addExample["name"] = removeFileExtension(fileName)
    addExample["fn"]["kwargs"]["funcProps"]["kwargs"]["pathToFile"] = os.path.join(exampleDir,fileName)
    addExample["fn"]["kwargs"]["funcProps"]["kwargs"]["fileName"] = removeFileExtension(fileName)
    exampleFuncs.append(addExample)

menuBarItems = [

    {
        "subM":"Help",
        "name":"Discussions (New Features)",
        "fn": lambda : webbrowser.open("https://github.com/hnolCol/instantclue/discussions")
    },
    {
        "subM":"Help",
        "name":"Tutorial/Wiki",
        "fn": lambda : webbrowser.open("https://github.com/hnolCol/instantclue/wiki")
    },
    {
        "subM":"Help",
        "name":"YouTube Videos",
        "fn": lambda : webbrowser.open("https://www.youtube.com/channel/UCjSfodDjhCMY2bw9_i6VOXA")
    },
    {
        "subM":"Help",
        "name":"Bug Report",
        "fn": lambda : webbrowser.open("https://github.com/hnolCol/instantclue/issues")
    },
    {
        "subM":"About",
        "name":"GitHub",
        "fn": lambda : webbrowser.open("https://github.com/hnolCol/instantclue")
    },
    {
        "subM":"About",
        "name":"Citation",
        "fn": lambda : webbrowser.open("https://www.nature.com/articles/s41598-018-31154-6/")
    },
    # {
    #     "subM":"About",
    #     "name":"Cite us (extend.)",
    #     "fn": lambda : webbrowser.open("https://www.nature.com/articles/s41598-018-31154-6/")
    # },
    {
        "subM":"About",
        "name":"{}".format(__VERSION__),
        "fn": lambda : pd.DataFrame([__VERSION__]).to_clipboard(index=False,header=False)
    },
    {
        "subM":"File",
        "name":"Load file(s)",
        "fn": {"obj":"self","fn":"askForFile","objName":"mainFrames","objKey":"data"}
    },
    {
        "subM":"File",
        "name":"Load dataset from MitoCube",
        "fn": {"obj":"self","fn":"fetchDataFromMitoCube","objName":"mainFrames","objKey":"data"}      
    },
    {
        "subM":"File",
        "name":"Load session",
        "fn": {"obj":"self","fn":"loadSession","objName":"mainFrames","objKey":"data"}
    },
    {
        "subM":"File",
        "name":"Save session",
        "fn": {"obj":"self","fn":"saveSession","objName":"mainFrames","objKey":"data"}
    },
    {
        "subM":"File",
        "name":"Load Examples",
        "fn": {"obj":"self","fn":"_createSubMenu"}
    },
    #{
    #    "subM":"Log",
    #    "name":"Save log",
    #    "fn": {"obj":"self","fn":"loadSession","objName":"mainFrames","objKey":"data"}
    #},
    # {
    #     "subM":"Share",
    #     "name":"Validate App",
    #     "fn": {"obj":"self","fn":"openAppValidationDialog"}
    # },
    # {
    #     "subM":"Share",
    #     "name":"Add app id",
    #     "fn": {"obj":"webAppComm","fn":"copyAppIDToClipboard"}
    # },
    # {
    #     "subM":"Share",
    #     "name":"Copy App ID",
    #     "fn": {"obj":"webAppComm","fn":"copyAppIDToClipboard"}
    # },
    # {
    #     "subM":"Share",
    #     "name":"Manage Graphs",
    #     "fn": {"obj":"webAppComm","fn":"copyAppIDToClipboard"}
    # },
    # {
    #     "subM":"Share",
    #     "name":"Retrieve Data",
    #     "fn": {"obj":"self","fn":"getChartData"}
    # },
    # {
    #     "subM":"Share",
    #     "name":"Display shared charts",
    #     "fn": {"obj":"webAppComm","fn":"displaySharedCharts"}
    # },
    {
        "subM":"Windows",
        "name":"Main Window",
        "fn": {"obj":"self","fn":"showWindow"}
    },
    # {
    #     "subM":"Workflow",
    #     "name":"Build Workflow",
    #     "fn": {"obj":"self","fn":"startBuildWorkflowDialog"}
    # }
    
] + exampleFuncs 


class InstantClue(QMainWindow):
    
    #define signals to clear interactive TableViews
    resetGroupColorTable = pyqtSignal()
    resetGroupSizeTable = pyqtSignal()
    resetQuickSelectTable = pyqtSignal()
    resetLabelTable = pyqtSignal()
    resetTooltipTable = pyqtSignal()
    resetStatisticTable = pyqtSignal() 
    resetMarkerTable = pyqtSignal()
    quickSelectTrigger = pyqtSignal()

    def __init__(self, parent=None):
        super(InstantClue, self).__init__(parent)

        self.mainPath = os.path.dirname(sys.argv[0])
        self.setWindowIcon(self.getWindowIcon())
        
        self.config = Config(mainController = self)
        self.version = __VERSION__
        self._setupFontStyle()
        #set up data collection
        self._setupData()
        #setup filter center
        self._setupFilters()
        #setup statustuc center
        self._setupStatistics()
        #setup normalizer
        self._setupNormalizer()
        self._setupTransformer()
        #plotter brain (calculates props for plos)
        self.plotterBrain = PlotterBrain(sourceData = self.data)
        #split widget
        self.splitterWidget = MainWindowSplitter(self)
       
        #set up logger 
        self.logger = ICLogger(self.config,__VERSION__)
        #setup web app communication
        self.webAppComm = ICAppValidator(self)
        self.updateChecker = UpdateChecker(__VERSION__)

        _widget = QWidget()
        _layout = QVBoxLayout(_widget)
        _layout.setContentsMargins(1,3,3,3)
        _layout.addWidget(self.splitterWidget)

        self.setCentralWidget(_widget)
        self._setupStyle()
    
        self._getMainFrames()
        self._setPlotter()
        self._addMenu()

        self.threadpool = QThreadPool()
        
        self.mainFrames["sliceMarks"].threadWidget.setMaxThreadNumber(self.threadpool.maxThreadCount())
 
        self._connectSignals()
        
        self.quickSelectTrigger.connect(self.mainFrames["data"].qS.updateDataSelection)
        self.setAcceptDrops(True)
        self.acceptDrop = False
        #update parameters saved in parents (e.g data, plotter etc)
        self.config.updateAllParamsInParent()
        #check for update
        
        self.sendRequestToThread({"key":"update::checkForUpdate","kwargs":{}})
        ##### 
        #self.webAppComm.isAppIDValidated()
        #self.webAppComm.getChartData()
        #self.webAppComm.getChartsByAppID()
        #self.validateApp()
        #print(self.config.getParentTypes())

    def _connectSignals(self):
        "Connects signals using the resetting of the tables defined in the sliceMarks frame."
        self.resetGroupColorTable.connect(self.mainFrames["sliceMarks"].colorTable.reset)
        self.resetGroupSizeTable.connect(self.mainFrames["sliceMarks"].sizeTable.reset)
        self.resetLabelTable.connect(self.mainFrames["sliceMarks"].labelTable.reset)
        self.resetTooltipTable.connect(self.mainFrames["sliceMarks"].tooltipTable.reset)
        self.resetStatisticTable.connect(self.mainFrames["sliceMarks"].statisticTable.reset)
        self.resetMarkerTable.connect(self.mainFrames["sliceMarks"].markerTable.reset)
        self.resetQuickSelectTable.connect(self.mainFrames["sliceMarks"].quickSelectTable.removeFromGraph)

    def _setupData(self):
        ""
        self.data = DataCollection(parent=self)
        self.grouping = ICGrouping(self.data)
        self.colorManager = self.data.colorManager
        self.sessionManager = ICSessionHandler(mainController = self)

    def _setupFontStyle(self):
        ""
        
        self.config.setParamRange("label.font.family",QFontDatabase().families())
        from ui import utils
        utils.standardFontSize = self.config.getParam("label.font.size")
        fontFamily = self.config.getParam("label.font.family") 
        if fontFamily in self.config.getParamRange("label.font.family"):
            utils.standardFontFamily = fontFamily
        else:
            #default to arial
            self.config.setParam("label.font.family","Arial") 
            utils.standardFontFamily = "Arial"

    def _setupFilters(self):
        ""
        if not hasattr(self,"data"):
            raise ValueError("No data object found.")

        self.categoricalFilter = self.data.categoricalFilter
        self.numericFilter = self.data.numericFilter
    
    def _setupNormalizer(self):

        self.normalizer = self.data.normalizer

    def _setupStatistics(self):

        self.statCenter = self.data.statCenter

    def _setupTransformer(self):

        self.transformer = self.data.transformer

    def _getMainFrames(self):

        self.mainFrames = self.splitterWidget.getMainFrames()
    
    def _setPlotter(self):

        self.data.setPlotter(self.mainFrames["middle"].ICPlotter)

    def _addMenu(self):
        "Main window menu."
        self.subMenus = {}
        subMenus = ["File","Settings","Help","About","Windows"] #"Workflow","Share""Log",
        for subM in subMenus:
            self.subMenus[subM] = QMenu(subM,self)
            self.menuBar().addMenu(self.subMenus[subM])

        for menuProps in menuBarItems:
            if "fn" in menuProps and isinstance(menuProps["fn"],dict) and "fn" in menuProps["fn"] and menuProps["fn"]["fn"]== "_createSubMenu":
                self._createSubMenu(menuProps["name"],menuProps["subM"])
                
            else:
                subMenu = self.subMenus[menuProps["subM"]]
                action = subMenu.addAction(menuProps["name"])
                if "fn" in menuProps:
                    if isinstance(menuProps["fn"],dict):

                        fn = self._getObjFunc(menuProps["fn"])
                        if "kwargs" in menuProps["fn"]:
                            action.triggered.connect(lambda bool, 
                                            fn = fn, 
                                            kwargs = menuProps["fn"]["kwargs"] : fn(**kwargs))
                        else:                        
                            action.triggered.connect(fn)
                    else:
                        action.triggered.connect(menuProps["fn"])
        
        #add Setting Headers
        subMenu = self.subMenus["Settings"]
        propItems = sorted(self.config.getParentTypes())
        for propItemName in propItems:
            action = subMenu.addAction(propItemName)
            action.triggered.connect(lambda _,propItem = propItemName:self.mainFrames["right"].openConfig(specificSettingsTab = propItem))

    def _createSubMenu(self,subMenuName,parentMenu):
        "Add a sub menu to a parent menu."
        if parentMenu in self.subMenus:
            parentMenu = self.subMenus[parentMenu]
            self.subMenus[subMenuName] = QMenu(subMenuName,parentMenu)
            parentMenu.addMenu(self.subMenus[subMenuName])

    def getWindowIcon(self):
        ""
        pathToIcon = os.path.join(self.mainPath,"icons","instantClueLogo.png")
        if os.path.exists(pathToIcon):
            return QIcon(pathToIcon)
        return QIcon()

    def getDataByDataID(self,dataID, columnNames = None, useClipping = False):
        ""
        if columnNames is None:
            columnNames = self.data.getPlainColumnNames(dataID)
        dataFrame = self.data.getDataByColumnNames(dataID,columnNames, ignore_clipping = not useClipping)["fnKwargs"]["data"]
        return dataFrame

    def progress_fn(self, progressText):
        ""
        

        
        
 
    def print_output(self, s):
        ""
        

    def errorInThread(self, errorType = None, v = None, e = None):
        "Error message if something went wrong in the calculation."
        "TO DO: Improve error handling."
        self.sendMessageRequest({"title":"Error ..","message":"There was an unknwon error."})


    def getChartData(self,*args,**kwargs):
        ""

        dlg = AskStringMessage(q="Please provide graphID from which you would like to retrieve the data.")
        if dlg.exec():
            fkey = "webApp::getChartData"
            kwargs = {"graphID":dlg.state}
            self.sendRequestToThread({"key":fkey,"kwargs":kwargs})

    def getFigureSize(self):
        ""
        canvasSizeDict = {}
        canvasSize = self.mainFrames["middle"].canvas.sizeHint()
        canvasSizeDict["width"] = canvasSize.width()
        canvasSizeDict["height"] = canvasSize.height()
        return canvasSizeDict



    def _getObjFunc(self,fnProps):
        ""
        if fnProps["obj"] == "self":
            if "objKey" in fnProps and "objName" in fnProps:
                subObj = getattr(self,fnProps["objName"])[fnProps["objKey"]]
                fn = getattr(subObj,fnProps["fn"])
            elif "objName" in fnProps:
                #objKey not in fnProps
                fn = getattr(getattr(self,fnProps["objName"]),fnProps["fn"])
            else:
                fn = getattr(self,fnProps["fn"])
        else:
            classObj = getattr(self,fnProps["obj"])
            fn = getattr(classObj,fnProps["fn"])
        return fn

    def _threadComplete(self,resultDict):
        ""
        #check if thread returns a dict or none
        if resultDict is None:
            return
        #get function to call afer thread completed calculations
        fnsComplete = funcPropControl[resultDict["funcKey"]]["completedRequest"]
        if "data" not in resultDict:
            print("data not found in result dict")
            return
        if resultDict["data"] is None:
            print("data in result dict is none..")
            data = {}
        else:
            data = copy.deepcopy(resultDict["data"]) #deep copy - thread safe? 
        #iteratre over functions. This is a list of dicts
        #containing function name and object name
        for fnComplete in fnsComplete:
            #get function from func dict
            fn = self._getObjFunc(fnComplete)
            #init kwargs : result dict should containg all kwargs for a function
            #careful with naming, since duplicates will be overwritten.
            if any(kw not in data for kw in fnComplete["requiredKwargs"]):
                continue
            kwargs = {}
            for kw in fnComplete["requiredKwargs"]:
                kwargs[kw] = data[kw]
            if "optionalKwargs" in fnComplete:
                for kw in fnComplete["optionalKwargs"]:
                    if kw in data:
                        kwargs[kw] = data[kw]
            try:
                #finnaly execute the function
                fn(**kwargs)
            except Exception as e:
                print(fn)
                print(fnComplete)
                print(e)
        
    
    def _threadFinished(self,threadID, messageText):
        "Indicate in the ui that a thread finished."
        self.mainFrames["sliceMarks"].threadWidget.threadFinished(threadID,messageText)
       # self.mainFrames["sliceMarks"].threadWidget
    

    def getTable(self,tableName):
        ""
        if hasattr(self.mainFrames["sliceMarks"],tableName):
            return getattr(self.mainFrames["sliceMarks"],tableName)

    def openSettings(self, specificSettingsTab : str | None):
        ""
        self.mainFrames["right"].openConfig(specificSettingsTab = specificSettingsTab)

    def openAppValidationDialog(self,*args,**kwargs):
        ""
        if self.webAppComm.isAppIDValidated():
            dlg = ICValidateEmail(self)
            dlg.exec()
        else:
            self.sendToWarningDialog(infoText="Web AppID is already validated.")
    
    def askForExcelFileName(self,title="Save Data to Excel",defaultName="ExcelExport.xlsx",*args,**kwargs):
        ""
        workingDir = self.config.getParam("WorkingDirectory")
        fileName,_ = QFileDialog.getSaveFileName(self,title,os.path.join(workingDir,defaultName),"Excel Files (*.xlsx)")
        return fileName
    
    def askForItemSelection(self, items ,title = "Categorical column selection.",**kwargs):
        "Opens a Dialog to the user and asks for some item select."
        dataFrame = pd.DataFrame(items)
        if not dataFrame.empty:
            dlg = ICDSelectItems(data = dataFrame, title = title, **kwargs)
            if dlg.exec():
               
           
                selectedItems = dlg.getSelection()
               
                if selectedItems.size > 0: #check
                    return selectedItems

    def askForGroupingSelection(self, funcKey, numericColumnsInKwargs = True, title = "Groupings to display in h. clustering.", kwargName = "groupingName", deleteFromFuncKeyIfDialogClosed : bool = False, **kwargs):
        """
        
        """
        if numericColumnsInKwargs:
            groupings = self.grouping.getGroupingsByColumnNames(columnNames=funcKey["kwargs"]["numericColumns"])
        else:
            groupings = self.grouping.groups.copy()
        if len(groupings) > 0:
            dlg = ICDSelectItems(data = pd.DataFrame(list(groupings.keys())), title = title, **kwargs)
            if dlg.exec():
                selectedGrupings = dlg.getSelection().values.flatten()
                if selectedGrupings.size > 0: #check
                    funcKey["kwargs"][kwargName] = selectedGrupings
            else:
                if kwargName in funcKey["kwargs"] and deleteFromFuncKeyIfDialogClosed:
                   del funcKey["kwargs"][kwargName]
        return funcKey

    def sendRequest(self,funcProps):
        ""
        try:
            if "key" not in funcProps:
                return
            else:
                funcKey = funcProps["key"]

            if  funcKey in funcPropControl:

                fnRequest = funcPropControl[funcKey]["threadRequest"]
                fn = self._getObjFunc(fnRequest)

                if all(reqKwarg in funcProps["kwargs"] for reqKwarg in fnRequest["requiredKwargs"]):
                    data = fn(**funcProps["kwargs"])
                    if data is not None:
                        self._threadComplete({"funcKey":funcKey,"data":data })
                else:
                    print("not all kwargs found.")
                    print(fnRequest["requiredKwargs"])
                    print(funcProps["kwargs"])
        except Exception as e:
            print(e)
    

    def dragMoveEvent(self, e):
        "Ignore/acccept drag Move Event"
        if self.acceptDrop:
            e.accept()
        else:
            e.ignore()
    
    def dragEnterEvent(self,event):
        "check if drag event has urls (e.g. files)"
        #check if drop event has Urls
        if event.mimeData().hasUrls():
            event.accept()
            self.acceptDrop = True
        else:
            event.ignore()
            self.acceptDrop = False
    
    def dropEvent(self,event):
        "Allows drop of files from os"
        #find Urls of dragged objects
        droppedFiles = [url.path() for url in event.mimeData().urls() if url.isValid()]
        #check if file ends with proper fileExtension
        checkedDroppedFiles = areFilesSuitableToLoad(droppedFiles)
        if len(checkedDroppedFiles) > 0:
            event.accept()
            self.mainFrames["data"].addTxtFiles(checkedDroppedFiles)
            self.mainFrames["data"].addExcelFiles(checkedDroppedFiles)


    def sendRequestToThread(self, funcProps = None, **kwargs):
        # Pass the function to execute
        try:
            if "key" not in funcProps:
                return
            else:
                funcKey = funcProps["key"]

            if  funcKey in funcPropControl:

                fnRequest = funcPropControl[funcKey]["threadRequest"]
                fn = self._getObjFunc(fnRequest)

                #print(fnRequest["requiredKwargs"])

                if all(reqKwarg in funcProps["kwargs"] for reqKwarg in fnRequest["requiredKwargs"]):
                    # Any other kwargs are passed to the run function
                    threadID = getRandomString()

                    worker = Worker(fn = fn, funcKey = funcKey, ID = threadID,**funcProps["kwargs"]) 
                    worker.signals.result.connect(self._threadComplete)
                    worker.signals.finished.connect(self._threadFinished)
                    worker.signals.progress.connect(self.progress_fn)
                    worker.signals.error.connect(self.errorInThread)
                    self.threadpool.start(worker)
                    self.mainFrames["sliceMarks"].threadWidget.addActiveThread(threadID, funcKey)
                   # self.logger.add(funcKey,funcProps["kwargs"])
                    #Count.setText(str(self.threadpool.activeThreadCount()))
                
                else:
                    print("not all required kwargs found...")

        except Exception as e:
            print(e)

    def createSettingMenu(self,settingKeyWord,menu=None):
        ""
        if menu is None:
            menu = createMenu()
        action = menu.addAction("Settings")
        action.triggered.connect(lambda _,keyWord = settingKeyWord:self.mainFrames["right"].openConfig(specificSettingsTab=keyWord))
        return menu 

    def closeEvent(self, event = None, *args, **kwargs):
        """Overwrite close event"""
       
        msgText = "Are you sure you want to exit Instant Clue? Please confirm?"
        
        w = AskQuestionMessage(
            parent=self,
            infoText = msgText, 
            title="Question",
            iconDir = self.mainPath,
            yesCallback = lambda e = event: self.saveParameterAndClose(e))
        w.exec()
        if w.state is None:
            event.ignore()

    def saveParameterAndClose(self, event):
        ""
        self.config.saveParameters()
        event.accept()

    def startBuildWorkflowDialog(self,event=None):
        ""
        dlg = ICWorkflowBuilder(self)
        dlg.exec()

    def getUserLoginInfo(self):
        "Visionary ..."
        try:
            URL = "http://127.0.0.1:5000/api/v1/projects"
            r = requests.get(URL)
            return True, r.json()
        except:
            return False, []

    def sendTextEntryToWebApp(self, projectID = 1, title = "", text = "Hi", isMarkDown = True):
        "Visionary ..."
        URL = "http://127.0.0.1:5000/api/v1/projects/entries"
        jsonData = {
                "ID"            :   projectID,
                "title"         :   title,
                "text"          :   text,
                "isMarkDown"    :   isMarkDown,
                "time"          :   time.time(),
                "timeFrmt"      :   datetime.now().strftime("%d-%m-%Y :: %H:%M:%S")
                }
        id = getRandomString()
        
        r = requests.post(URL, json = jsonData)

        if r.ok:
            self.sendMessageRequest({"title":"Done","message":"Text entry transfered to WebApp. "})


    def getDataID(self):
        ""
        return self.mainFrames["data"].getDataID()

    def getGraph(self):
        "Returns the graph object from the figure mainFrame (middle)."
        graph = None
        exists = hasattr(self.mainFrames["middle"].ICPlotter,"graph")
        if exists:
            graph = self.mainFrames["middle"].ICPlotter.graph
        return exists, graph

    def getTreeView(self,dataHeader = "Numeric Floats"):
        "Returns the tree view for a specific data type"
        return self.mainFrames["data"].getTreeView(dataHeader)

    def getPlotType(self):
        "Returns the the current plot type as a string"
        return self.mainFrames["right"].getCurrentPlotType()

    def groupingActive(self):
        "Returns bools, indicating if grouping is active."
        return self.getTreeView().table.isGroupigActive()

    def isDataLoaded(self):
        "Checks if there is any data loaded."
        return len(self.mainFrames["data"].dataTreeView.dfs) > 0
    
    def getQuickSelect(self):
        ""
        return self.mainFrames["data"].qS

    def getLiveGraph(self):
        ""
        return self.mainFrames["data"].liveGraph

    def showMessageForNewVersion(self,releaseURL):
        ""
        w = AskQuestionMessage(
            parent=self,
            infoText = "A new version of Instant Clue is available. Download now?", 
            title="Information",
            iconDir = self.mainPath,
            yesCallback = lambda : webbrowser.open(releaseURL))
        w.exec()


    def setBufToClipboardImage(self,buf):
        ""
        QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
        buf.close()

    def sendMessageRequest(self,messageProps = dict()):
        "Display message on user screen in the top right corner"
        # check if all keys present
        if all(x in messageProps for x in ["title","message"]): 
            if self.config.getParam("errorShownInDialog") and "Error" in messageProps["title"]:
                self.sendToWarningDialog(infoText=messageProps["message"])
            # else:
            #     self.notification.setNotify(
            #         messageProps["title"],
            #         messageProps["message"])

    def sendToWarningDialog(self,infoText="",textIsSelectable=False,*args,**kwargs):
        ""
        w = WarningMessage(title="Warning", infoText=infoText,iconDir=self.mainPath, textIsSelectable = textIsSelectable, *args,**kwargs)
        w.exec()

    def sendToInformationDialog(self,infoText="",textIsSelectable=False,*args,**kwargs):
        ""
        w = WarningMessage(title="Information", infoText=infoText,iconDir=self.mainPath, textIsSelectable = textIsSelectable,*args,**kwargs)
        w.exec()

    def addWindowMenu(self,actionName,actionFn,fnKwargs):
        ""
        action = self.subMenus["Windows"].addAction(actionName)
        action.triggered.connect(lambda _,kwargs= fnKwargs: actionFn(**kwargs))
        return action 

    def deleteWindowAction(self,action):
        "Delete an aciton"
        self.subMenus["Windows"].removeAction(action)
        
    def showWindow(self,*args,**kwargs):
        ""
        if hasattr(self,"raise_"):
            self.raise_()

    def _setupStyle(self):
        "Style setup of the graphical user interface."
        
        self.setWindowTitle("Instant Clue")
        #self.setStyleSheet("QToolTip{ background-color: white ; color: black;font: 12pt;font-family: Arial;margin: 3px 3px 3px 3px;border: 5px}")
        self.setStyleSheet("""
                QToolTip {
                    font-family: Arial;
                    line-height: 1.75;
                    padding: 2px;
                    border: 0.5px;
                }

                QScrollBar:horizontal {
                    border: none;
                    background: none;
                    height: 11px;
                    margin: 0px 11px 0 11px;
                }

                QScrollBar::handle:horizontal {
                    background: darkgrey;
                    min-width: 11px;
                   
                }
                QScrollBar::handle:horizontal:hover {
                    background: #286FA4;
                }

                QScrollBar::add-line:horizontal {
                    background: none;
                    width: 11px;
                    subcontrol-position: right;
                    subcontrol-origin: margin;
                    
                }

                QScrollBar::sub-line:horizontal {
                    background: none;
                    width: 11px;
                    subcontrol-position: top left;
                    subcontrol-origin: margin;
                    position: absolute;
                }

                QScrollBar:left-arrow:horizontal{
                    width: 11px;
                    height: 11px;
                    background: #7c7c7b;
                    
                }

                QScrollBar:right-arrow:horizontal {
                    width: 11px;
                    height: 11px;
                    background: #7c7c7b;
                    
                }
                

                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                    background: none;
                }

                /* VERTICAL */
                QScrollBar:vertical {
                    border: none;
                    background: none;
                    width: 11px;
                    margin: 11px 0 11px 0;
                }

                QScrollBar::handle:vertical {
                    background: darkgrey;
                    min-height: 11px;
                    border-radius: 1px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #286FA4;
                }

                QScrollBar::add-line:vertical {
                    background: none;
                    height: 11px;
                    subcontrol-position: bottom;
                    subcontrol-origin: margin;
                }

                QScrollBar::sub-line:vertical {
                    background: none;
                    height: 11px;
                    subcontrol-position: top left;
                    subcontrol-origin: margin;
                    position: absolute;
                }

                QScrollBar:up-arrow:vertical {
                    width: 11px;
                    height: 11px;
                    background: #7c7c7b;
                    
                }

                QScrollBar:down-arrow:vertical {
                    width: 11px;
                    height: 11px;
                    background: #7c7c7b;
                    
                }

                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                }

                """)

class MainWindowSplitter(QWidget):
    "Main Window Splitter to separate ui in different frames."
    def __init__(self, *args,**kwargs):
        super(MainWindowSplitter, self).__init__(*args,**kwargs)

        self.mC = self.parent()
        self.__controls()
        self.__layout()


    def __controls(self):
        "Creates widgets"
        
        #self.ICwelcome = ICWelcomeScreen(parent=self,version=__VERSION__)
        self.mainSplitter = QSplitter(parent=self)
        
        mainWindowWidth = self.parent().frameGeometry().width()
        sizeCalculation = []
        self.mainFrames = dict()
        mainFrameProps = [("data",0.25),("sliceMarks",0.1),("middle",0.55),("right",0.1)]
        for layoutId, sizeFrac in mainFrameProps:
            if layoutId == "data":
                w = DataHandleFrame(self, mainController = self.mC)
            elif layoutId == "sliceMarks":
                w = SliceMarksFrame(self, mainController = self.mC)
            elif layoutId == "middle":
                w = MatplotlibFigure(self, mainController= self.mC)
            elif layoutId == "right":
                w = PlotOptionFrame(self, mainController= self.mC)
            self.mainFrames[layoutId] = w 
            self.mainSplitter.addWidget(w)
            sizeCalculation.append(int(mainWindowWidth * sizeFrac*1000)) #hack, do not know why it is not working properly with small numbers
        #make splitter expand
        self.mainSplitter.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.mainSplitter.setSizes(sizeCalculation)

    def __layout(self):
        "Adds widgets to layout"
        vbox = QVBoxLayout(self)
       # vbox.addWidget(self.ICwelcome)#self.mainSplitter)
        self.setLayout(vbox)
        self.layout().addWidget(self.mainSplitter)

    def getMainFrames(self):
        ""
        return self.mainFrames

    def welcomeScreenDone(self):
        "Indicate layout changes once Welcome Screen finished."
      # self.layout().removeWidget(self.ICwelcome)
       # self.layout().addWidget(self.mainSplitter)
       # self.ICwelcome.deleteLater()


def main():
    "Start the main window."
    if '_PYIBoot_SPLASH' in os.environ and importlib.util.find_spec("pyi_splash") and isWindows():
        import pyi_splash
        if pyi_splash.is_alive():
            pyi_splash.update_text('UI Loaded ...')
            pyi_splash.close()
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    #QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True)
    app = QApplication(sys.argv)
    app.setStyle("Windows") # set Fusion Style
    iconPath = os.path.join("..","icons","base","32.png")
    if os.path.exists(iconPath):
        app.setWindowIcon(QIcon(iconPath))
    win = InstantClue() # Inherits QMainWindow
    screenGeom = app.primaryScreen().geometry()
    win.setGeometry(10,10,screenGeom.width()-100,screenGeom.height()-140)
    #win.showMaximized()
    win.show()    
    win.raise_()
    app.exec()

if __name__ == '__main__':
    multiprocessing.freeze_support()
   # multiprocessing.set_start_method('spawn')
    sys.exit(main())
