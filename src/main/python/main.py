
# ---------------------------------------------------------------------------
# Item 55 (report A0): error and crash logging to text files in the
# installation folder.
#
# The shipped build runs without a console (console=False), so an error that
# escapes a Qt slot, event handler or model method used to terminate the
# program instantly with no message anywhere (PyQt6 calls qFatal() on an
# unhandled exception). This block is deliberately placed BEFORE every other
# import and uses only the standard library, so that even a failing import at
# startup is recorded.
#
#   InstantClue_error_log.txt  - Python errors, with full traceback. After such
#                                an error the program keeps running and shows a
#                                dialog naming this file.
#   InstantClue_crash_log.txt  - hard crashes inside compiled code (e.g. an
#                                access violation in Qt/numpy), which no Python
#                                handler can catch; written by faulthandler.
#                                Removed again on a normal exit if nothing
#                                was written to it.
#
# Both go to the installation folder (the folder containing InstantClue.exe,
# or containing main.py when run from source). If that folder is not
# writable (e.g. under Program Files) they fall back to
# %APPDATA%\InstantClue and then to the system temp folder.
# ---------------------------------------------------------------------------
__VERSION__ = "v0.13.0"

import sys, os, traceback, faulthandler, threading, atexit
from datetime import datetime as _icDatetime

IC_ERROR_LOG_FILENAME = "InstantClue_error_log.txt"
IC_CRASH_LOG_FILENAME = "InstantClue_crash_log.txt"

# console=False leaves sys.stdout/sys.stderr as None. print() tolerates that,
# but any library calling sys.stderr.write(...) directly raises AttributeError.
for _icStreamName in ("stdout", "stderr"):
    if getattr(sys, _icStreamName, None) is None:
        try:
            setattr(sys, _icStreamName, open(os.devnull, "w", encoding="utf-8"))
        except Exception:
            pass

def icGetInstallDir():
    "Folder containing InstantClue.exe (frozen build) or main.py (source run)."
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

def _icCandidateLogDirs():
    candidateDirs = [icGetInstallDir()]
    appData = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
    if appData:
        candidateDirs.append(os.path.join(appData, "InstantClue"))
    try:
        import tempfile as _icTempfile
        candidateDirs.append(os.path.join(_icTempfile.gettempdir(), "InstantClue"))
    except Exception:
        pass
    return candidateDirs

def icGetErrorLogPath(fileName = IC_ERROR_LOG_FILENAME):
    "First writable location for the log file, or None."
    for candidateDir in _icCandidateLogDirs():
        try:
            os.makedirs(candidateDir, exist_ok=True)
            logPath = os.path.join(candidateDir, fileName)
            with open(logPath, "a", encoding="utf-8"):
                pass
            return logPath
        except Exception:
            continue
    return None

def icWriteErrorLog(details, heading = None):
    "Append a timestamped entry to InstantClue_error_log.txt. Returns the path or None."
    logPath = icGetErrorLogPath()
    if logPath is None:
        return None
    try:
        with open(logPath, "a", encoding="utf-8") as logFile:
            logFile.write("\n" + "=" * 72 + "\n")
            logFile.write("{} - InstantClue {}\n".format(
                _icDatetime.now().strftime("%Y-%m-%d %H:%M:%S"), globals().get("__VERSION__", "")))
            if heading:
                logFile.write(heading + "\n")
            logFile.write("=" * 72 + "\n")
            logFile.write(details if isinstance(details, str) else str(details))
            logFile.write("\n")
        return logPath
    except Exception:
        return None

_icShownErrorSummaries = set()
_icErrorDialogOpen = [False]

def _icShowErrorDialog(summary, logPath):
    """Show one dialog per distinct error message and session.

    Deferred with QTimer.singleShot so it never opens inside a paint or key
    handler, and de-duplicated so an error repeating on every repaint cannot
    produce an endless series of dialogs (every occurrence is still logged).
    """
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        from PyQt6.QtCore import QTimer
    except Exception:
        return
    if QApplication.instance() is None:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    if summary in _icShownErrorSummaries or _icErrorDialogOpen[0]:
        return
    _icShownErrorSummaries.add(summary)

    def _show():
        _icErrorDialogOpen[0] = True
        try:
            text = "An unexpected error occurred. InstantClue is still running, " \
                   "but the last action may not have completed.\n\n{}".format(summary)
            if logPath is not None:
                text += "\n\nFull details were written to:\n{}".format(logPath)
            QMessageBox.critical(None, "InstantClue - Error", text)
        except Exception:
            pass
        finally:
            _icErrorDialogOpen[0] = False
    try:
        QTimer.singleShot(0, _show)
    except Exception:
        pass

def _icExceptHook(excType, excValue, excTraceback):
    "Replaces the default handler: log the error instead of letting Qt abort."
    if excType is not None and issubclass(excType, KeyboardInterrupt):
        sys.__excepthook__(excType, excValue, excTraceback)
        return
    try:
        if getattr(excValue, "_icAlreadyLogged", False):
            return
        details = "".join(traceback.format_exception(excType, excValue, excTraceback))
        logPath = icWriteErrorLog(details, heading = "Unhandled error")
        summary = "".join(traceback.format_exception_only(excType, excValue)).strip()
        _icShowErrorDialog(summary, logPath)
    except Exception:
        pass

sys.excepthook = _icExceptHook
threading.excepthook = lambda args: _icExceptHook(args.exc_type, args.exc_value, args.exc_traceback)

_icCrashLogFile = None

def _icEnableCrashLog():
    """Route native crash dumps to InstantClue_crash_log.txt.

    A session header is written at startup; on a normal exit the file is cut
    back to its previous length (and deleted if it was empty), so the file
    only accumulates entries for sessions that actually crashed.
    """
    global _icCrashLogFile
    for candidateDir in _icCandidateLogDirs():
        crashPath = os.path.join(candidateDir, IC_CRASH_LOG_FILENAME)
        try:
            os.makedirs(candidateDir, exist_ok=True)
            startSize = os.path.getsize(crashPath) if os.path.exists(crashPath) else 0
            crashFile = open(crashPath, "a", encoding="utf-8")
            crashFile.write("\n" + "=" * 72 + "\n")
            crashFile.write("{} - InstantClue {} - session started; a crash dump "
                            "follows only if the program crashed\n".format(
                _icDatetime.now().strftime("%Y-%m-%d %H:%M:%S"), globals().get("__VERSION__", "")))
            crashFile.write("=" * 72 + "\n")
            crashFile.flush()
            faulthandler.enable(file=crashFile, all_threads=True)
            _icCrashLogFile = crashFile

            def _icCleanupCrashLog(crashPath = crashPath, startSize = startSize):
                try:
                    faulthandler.disable()
                    _icCrashLogFile.close()
                except Exception:
                    pass
                try:
                    if startSize == 0:
                        os.remove(crashPath)
                    else:
                        with open(crashPath, "r+", encoding="utf-8") as f:
                            f.truncate(startSize)
                except Exception:
                    pass
            atexit.register(_icCleanupCrashLog)
            return
        except Exception:
            continue

# only in the main process: multiprocessing children (spawn) import this file
# as "__mp_main__" and must not rewrite or truncate the parent's crash log
if __name__ == "__main__":
    _icEnableCrashLog()

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *


    from pynndescent import NNDescent, PyNNDescentTransformer #try to remove error got with pyinstaller.

    from ui.notifications.messageWindow import Notification
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
    from backend.data.ICGrouping import ICGrouping
    from backend.utils.funcControl import funcPropControl
    from backend.utils.misc import getTxtFilesFromDir
    from backend.utils.stringOperations import getRandomString
    from backend.utils.Logger import ICLogger
    from backend.config.config import Config
    from backend.saver.ICSessionHandler import ICSessionHandler
    from backend.webapp.ICAppValidator import ICAppValidator
    from backend.plotting.plotterCalculations import PlotterBrain
    from backend.update.Update import UpdateChecker

    import sys, os
    import tempfile
    import numpy as np
    import pandas as pd
    import time
    from datetime import datetime
    import webbrowser
    import requests
    import warnings
    import multiprocessing
    import importlib
except BaseException as _icStartupError:
    #a failing import at startup (e.g. a module missing from the frozen build)
    #happens before any window exists - record it, then let it propagate
    try:
        icWriteErrorLog(traceback.format_exc(), heading = "Error while starting InstantClue (import failed)")
        _icStartupError._icAlreadyLogged = True
    except Exception:
        pass
    raise


os.environ["OUTDATED_IGNORE"] = "1"
#ignore some warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

#__VERSION__ is defined at the top of this file (item 55), so the error/crash logs can use it

# In a normal dev run, sys.argv[0] is main.py, sitting right next to examples/.
# In a PyInstaller ONEFILE frozen build, sys.argv[0] is the .exe path, and
# examples/ is not extracted next to it - PyInstaller unpacks bundled data
# (see main-windows.spec's a.datas += Tree(...,'examples')) into its own
# runtime temp directory, sys._MEIPASS. Without this, exampleDir never
# exists on disk in a frozen build, getTxtFilesFromDir() silently returns
# [], and the "Load Examples" submenu ends up with zero items (see item 31,
# same root cause for self.mainPath in InstantClue.__init__).
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    filePath = sys._MEIPASS
else:
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

        # Bug fix: resource lookup (icons, conf/, quickSelectLists/, etc.) was
        # using os.path.dirname(sys.argv[0]) unconditionally. In a normal dev
        # run that's the source folder next to those resource directories, so
        # it worked - but in a PyInstaller ONEFILE frozen build, sys.argv[0]
        # is the path to the .exe itself, and the bundled resource folders are
        # NOT extracted next to the exe - they're unpacked into PyInstaller's
        # own runtime temp folder (sys._MEIPASS), which is a different path
        # entirely. The old code silently resolved to a folder that never had
        # those resources, so every os.path.exists() guard downstream quietly
        # failed and fell back to an empty/default icon (this is why the
        # taskbar/window icon showed a generic icon instead of the InstantClue
        # logo - main-windows.spec bundles icons/, conf/, quickSelectLists/,
        # examples/, annotations/ directly under _MEIPASS in a frozen build).
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            self.mainPath = sys._MEIPASS
        else:
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
        #set up notifcation handler
        self.notification = Notification()
        #set up logger 
        self.logger = ICLogger(self.config,__VERSION__)
        self.updateChecker = UpdateChecker(__VERSION__) #item 62: update check re-enabled
        #setup web app communication
        self.webAppComm = ICAppValidator(self)

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
        #Item 62: update check re-enabled (removed in item 48). Runs once on a
        #worker thread at startup; it only shows something if GitHub has a release
        #with a strictly higher version number than __VERSION__ (offline / up to
        #date = silent). See backend/update/Update.py.
        self.sendRequestToThread({"key":"update::checkForUpdate","kwargs":{"silent":True}})
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
        
        self.config.setParamRange("label.font.family",QFontDatabase.families())
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

    def progress_fn(self, n):
        ""
        
        
 
    def print_output(self, s):
        ""
        

    def getErrorLogPath(self):
        """Return a writable path for the error log, or None if nowhere is writable.

        Item 55: now the same file as the startup/crash handler at the top of
        this module - InstantClue_error_log.txt in the installation folder
        (the folder of InstantClue.exe), falling back to %APPDATA%\\InstantClue
        and then the temp folder when the installation folder is read-only.
        (Item 50 wrote logs\\ic_errors.log next to self.mainPath, which in a
        frozen build is PyInstaller's bundle folder rather than the folder the
        user sees.)
        """
        return icGetErrorLogPath()

    def writeErrorLog(self, details):
        "Append details to the error log. Returns the path written, else None."
        logPath = self.getErrorLogPath()
        if logPath is None:
            return None
        try:
            with open(logPath,"a",encoding="utf-8") as logFile:
                logFile.write("\n" + "="*72 + "\n")
                logFile.write("{} - InstantClue {}\n".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), __VERSION__))
                logFile.write("="*72 + "\n")
                logFile.write(details if isinstance(details,str) else str(details))
                logFile.write("\n")
            return logPath
        except Exception:
            return None

    def errorInThread(self, errorInfo = None):
        """Report an exception raised on a worker thread.

        Bug fix (item 50 / report 2.1). This used to take three parameters and
        ignore all of them, showing only "There was an unknwon error." - which
        discarded the exception type, message and traceback that
        backend/utils/worker.py had already captured and emitted. The signature
        was also wrong: WorkerSignals.error is pyqtSignal(tuple), i.e. exactly
        one argument, so "v" and "e" were always None.

        errorInfo is that single tuple: (exception type, exception value,
        formatted traceback).
        """
        excType = excValue = excTraceback = None
        if isinstance(errorInfo,(tuple,list)):
            if len(errorInfo) > 0: excType = errorInfo[0]
            if len(errorInfo) > 1: excValue = errorInfo[1]
            if len(errorInfo) > 2: excTraceback = errorInfo[2]

        typeName = getattr(excType,"__name__",None)
        if typeName is None:
            typeName = "Unknown error" if excType is None else str(excType)
        summary = typeName if excValue is None else "{}: {}".format(typeName,excValue)

        #write the full traceback if we have one, otherwise at least the summary
        logPath = self.writeErrorLog(excTraceback if excTraceback else summary)

        message = "An error occurred in a background calculation:\n\n{}".format(summary)
        if logPath is not None:
            message += "\n\nFull details (traceback) were written to:\n{}".format(logPath)
        self.sendMessageRequest({"title":"Error ..","message":message})


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
        data = resultDict["data"].copy()
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
                #item 56 (report C1): errors while updating the GUI after a calculation
                #(plot/table not refreshing) used to leave no trace at all
                self.writeErrorLog("Error while applying the result of '{}' ({}):\n{}".format(
                    resultDict.get("funcKey"), fnComplete.get("fn") if isinstance(fnComplete,dict) else fnComplete,
                    traceback.format_exc()))
        
    
    def _threadFinished(self,threadID):
        "Indicate in the ui that a thread finished."
        self.mainFrames["sliceMarks"].threadWidget.threadFinished(threadID)
    

    def getTable(self,tableName):
        ""
        if hasattr(self.mainFrames["sliceMarks"],tableName):
            return getattr(self.mainFrames["sliceMarks"],tableName)


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

    def askForGroupingSelection(self, funcKey, numericColumnsInKwargs = True, title = "Groupings to display in h. clustering.", kwargName = "groupingName", **kwargs):
        ""
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
                    #item 56 (report C1): used to be print() only, i.e. invisible
                    missingKwargs = [kw for kw in fnRequest["requiredKwargs"] if kw not in funcProps["kwargs"]]
                    print("not all required kwargs found...")
                    self.writeErrorLog("Request '{}' was not started: missing required argument(s) {}".format(funcKey, missingKwargs))

        except Exception as e:
            #item 56 (report C1): a failing request (e.g. a dispatch entry pointing at a
            #method that does not exist) used to vanish silently
            print(e)
            self.writeErrorLog("Error while starting request '{}':\n{}".format(
                funcProps.get("key") if isinstance(funcProps,dict) else funcProps, traceback.format_exc()))

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
            r = requests.get(URL, timeout=30)
            return True, r.json()
        except Exception:
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
        
        r = requests.post(URL, json = jsonData, timeout=30)

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

    def setBufToClipboardImage(self,buf):
        ""
        QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
        buf.close()

    def showMessageForNewVersion(self,releaseURL,latestVersion=None):
        "Item 62: ask whether to open the GitHub page of a newer release."
        infoText = "A new version of Instant Clue is available"
        if latestVersion is not None:
            infoText += " ({}; this version is {})".format(latestVersion, __VERSION__)
        infoText += ". Open the download page now?"
        w = AskQuestionMessage(
            parent=self,
            infoText = infoText,
            title="Information",
            iconDir = self.mainPath,
            yesCallback = lambda : webbrowser.open(releaseURL))
        w.exec()

    def sendMessageRequest(self,messageProps = dict()):
        "Display message on user screen in the top right corner"
        # check if all keys present
        if all(x in messageProps for x in ["title","message"]): 
            if self.config.getParam("errorShownInDialog") and "Error" in messageProps["title"]:
                self.sendToWarningDialog(infoText=messageProps["message"])
            else:
                self.notification.setNotify(
                    messageProps["title"],
                    messageProps["message"])

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
    # Bug fix: iconPath was a relative path ("../icons/base/32.png") to a file
    # that does not exist anywhere in this codebase (verified: no "base/32.png"
    # under icons/ at all), and even if it existed, a relative path depends on
    # the process's current working directory - arbitrary for a double-clicked
    # frozen .exe. os.path.exists() therefore always returned False,
    # app.setWindowIcon() was never called, and Windows fell back to its own
    # generic default icon in the taskbar (a blank, detail-less window icon)
    # instead of the InstantClue logo. Same root-cause class as item 31/40
    # (self.mainPath / filePath), but a separate variable those fixes never
    # touched. Fixed the same way: resolve relative to sys._MEIPASS when
    # frozen, else relative to this file's own directory (robust regardless
    # of CWD), and point at the logo file that is actually bundled and known
    # to exist (icons/instantClueLogo.png - the same file getWindowIcon() uses).
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        _iconBaseDir = sys._MEIPASS
    else:
        _iconBaseDir = os.path.dirname(os.path.abspath(__file__))
    iconPath = os.path.join(_iconBaseDir,"icons","instantClueLogo.png")
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
    multiprocessing.set_start_method('spawn')
    try:
        exitCode = main()
    except BaseException as _icMainError:
        #item 55: an error while building the main window (before the event
        #loop runs) would otherwise end the program with no trace
        if not isinstance(_icMainError, (SystemExit, KeyboardInterrupt)):
            try:
                icWriteErrorLog(traceback.format_exc(), heading = "Error while starting InstantClue")
                _icMainError._icAlreadyLogged = True
            except Exception:
                pass
        raise
    sys.exit(exitCode)
