from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# internal imports
from ui.custom.buttonDesigns import LabelButton, TooltipButton, SizeButton, ColorButton, FilterButton, SelectButton , MarkerButton
from ui.custom.ICColorTable import ICColorTable 
from ui.custom.ICSizeTable import ICSizeTable
from ui.custom.ICLabelTable import ICLabelTable
from ui.custom.ICMarkerTable import ICMarkerTable
from ui.custom.ICStatisticTable import ICStatisticTable
from ..dialogs.ICColorChooser import ColorChooserDialog
from ..dialogs.ICCategoricalFilter import CategoricalFilter, FindStrings
from ..dialogs.numericalFilter import NumericFilter
from ui.custom.warnMessage import WarningMessage
from ..utils import createSubMenu
from ..tooltips import SLICE_MARKS_TOOLTIPSTR

from backend.utils.stringOperations import mergeListToString
#selection sqaure size in percentage of axis limits
selectValueMatch = {"Single":0,"small (2%)":0.02,"middle (5%)":0.05,"huge (10%)":0.1,"extra huge (15%)":0.15}


class SliceMarksFrame(QWidget):
    def __init__(self,parent=None, mainController = None):
        
        super(SliceMarksFrame, self).__init__(parent)

        self.mC = mainController
        self.__controls()
        self.__layout() 
        self.__connectEvents()
      
    def __controls(self):
        #control background role
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor("#f6f6f6"))
        self.setPalette(p)
        self.setAutoFillBackground(True)
        #create buttons
        self.colorButton = ColorButton(self, 
                                tooltipStr=SLICE_MARKS_TOOLTIPSTR["color"], 
                                callback = self.handleColorDrop, 
                                getDragType = self.getDragType, 
                                acceptDrops=True,
                                acceptedDragTypes= ["Categories" , "Numeric Floats","Integers"])

        self.sizeButton = SizeButton(self, 
                                tooltipStr = SLICE_MARKS_TOOLTIPSTR["size"], 
                                callback = self.handleSizeDrop, 
                                getDragType = self.getDragType, 
                                acceptDrops=True,
                                acceptedDragTypes= ["Categories" , "Numeric Floats","Integers"])

        self.markerButton = MarkerButton(self,
                                tooltipStr = SLICE_MARKS_TOOLTIPSTR["marker"], 
                                acceptDrops = True,
                                callback = self.handleMarkerDrop, 
                                getDragType = self.getDragType,
                                acceptedDragTypes= ["Categories"])

        self.toolTipButton = TooltipButton(self, tooltipStr=SLICE_MARKS_TOOLTIPSTR["tooltip"],
                                callback= self.handleTooltipDrop,
                                getDragType = self.getDragType,
                                acceptDrops = True,
                                acceptedDragTypes= ["Categories" , "Numeric Floats", "Integers"])



        self.labelButton = LabelButton(self, 
                                tooltipStr=SLICE_MARKS_TOOLTIPSTR["label"],
                                callback= self.handleLabelDrop,
                                getDragType = self.getDragType,
                                acceptDrops = True,
                                acceptedDragTypes= ["Categories" , "Numeric Floats", "Integers"])
                                

        self.filterButton = FilterButton(self, callback=self.applyFilter, getDragType = self.getDragType ,acceptDrops=True ,tooltipStr=SLICE_MARKS_TOOLTIPSTR["filter"], acceptedDragTypes= ["Categories" , "Numeric Floats"])
        self.selectButton = SelectButton(self, tooltipStr=SLICE_MARKS_TOOLTIPSTR["select"])

        self.colorTable = ICColorTable(mainController=self.mC)
        self.sizeTable = ICSizeTable(mainController=self.mC)
        self.labelTable = ICLabelTable(mainController=self.mC)
        self.tooltipTable = ICLabelTable(mainController=self.mC, header="Tooltip")
        self.statisticTable = ICStatisticTable(mainController=self.mC)
        self.markerTable = ICMarkerTable(mainController=self.mC)

    def __layout(self):
        ""

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(5,0,5,0)

        topGrid = QGridLayout()
        topGrid.setContentsMargins(2,2,2,2)
        topGrid.setSpacing(10)
        topGrid.addWidget(self.filterButton,0,1)
        topGrid.addWidget(self.selectButton,0,0)
        topGrid.setAlignment(Qt.AlignCenter)

        bottomGrid = QGridLayout()
        bottomGrid.setContentsMargins(2,2,2,2)
        bottomGrid.setSpacing(10)
        bottomGrid.addWidget(self.colorButton,0,0)
        bottomGrid.addWidget(self.sizeButton,0,1)
        bottomGrid.addWidget(self.markerButton,0,2)
        bottomGrid.addWidget(self.labelButton,2,0)
        bottomGrid.addWidget(self.toolTipButton,2,1)
        bottomGrid.setAlignment(Qt.AlignCenter)
       

        self.layout().addWidget(QLabel("Slice"))
        self.layout().addLayout(topGrid)
        self.layout().addWidget(QLabel("Marks"))
        self.layout().addLayout(bottomGrid)
        
        self.layout().addWidget(self.colorTable)
        self.layout().addWidget(self.sizeTable)
        self.layout().addWidget(self.markerTable)
        self.layout().addWidget(self.labelTable)
        self.layout().addWidget(self.tooltipTable)
        self.layout().addWidget(self.statisticTable)
        #self.layout().addStretch(1)
        self.layout().setAlignment(Qt.AlignTop)

    def __connectEvents(self):
        ""
        self.selectButton.clicked.connect(self.chooseSelectMode)
        self.colorButton.clicked.connect(self.chooseColor)
        #self.filterButton.clicked.connect(self.applyFilter)


    def chooseSelectMode(self,e = None):
        ""
        menu = createSubMenu(subMenus=["Point Selection","Rectangle Selection"])
        menu["main"].addMenu(menu["Rectangle Selection"])
        for ellipseSize in ["small (2%)","middle (5%)","huge (10%)","extra huge (15%)"]:
            menu["Rectangle Selection"].addAction(ellipseSize)
        menu["Point Selection"].addAction("Single")
        #menu["Point Selection"].addAction("Lasso")
        #find bottom left corner
        senderGeom = self.sender().geometry()
        topLeft = self.mapToGlobal(senderGeom.bottomLeft())
        #set sender status 
        if hasattr(self.sender(),"mouseOver"):
            self.sender().mouseOver = False
        #cast menu
        action = menu["main"].exec_(topLeft)
        if action:
            self.adjustSelectMode(mode = action.text())

    def chooseColor(self,event=None):
        ""
        #find bottom left corner
        senderGeom = self.sender().geometry()
        topLeft = self.mapToGlobal(senderGeom.bottomLeft())
        #set sender status 
        self.sender().mouseOver = False
        #cast menu
        dlg = ColorChooserDialog(mainController = self.mC)
        dlg.setGeometry(topLeft.x(),topLeft.y(),350,300)
        dlg.exec_() 
    
    def checkGraph(self):
        ""
        exists, graph = self.mC.getGraph()
        if not exists:
            w = WarningMessage(infoText="Create a chart first.")
            w.exec_() 
        return exists, graph
    
    def applyFilter(self, event = None, columnNames = None, dragType = None, dataID = None, filterType = "category"):
        ""            
        dataFrame = self.mC.mainFrames["data"]
        if dataID is None:
            dataID = dataFrame.getDataID()
        if columnNames is None:
            columnNames = dataFrame.getDragColumns()

        if dragType is None:
            dragType = dataFrame.getDragType()
            
        if dragType != "Numeric Floats":
            
                self.mC.data.categoricalFilter.setupLiveStringFilter(dataID,columnNames, filterType = filterType)
                if filterType == "category":
                    self.filterDlg = CategoricalFilter(mainController = self.mC, categoricalColumns = columnNames)
                elif filterType == "string":
                    self.filterDlg = FindStrings(mainController = self.mC, categoricalColumns = columnNames)
            
        else:
            
                self.filterDlg = NumericFilter(mainController = self.mC, selectedNumericColumn = columnNames)
            
        self.filterDlg.exec_()
    
    def handleColorDrop(self):
        ""
        plotType = self.mC.getPlotType()
        if plotType == "scatter":
            fkey = "plotter:getScatterColorGroups"
        elif plotType == "hclust":
            fkey = "plotter:getHclustColorGroups"
        else:
            return
        columnNames = self.mC.mainFrames["data"].getDragColumns()
        dragType = self.mC.mainFrames["data"].getDragType()
        dataID = self.mC.mainFrames["data"].getDataID()
        funcProps = {"key":fkey,"kwargs":{"dataID":dataID,"colorColumn":columnNames,"colorColumnType":dragType}}
        self.mC.sendRequestToThread(funcProps)

    def handleLabelDrop(self):
        ""
        try:
            plotType = self.mC.getPlotType()
            columnNames = self.mC.mainFrames["data"].getDragColumns()
            dataID = self.mC.mainFrames["data"].getDataID()
            exists, graph = self.checkGraph()
            if exists:
                if plotType not in ["scatter","hclust"]:
                    w = WarningMessage(infoText="Labels can not be assigned to this plot type.")
                    w.exec_()
                else:
                    graph.addAnnotations(columnNames,dataID)
            
        except Exception as e:
            print(e)

    def handleMarkerDrop(self):
        ""
        plotType = self.mC.getPlotType()
        
        columnNames = self.mC.mainFrames["data"].getDragColumns()
        dragType = self.mC.mainFrames["data"].getDragType()
        dataID = self.mC.mainFrames["data"].getDataID()
        fkey = "plotter:getScatterMarkerGroups"
        funcProps = {"key":fkey,"kwargs":{"dataID":dataID,"markerColumn":columnNames,"markerColumnType":dragType}}
        self.mC.sendRequestToThread(funcProps)

    def handleSizeDrop(self):
        ""
        plotType = self.mC.getPlotType()
        if plotType == "scatter":
            fkey = "plotter:getScatterSizeGroups"
        elif plotType == "hclust":
            fkey = "plotter:getHclustSizeGroups"
        else:
            return
        columnNames = self.mC.mainFrames["data"].getDragColumns()
        dragType = self.mC.mainFrames["data"].getDragType()
        dataID = self.mC.mainFrames["data"].getDataID()
        funcProps = {"key":fkey,"kwargs":{"dataID":dataID,"sizeColumn":columnNames,"sizeColumnType":dragType}}
        self.mC.sendRequestToThread(funcProps)

    def handleTooltipDrop(self):
        ""
        plotType = self.mC.getPlotType()
        columnNames = self.mC.mainFrames["data"].getDragColumns()
        dataID = self.mC.mainFrames["data"].getDataID()
        exists, graph = self.checkGraph()
        if exists:
            if plotType not in ["scatter","hclust","swarmplot"]:
                w = WarningMessage(infoText="Tooltips can not be assigned to this plot type.")
                w.exec_()
            else:
                graph.addTooltip(columnNames,dataID)


    def updateFilter(self,boolIndicator,resetData=False):
        ""
        if hasattr(self,"filterDlg"):
            self.filterDlg.updateModelDataByBool(boolIndicator,resetData)

    def adjustSelectMode(self, mode):
        ""
        if mode in selectValueMatch:
            r = selectValueMatch[mode]
            self.mC.config.setParam("selectionRectangleSize",r)

    def getDragType(self):
        ""
        return self.mC.mainFrames["data"].getDragType()

    def sendRequestToThread(self,funcProps, addDraggedColumns = True, addDataID = True):
        try:
            if "kwargs" not in funcProps:
                funcProps["kwargs"] = dict()
            if addDraggedColumns:
                funcProps["kwargs"]["columnNames"] = self.mC.mainFrames["data"].getDragColumns()
            if addDataID:
                funcProps["kwargs"]["dataID"] = self.mC.mainFrames["data"].getDataID()
            self.mC.sendRequestToThread(funcProps)
        except Exception as e:
            print(e)
  
    def setColorGroupData(self,colorGroupData,title="",isEditable = True):
        ""
        self.colorTable.setData(colorGroupData,title,isEditable)

    def setSizeGroupData(self,sizeGroupData,title=""):
        ""
        self.sizeTable.setData(sizeGroupData,title)

    def setMarkerGroupData(self,markerGroupData,title=""):
        ""
        self.markerTable.setData(markerGroupData,title)