from re import split
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


from ui.utils import INSTANT_CLUE_BLUE, getRandomString, createTitleLabel


from ..custom.resortableTable import ResortTableWidget, ResortTableModel
import pandas as pd 
from ..custom.ICReceiverBox import BoxItem
from ..custom.buttonDesigns import ResetButton, PushHoverButton, ResortButton
from ..custom.utils import PropertyChooser



AVAILABLE_FUNCTIONS = [

    {
        "text" : "Select columns",
        "type" : "User Input",
        "tooltip" : "Selection of columns",
        "fkey" : ""
    },
    {
        "text" : "Define Group (User input)",
        "type" : "User Input",
        "tooltip" : "User will be prompted with a dialog to define grouping of columns",
        "fkey" : ""
    },
    {
        "text" : "Transform values",
        "type" : "Transformation",
        "tooltip" : "Transforms selected data.",
        "options" : ["log2","log10","2^x","ln"],
        "fkey" : ""
    },
    {
        "text" : "Median normalization",
        "type" : "Normalization",
        "tooltip" : "Transforms selected data.",
        "fkey" : ""
    },
    {
        "text" : "Perform Analysis of Variance (ANOVA)",
        "type" : "Statistics",
        "tooltip" : "Performs a 1W ANOVA using all or specified groupings. This step requires a defined grouping",
        "fkey" : ""
    },
    {
        "text" : "Save as Excel File",
        "type" : "General",
        "tooltip" : "Saves file to a specified excel path.",
        "requiredInput" : [
                {"type":"path","text":"Provide the location to which the file should be saved."},
                {"type":"","text":""}
                ],
        "fkey" : ""
    }
    
]


class StepOptionDialog(QDialog):
    def __init__(self,options,*args,**kwargs):
        super(StepOptionDialog,self).__init__(*args,**kwargs)
        self.options = options 

    def __control(self):
        ""
                


class WorkflowItem(BoxItem):
    def __init__(self,setID, *args,**kwargs):
        super(WorkflowItem,self).__init__(*args,**kwargs)
        
        self.setID = setID

    def getWidthAndHeight(self):
        #get size
        
        self.width = 230
        self.height = 20

class ICWorkflowLine(PushHoverButton):
    def __init__(self,dropCallback, tableToResetDrag, rowInGrid, setID, rightClickCallback, *args,**kwargs):
        super(ICWorkflowLine,self).__init__(*args,**kwargs)
        #self.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.setFixedSize(230,35)
        #self.setFixedHeight(35)
        self.color = INSTANT_CLUE_BLUE
        self.dropCallback = dropCallback
        self.rowInGrid = rowInGrid
        self.tableToResetDrag = tableToResetDrag
        self.rightClickCallback = rightClickCallback
        self.setID = setID
        self.setAcceptDrops(True)

    def getRectProps(self,event):
        ""
        #calculate rectangle props
        rect = event.rect()
        h = rect.height()
        w = h
        x0 = rect.center().x()
        y0 = rect.center().y()    
        return rect, h, w, x0, y0 

    def paintEvent(self, ev: QPaintEvent) -> None:
        #get rect props
        rect, h, w, x0, y0 = self.getRectProps(ev)
        #setup painter
        
        painter = QPainter(self)
        pen = QPen(QColor(self.color))
        pen.setWidthF(2)
        painter.setFont(self.getStandardFont())
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.drawLine(QLineF(x0,y0-h/2,x0,y0+h/2))
        painter.drawEllipse(QPointF(x0,3),2,2)
        painter.drawEllipse(QPointF(x0,h-3),2,2)

    def dragEnterEvent(self,event):
        self.acceptDrop = True
        setattr(self,"color","red")
        self.update()
        event.accept()
    
    def dragLeaveEvent(self, event):
        setattr(self,"color",INSTANT_CLUE_BLUE)
        self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if e.button() == 2:
            self.rightClickCallback(self.setID)
       
    def dragMoveEvent(self, e):
        if self.acceptDrop:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self,event):
        ""
        try:
            event.accept()
            self.dropCallback(self.rowInGrid)
            self.tableToResetDrag.resetDragEvent()
            setattr(self,"color",INSTANT_CLUE_BLUE)
            self.update()
        except Exception as e:
            print(e)

class ICGroupFrame(QFrame):
    def __init__(self,groupID,updateFn,tableToResetDrag,*args,**kwargs):
        ""
        super(ICGroupFrame,self).__init__(*args,**kwargs)
        self.groupID = groupID
        self.setAcceptDrops(True)
        self.updateFn = updateFn
        self.tableToResetDrag = tableToResetDrag

    def dragEnterEvent(self,event):
        ""
        self.acceptDrop = False
        event.accept()
    
       
    def dragMoveEvent(self, e):
        ""
        if self.acceptDrop:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self,event):
        ""
        try:
            event.accept()
            self.updateFn(groupID = self.groupID)
            self.tableToResetDrag.resetDragEvent()
        except Exception as e:
            print(e)
    

class ICWorkflowBuilder(QDialog):

    def __init__(self, mainController, *args,**kwargs):
        ""
        super(ICWorkflowBuilder,self).__init__(*args,**kwargs)

        self.mC = mainController
        self.itemSets = dict() 
        self.__controls()
        self.__layout()

    def __controls(self):
        ""

        self.table =  ResortTableWidget(parent = self, menu = None)
        self.model = ResortTableModel(parent = self.table,
                                      inputLabels=pd.Series([f["text"] for f in AVAILABLE_FUNCTIONS]),
                                      title="Workflow Processes")
        self.model.onlyDragNoResort = True
        self.table.setModel(self.model)

        self.rightFrame = QFrame(parent=self)
        self.scrollArea = QScrollArea(parent=self.rightFrame)

        self.scrollFrame = ICGroupFrame("hi",self.addStep,self.table,parent=self.scrollArea) 
        #self.scrollFrame.setMinimumSize(200,300)
        self.scrollFrame.setMinimumHeight(200)
        
        item = WorkflowItem(itemName = "Load Data",parent=self.scrollFrame,itemBorder=5,setID=None)
        sep = ICWorkflowLine(self.addStep,self.table,1,setID=None, rightClickCallback = self.deleteStep )

        self.parentStepLayout = QVBoxLayout()

        steplayout = QVBoxLayout()
        steplayout.addWidget(item)
        steplayout.addWidget(sep)
        steplayout.setAlignment(Qt.AlignVCenter)
        
        self.parentStepLayout.addLayout(steplayout)

        self.scrollFrame.setLayout(QGridLayout())
        self.scrollFrame.layout().setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.scrollFrame.layout().addLayout(self.parentStepLayout,0,0)
        self.scrollFrame.layout().setRowStretch(1,1)
       

        #self.scrollFrame.layout().addStretch()

        #self.scrollFrame.layout().setRowStretch(200,2)
        
       # self.scrollFrame.layout().setAlignment(Qt.AlignTop) # | Qt.AlignTop

        self.scrollArea.setWidget(self.scrollFrame)
        self.scrollArea.setWidgetResizable(True)
        #
       

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        hboxMain = QHBoxLayout()
        #self.groupLayout = QVBoxLayout()
        #self.groupLayout.setAlignment(Qt.AlignTop)

        self.rightFrame.setLayout(QVBoxLayout())
        self.rightFrame.layout().addWidget(self.scrollArea)

        hboxMain.addWidget(self.table)
        hboxMain.addWidget(self.rightFrame)
        
        self.layout().addLayout(hboxMain)

    def deleteStep(self,stepID):
        ""
        if stepID is not None and stepID in self.itemSets:
            widgets = self.itemSets[stepID]
            self.scrollFrame.layout().removeWidget(widgets["item"])
            self.scrollFrame.layout().removeWidget(widgets["sep"])
            widgets["item"].deleteLater()
            widgets["sep"].deleteLater()
            del self.itemSets[stepID]

    def addStep(self,rowInGrid,*args,**kwargs):
        ""
        labels = self.model.getDraggedlabels()
        setID = getRandomString()
        item = WorkflowItem(itemName=labels.values[0],parent=self.scrollFrame,setID = setID)
        sep = ICWorkflowLine(self.addStep,self.table,rowInGrid+2,setID = setID, rightClickCallback = self.deleteStep)
        
        steplayout = QVBoxLayout()
        steplayout.addWidget(item)
        steplayout.addWidget(sep)
        steplayout.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.parentStepLayout.insertLayout(rowInGrid,steplayout)
        
        self.itemSets[setID] = {"item":item,"sep":sep,"rowInGrid":rowInGrid}
        #QVBoxLayout().insertLayout()
        #self.scrollFrame.layout().addWidget(sep,rowInGrid+2,0,1,1,Qt.AlignTop | Qt.AlignCenter)
        
        