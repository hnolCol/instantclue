from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 
from ..utils import createLabel, INSTANT_CLUE_BLUE, WIDGET_HOVER_COLOR, getStandardFont, createMenu, createCombobox, createLineEdit, isWindows
from .buttonDesigns import LabelLikeButton
from ..dialogs.ICColorChooser import ColorLabel
import numpy as np

INSTANT_CLUE_ANAYLSIS = [
                    {"Model":
                    [
                        "Axes Diagonal",
                        "Line (slope=1)",
                        "Line (y = m*x + b)",
                        "Vertical Line",
                        "Horizontal Line",
                        "Cross",
                        "Quadrant Lines",
                        "Line from file",
                        "Line from clipboard",
                        "lowess",
                        "linear fit"]},
                    {"Compare two groups":
                    [
                        "t-test",
                        "Welch-test",
                        "Wilcoxon",
                        "(Whitney-Mann) U-test"]},
                    {"One group tests":
                    [
                        "One-sample t-test",
                        "Wilcoxon signed-rank test"]}

    #{"Compare multiple groups":["1W-ANOVA","1W-ANOVA (rep. measures)"]},
   # {"Dimensional Reduction":}
   # {"Transformation":["TSNE","PCA"]},
    #{"Cluster Analysis":["k-means","DBSCAN","Birch","Affinity Propagation","Agglomerative Clustering"]}
]

dataFileExport = [("txt","Text File"),
                    ("xlsx", "Excel file"),
                    ("xlsx-multiple", "Excel file (multiple data frames)"),
                    ("json", "Json file"),
                    ("md","Markdown file")]

def clearLayout(layout):
    "Clears all widgets from layout"
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


class ICSCrollArea(QScrollArea):
    def __init__(self,getUpdatabelWidgets = None, updateSrollbar = "V",*args,**kwargs):
        super(ICSCrollArea,self).__init__(*args,**kwargs)
        if updateSrollbar == "V":
            self.verticalScrollBar().valueChanged.connect(self.sliderMoved)
        elif updateSrollbar == "H":
            self.horizontalScrollBar().valueChanged.connect(self.sliderMoved)
        elif updateSrollbar == "B":
            self.verticalScrollBar().valueChanged.connect(self.sliderMoved)
            self.horizontalScrollBar().valueChanged.connect(self.sliderMoved)
        self.getUpdatabelWidgets = getUpdatabelWidgets

    def sliderMoved(self,*args,**kwargs):
        ""
        self.updateWidgetsOnWindows()

    def viewportEvent(self, a0: QEvent) -> bool:
        
        return super().viewportEvent(a0)

    def updateWidgetsOnWindows(self):
        if isWindows() and self.getUpdatabelWidgets is not None:
        #print(self.parent().filterProps)
            
            for _, widgetCollection in self.getUpdatabelWidgets().items():
                for w in widgetCollection["widgetsToUpdate"]:
                    w.update()
           
      

class BuddyLabel(QLabel):
    def __init__(self, buddy, parent = None):
        super(BuddyLabel, self).__init__(parent)
        self.buddy = buddy
        # When it's clicked, hide itself and show its buddy
        font = getStandardFont()
        self.setFont(font)

    def mousePressEvent(self, event):
        self.hide()
        self.buddy.show()
        self.buddy.setFocus() # Set focus on buddy so user doesn't have to click again


class QHLine(QFrame):
    def __init__(self,parent=None):
        super(QHLine, self).__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)


class QVLine(QFrame):
    def __init__(self,parent=None):
        super(QVLine, self).__init__(parent)
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(QFrame.Shape.Sunken)


class QToggle(QPushButton):
    def __init__(self, parent = None,*args,**kwargs):
        super().__init__(parent=parent,*args,**kwargs)
        self.setCheckable(True)
        self.setMinimumWidth(66)
        self.setMinimumHeight(22)

    def paintEvent(self, event):
        label = "True" if self.isChecked() else "False"
        bg_color = QColor(INSTANT_CLUE_BLUE) if self.isChecked() else QColor(WIDGET_HOVER_COLOR)

        radius = 15
        width = 32
        center = self.rect().center()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(center)
        painter.setBrush(QColor(0,0,0))

        pen = QPen(Qt.GlobalColor.black)
        pen.setWidthF(0.5)
        painter.setPen(pen)

       # painter.drawRoundedRect(QRect(-width, -radius, 2*width, 2*radius), radius, radius)
        painter.setBrush(QBrush(bg_color))
        painter.drawEllipse(QRectF(-width, -radius/2, radius, radius))
        
        sw_rect = QRectF(-radius, -radius, width + radius, 2*radius)
      
        painter.setFont(getStandardFont())
        painter.drawText(sw_rect, Qt.AlignmentFlag.AlignVCenter, label)



class PropertyChooser(QWidget):
    def __init__(self,mainController, parent=None,*args,**kwargs):
        super(PropertyChooser,self).__init__(parent=parent,*args,**kwargs)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.setLayout(QGridLayout())
        self.layout().setContentsMargins(2,1,2,1)
        self.mC = mainController
      
    def addItem(self,boxItem):
        ""
        self.layout().addWidget(boxItem)

    def chooseColor(self,event ,parentLineEdit ,initialColor,sender,*args,**kwargs):
        ""
        
        updatadColor = QColorDialog(parent=self).getColor(QColor(initialColor))
        if updatadColor.isValid():
            updatedHexColor = updatadColor.name()
            if updatedHexColor != initialColor:
                if hasattr(parentLineEdit,"setText") and isinstance(updatedHexColor,str):
                    parentLineEdit.setText(updatedHexColor)
                if hasattr(sender,"setBackgroundColor"):
                    sender.setBackgroundColor(updatedHexColor)
        

    def addProperties(self, parameters, clearLayoutBefore = True, updateParamsBefore = True):
        ""
        
        if updateParamsBefore:
            self.updateParams()

        if clearLayoutBefore:
            clearLayout(self.layout())
            self.inputValues = []

        for n, p in  enumerate(parameters):
            colorButton = None
            propLabel = createLabel("{}:".format(p.getAttr("name")))
            if p.getAttr("dtype") == str:
                if isinstance(p.getAttr("range"),list):
                    vInput = createCombobox(self,p.getAttr("range"))
                    vInput.setCurrentText(p.getAttr("value"))
                elif isinstance(p.getAttr("range"),str):
                    vInput = createLineEdit(p.getAttr("name"),"")
                    vInput.setText(p.getAttr("value"))
                    if p.getAttr("isColor"):
                        hexColor = p.getAttr("value")
                        colorButton = ColorLabel(backgroundColor=hexColor)
                        colorButton.setFixedSize(QSize(20,20))
                        colorButton.mousePressEvent = lambda event,parentLineEdit = vInput, hexColor = hexColor, sender = colorButton: self.chooseColor(event, parentLineEdit,hexColor,sender)
                       
            elif p.getAttr("dtype") == bool:

                vInput = QToggle()
                vInput.setChecked(p.getAttr("value"))

            else:
                vInput = createLineEdit(p.getAttr("name"),"")
                vInput.setText(str(p.getAttr("value")))
            propLabel.setToolTip(p.getAttr("description"))

            if colorButton is not None:
                self.layout().addWidget(colorButton,n,2,1,1)
                self.layout().addWidget(vInput,n,1,1,1)
            else:
                self.layout().addWidget(vInput,n,1,1,2)
            self.layout().addWidget(propLabel,n,0,1,1,Qt.AlignmentFlag.AlignRight)
            
            #save input label
            self.inputValues.append(vInput)
        self.parameters = parameters
            
    def clear(self):
        clearLayout(self.layout())  

    def updateParams(self):
        ""
        if hasattr(self,"parameters"):
            for n,p in enumerate(self.parameters):
                vInput = self.inputValues[n]
                if p.getAttr("dtype") == bool:
                    newValue = vInput.isChecked()
                elif p.getAttr("dtype") == str:
                    if isinstance(p.getAttr("range"),list):
                        newValue = vInput.currentText()
                    elif isinstance(p.getAttr("range"),str):
                        newValue = vInput.text() 
                else:
                    try:
                        newValue = np.int(np.float(vInput.text())) if p.getAttr("dtype") == int else np.float(vInput.text())
                    except:

                        
                        self.mC.sendToWarningDialog(infoText="Could not convert input to float or integer. Paramater not updated")
                        continue
                            
                p.setAttr("value",newValue)
                p.updateAttrInParent()


class LabelLikeCombo(LabelLikeButton):
    selectionChanged = pyqtSignal(tuple)
    def __init__(self,items, *args,**kwargs):

        
        super(LabelLikeCombo,self).__init__(*args,**kwargs)
        self.items = items
        self.itemID = None
        self.addMenu()
        self.clicked.connect(self.castMenu)
    
    def addMenu(self):
        ""
        self.menu = createMenu(parent = self.parent())
        for itemID, itemName in self.items.items():
                if self.itemID is None:
                    self.itemID = itemID
                action = self.menu.addAction(itemName)
                action.triggered.connect(lambda _, ID = itemID,text = itemName : self.emitSignal(ID,text))

    def castMenu(self):
        #reset button
        self.mouseOver = False
        #find menu position
        senderGeom = self.geometry()
        topLeft = self.parent().mapToGlobal(senderGeom.bottomLeft())
        #cast menu
        self.menu.popup(topLeft)
    
    def getItemID(self):
        ""
        return self.itemID
           
    def emitSignal(self,itemID,itemText):
        ""
        self.itemID = itemID
        self.setText(itemText)
        self.selectionChanged.emit((itemID,itemText))
       
      