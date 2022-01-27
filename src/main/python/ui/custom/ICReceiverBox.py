from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from collections import OrderedDict

from .buttonDesigns import ResetButton, PushHoverButton, ResortButton
from .resortableTable import ResortableTable
from .utils import clearLayout
from ..utils import INSTANT_CLUE_BLUE, HOVER_COLOR ,WIDGET_HOVER_COLOR, createLabel, createTitleLabel

class BoxItem(PushHoverButton):

    def __init__(self,itemName = "", parent=None, itemBorder = 5, fColor = "black", bgColor = "white", *args,**kwargs):
        super(BoxItem,self).__init__(parent=parent,*args,**kwargs)
        self.itemName = itemName
        self.itemBorder = itemBorder
        self.bgColor = bgColor
        self.fColor = fColor
        self.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.setToolTip(itemName)
        self.getWidthAndHeight()

    def getWidthAndHeight(self):
        #get size
        font = self.getStandardFont()
        fontMetric = QFontMetrics(font)
        self.width = fontMetric.width(self.itemName) + self.itemBorder
        self.height = fontMetric.height() + self.itemBorder

    def getRectProps(self,event):
        ""
        #calculate rectangle props
        rect = event.rect()
        h = rect.height()
        w = h
        x0 = rect.center().x()
        y0 = rect.center().y()    
        return rect, h, w, x0, y0 

    def sizeHint(self):
        ""
        return QSize(self.width,self.height)

    def getMainLabelRect(self,x0,y0,w,h):
        ""
        return QRect(x0-w/2,y0+h/5,w,h/4)

    def paintEvent(self,event):
        ""
        #get rect props
        rect, h, w, x0, y0 = self.getRectProps(event)
        #setup painter
        
        painter = QPainter(self)
        pen = QPen(QColor(self.fColor))
        pen.setWidthF(0.5)
        painter.setFont(self.getStandardFont())
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        if self.mouseOver:
            brush = QBrush(QColor(HOVER_COLOR))
        else:
            brush = QBrush(QColor(self.bgColor))

        painter.setBrush(brush)
        if self.drawFrame:
            painter.drawRect(rect)

        painter.drawText(rect,
                         Qt.AlignCenter | Qt.AlignTop, 
                         self.itemName)
        
    def setText(self,itemName):
        ""
        self.itemName = itemName
        self.getWidthAndHeight()
        self.setFixedSize(self.sizeHint())

class ItemHolder(QWidget):
    def __init__(self,direction = "H",title = "Drag column headers here.", *args,**kwargs):
        super(ItemHolder,self).__init__(*args,**kwargs)
        #self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        if direction == "H":
            self.setLayout(QHBoxLayout())
            self.layout().setAlignment(Qt.AlignLeft)
            self.itemLayout = QHBoxLayout()
            self.itemLayout.setAlignment(Qt.AlignLeft)
            
        else:
            self.setLayout(QVBoxLayout())
            self.layout().setAlignment(Qt.AlignTop)
            self.itemLayout = QVBoxLayout()
            self.itemLayout.setAlignment(Qt.AlignTop)

        self.layout().addLayout(self.itemLayout)
        self.itemLayout.setContentsMargins(2,1,2,1)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor("#f6f6f6"))
        self.setPalette(p)

        self.dragLabel = createTitleLabel(title,fontSize=14)

        self.layout().addWidget(self.dragLabel)
        
    def toggleDragLabelVisibility(self):
        ""
        self.dragLabel.setVisible(not self.dragLabel.isVisible())

    def setDragLabelVisibility(self,visible):
        ""
        self.dragLabel.setVisible(visible)

    def addItem(self,boxItem):
        ""        
        self.itemLayout.addWidget(boxItem)
    
    def deleteItem(self,boxItem):
        boxItem.deleteLater()

    def clear(self):
        clearLayout(self.itemLayout)
        


class ReceiverBox(QFrame):
    def __init__(self,parent=None,
                title="Numeric", 
                acceptedDragTypes = ["Numeric Floats", "Integers"], 
                *args,**kwargs):
        
        super(ReceiverBox, self).__init__(parent,*args,**kwargs)

        self.title = title
        self.acceptedDragTypes = acceptedDragTypes
        self.items = OrderedDict()
        self.performUpdate = False

        #create widget
        self.__controls()
        self.__layout()
        self.__connectEvents() 

        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        p = self.palette()
        p.setColor(self.backgroundRole(),QColor("#f6f6f6"))
        self.setPalette(p)
        
        self.setAutoFillBackground(True)
        self.setToolTip("Drag & Drop column headers")
        
    def __controls(self):
        ""
        self.titleLabel = createLabel(self.title)
        self.clearButton = ResetButton(self, tooltipStr="Remove all items from receiver box.")
        self.sortButton = ResortButton(parent=self,tooltipStr="Resort items in receiver box.")

        self.itemHolder = ItemHolder()

        self.itemFrame = QScrollArea()
        self.itemFrame.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.itemFrame.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.itemFrame.setWidgetResizable(True)
        self.itemFrame.setWidget(self.itemHolder)
        #self.itemFrame.setStyleSheet("background-color:red;");
        self.itemFrame.setFrameShape(QFrame.NoFrame)
        self.itemFrame.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.itemFrame.setMaximumHeight(25)

        

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(1,1,1,1)
        self.layout().setSpacing(1)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(2,2,2,2)
        hbox.setSpacing(4)
        hbox.addWidget(self.titleLabel)
        hbox.addStretch(1)
        hbox.addWidget(self.sortButton)
        hbox.addWidget(self.clearButton)
        self.layout().addLayout(hbox)
        self.layout().addWidget(self.itemFrame)
        self.layout().addStretch(1)

    def __connectEvents(self):
        ""
        self.clearButton.clicked.connect(self.clearDroppedItems)
        self.sortButton.clicked.connect(self.openSortDialog)

    def sizeHint(self):
        ""
        return QSize(50,50)

    def dropEvent(self,event = None):
        ""
        items = self.parent().getDragColumns()
        self.addItems(items)
        self.reportStateBackToTreeView(event.source())
        self.update()
        self.parent().recieverBoxItemsChanged()
        event.accept()
        
    def dragEnterEvent(self,e):
        ""
        dragType = self.parent().getDragType()
        #check if type is accpeted and check if not all items are anyway already there
        if dragType in self.acceptedDragTypes and \
            not all(columnName in self.items for columnName in self.parent().getDragColumns()):
            self.acceptDrop = True
        else:
            self.acceptDrop = False
        e.accept()

    def dragMoveEvent(self, e):
        "Ignore/acccept drag Move Event"
        if self.acceptDrop:
            e.accept()
        else:
            e.ignore()

    def getItems(self):
        ""
        return self.items

    def getItemNames(self):
        ""
        return list(self.items.keys())

    def clearDroppedItems(self, event = None, reportStateToTreeView = True, emitReceiverBoxChangeSignal = True):
        ""
        if len(self.items) == 0:
            return
        self.itemHolder.clear()#   clearLayout(self.itemBox)
        
        if reportStateToTreeView:
            self.reportItemRemovalToTreeView(list(self.items.keys()))

        self.items.clear()
        self.handleDragLabelVisibility()
        self.update()
        if emitReceiverBoxChangeSignal:
            self.itemsChanged()

    def addItems(self,items):
        "Adds items (list) to widget"
        
        for itemName in items:
            if itemName not in self.items:
                try:
                    w = BoxItem(itemName)
                    w.setContextMenuPolicy(Qt.CustomContextMenu)
                    w.customContextMenuRequested.connect(self.deleteBoxItem)
                except Exception as e:

                    pass 
                
                self.items[itemName] = w
                self.itemHolder.addItem(w)
        self.handleDragLabelVisibility()

    def deleteBoxItem(self,event):
        ""
        itemName = [self.sender().itemName]
        self.removeItems(itemName, reportStateToTreeView=True)
        self.handleDragLabelVisibility()
        self.itemsChanged()

    def updateItems(self):
        ""
        if not self.performUpdate:
            return
        if len(self.items) == 0:
            return
        updatedItems = OrderedDict()
        for v in self.items.values():
            itemText = v.itemName
            updatedItems[itemText] = v
        self.items = updatedItems
        self.performUpdate = False

    def itemsChanged(self):
        "Notify plotter that items changed."
        
        self.parent().recieverBoxItemsChanged()

    def handleDragLabelVisibility(self):
        ""
       
        if len(self.items) == 0:

            self.itemHolder.setDragLabelVisibility(True)
        else:
            self.itemHolder.setDragLabelVisibility(False)

    def renameItem(self,itemName,newItemName):
        ""
        
        if itemName in self.items:
            self.items[itemName].setText(newItemName) 
            self.performUpdate = True
                
    def removeItems(self, items, reportStateToTreeView = False):
        ""
        for itemName in items:
            if itemName in self.items:
                self.items[itemName].deleteLater()
                del self.items[itemName]
        if reportStateToTreeView:
            self.reportItemRemovalToTreeView(items)
        self.update()

    def reportStateBackToTreeView(self,treeView):
        ""
        if hasattr(treeView, "setColumnStateOfDraggedColumns"):
            treeView.setColumnStateOfDraggedColumns()

    def reportItemRemovalToTreeView(self,items):
        ""
        self.parent().setColumnStateInTreeView(items)

    def openSortDialog(self,event=None):
        ""
        inputLabels = list(self.items.keys())
        if len(inputLabels) >= 2:
            sortDialog = ResortableTable(inputLabels = inputLabels)
            sortDialog.exec_()
            if sortDialog.savedData is not None:
                #user clicked save
                resortedLabels = sortDialog.savedData.values.tolist()
                #actually resorted
                if inputLabels != resortedLabels:
                    self.clearDroppedItems(emitReceiverBoxChangeSignal=False)
                    self.addItems(resortedLabels)
                    self.itemsChanged()

                
        