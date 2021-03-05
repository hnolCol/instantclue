from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..custom.buttonDesigns import BigPlusButton, ResetButton, LabelLikeButton, ICStandardButton
from ..utils import createLabel, createTitleLabel, createLineEdit, getMessageProps, createMenu
from ..custom.utils import clearLayout, BuddyLabel
from ..custom.resortableTable import ResortTableWidget, ResortTableModel
from ..custom.ICReceiverBox import ItemHolder, BoxItem
from ..custom.warnMessage import WarningMessage

from collections import OrderedDict
import pandas as pd

class ICGroupFrame(QFrame):
    def __init__(self,groupID,*args,**kwargs):
        ""
        super(ICGroupFrame,self).__init__(*args,**kwargs)
        self.groupID = groupID
        self.setAcceptDrops(True)

    def dragEnterEvent(self,event):
        "Check if drag items is of correct datatype"
        self.acceptDrop = True
           # else:
            #    self.acceptDrop = False
        event.accept()
       
    def dragMoveEvent(self, e):
        "Ignore/acccept drag Move Event"
        if self.acceptDrop:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self,event):
        ""
        try:
            event.accept()
            self.parent().updateItems(groupID = self.groupID)
            self.parent().table.resetDragEvent()
        except Exception as e:
            print(e)
    





class ICGrouper(QDialog):

    def __init__(self, mainController, initNGroups = 2, *args,**kwargs):
        ""
        super(ICGrouper,self).__init__(*args,**kwargs)

        self.mC = mainController
        self.groupItems = OrderedDict()
        self.initNGroups = initNGroups
        
        self.__createMenu()
        self.__controls()
        self.__layout()
        self.__connectEvents()
        

        for n in range(initNGroups):
            self.addGroupArea(groupID = "Group {}".format(n))

    def __controls(self):
        ""
        self.groupingEdit = createLineEdit("Enter Grouping Name",
                    tooltipText="Provide name for selected grouping.\nWhen a test requires grouping, the grouping has to be choosen by this name.")
        self.addGroup = BigPlusButton(buttonSize=(25,25), tooltipStr="Add an additional group.")

        dataID = self.mC.getDataID()
        numericColumns = self.mC.mainFrames["data"].dataTreeView.getColumns("Numeric Floats")["Numeric Floats"]
        self.table =  ResortTableWidget(parent = self, menu = self.menu)
        self.model = ResortTableModel(parent = self.table,
                                      inputLabels=numericColumns,
                                      title="Numeric Columns")
        self.table.setModel(self.model)

        self.okButton = ICStandardButton(itemName="Okay")
        self.cancelButton = ICStandardButton(itemName = "Cancel")

    def __createMenu(self):
        ""

        self.menu = createMenu()
        self.menu.addAction("Infer grouping")
       
    
    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        hboxMain = QHBoxLayout()
        self.groupLayout = QVBoxLayout()
        self.groupLayout.setAlignment(Qt.AlignTop)

        hboxMain.addWidget(self.table)
        hbox = QHBoxLayout()
        hbox.addWidget(self.groupingEdit)
        hbox.addWidget(self.addGroup)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(self.groupLayout)

        hboxMain.addLayout(vbox)
        self.layout().addLayout(hboxMain)

        hboxButtons = QHBoxLayout()
        hboxButtons.addWidget(self.okButton)
        hboxButtons.addWidget(self.cancelButton)

        self.layout().addLayout(hboxButtons)
       
        
    def __connectEvents(self):
        ""
        self.addGroup.clicked.connect(self.addNewGroup)
        self.okButton.clicked.connect(self.saveGrouping)
        self.cancelButton.clicked.connect(self.close)

      #self.layout().addLayout(self.groupLayout)


    
    def addGroupArea(self,groupID = "2"):
        ""
        self.groupItems[groupID] = {}
        groupFrame = ICGroupFrame(parent = self, groupID = groupID) 

        try:
            groupEdit = createLineEdit("Enter group name ..","Define name of group. This name will be used to indicate comparisions.")
            groupEdit.hide() # Hide line edit
            groupEdit.editingFinished.connect(lambda groupID = groupID : self.textEdited(groupID=groupID))
            groupLabel = BuddyLabel(groupEdit) # Create our custom label, and assign myEdit as its buddy
            groupLabel.setText(groupID)
            groupLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed) # Change vertical size policy so they both match and you don't get popping when switching

            # Put them under a layout together
            hLabelLayout = QHBoxLayout()
            hLabelLayout.addWidget(groupLabel)
            hLabelLayout.addWidget(groupEdit)
            
            deleteButton = ResetButton(parent=groupFrame)
            deleteButton.clicked.connect(lambda _,groupID = groupID:self.deleteGroup(groupID))
            
            hbox = QHBoxLayout()
            hbox.addLayout(hLabelLayout)
            hbox.addWidget(deleteButton)
            
            groupFrame.setLayout(QVBoxLayout())

            itemFrame = QScrollArea(parent=groupFrame)
            itemHolder =  ItemHolder(direction="V",parent=itemFrame)
            
            itemFrame.setWidgetResizable(True)
            itemFrame.setWidget(itemHolder)
            self.groupItems[groupID]["itemFrame"] = itemFrame
            self.groupItems[groupID]["itemHolder"] = itemHolder
            self.groupItems[groupID]["main"] = groupFrame
            self.groupItems[groupID]["items"] = pd.Series()
            self.groupItems[groupID]["edit"] = groupEdit
            self.groupItems[groupID]["label"] = groupLabel
            self.groupItems[groupID]["name"] = groupID
        except Exception as e:
            print(e)

        groupFrame.layout().addLayout(hbox)
        groupFrame.layout().addWidget(itemFrame)
        self.groupLayout.addWidget(groupFrame)


    def addNewGroup(self):
        ""
        groupN = len(self.groupItems)
        groupID = "Group {}".format(groupN)
        while groupID in self.groupItems:
            groupN += 1
            groupID = "Group {}".format(groupN)
        
        self.addGroupArea(groupID)

    def deleteGroup(self,groupID):
        "Delete Frame that Carries Group Frame"
        if groupID in self.groupItems:
            self.groupLayout.removeWidget(self.groupItems[groupID]["main"])
            self.groupItems[groupID]["main"].deleteLater()
            if self.groupItems[groupID]["items"].index.size > 0:
                
                self.model.layoutAboutToBeChanged.emit()
                self.model.showHiddenLabels(self.groupItems[groupID]["items"])
                self.model.layoutChanged.emit()
                self.model.completeDataChanged()
            del self.groupItems[groupID]
           
    def saveGrouping(self,event=None):
        ""
        groupingName = self.groupingEdit.text()
        
        if  groupingName == "":
            w = WarningMessage(infoText = "No name for Grouping found.")
            w.exec_()
            return
        
        elif self.mC.grouping.nameExists(groupingName):
            w = WarningMessage(infoText = "The name of grouping exists already.")
            w.exec_()
            return
            
        elif any(self.groupItems[groupID]["items"].size < 2 for groupID in self.groupItems.keys()):
            w = WarningMessage(infoText = "One or more groups contain only a single item. Either remove group or add items.")
            w.exec_()
            return
        
        groupedItems = OrderedDict([(self.groupItems[groupID]["name"],self.groupItems[groupID]["items"]) for groupID in self.groupItems.keys()])
        self.mC.grouping.addGrouping(groupingName,groupedItems)
        treeView = self.mC.getTreeView("Numeric Floats")
        treeView.setGrouping(groupedItems,groupingName)
        self.mC.sendMessageRequest(getMessageProps("Done ..","Grouping {} saved.".format(groupingName)))
        self.close()
        
    def sizeHint(self):
        ""
        return QSize(600,500)

    def updateItems(self,event=None,groupID = None):
        ""
        labels = self.model.getDraggedlabels()
        items = self.groupItems[groupID]["items"]
        #find items that are not already in group
        idxToAdd = labels.index.difference(items.index)
        if idxToAdd.size > 0:

            addedItems = items.append(labels.loc[idxToAdd],ignore_index = False)
            self.groupItems[groupID]["items"] = addedItems
            self.model.layoutAboutToBeChanged.emit()
            self.model.hideLabels(labels)
            self.model.layoutChanged.emit()
            self.model.completeDataChanged()
            newLabels = labels.loc[idxToAdd]
            

            for l in newLabels.values:
                bItem = BoxItem(itemName=l,tooltipStr="Right-click to remove item from list.")
                self.groupItems[groupID]["itemHolder"].addItem(bItem)

            
    def textEdited(self, event = None, groupID = None):
        ""            
        if groupID in self.groupItems:
            edit = self.groupItems[groupID]["edit"]
            label = self.groupItems[groupID]["label"]
            if not edit.text():
                edit.hide()
                label.setText(self.groupItems[groupID]["name"])
                label.show()
            else:
                edit.hide()
                label.setText(edit.text())
                #save name
                self.groupItems[groupID]["name"] = edit.text()
                label.show()



