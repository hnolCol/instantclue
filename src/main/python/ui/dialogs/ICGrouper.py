from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..custom.buttonDesigns import BigPlusButton, ResetButton, LabelLikeButton, ICStandardButton
from ..utils import createLabel, createTitleLabel, createLineEdit, getMessageProps, createMenu
from ..custom.utils import clearLayout, BuddyLabel
from ..custom.resortableTable import ResortTableWidget, ResortTableModel
from ..custom.ICReceiverBox import ItemHolder, BoxItem
from ..custom.warnMessage import WarningMessage
from .selectionDialog import SelectionDialog
from collections import OrderedDict
import pandas as pd
import numpy as np

class ICGroupFrame(QFrame):
    def __init__(self,groupID,updateFn,tableToResetDrag,*args,**kwargs):
        ""
        super(ICGroupFrame,self).__init__(*args,**kwargs)
        self.groupID = groupID
        self.setAcceptDrops(True)
        self.updateFn = updateFn
        self.tableToResetDrag = tableToResetDrag

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
            self.updateFn(groupID = self.groupID)
            self.tableToResetDrag.resetDragEvent()
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
        self.rightFrame = QFrame(parent=self)
        
        self.scrollArea = QScrollArea(parent=self.rightFrame)
        
        self.scrollFrame = QFrame(parent=self.scrollArea) 
       
        self.scrollArea.setWidget(self.scrollFrame)
        self.scrollArea.setWidgetResizable(True)

        self.groupingEdit = createLineEdit("Enter Grouping Name",
                    tooltipText="Provide name for selected grouping.\nWhen a test requires grouping, the grouping has to be choosen by this name.",parent=self.rightFrame)
        self.addGroup = BigPlusButton(buttonSize=(25,25), tooltipStr="Add an additional group.", parent=self.rightFrame)
        self.addGroup.setDefault(False)
        self.addGroup.setAutoDefault(False)

        numericColumns = self.mC.mainFrames["data"].dataTreeView.getColumns("Numeric Floats")["Numeric Floats"]
        self.table =  ResortTableWidget(parent = self, menu = self.menu)
        self.model = ResortTableModel(parent = self.table,
                                      inputLabels=numericColumns,
                                      title="Numeric Columns")
        self.model.onlyDragNoResort = True
        self.table.setModel(self.model)

        self.okButton = ICStandardButton(itemName="Okay",default=False,autoDefault = False)
        
        self.cancelButton = ICStandardButton(itemName = "Cancel",default=False,autoDefault = False)


    def __createMenu(self):
        ""
        self.menu = createMenu(parent=self)
       # self.menu.addAction("Infer grouping",)
        self.menu.addAction("Grouping by sample name", self.groupingBySampleName)
        self.menu.addAction("Group by split string", self.groupBySplitString)
       
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

        self.rightFrame.setLayout(QVBoxLayout())
        self.rightFrame.layout().addLayout(hbox)
        self.rightFrame.layout().addWidget(self.scrollArea)
        self.scrollFrame.setLayout(QVBoxLayout())
        
        self.scrollFrame.layout().addLayout(self.groupLayout)

        
        hboxMain.addWidget(self.rightFrame)
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

    def groupBySplitString(self,event=None):
        ""
        self.deleteEmptyGroups()
        selectedColumnNames = self.findSelectedColumns()
        if selectedColumnNames is None:
            return
        selDialog = SelectionDialog(
            title="Select split string extraction props.",
            selectionNames = ["splitString","index"],
            selectionOptions={"splitString":["_","__","-","//",";","space"],"index":[str(x) for x in range(30)]},
            selectionDefaultIndex={"splitString":"_","index":"0"}
            )
        if selDialog.exec_():
            selectedItems = selDialog.savedSelection
            try:
                splitIndex = int(float(selectedItems["index"]))
            except:
                w = WarningMessage(infoText = "Index could be itnerpreted as an integer (0,1,2,3).",iconDir = self.mC.mainPath)
                w.exec_()
                return 
            if selectedItems["splitString"] == "space":
                selectedItems["splitString"] = " "
            groupNames = self.extractGroupsByColumnNames(selectedColumnNames,selectedItems["splitString"],index=splitIndex)
            if groupNames is not None:
                self.addGroupsByDataFrame(groupNames)

    def extractGroupsByColumnNames(self,selectedColumnNames,splitString = "_",index = 0,rsplit = False,maxsplit=-1):
        ""
        try:
            if rsplit:
                groupNames = pd.DataFrame([colName.rsplit(splitString,maxsplit=maxsplit)[index] for colName in selectedColumnNames.values],
                    columns=["GroupName"], index=selectedColumnNames.index)
            else:
                groupNames = pd.DataFrame([colName.split(splitString,maxsplit=maxsplit)[index] for colName in selectedColumnNames.values],
                    columns=["GroupName"], index=selectedColumnNames.index)
        except:
            w = WarningMessage(infoText = "Splitting resulted in an error. Index out of range? Indexing starts with 0.",iconDir = self.mC.mainPath)
            w.exec_()
            return
        groupNames["ColumnNames"] = selectedColumnNames.values
        return groupNames

    def deleteEmptyGroups(self):
        ""
        groupItems = list(self.groupItems.keys())
        for groupName in groupItems:
            if self.groupItems[groupName]["items"].empty:
                self.deleteGroup(groupName)

    def findSelectedColumns(self):
        ""
        selIdx = self.table.getSelectedRows()
        if len(selIdx) < 2:
            w = WarningMessage(infoText = "Please select at least two columns.",iconDir = self.mC.mainPath)
            w.exec_()
            return

        selRow = [idx.row() for idx in selIdx]
        selectedColumnNames = self.model._labels.iloc[selRow]
        return selectedColumnNames

    def groupingBySampleName(self,event=None):
        ""
        #remove existing empty groupings
        
        self.deleteEmptyGroups()

        selectedColumnNames = self.findSelectedColumns()
        if selectedColumnNames is None:
            return
        groupNames = self.extractGroupsByColumnNames(selectedColumnNames,maxsplit=1,rsplit=True)
        self.addGroupsByDataFrame(groupNames)
        
    def addGroupsByDataFrame(self,groupNames):
        ""
        warn = False
        for groupName,groupData in groupNames.groupby("GroupName",sort=False):
            if groupData.index.size > 1:
                self.addNewGroup(groupID = groupName)
                self.updateItems(groupID=groupName,labels=groupData["ColumnNames"])
            else:
                warn = True
        if warn:
            w = WarningMessage(infoText = "Some groups not created because only a single column was found.",iconDir = self.mC.mainPath)
            w.exec_()
        

    
    def addGroupArea(self,groupID = "2"):
        ""
        self.groupItems[groupID] = {}
        groupFrame = ICGroupFrame(parent = self.scrollFrame, groupID = groupID, updateFn=self.updateItems, tableToResetDrag=self.table) 
        
        try:
            groupEdit = createLineEdit("Enter group name ..","Define name of group. This name will be used to indicate comparisions.",parent=groupFrame)
            groupEdit.hide() # Hide line edit
            groupEdit.returnPressed.connect(lambda:print("return"))
            groupEdit.editingFinished.connect(lambda groupID = groupID : self.textEdited(groupID=groupID))
            groupLabel = BuddyLabel(groupEdit,parent=groupFrame) # Create our custom label, and assign myEdit as its buddy
            groupLabel.setText(groupID)
            groupLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed) # Change vertical size policy so they both match and you don't get popping when switching

            # Put them under a layout together
            hLabelLayout = QHBoxLayout()
            hLabelLayout.addWidget(groupLabel)
            hLabelLayout.addWidget(groupEdit)
            
            deleteButton = ResetButton(parent=groupFrame)
            deleteButton.clicked.connect(lambda _,groupID = groupID:self.deleteGroup(groupID))
            deleteButton.setDefault(False)
            deleteButton.setAutoDefault(False)
            
            
            hbox = QHBoxLayout()
            hbox.addLayout(hLabelLayout)
            hbox.addWidget(deleteButton)
            
            groupFrame.setLayout(QVBoxLayout())

            itemFrame = QScrollArea(parent=groupFrame)
            itemHolder =  ItemHolder(direction="V",parent=itemFrame)
            itemFrame.setMinimumHeight(200)
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


    def addNewGroup(self, event = None, groupID = None):
        ""
        groupN = len(self.groupItems)
        if groupID is None or groupID in self.groupItems:
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

    def deleteBoxTimeFromGroup(self,event=None,groupID=None,itemName=None):
        ""
        if groupID is not None and itemName is not None and groupID in self.groupItems:
            boolIdx = self.groupItems[groupID]["items"] == itemName
            if np.any(boolIdx):
                self.model.layoutAboutToBeChanged.emit()
                self.model.showHiddenLabels(self.groupItems[groupID]["items"].loc[boolIdx])
                self.model.layoutChanged.emit()
                self.model.completeDataChanged()
            self.groupItems[groupID]["itemHolder"].deleteItem(self.sender())
            self.groupItems[groupID]["items"] = self.groupItems[groupID]["items"].loc[~boolIdx]
            

    def saveGrouping(self,event=None):
        ""
        groupingName = self.groupingEdit.text()
        
        if  groupingName == "":
            w = WarningMessage(infoText = "No name for Grouping found.",iconDir = self.mC.mainPath)
            w.exec_()
            return
        
        elif  groupingName == "None":
            w = WarningMessage(infoText = "None is not allowed as a group name.",iconDir = self.mC.mainPath)
            w.exec_()
            return


        elif self.mC.grouping.nameExists(groupingName):
            w = WarningMessage(infoText = "The name of grouping exists already.",iconDir = self.mC.mainPath)
            w.exec_()
            return
            
        elif any(self.groupItems[groupID]["items"].size < 2 for groupID in self.groupItems.keys()):
            w = WarningMessage(infoText = "One or more groups contain only a single item. Either remove group or add items.",iconDir = self.mC.mainPath)
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

    def updateItems(self,event=None,groupID = None, labels = None):
        ""
        if labels is None:
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
                bItem.setContextMenuPolicy(Qt.CustomContextMenu)
                bItem.customContextMenuRequested.connect(lambda _,groupID = groupID, itemName = l: self.deleteBoxTimeFromGroup(groupID=groupID,itemName=itemName))

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



