from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 


#ui utils
from ...utils import INSTANT_CLUE_BLUE, TABLE_ODD_ROW_COLOR, WIDGET_HOVER_COLOR, getHoverColor, getStdTextColor, getMessageProps, createMenu, getStandardFont, getDefaultWidgetBGColor

from ..warnMessage import AskQuestionMessage, AskStringMessage

#external imports
import pandas as pd 
import numpy as np
from collections import OrderedDict

contextMenuData = OrderedDict([
            ("deleteRows",{"label":"Delete Row(s)","fn":"deleteRows"}),
            ("copyRows",{"label":"Copy Row(s)","fn":"copyRows"}),
            ("copyData",{"label":"To Clipboard","fn":"copyDf"}),
            ("addDataFrame",{"label":"Add as new data frame","fn":"addDf"})

        ])

class PandaTable(QTableView):
    
    def __init__(self, parent=None, mainController = None,  cornerButton = True, hideMenu = False, rightClickOnHeaderCallBack = None, onHoverCallback = None, forwardSelectionToGraph =True):
        super(PandaTable, self).__init__(parent)
        self.highlightRow = None
        self.forwardSelectionToGraph = forwardSelectionToGraph 
        self.setMouseTracking(True)
        self.setShowGrid(True)
        self.shiftHold = False
        self.rightClickOnHeaderCallBack = rightClickOnHeaderCallBack
        self.onHoverCallback = onHoverCallback

        self.mC = mainController
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.verticalHeader().setDefaultSectionSize(15)
        self.horizontalHeader().sectionClicked.connect(self.headerClicked)
        self.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self.handleHeaderRightClick)
        if not hideMenu:
            self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.customContextMenuRequested.connect( self.showHeaderMenu )
        self.setItemDelegate(EditorDelegate(self))
        self.setCornerButtonEnabled(cornerButton)
        
        self.setStyleSheet("""
            QTableView::item:selected {
                background: #E4DED4;
                color: black;
                }
            QTableView QTableCornerButton::section {
                background-color:#4C626F;
                }
            """
            )

    def showHeaderMenu( self, point ):
        """ """  
        menu = createMenu(parent=self)     

        for k, v in contextMenuData.items():
            action = menu.addAction(v["label"])
            fn = getattr(self,v["fn"])
            action.triggered.connect(fn)

        menu.exec(QCursor.pos()+QPoint(3,3))

    def handleHeaderRightClick(self, point):
        ""
        if self.rightClickOnHeaderCallBack is not None:
            idxClicked = self.horizontalHeader().logicalIndexAt(point)
            self.rightClickOnHeaderCallBack(idxClicked)

    def headerClicked(self,columnIndex):
        ""
        if hasattr(self.model(),"sortByColumnIndex"):
            self.model().sortByColumnIndex(columnIndex)

    def getSelectedRows(self):
        ""
        return self.selectionModel().selectedRows()

    def copyRows(self,e=None,selectedRows = None):
        """ """
        try:
            data = None

            if selectedRows is None:
                selectedRows = self.getSelectedRows()
                indexes = [idx.row() for idx in selectedRows if idx.isValid()]
                data = self.model().getSelectedRows(indexes)
            
            elif isinstance(selectedRows,list):
                indexes = [idx.row() for idx in selectedRows if idx.isValid()]
                data = self.model().getSelectedRows(indexes)
            else:
                if isinstance(selectedRows,str):
                    if selectedRows == "all":
                        data = self.model().getSelectedRows(selectedRows)

            if data is not None:

                funcProp = {"key":"copyDataFrameToClipboard",
                            "kwargs":{"data":data}}
        
                self.mC.sendRequestToThread(funcProp)
                        
            else:
                self.parent().sendMessage(getMessageProps("Error","No selected rows.."))
        except Exception as e:
            print(e)


    def keyPressEvent(self, e: QKeyEvent) -> None:
        "Overwrite key press event to copy rows by defualt not the actual value"
        if e.matches(QKeySequence.StandardKey.Copy):
            selectedRows = self.getSelectedRows()
            if len(selectedRows) >0:
                self.copyRows(selectedRows=selectedRows)
        else:
            super().keyPressEvent(e)

    def addDf(self,e=None):
        ""
        
        dialog = AskStringMessage(q="Please provide a file name")
        if dialog.exec():
            fileName = dialog.text
            funcKey = "data::addDataFrame"
            funcKwargs = {"dataFrame":self.model().getCurrentData(),"fileName":fileName}
            self.mC.sendRequestToThread({"key":funcKey,"kwargs":funcKwargs}) 

    def copyDf(self,e=None):
        ""
        self.copyRows(selectedRows="all")
        
    def deleteRows(self,e=None):
        """ """
        selectedRows = self.getSelectedRows()
        self.model().layoutAboutToBeChanged.emit()
        self.model().dropRows(selectedRows)
        self.model().layoutChanged.emit()
        self.model().completeDataChanged()
        try:
            self.parent().sendMessage(getMessageProps("Deleted","Selected rows deleted."))
        except Exception as e:
            print(e)

    def mouseReleaseEvent(self,event):
        ""
        if isinstance(self.model(), SelectablePandaModel):
            
            eventIndex = self.mouseEventToIndex(event)
            self.model().setCheckState(eventIndex)
            self.model().completeDataChanged() 
            
        else:
            super(QTableView,self).mouseReleaseEvent(event)

    def mouseMoveEvent(self,event):
        ""
        if event.buttons() == Qt.MouseButton.LeftButton:         
            super(QTableView,self).mouseMoveEvent(event)
            
        else:
            eventIndex = self.mouseEventToIndex(event)
            if self.highlightRow != eventIndex.row():
                self.highlightRow  = eventIndex.row() 
                if self.onHoverCallback is not None and self.highlightRow > -1:
                    dataIndex = self.model().getRowDataIndexByTableIndex(eventIndex)
                    self.onHoverCallback(self.highlightRow, dataIndex)
                self.model().completeDataChanged() 

    def mouseEventToIndex(self,event):
        "Converts mouse event on table to tableIndex"
        row = self.rowAt(event.pos().y())
        column = self.columnAt(event.pos().x())
        return self.model().index(row,column)

    def leaveEvent(self,event):
        ""
        if hasattr(self,"highlightRow"):
            self.highlightRow = None  
            self.model().completeDataChanged()

    def selectionChanged(self,selected,deselected):
        "Mark Datapoints in Selection"

        selectedRows = self.getSelectedRows()
        dataIndex = np.array([self.model().getRowDataIndexByTableIndex(idx) for idx in selectedRows])
        self.markSelectionInDataAndSetLabel(dataIndex)
        self.model().completeDataChanged() 
        
    def markSelectionInDataAndSetLabel(self,dataIndex):
        ""
        if self.mC is not None and hasattr(self.mC,"getGraph") and self.forwardSelectionToGraph:
            exists,graph = self.mC.getGraph()
            if exists:
                graph.setHoverData(dataIndex)
        if hasattr(self.parent(),"setSelectedRowsLabel"):
            self.parent().setSelectedRowsLabel(int(dataIndex.size))

    # def currentChanged(self,selected,deselected):
    #     ""
    #     print(selected)
    #     dataIndex = np.array([self.model().getRowDataIndexByTableIndex(idx) for idx in selected])
    #     self.markSelectionInDataAndSetLabel(dataIndex)

class EditorDelegate(QStyledItemDelegate):

    def paint(self, painter, option, index):
            
        QStyledItemDelegate.paint(self, painter, option, index)

    def createEditor(self, parent, option, index):
        
        return super(EditorDelegate, self).createEditor(parent, option, index)
       
    def setEditorData(self,editor,index):
        editor.setAutoFillBackground(True)
        editor.setText(self.parent().model().data(index,Qt.ItemDataRole.DisplayRole))
     

class PandaModel(QAbstractTableModel):
    
    def __init__(self, df = pd.DataFrame(), parent=None):
        super(PandaModel, self).__init__(parent)
        self.initData(df)
        self.sortMemory = dict()
        self.highlightBackgroundHeaderColors = dict() 
        

    def initData(self,df):
        ""
        self.df = df.copy()
        self.__df = df.copy() 
        self.filters = pd.DataFrame(index=df.index)

    def rowCount(self, parent=QModelIndex()):
        
        return self.df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        
        return self.df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole): 
        ""
        columnIndex = index.column()
        rowIndex = index.row()
        if not index.isValid(): 
            return QVariant()
        elif role == Qt.ItemDataRole.DisplayRole:
            return str(self.df.iloc[rowIndex ,columnIndex])
        
        elif role == Qt.ItemDataRole.FontRole:
            return self.getFont()
        
        elif role == Qt.ItemDataRole.BackgroundRole:
           # print(self.df.iloc[index.row(),index.column()] == "+")
            if self.df.iloc[rowIndex,columnIndex] == "+":
                    return QBrush(QColor(INSTANT_CLUE_BLUE))
            elif columnIndex in self.highlightBackgroundHeaderColors:
                return QBrush(QColor(self.highlightBackgroundHeaderColors[columnIndex]))
            return QBrush(QColor("white"))
        elif role == Qt.ItemDataRole.ForegroundRole:

            return QBrush(QColor(getStdTextColor()))
            
    def setData(self,index,value,role):
        ""
        if role == Qt.ItemDataRole.EditRole:
        
            self.updateData(value,index)
            
            self.dataChanged.emit(index,index)
            return True

    def sortByColumnIndex(self,columnIndex):
        ""
        columnName = self.getColumnNameByColumnIndex(columnIndex)
        if columnName is None: return
        if columnIndex not in self.sortMemory:
                self.df = self.df.sort_values(by = columnName)
                self.sortMemory[columnIndex] = "ascending"
        elif self.sortMemory[columnIndex] == "ascending":
            self.df = self.df.sort_values(by = columnName, ascending=False)
            self.sortMemory[columnIndex] = "descending"
        elif self.sortMemory[columnIndex] == "descending":
            self.df = self.df.sort_index()
            del self.sortMemory[columnIndex]

        self.completeDataChanged()
        self.completeHeaderDataChanged()
        
    
    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return str(self.df.columns[col])
        elif orientation == Qt.Orientation.Vertical and role == Qt.ItemDataRole.DisplayRole:
            return str(self.df.index[col])
        elif role == Qt.ItemDataRole.BackgroundRole:
            if orientation == Qt.Orientation.Horizontal:
                if col in self.highlightBackgroundHeaderColors:
                    return QBrush(QColor(self.highlightBackgroundHeaderColors[col]))
                return QBrush(QColor("lightgrey"))
            else:
                return QBrush(QColor("lightgrey"))

        elif role == Qt.ItemDataRole.ForegroundRole:

            return QBrush(QColor(getStdTextColor()))

        elif role == Qt.ItemDataRole.FontRole:
            font = getStandardFont()
            return font
        return None

    def setNumericFilterByColumnIndex(self,columnIndex,minValue,maxValue,tagID):
        ""
        columnName = self.getColumnNameByColumnIndex(columnIndex)
        if columnName is not None:
            boolIdx = self.__df[columnName].between(minValue,maxValue)
            boolIdx.name = tagID
            self.filters = self.filters.join(boolIdx)
            self.updateFilter() 

    def setFilterByColumnIndex(self,columnIndex,filterString,tagID,exactMatch=False):
        ""
        columnName = self.getColumnNameByColumnIndex(columnIndex)
        if columnName is not None:
            if exactMatch:
                boolIdx = self.__df[columnName] == filterString
            else:
                boolIdx = self.__df[columnName].str.contains(filterString,regex=False).sort_values(ascending=False)
            boolIdx.name = tagID
            self.filters = self.filters.join(boolIdx)
            self.updateFilter() 

    def updateFilter(self):
        "Update data frame based on filtering"
        if self.filters.columns.size == 0:
            self.df = self.__df
        else:
            if self.parent().mC.config.getParam("source.data.filter.logical.op") == "and":
                boolIdx = self.filters.index[self.filters.sum(axis=1).values == self.filters.columns.size]
            else:
                boolIdx = self.filters.index[self.filters.sum(axis=1).values > 0]
            self.df = self.__df.loc[boolIdx]

    def removeFilter(self,tagID):
        "Drop filter and updates the dataframe."
        if tagID in self.filters.columns:
            self.filters = self.filters.drop(tagID, axis="columns")
            self.updateFilter()
        
    def getCurrentShape(self):
        ""
        return self.df.shape
    
    def setHighlightBackground(self,columnIndex,hexColor):
        ""
        self.highlightBackgroundHeaderColors[columnIndex] = hexColor
    
    def getHighlightBackgroundcolor(self,columnIndex):
        ""
        if columnIndex in self.highlightBackgroundHeaderColors:
            return self.highlightBackgroundHeaderColors[columnIndex]

    def getNumberOfHighlightedBackgrounds(self):
        ""
        return len(self.highlightBackgroundHeaderColors)
    
    def removeHighlightBackground(self,columnIndex):
        if columnIndex in self.highlightBackgroundHeaderColors:
            del self.highlightBackgroundHeaderColors[columnIndex]

    def getCurrentData(self):
        "Returns data even if filtering is applied."
        return self.df.copy()


    def getColumnNameByColumnIndex(self,columnIndex):
        ""
        if columnIndex < self.df.columns.size:
            return self.df.columns[columnIndex]

    def getColumnNameByTableIndex(self,index):
        ""
        columnName = self.df.columns[index.column()]
        return columnName

    def getDataTypeByColumnIndex(self,columnIndex):
        ""
        columnName = self.getColumnNameByColumnIndex(columnIndex)
        if columnName is not None:
            return self.df[columnName].dtype

    def getFont(self):
        ""
        return getStandardFont()

    def getRowDataIndexByTableIndex(self,index):
        ""
        return self.df.index[index.row()]

    def updateData(self,value,index):
        """
        Updates data in df. Checks if data type is matched. 
        Changing the datatype is not prevented.
        """
        columnName = self.df.columns[index.column()]
        dtype = self.df.dtypes.loc[columnName]
        try:
            if dtype == np.float64:
                v = float(value)
            elif dtype == np.int64:
                v = int(value)
            else:
                v = value
        except:
            
            if hasattr(self.parent(),"mC") and hasattr(self.parent().mC,"sendToWarningDialog"):
                self.parent().mC.sendToWarningDialog(
                    infoText="Input could not be interpretet as required data type ({}).".format(dtype),parent=self.parent()
                )
            return

        self.df.iloc[index.row(),index.column()] = v

    def getSelectedRows(self,selectedRows="all"):
        ""
        
        if isinstance(selectedRows,list):
            return self.df.iloc[selectedRows,:]
        elif isinstance(selectedRows,str):
            if selectedRows == "all":
                return self.df

    def dropRows(self,selectedRows):
        ""
        if isinstance(selectedRows,list):
            dataIndex = [self.getRowDataIndexByTableIndex(idx) for idx in selectedRows]
            self.df = self.df.drop(dataIndex)
        else:
            raise ValueError("selected rows must be a list!")

    def completeDataChanged(self):
        ""
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1))

    def completeHeaderDataChanged(self):
        ""
        self.headerDataChanged.emit(Qt.Orientation.Vertical,0,self.rowCount()-1)
    
    def updateDataFrame(self,df):
        ""
        self.layoutAboutToBeChanged.emit()
        self.initData(df)
        self.completeDataChanged()
        self.layoutChanged.emit()

    def flags(self, index):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable  | Qt.ItemFlag.ItemIsSelectable




class SelectablePandaModel(PandaModel):

    def __init__(self, singleSelection = False, *args, **kwargs):

        super(SelectablePandaModel,self).__init__(*args, **kwargs)
        
        self.__df = self.df.copy()
        self.setCheckedSeries()
        self.lastClicked = None
        self.singleSelection = singleSelection

    def initData(self,df):
        ""
        self.__df = df.copy()
        self.df = df.copy()
        self.setCheckedSeries() 
        self.lastClicked = None


    def data(self, index, role=Qt.ItemDataRole.DisplayRole): 
        ""
        # use default display and font role for consistedn look
        if not index.isValid(): 
            return QVariant()
        elif role == Qt.ItemDataRole.DisplayRole:
            return str(self.df.iloc[index.row(),index.column()])
        
        elif role == Qt.ItemDataRole.FontRole:
            return self.getFont()
        
        elif role == Qt.ItemDataRole.CheckStateRole:
            if index.column() != 0:
                return QVariant()
            elif self.getCheckStateByTableIndex(index):
                return Qt.CheckState.Checked
            else:
                return Qt.CheckState.Unchecked
            
        elif role == Qt.ItemDataRole.ForegroundRole:
            return QBrush(QColor(getStdTextColor()))
        elif role == Qt.ItemDataRole.BackgroundRole:

            if self.getCheckStateByTableIndex(index) or \
                (self.parent() is not None and self.parent().highlightRow is not None and self.parent().highlightRow == index.row()):

                return QBrush(QColor(getHoverColor()))
            else:
                
                return QBrush(QColor(getDefaultWidgetBGColor()))
       
    def setData(self,index,value,role):

        if role != Qt.ItemDataRole.CheckStateRole:
            #use default setData
            return super().setData(index,role)

        if role == Qt.ItemDataRole.CheckStateRole:
            #this model uses first column to check complet row
            indexBottomRight = self.index(index.row(),self.columnCount())
            self.setCheckState(index)
            
            self.dataChanged.emit(index,indexBottomRight)
            return True

    def getCheckStateByDataIndex(self,dataIndex):
        "Returns current check state by data row index"
        return self.checkedLabels.loc[dataIndex]

    def getCheckStateByTableIndex(self,tableIndex):
        "Returns current check state by table index"
        dataIndex = self.getRowDataIndexByTableIndex(tableIndex)
        return self.getCheckStateByDataIndex(dataIndex)

    def getCheckedData(self):
        "Returns checked values"

        
        return self.__df.loc[self.checkedLabels.values,:]

    def setCheckState(self,tableIndex):
        "Sets check state by table index."
       
        try:
            if self.singleSelection:
                self.setCheckedSeries()
            dataIndex = self.getRowDataIndexByTableIndex(tableIndex)
            newState = not self.checkedLabels.loc[dataIndex]
            if newState and self.lastClicked is None:
                self.lastClicked = tableIndex
           
            elif hasattr(self.parent(),"shiftHold") and not self.parent().shiftHold:
                self.lastClicked = None
            
            if hasattr(self.parent(),"shiftHold") and self.parent().shiftHold and self.lastClicked is not None:
                if tableIndex.row() > self.lastClicked.row():
                    dataIndices = self.checkedLabels.index[self.lastClicked.row():tableIndex.row()+1]
                else:
                    dataIndices = self.checkedLabels.index[tableIndex.row():self.lastClicked.row()]
                if all(self.getCheckStateByDataIndex(dataIndex) for dataIndex in dataIndices):
                    self.setCheckStateByDataIndex(dataIndices, state = 0)
                else:
                    self.setCheckStateByDataIndex(dataIndices)
                self.lastClicked = None
            else:
               
                self.checkedLabels.loc[dataIndex] = newState
            return newState
        except Exception as e:
            print(e)
            return False

    def setCheckStateByDataIndex(self,dataIndex, state = 1):
        ""
        matchedIdx = dataIndex.intersection(self.checkedLabels.index)
        self.checkedLabels.loc[matchedIdx] = state

    def setCheckedSeries(self):
        ""
        if self.rowCount() == 0:
            self.checkedLabels = pd.Series(dtype=bool)
        else:
            self.checkedLabels = pd.Series(np.zeros(shape=self.__df.index.size), index=self.__df.index)
            self.checkedLabels = self.checkedLabels.astype(bool)

    def setAllCheckStates(self,newState):
        ""
        if newState:
            self.checkedLabels = pd.Series(np.ones(shape=self.__df.index.size), index=self.__df.index)
            self.checkedLabels = self.checkedLabels.astype(bool)
        else:
            self.setCheckedSeries()

    def flags(self,index):
        if index.column() == 0:
            return Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled #|  Qt.ItemIsEnabled |  
        else:
            return Qt.ItemFlag.ItemIsEnabled #| Qt.ItemIsSelectable

    def updateDataByBool(self, boolIndicator,resetData):
        ""
        if resetData:
            self.df = self.__df.copy()
        if self.df.index.size == boolIndicator.size:
            self.df = self.df[boolIndicator]
            self.completeDataChanged()



class MultiColumnSelectablePandaModel(PandaModel):

    def __init__(self, selectionCallBack = None, *args, **kwargs):

        super(MultiColumnSelectablePandaModel,self).__init__(*args, **kwargs)
        self.setCheckedSeries()
        self.selectionCallBack = selectionCallBack
    
    def initData(self,df):
        self.df = df
        self.__df = self.df.copy()
        self.setCheckedSeries()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole): 
        ""
        # use default display and font role for consistedn look
        if not index.isValid(): 
            return QVariant()
        elif role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(self.df.iloc[index.row(),index.column()]):
                return QVariant()
            else:
                return str(self.df.iloc[index.row(),index.column()])
        
        elif role == Qt.FontRole:
            return self.getFont()
        
        elif role == Qt.CheckStateRole:
            if pd.isna(self.df.iloc[index.row(),index.column()]):
                return QVariant()
            if self.getCheckStateByTableIndex(index):
                return Qt.Checked
            else:
                return Qt.Unchecked
        elif role == Qt.BackgroundRole:
            if self.getCheckStateByTableIndex(index) or \
                (self.parent() is not None and self.parent().highlightRow is not None and self.parent().highlightRow == index.row()):

                return QBrush(QColor(getHoverColor()))
            else:
                return QBrush(QColor("white"))
       
    def setData(self,index,value,role):

        if role != Qt.CheckStateRole:
            #use default setData
            return super().setData(index,role)

        if role == Qt.CheckStateRole:
            #this model uses first column to check complet row
            indexBottomRight = self.index(index.row(),self.columnCount())
            _, dataIndex, columnIndex = self.setCheckState(index)
            if self.selectionCallBack is not None:
                self.selectionCallBack(dataIndex,columnIndex)
            self.dataChanged.emit(index,indexBottomRight)
            return True

    def getCheckStateByDataIndex(self,dataIndex,columnName):
        "Returns current check state by data row index"
        if columnName not in self.checkedLabels.columns:
            return False
        return self.checkedLabels[columnName].loc[dataIndex] == 1

    def getCheckStateByTableIndex(self,tableIndex):
        "Returns current check state by table index"
        dataIndex = self.getRowDataIndexByTableIndex(tableIndex)
        columnName = self.getColumnNameByTableIndex(tableIndex)
       
        return self.getCheckStateByDataIndex(dataIndex,columnName)

    def getCheckedData(self):
        "Returns checked values"
        checkedValues = OrderedDict() 
        for columnName in self.df.columns:
            boolInd = self.checkedLabels[columnName] == 1
            if np.any(boolInd):
                checkedValues[columnName] = self.__df.loc[boolInd,columnName].values.flatten()
        return checkedValues

    def setCheckState(self,tableIndex):
        "Sets check state by table index."
        dataIndex = self.getRowDataIndexByTableIndex(tableIndex)
        columnName = self.getColumnNameByTableIndex(tableIndex)
        newState = not self.checkedLabels[columnName].loc[dataIndex]
        self.checkedLabels.loc[dataIndex,columnName] = newState
        return newState, dataIndex, tableIndex.column()

    def setCheckStateByDataIndex(self,dataIndex):
        ""
        matchedIdx = dataIndex.intersection(self.checkedLabels.index)
        self.checkedLabels.loc[matchedIdx] = 1 

    def setCheckedSeries(self):
        ""
        if self.rowCount() == 0:
            self.checkedLabels = pd.DataFrame()
        else:
            self.checkedLabels = pd.DataFrame(np.zeros(shape=(self.rowCount(),self.columnCount())), index=self.__df.index, columns=self.__df.columns)
            self.checkedLabels = self.checkedLabels.astype(bool)
        
    def setAllCheckStates(self,newState):
        ""
        if newState:
            self.checkedLabels = pd.DataFrame(np.ones(shape=(self.rowCount(),self.columnCount())), index=self.__df.index, columns=self.__df.columns)
            self.checkedLabels = self.checkedLabels.astype(bool)
        else:
            self.setCheckedSeries()

    def setCheckStateByColumnNameAndIndex(self,columnNameIndexMapper,selecteded = True):
        ""
        if isinstance(columnNameIndexMapper,dict):
            for columnName, idx in columnNameIndexMapper.items():
                if columnName in self.checkedLabels.columns:
                    self.checkedLabels.loc[idx,columnName] = selecteded

    def flags(self,index):
        if pd.isna(self.df.iloc[index.row(),index.column()]):
            return Qt.ItemFlag.ItemIsEnabled
        else:
            return Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled #|  Qt.ItemIsEnabled |  
        

    def updateDataByBool(self, boolIndicator,resetData):
        ""
        if resetData:
            self.df = self.__df
        if self.df.index.size == boolIndicator.size:
            self.df = self.df[boolIndicator]
            self.completeDataChanged()