from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

#ui utils
from ...utils import TABLE_ODD_ROW_COLOR, WIDGET_HOVER_COLOR, HOVER_COLOR, createTitleLabel, getMessageProps, createMenu, createSubMenu, getStandardFont
from ..warnMessage import AskQuestionMessage

#external imports
import pandas as pd 
import numpy as np
from collections import OrderedDict

contextMenuData = OrderedDict([
            ("deleteRows",{"label":"Delete Row(s)","fn":"deleteRows"}),
            ("copyRows",{"label":"Copy Row(s)","fn":"copyRows"}),
            ("copyData",{"label":"Copy Data Frame","fn":"copyDf"})
        ])



class PandaTable(QTableView):
    
    def __init__(self, parent=None, mainController = None,  cornerButton = True):
        super(PandaTable, self).__init__(parent)
        self.highlightRow = None
        self.setMouseTracking(True)
        self.setShowGrid(True)

        self.mC = mainController
        
        self.verticalHeader().setDefaultSectionSize(15)
        self.verticalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.verticalHeader().customContextMenuRequested.connect( self.showHeaderMenu )
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
        menus = createSubMenu(menu,subMenus=["File .. "])
        for k, v in contextMenuData.items():
            action = menus["File .. "].addAction(v["label"])
            fn = getattr(self,v["fn"])
            action.triggered.connect(fn)

        menus["main"].exec_(QCursor.pos())

  
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
                
            else:
                if isinstance(selectedRows,str):
                    if selectedRows == "all":
                        data = self.model().getSelectedRows(selectedRows)

            if data is not None:

                funcProp = {"key":"copyDataFrameToClipboard",
                            "kwargs":{"data":data}}
        
                self.parent().sendToThread(funcProp)
                        
            else:
                self.parent().sendMessage(getMessageProps("Error","No selected rows.."))
        except Exception as e:
            print(e)


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
        if event.buttons() == Qt.LeftButton:         
            super(QTableView,self).mouseMoveEvent(event)
            
        else:
            eventIndex = self.mouseEventToIndex(event)
            self.highlightRow  = eventIndex.row() 
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
        selectedRows = np.unique([idx.row() for idx in selected.indexes()])
        if self.mC is not None and hasattr(self.mC,"getGraph"):
            exists,graph = self.mC.getGraph()
            if exists:
                selectedRows = self.getSelectedRows()
                dataIndex = np.array([self.model().getRowDataIndexByTableIndex(idx) for idx in selectedRows])
                graph.setHoverData(dataIndex)

class EditorDelegate(QStyledItemDelegate):

    def paint(self, painter, option, index):
            
        QStyledItemDelegate.paint(self, painter, option, index)

    def createEditor(self, parent, option, index):
        
        return super(EditorDelegate, self).createEditor(parent, option, index)
       
    def setEditorData(self,editor,index):
        editor.setAutoFillBackground(True)
        editor.setText(self.parent().model().data(index,Qt.DisplayRole))
     

class PandaModel(QAbstractTableModel):
    
    def __init__(self, df = pd.DataFrame(), parent=None):
        super(PandaModel, self).__init__(parent)
        self.initData(df)

    def initData(self,df):
        self.df = df.copy()

    def rowCount(self, parent=QModelIndex()):
        
        return self.df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        
        return self.df.shape[1]

    def data(self, index, role=Qt.DisplayRole): 
        ""
        if not index.isValid(): 
            return QVariant()
        elif role == Qt.DisplayRole:
            return str(self.df.iloc[index.row(),index.column()])
        
        elif role == Qt.FontRole:
            return self.getFont()
        
        elif role == Qt.BackgroundRole:
            
            return QBrush(QColor("white"))#TABLE_ODD_ROW_COLOR
            
    def setData(self,index,value,role):
        ""
        if role == Qt.EditRole:
        
            self.updateData(value,index)
            
            self.dataChanged.emit(index,index)
            return True
            
    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self.df.columns[col])
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(self.df.index[col])
        elif role == Qt.BackgroundRole:
            if orientation == Qt.Horizontal:

                return QBrush(QColor("lightgrey"))
            else:
                return QBrush(QColor("lightgrey"))

        elif role == Qt.ForegroundRole:

            return QBrush(QColor("black"))

        elif role == Qt.FontRole:
            font = getStandardFont()
            return font
        return None
    
    def getColumnNameByTableIndex(self,index):
        ""
        columnName = self.df.columns[index.column()]
        return columnName

    def getFont(self):
        ""
        return getStandardFont()

    def getRowDataIndexByTableIndex(self,index):
        ""
        return self.df.index[index.row()]

    def updateData(self,value,index):
        ""
        self.df.iloc[index.row(),index.column()] = value

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
    
    def updateDataFrame(self,df):
        ""
        self.initData(df)
        self.completeDataChanged()

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsEditable  | Qt.ItemIsSelectable




class SelectablePandaModel(PandaModel):

    def __init__(self, *args, **kwargs):

        super(SelectablePandaModel,self).__init__(*args, **kwargs)
        self.setCheckedSeries()
        self._df = self.df.copy()

    def data(self, index, role=Qt.DisplayRole): 
        ""
        # use default display and font role for consistedn look
        if not index.isValid(): 
            return QVariant()
        elif role == Qt.DisplayRole:
            return str(self.df.iloc[index.row(),index.column()])
        
        elif role == Qt.FontRole:
            return self.getFont()
        
        elif role == Qt.CheckStateRole:
            if index.column() != 0:
                return QVariant()
            elif self.getCheckStateByTableIndex(index):
                return Qt.Checked
            else:
                return Qt.Unchecked
        elif role == Qt.BackgroundRole:
            if self.getCheckStateByTableIndex(index) or \
                (self.parent() is not None and self.parent().highlightRow is not None and self.parent().highlightRow == index.row()):

                return QBrush(QColor(HOVER_COLOR))
            else:
                return QBrush(QColor("white"))
       
    def setData(self,index,value,role):

        if role != Qt.CheckStateRole:
            #use default setData
            return super().setData(index,role)

        if role == Qt.CheckStateRole:
            #this model uses first column to check complet row
            indexBottomRight = self.index(index.row(),self.columnCount())
            self.setCheckState(index)
            self.dataChanged.emit(index,indexBottomRight)
            return True

    def getCheckStateByDataIndex(self,dataIndex):
        "Returns current check state by data row index"
        return self.checkedLabels.loc[dataIndex] == 1

    def getCheckStateByTableIndex(self,tableIndex):
        "Returns current check state by table index"
        dataIndex = self.getRowDataIndexByTableIndex(tableIndex)
        return self.getCheckStateByDataIndex(dataIndex)

    def getCheckedData(self):
        "Returns checked values"
        boolInd = self.checkedLabels == 1
        return self.df.loc[boolInd,:]

    def setCheckState(self,tableIndex):
        "Sets check state by table index."
        dataIndex = self.getRowDataIndexByTableIndex(tableIndex)
        newState = not self.checkedLabels.loc[dataIndex]
        self.checkedLabels.loc[dataIndex] = newState
        return newState

    def setCheckStateByDataIndex(self,dataIndex):
        ""
        matchedIdx = dataIndex.intersection(self.checkedLabels.index)
        self.checkedLabels.loc[matchedIdx] = 1 

    def setCheckedSeries(self):
        ""
        if self.rowCount() == 0:
            self.checkedLabels = pd.Series()
        else:
            self.checkedLabels = pd.Series(np.zeros(shape=self.rowCount()), index=self.df.index)
            self.checkedLabels = self.checkedLabels.astype(bool)

    def setAllCheckStates(self,newState):
        ""
        if newState:
            self.checkedLabels = pd.Series(np.ones(shape=self.rowCount()), index=self.df.index)
            self.checkedLabels = self.checkedLabels.astype(bool)
        else:
            self.setCheckedSeries()

        

    def flags(self,index):
        if index.column() == 0:
            return Qt.ItemIsUserCheckable | Qt.ItemIsEnabled #|  Qt.ItemIsEnabled |  
        else:
            return Qt.ItemIsEnabled #| Qt.ItemIsSelectable

    def updateDataByBool(self, boolIndicator,resetData):
        ""
        if resetData:
            self.df = self._df
        if self.df.index.size == boolIndicator.size:
            self.df = self.df[boolIndicator]
            self.completeDataChanged()



class MultiColumnSelectablePandaModel(PandaModel):

    def __init__(self, *args, **kwargs):

        super(MultiColumnSelectablePandaModel,self).__init__(*args, **kwargs)
        self.setCheckedSeries()
    
    def initData(self,df):
        self.df = df
        self._df = self.df.copy()

    def data(self, index, role=Qt.DisplayRole): 
        ""
        # use default display and font role for consistedn look
        if not index.isValid(): 
            return QVariant()
        elif role == Qt.DisplayRole:
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

                return QBrush(QColor(HOVER_COLOR))
            else:
                return QBrush(QColor("white"))
       
    def setData(self,index,value,role):

        if role != Qt.CheckStateRole:
            #use default setData
            return super().setData(index,role)

        if role == Qt.CheckStateRole:
            #this model uses first column to check complet row
            indexBottomRight = self.index(index.row(),self.columnCount())
            self.setCheckState(index)
            self.dataChanged.emit(index,indexBottomRight)
            return True

    def getCheckStateByDataIndex(self,dataIndex,columnName):
        "Returns current check state by data row index"
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
                checkedValues[columnName] = self._df.loc[boolInd,columnName].values.flatten()
        return checkedValues

    def setCheckState(self,tableIndex):
        "Sets check state by table index."
        dataIndex = self.getRowDataIndexByTableIndex(tableIndex)
        columnName = self.getColumnNameByTableIndex(tableIndex)
        newState = not self.checkedLabels[columnName].loc[dataIndex]
        self.checkedLabels.loc[dataIndex,columnName] = newState
        return newState

    def setCheckStateByDataIndex(self,dataIndex):
        ""
        matchedIdx = dataIndex.intersection(self.checkedLabels.index)
        self.checkedLabels.loc[matchedIdx] = 1 

    def setCheckedSeries(self):
        ""
        if self.rowCount() == 0:
            self.checkedLabels = pd.DataFrame()
        else:
            self.checkedLabels = pd.DataFrame(np.zeros(shape=(self.rowCount(),self.columnCount())), index=self._df.index, columns=self._df.columns)
            self.checkedLabels = self.checkedLabels.astype(bool)

    def setAllCheckStates(self,newState):
        ""
        if newState:
            self.checkedLabels = pd.DataFrame(np.ones(shape=(self.rowCount(),self.columnCount())), index=self._df.index, columns=self._df.columns)
            self.checkedLabels = self.checkedLabels.astype(bool)
        else:
            self.setCheckedSeries()


    def flags(self,index):
        if pd.isna(self.df.iloc[index.row(),index.column()]):
            return Qt.ItemIsEnabled
        else:
            return Qt.ItemIsUserCheckable | Qt.ItemIsEnabled #|  Qt.ItemIsEnabled |  
        

    def updateDataByBool(self, boolIndicator,resetData):
        ""
        if resetData:
            self.df = self._df
        if self.df.index.size == boolIndicator.size:
            self.df = self.df[boolIndicator]
            self.completeDataChanged()