from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.pyplot import title 
from ..custom.buttonDesigns import ICStandardButton, LabelLikeButton
from ..custom.warnMessage import AskStringMessage
from .ICDSelectItems import ICDSelectItems
from ..utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR, createCombobox

import requests
import pandas as pd 

class ICShareGraph(QDialog):
    ""
    def __init__(self, mainController, title = "Share Graph.", *args,**kwargs):
        ""
        super(ICShareGraph, self).__init__(*args,**kwargs)
        self.title = title
        self.mC = mainController
        self.sharedCharts = self.mC.webAppComm.getChartsByAppID()
        self.__controls()
        self.__layout()
        self.__connectEvents()

    def __controls(self):
        ""
        self.title = createTitleLabel(self.title)

        self.updateGraph = createCombobox(self, items= ["New Entry"] + self.sharedCharts["title"].values.tolist())

        self.titleEdit = createLineEdit("Project/Graph Title","Title of the project.")
        self.subTitleEdit = createLineEdit("Project/Graph Subtitle","Subtitle of the project. Will be displayed below header.")
        self.keywordsEdit = createLineEdit("Keywords separated by comma (,)","Keywords will be displayed and allow you to identify graphs in your collections.")
        self.doiEdit = createLineEdit("DOI (optional)","Reference DOI if graph is for a specific publication. Will be clickable.")
        self.infoTextEdit = QTextEdit()
        self.infoTextEdit.setPlaceholderText("Enter information of the experiment / project ..")
        self.markDownCheckbox = QCheckBox("Text is markdown style")
        self.enableLabel = createLabel("Enable on web-graph:")
  #self.searchableColumn = createCombobox(self, items=self.mC.data.getCategoricalColumns(dataID=self.mC.getDataID()))
        self.selectSearchableLabel = createLabel("Searchable columns:")
        self.searchableColumns = LabelLikeButton(parent = self, text = "Select column(s)", tooltipStr="Select columns that should be selectable in the web application for searching..", itemBorder=5)                     

        
        self.qsCheckbox = QCheckBox("Quick Select")
        self.dataDownloadCheckbox = QCheckBox("Data Download")
        self.dataDownloadCheckbox.setToolTip("If enabled, users can enter the graphID and download the data directly in InstantClue. Please note that if you dont set a password, users can still download the data using the API. It is recommended to enable this.")

        self.pwLineEdit = createLineEdit("Enter password","In you enter a password, users will be asked to enter a password before they can see the graph/chart.")
        self.pwLineEdit.setEchoMode(QLineEdit.Password)

        self.validCombo = createCombobox(self,items=["90 days","30 days","ulimited (publication)"])
        
        self.applyButton = ICStandardButton("Share")
        self.closeButton = ICStandardButton("Close")

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.title)
        self.layout().addWidget(self.updateGraph)
        grid = QGridLayout()

        grid.addWidget(self.titleEdit)
        grid.addWidget(self.subTitleEdit)
        grid.addWidget(self.infoTextEdit)
        grid.addWidget(self.markDownCheckbox)
        grid.addWidget(self.doiEdit)
        grid.addWidget(self.selectSearchableLabel)
        grid.addWidget(self.searchableColumns)
        grid.addWidget(self.enableLabel)
        grid.addWidget(self.dataDownloadCheckbox)
        grid.addWidget(self.validCombo)
        grid.addWidget(self.pwLineEdit)

        self.layout().addLayout(grid)

        hbox = QHBoxLayout()
        hbox.addWidget(self.applyButton)
        hbox.addWidget(self.closeButton)

        self.layout().addLayout(hbox)

        grid.setColumnStretch(0,1)
        grid.setRowStretch(7,2)

    def __connectEvents(self):
        ""

        self.updateGraph.currentIndexChanged.connect(self.updateGraphSelectionChanged)

        self.closeButton.clicked.connect(self.close)
        self.applyButton.clicked.connect(self.shareGraph)
        #choose specific column
        self.searchableColumns.clicked.connect(self.chooseSearchableColumns)
        

    def closeEvent(self,event=None):
        "Overwrite close event"
        event.accept()

    def chooseSearchableColumns(self, e = None):
        selectableColumns = pd.DataFrame(self.mC.data.getCategoricalColumns(self.mC.getDataID()))
        dlg = ICDSelectItems(data = selectableColumns, selectAll=False, singleSelection=False)
        # handle position and geomettry
        senderGeom = self.sender().geometry()
        bottomRight = self.mapToGlobal(senderGeom.bottomRight())
        h = dlg.getApparentHeight()
        dlg.setGeometry(bottomRight.x() + 15, bottomRight.y()-int(h/2), 185, h)
        #handle result
        if dlg.exec_():
            selectedColumns = dlg.getSelection()
            self.selectedSearchableColumns = selectedColumns.values.flatten().tolist()
            numColumnsSelected = len(self.selectedSearchableColumns)
            if hasattr(self.sender(),"setText"):
                if numColumnsSelected > 1:
                    self.sender().setText("{} columns selected".format(numColumnsSelected))
                else:
                    self.sender().setText(self.selectedSearchableColumns[0][:20])

    def collectProjectInfoData(self):
        ""
        graphProps = {}
        for widgetName, widget in [ ("title",self.titleEdit),
                                    ("subtitle",self.subTitleEdit),
                                    ("info",self.infoTextEdit),
                                    ("pwd",self.pwLineEdit),
                                    ("keywords",self.keywordsEdit),
                                    ("valid",self.validCombo)]:
            if widgetName == "pwd":
                if widget.text() != "":
                    graphProps[widgetName] = self.mC.webAppComm.encryptStringWithPublicKey(widget.text().encode("utf-8"))
                else:
                    graphProps[widgetName] = ""
            else:
                if hasattr(widget,"text"):

                    graphProps[widgetName] = widget.text()

                elif hasattr(widget,"toPlainText"):

                    graphProps[widgetName] = widget.toPlainText()
                
                elif hasattr(widget,"currentText"):

                    graphProps[widgetName] = widget.currentText()

       
        graphProps["data-download"] = self.dataDownloadCheckbox.checkState()
        graphProps["info-is-markdown"] = self.markDownCheckbox.checkState()

        return graphProps

    def shareGraph(self):
        ""
        if self.requiredWidgetsFilled():
            URL = "https://www.instantclue.de/api/v1/graph"
            appID = self.mC.webAppComm.getAppID()
            exists, graph = self.mC.getGraph()
            if exists:
                graphProps = self.collectProjectInfoData()
                data, columnNames, graphLimits,annotatedIdx,annotationProps = graph.getDataForWebApp()
                
                
                searchColumnName = self.selectedSearchableColumns[0]

                graphProps["xLabel"] = columnNames[0]
                graphProps["yLabel"] = columnNames[1]
                graphProps["domains"] = graphLimits
                graphProps["annotIdx"] = annotatedIdx
                graphProps["annotProps"] = annotationProps
                graphProps["searchColumnName"] = searchColumnName
                graphProps["hoverColumnName"] = searchColumnName
                graphProps["hoverColor"] = self.mC.config.getParam("scatter.hover.color")
                graphProps["valid"] = self.validCombo.currentText()
                
                searchData = self.mC.data.getDataByColumnNameForWebApp(self.mC.getDataID(),searchColumnName)
                data = self.mC.data.joinColumnToData(data,self.mC.getDataID(),searchColumnName)
                if data is not None:
                    try:
                        r = requests.put(URL,json={
                            "appID":appID,
                            "data":data.to_json(orient="records"),
                            "graph-props":graphProps,
                            "search-data":searchData}
                            )
                    except:
                        self.mC.sendToInformationDialog(infoText="Error in http request. Server not reachable.")
                        return
                    if r.status_code == 200:
                        self.mC.sendToInformationDialog("Graph shared succesfully. You can now access the graph using:\n\n https://www.instantclue.de/s/{}\n\nThe link is valid for the time duration of: {}".format(r.text.replace('"',""),self.validCombo.currentText()),textIsSelectable=True)
                        

    def updateGraphSelectionChanged(self,currentIdx):
        ""
        
        pwd = ""
        if currentIdx > 0:
            chartDetails = self.sharedCharts.iloc[currentIdx-1]
            graphIsProtected = self.mC.webAppComm.isChartProtected(chartDetails["short-url"])
            if graphIsProtected is not None and isinstance(graphIsProtected,bool):

                if self.mC.webAppComm.isChartProtected(chartDetails["short-url"]):
                   
                    qs = AskStringMessage(q="Please enter password.",passwordMode=True)
                    if qs.exec_():
                        pwd = qs.text.encode("utf-8")
                    else:
                        return
                else:
                    pass
                self.mC.webAppComm.getChartTextInfo(chartDetails,graphIsProtected,pwd)
            
            else:
                self.mC.sendToWarningDialog(infoText = "Couldn't reach the server. Cannot update chart.")


    def requiredWidgetsFilled(self):
        ""
        if len(self.titleEdit.text()) > 0:
            return True
        else:
            self.mC.sendToWarningDialog(infoText="Please set a chart title.")
           
            return False