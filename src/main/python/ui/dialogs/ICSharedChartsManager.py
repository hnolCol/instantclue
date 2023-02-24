from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 
from matplotlib.pyplot import title 
from ..custom.buttonDesigns import ICStandardButton
from ..custom.warnMessage import WarningMessage

from ..utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR, createCombobox

import requests


class ICShareGraph(QDialog):
    ""
    def __init__(self, mainController, title = "Shared Charts Manager.", *args,**kwargs):
        ""
        super(ICShareGraph, self).__init__(*args,**kwargs)
        self.title = title
        self.mC = mainController
        self.__controls()
        self.__layout()
        self.__connectEvents()

    def __controls(self):
        ""
        self.title = createTitleLabel(self.title)
        self.infoText = createLabel("Within this dialog you can delete and publish (remove password) graphs. You can also add a link to a paper.")
        
        self.deleteButton = ICStandardButton("Delete")
        self.closeButton = ICStandardButton("Close")

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.title)
        self.layout().addWidget(self.infoText)
       

        hbox = QHBoxLayout()
        hbox.addWidget(self.deleteButton)
        hbox.addWidget(self.publishButton)
        hbox.addWidget(self.closeButton)

        self.layout().addLayout(hbox)

   

    def __connectEvents(self):
        ""
        self.closeButton.clicked.connect(self.close)
        self.applyButton.clicked.connect(self.updateGraphs)
   

    def shareGraph(self):
        ""



        URL = "http://127.0.0.1:5000/api/v1/graph"
        appID = self.mC.getAppID()
        exists, graph = self.mC.getGraph()
        if exists:
            data, columnNames, graphLimits,annotatedIdx,annotationProps = graph.getDataForWebApp()
            
            graphProps = self.collectProjectInfoData()
            searchColumnName = self.searchableColumn.currentText()

            graphProps["xLabel"] = columnNames[0]
            graphProps["yLabel"] = columnNames[1]
            graphProps["domains"] = graphLimits
            graphProps["annotIdx"] = annotatedIdx
            graphProps["annotProps"] = annotationProps
            graphProps["searchColumnName"] = searchColumnName
            graphProps["hoverColumnName"] = searchColumnName
            graphProps["hoverColor"] = self.mC.config.getParam("scatter.hover.color")


            
            searchData = self.mC.data.getDataByColumnNameForWebApp(self.mC.getDataID(),searchColumnName)
            data = self.mC.data.joinColumnToData(data,self.mC.getDataID(),searchColumnName)
            if data is not None:
                try:
                    r = requests.put(URL,json={
                        "app-id":appID,
                        "data":data.to_json(orient="records"),
                        "graph-props":graphProps,
                        "search-data":searchData}
                        )
                except:
                    self.mC.sendToInformationDialog(infoText="Error in http request. Server not reachable.")
                    return
                    
                if r.status_code == 200:
                     self.mC.sendToInformationDialog("Graph shared succesfully. You can now access the graph using:\n\n http://instantclue.age.mpg.de/s/{}\n\nThe link is valid for the time duration of: {}".format(r.text.replace('"',""),self.validCheckbox.currentText()),textIsSelectable=True)