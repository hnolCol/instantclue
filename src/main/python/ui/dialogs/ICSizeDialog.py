from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from ..custom.buttonDesigns import ICStandardButton, ResetButton
from ..utils import createTitleLabel, createLabel, createLineEdit, createCombobox, getMessageProps
import numpy as np 



widgetParamMatches = [
    {"label": "Size of scatter points:", "paramName": "scatterSize"},
    {"label": "Size in pointplots:", "paramName": "pointplot.marker.size"},
    {"label": "Min scatter point in range:", "paramName": "minScatterSize"},
    {"label": "Max scatter point in range:", "paramName": "maxScatterSize"},
    {"label":"Add swarm scatter size", "paramName": "swarm.scatterSize"},
    {"label":"XYPlot marker size","paramName":"xy.plot.marker.size"}
]


class SlideLineEdit(QWidget):
    ""
    def __init__(self, parent, header = "Scatter Size", initValue = 10, minValue = 0.5, maxValue = 20, precision = 0.5, *args,**kwargs):
        ""
        super(SlideLineEdit,self).__init__(parent,*args,**kwargs)

        self.header = header 
        self.value = initValue

        self.maxValue = maxValue if maxValue != np.inf else initValue + (initValue - minValue) * 2 
        self.minValue = minValue if minValue != -np.inf else initValue - (initValue - maxValue) * 2 
        self.precision = precision 

        self.justSetNewValue = False 

        self.minSliderValue, self.maxSliderValue = self.calculateRange()

        self.__controls()
        self.__layout()
        self.__connectEvents()
        


    def __controls(self):
        ""
        
        self.headerTitle = createLabel(self.header)

        self.lineEdit = createLineEdit("Enter value..")
        self.lineEdit.setText(str(self.value))
        self.lineEdit.setValidator(QDoubleValidator(self.minValue,self.maxValue,2))
        
        #
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setToolTip("Slider with range from {} to {}. Precision is {}".format(self.minValue,self.maxValue, self.precision))
        self.slider.setValue(self.transformValueForSlider(self.value))
        self.slider.setSingleStep(1)
        self.slider.setMaximum(self.maxSliderValue)
        self.slider.setMinimum(self.minSliderValue)

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        hbox = QHBoxLayout()
        self.layout().addWidget(self.headerTitle)
        hbox.addWidget(self.lineEdit)
        hbox.addWidget(self.slider)
        self.layout().addLayout(hbox)
        
    def __connectEvents(self):
        ""
        self.slider.valueChanged.connect(self.onValueChangeBySlider)
        
    
    def calculateRange(self):
        ""
        self.valueRange = np.arange(self.minValue,self.maxValue + self.precision,self.precision)
        minValue, maxValue  = 0, self.valueRange.size-1
        return minValue, maxValue

    def calculateValueFromSlider(self, newValue):
        ""

        return self.valueRange[int(newValue)]#((newValue - self.minSliderValue) / (self.maxSliderValue - self.minSliderValue)) * (self.maxValue - self.minValue) + self.minValue

    def transformValueForSlider(self,value):
        ""
        return np.abs(self.valueRange - value).argmin()

    def getValue(self):
        ""
        return self.value
    
    def onValueChangeBySlider(self, newValue):
        ""
        if not self.justSetNewValue:
            self.value = self.calculateValueFromSlider(newValue)
            self.lineEdit.setText(str(self.value))
        self.justSetNewValue = False



class ICSizeDialog(QDialog):
    def __init__(self, mainController, *args, **kwargs):
        super(ICSizeDialog, self).__init__(*args, **kwargs)
        self.mC = mainController

        self.lineEditSliders = []
        
        self.__controls()
        self.__layout()
        self.__connectEvents()
        self.__windowUpdate()
    
    def __controls(self):
        """Init widgets"""
        
        self.headerLabel = createTitleLabel("Set Default Sizes")
        self.closeButton = ResetButton()

        
        for props in widgetParamMatches:
            paramName = props["paramName"]
            initValue = self.mC.config.getParam(paramName)
            paramRange = self.mC.config.getParamRange(paramName)
            w = SlideLineEdit(self, header=props["label"],initValue=initValue, minValue = paramRange[0],maxValue = paramRange[1])
            self.lineEditSliders.append(w)

        self.acceptButton = ICStandardButton(itemName="Save")
        self.discardButton = ICStandardButton(itemName = "Discard")


    def __layout(self):

        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())

        self.layout().setContentsMargins(10,10,10,10)
        self.layout().setSpacing(3)
        hboxTop = QHBoxLayout()
        hboxTop.addWidget(self.headerLabel )
        hboxTop.addStretch()
        hboxTop.addWidget(self.closeButton)
    
        self.layout().addLayout(hboxTop)
        for w in self.lineEditSliders:
            self.layout().addWidget(w)

        hboxBottom = QHBoxLayout()
        hboxBottom.addWidget(self.acceptButton)
        hboxBottom.addWidget(self.discardButton)
        self.layout().addLayout(hboxBottom)
        self.layout().addStretch()
        
       
    def __connectEvents(self):
        """Connect events to functions"""
        self.closeButton.clicked.connect(self.close)
        self.discardButton.clicked.connect(self.close)
        self.acceptButton.clicked.connect(self.accept)
    
    def __windowUpdate(self):
        ""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.setWindowOpacity(0.98)

    def keyPressEvent(self,event=None):
        ""
        if event.key() == Qt.Key_Enter:
            return
        elif event.key() == Qt.Key_Escape:
            self.close() 

    def accept(self):
        ""
        for n,w in enumerate(self.lineEditSliders):
            value = w.getValue()
            paramName = widgetParamMatches[n]["paramName"]
            self.mC.config.setParam(paramName,value)
    
        self.mC.sendMessageRequest(getMessageProps("Done..","Sizes changed and saved."))