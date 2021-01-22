from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from backend.color.colorHelper import ColorHelper
from ..utils import createLabel, createLineEdit, createTitleLabel, WIDGET_HOVER_COLOR
from ..custom.buttonDesigns import ResetButton

import seaborn as sns


class ColorLabel(QWidget):
    def __init__(self,backgroundColor = "black", *args,**kwargs):
        super(ColorLabel,self).__init__(*args,**kwargs)
        self.mouseOver = False
        self.backgroundColor = backgroundColor

    def paintEvent(self,event):
        ""
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5 if not self.mouseOver else 1)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)
        rect, h, w, x0, y0 = self.getRectProps(event)

        adjRect = rect
        if self.mouseOver:
            adjRect.adjust(0.5,0.5,-0.5,-0.5)
        else:
            adjRect.adjust(1.5,1.5,-1.5,-1.5)
        #painter.drawRoundedRect(adjRect,4,4)
        brush = QBrush(QColor(self.backgroundColor))
        painter.setBrush(brush)
        painter.drawRect(adjRect)

    def getRectProps(self,event):
        ""
        #calculate rectangle props
        rect = event.rect()
        h = rect.height()
        w = h
        x0 = rect.center().x()
        y0 = rect.center().y()    
        return rect, h, w, x0, y0 
    
    def enterEvent(self,event):
        ""
        self.mouseOver = True
        self.update()

    def leaveEvent(self,event):
        ""
        self.mouseOver = False
        self.update()

class ColorChooserDialog(QDialog):
    def __init__(self,mainController,*args, **kwargs):
        super(ColorChooserDialog,self).__init__(*args, **kwargs)

        self.mC = mainController
        self.colorHelper = ColorHelper()
        self.selectedColorMap = self.mC.data.colorManager.colorMap
        self.selectedAlpha = self.mC.data.colorManager.alpha
        self.colorMapCBs = dict()

        self.__windowUpdate()
        self.__controls()
        self.__layout()
        self.__connectEvents()
        

    def __windowUpdate(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.setWindowOpacity(0.98)

    
    
    def __controls(self):
        """Init widgets"""
        self.titleLabel = createTitleLabel("Color palettes")
        self.closeButton = ResetButton()
        
        
    def __layout(self):
        """Put widgets in layout"""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(10,10,10,10)
        self.layout().setSpacing(3)
        hboxTop = QHBoxLayout()
        hboxTop.addWidget(self.titleLabel)
        hboxTop.addStretch()
        hboxTop.addWidget(self.closeButton)
        self.layout().addLayout(hboxTop)
        hbox = QHBoxLayout() 
        try:
            for header, colorPalettes in self.colorHelper.getColorPalettes().items():
                vbox = QVBoxLayout()
                vbox.setAlignment(Qt.AlignTop)
                vbox.addWidget(createTitleLabel(header,fontSize=12))
                for colorPaletteName in colorPalettes:
                    layout = self.createColorPalette(colorPaletteName,checkboxState=self.selectedColorMap == colorPaletteName)
                    vbox.addLayout(layout)
                hbox.addLayout(vbox)
        except Exception as e:
            print(e)
        self.layout().addLayout(hbox)

        alphaLayout = self.createAlphaWidgets()
        self.layout().addLayout(alphaLayout)

    def createAlphaWidgets(self):
        ""
        gridBox = QGridLayout() 
        alphaLabel = createLabel("Transparency: ", fontSize=12)
        self.alphaLineEdit = createLineEdit("Set alpha ..","Set Transparency (alpha) of plot items.")
        #validator for float input
        validator = QDoubleValidator()
        validator.setRange(0.00,1.00,2)
        validator.setDecimals(2)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.alphaLineEdit.setValidator(validator)
        self.alphaLineEdit.textChanged.connect(self.alphaChanged)
        #set widget size
        self.alphaLineEdit.setFixedSize(QSize(80,20))
        
        gridBox.addWidget(alphaLabel,0,0)
        gridBox.addWidget(self.alphaLineEdit,0,1)
        gridBox.setColumnStretch(2,2)
        #x.addStretch(1)
        self.alphaSlider = QSlider(Qt.Horizontal)
        self.alphaSlider.setMinimum(0)
        self.alphaSlider.setMaximum(100)
        self.alphaSlider.setSingleStep(5)
        self.alphaSlider.setValue(self.selectedAlpha * 100)
        self.alphaSlider.setTickPosition(QSlider.TicksBelow)
        self.alphaSlider.setTickInterval(10)
        self.alphaSlider.valueChanged.connect(self.sliderMoved)   
        gridBox.addWidget(self.alphaSlider,1,0,1,2)
        #set value, text changed is connect to alphaSlider
        self.alphaLineEdit.setText(str(self.selectedAlpha))
        return gridBox
        

    def __connectEvents(self):
        """Connect events to functions"""
        self.closeButton.clicked.connect(self.close)

    def alphaChanged(self,event=None):
        
        #set text color to red if input is not valid
        if self.alphaLineEdit.text() != "" and not self.alphaLineEdit.hasAcceptableInput():
            self.alphaLineEdit.setStyleSheet("color: {}".format(WIDGET_HOVER_COLOR))
        else:
           
            if self.alphaLineEdit.text() not in ["","0."]:
                alpha = float(self.alphaLineEdit.text()) * 100 
                self.alphaSlider.setValue(alpha)
            #otherwise set it back to black
            
            self.alphaLineEdit.setStyleSheet("color: black")
            
    def closeEvent(self,event=None):
        
        if self.alphaLineEdit.hasAcceptableInput():
            newAlpha = float(self.alphaLineEdit.text())
            if newAlpha != self.selectedAlpha:
                self.updateAlpha(newAlpha)

        self.mC.mainFrames["middle"].plotter.update_scatter_point_properties()
    
    def createColorPalette(self, paletteName, checkboxState = False):
        ""
        hbox = QHBoxLayout()
        checkBox = QCheckBox(paletteName)
        checkBox.setCheckState(checkboxState)
        checkBox.setTristate(False)
        checkBox.clicked.connect(self.setColorMap)
        self.colorMapCBs[paletteName] = checkBox

        colorList = sns.color_palette(paletteName,5,self.mC.data.colorManager.desat).as_hex()
        
        for color in colorList:
            labelColorWidget = ColorLabel(backgroundColor = color)
            labelColorWidget.setFixedSize(QSize(18,18))
            hbox.addWidget(labelColorWidget)
            
        hbox.addWidget(checkBox)
        return hbox

    def keyPressEvent(self,e):
        """Handle key press event"""
        if e.key() == Qt.Key_Escape:
            self.close()
        
    def setColorMap(self,event=None):
        ""
        newColorMap = self.sender().text()

        self.colorMapCBs[newColorMap].setCheckState(True)
        self.colorMapCBs[self.selectedColorMap].setCheckState(False)
        self.selectedColorMap = newColorMap
        self.updateColorMap()

    def sliderMoved(self,event=None):
        ""
        alpha = round(self.sender().value() / 100,2)
        self.alphaLineEdit.setText(str(alpha))

    def updateColorMap(self,paletteName = None):
        ""
        if paletteName is None:
            paletteName = self.selectedColorMap
        #save color map selection
        setattr(self.mC.data.colorManager, "colorMap", paletteName)
        #apply color map to color table (if existent)
        self.mC.mainFrames["sliceMarks"].colorTable.clorMapChanged.emit()

    def updateAlpha(self,alpha):
        ""
        if isinstance(alpha,float) and alpha > 0 and alpha <= 1:
            self.mC.config.setParam("alpha",alpha)

