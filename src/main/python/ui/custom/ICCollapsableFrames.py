from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

import numpy as np
from collections import OrderedDict
from ..utils import getMainWindowBGColor

class CollapsableFrames(QWidget):
    def __init__(self,buttonDesign, buttonMenu = None, parent = None, animationDuration = 550, spacing = 0):
        super(CollapsableFrames,self).__init__(parent)
        self.frameProps = OrderedDict()
        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(spacing)
        self.layout().setContentsMargins(0,0,0,0)
        self.buttonMenu = buttonMenu
        self.buttonDesign = buttonDesign
        self.animation = QParallelAnimationGroup()
        self.parentGeometry()
        self.animationDuration = animationDuration
        self.spacing = spacing

        
        
    def parentGeometry(self, justReturn = False):
        
        if justReturn:
            return self.parent().frameGeometry().height(), self.parent().frameGeometry().width()
        else:
            self.parentHeight = self.parent().frameGeometry().height()
            self.parentWidth = self.parent().frameGeometry().width()

            return self.parentHeight, self.parentWidth
        
    def resizeHeaders(self):
        ""
        
        if hasattr(self,"frameProps"):
            currentWidthHeader = self.currentFrameWidth(0)
            for intFrameID in self.frameProps.keys():
                if currentWidthHeader < 0:
                    currentWidthHeader = 0

                self.frameProps[intFrameID]["contentArea"].setMaximumWidth(currentWidthHeader)
                self.frameProps[intFrameID]["contentArea"].setMinimumWidth(currentWidthHeader)
               

    def closeFrames(self):
        ""
        d = self.frameProps.copy() 

        for k in d.keys():
            self.frameProps[k]["open"] = False

    def resizeEvent(self,event):
        ""
        self.parentGeometry()
        self.startAnimation() 
        self.update()
        
    
    def addCollapsableFrame(self,frameProps, stackFrameTop = True,*args, **kwargs):
        ""
        if isinstance(stackFrameTop,bool) and not stackFrameTop:
            self.layout().addStretch(1)
        #self.colors = ["#E8E8E8","#a6cee3","#A0D4CB","#2776BC"]#sns.color_palette("Paired",len(frameProps)).as_hex()
        self.frameInitHeight = self.calculateFrameHeight(frameProps)
        for frameID, frameInfo in enumerate(frameProps):
            if isinstance(stackFrameTop,str) and frameID == int(stackFrameTop):
                self.layout().addStretch(1)
            self.frameProps[frameID] = dict() 
            self.frameProps[frameID]["open"] = frameInfo["open"]
            self.frameProps[frameID]["fixedHeight"] = frameInfo["fixedHeight"]
            self.frameProps[frameID]["height"] = frameInfo["height"]
            self.frameProps[frameID]["title"] = frameInfo["title"]
            self.frameProps[frameID]["active"] = True
            self.addHeaderFrame(frameID)
            self.addContentArea(frameID,frameInfo["layout"] ,frameInfo["open"])
            self.addToggleButton(frameID,frameInfo["title"],  *args, **kwargs)
        
        if isinstance(stackFrameTop,bool)  and stackFrameTop:

            self.layout().addStretch(1)
        
        
    def addContentArea(self, frameID, contentLayout, frameOpen = True):
        ""
        height = self.frameInitHeight if frameOpen else 0
        
        contentArea = QScrollArea(self)
        contentArea.setWidgetResizable(True)
        contentArea.setStyleSheet("QScrollArea {background-color:"+f" {getMainWindowBGColor()}"+";border:None};")

        self.animation.addAnimation(QPropertyAnimation(contentArea,b"maximumHeight"))
        self.animation.addAnimation(QPropertyAnimation(contentArea,b"minimumHeight"))
        
        contentArea.setMaximumHeight(height)
        contentArea.setMinimumHeight(height)
        contentArea.setLayout(contentLayout)
        self.frameProps[frameID]["contentArea"] = contentArea
        self.frameProps[frameID]["contentArea"].setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.layout().addWidget(self.frameProps[frameID]["contentArea"])

    def addHeaderFrame(self, frameID):
        ""
        self.frameProps[frameID]["headerFrame"] = QFrame(parent=self)
        self.frameProps[frameID]["headerFrame"].resize(self.parentWidth,30)
        self.frameProps[frameID]["headerFrame"].setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)

        self.layout().addWidget(self.frameProps[frameID]["headerFrame"])

    def addToggleButton(self, frameID, title, *args, **kwargs):
        ""
        vbox = QVBoxLayout(self.frameProps[frameID]["headerFrame"])
        vbox.setContentsMargins(0,0,0,0)
        toolButton = self.buttonDesign(
                    parent = self.frameProps[frameID]["headerFrame"],
                    text = title, 
                    openFrame=self.frameProps[frameID]["open"],
                    *args, **kwargs)#, openColor=self.colors[frameID])
        if self.buttonMenu is not None:
            toolButton.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            toolButton.customContextMenuRequested.connect(self.openContextMenu)

        self.frameProps[frameID]["button"] = toolButton
        toolButton.clicked.connect(lambda event, frameID = frameID: self.startAnimation(event,frameID))
        vbox.addWidget(toolButton)
        self.buttonHeight = toolButton.getWidgetHeight()
        

    def getFrameIDByTitle(self,frameTitle):
        ""
        for frameID, frameInfo in self.frameProps.items():
            
            if frameTitle == frameInfo["title"]:

                return frameID

    def setHeaderNameByFrameID(self,frameTitle,headerName):
        "Danger- only possible if title unique"
        frameID = self.getFrameIDByTitle(frameTitle)
        if frameID is not None:
            self.frameProps[frameID]["button"].setText(headerName)
            self.frameProps[frameID]["button"].active = True
            self.frameProps[frameID]["active"] = True
       
    def setInactiveByTitle(self,frameTitle):
        ""
        frameID = self.getFrameIDByTitle(frameTitle)
        if frameID is not None:
            self.frameProps[frameID]["button"].active = False
            self.frameProps[frameID]["active"] = False
            self.frameProps[frameID]["open"] = False
        
    
    def startAnimation(self, event = None, frameID = None):
        ""
        if frameID is not None:
            if not self.frameProps[frameID]["active"]:
                self.frameProps[frameID]["open"] = False
                return
            self.frameProps[frameID]["open"] = not self.frameProps[frameID]["open"]
            self.frameProps[frameID]["button"].setMouseEntered(False)
            
        frameHeights = self.calculateHeights()
        if frameHeights is None:
            return
        animationID = 0
        for intFrameID in self.frameProps.keys():
            currentHeight = self.currentFrameHeight(intFrameID,"contentArea")
            currentWidthHeader = self.currentFrameWidth(0)
            if currentWidthHeader < 0:
                currentWidthHeader = 0
            self.frameProps[intFrameID]["contentArea"].setMaximumWidth(currentWidthHeader)

            endValue = 0 if self.frameProps[intFrameID]["open"] == False else frameHeights[intFrameID] 
            
            for i in range(2):
                animation = self.animation.animationAt(animationID)
                animation.setStartValue(currentHeight)
                animation.setEndValue(endValue)
                animation.setDuration(self.animationDuration)
                animationID += 1
        self.animation.start()
        self.animation.finished.connect(lambda frameID= frameID:self.updateButtons(frameID))


    def updateButtons(self, frameID = None):
        ""
        
        if frameID is not None:
            self.frameProps[frameID]["button"].setFrameState(self.frameProps[frameID]["open"])

    def calculateFrameHeight(self, frameProps):
        ""
        if len(frameProps) == 0:
            return 0
        if not hasattr(self,"buttonHeight"):
            return 0 
        n = len(frameProps)
        maxHeight = self.parentHeight
        headerHeights = (self.buttonHeight + 1) * n + self.spacing * (n+1) # size hint is 24 from QPushButtons
        fixedHeights = [int(props["height"]) for props in frameProps if props["fixedHeight"] and props["open"]]
        fixedHeight = np.sum(fixedHeights) if len(fixedHeights) > 0 else 0 
        contentFrameHeight = maxHeight - headerHeights - fixedHeight

        openFrames = np.sum([frameProp["open"] for frameProp in frameProps if not frameProp["fixedHeight"]])

        if openFrames == 0:
            return 0
        heightPerFrame = contentFrameHeight / openFrames
        return heightPerFrame

    def calculateHeights(self):
        ""
        if not hasattr(self,"frameProps"):
            return
        calculatedHeights = dict() 
        heightPerFrame = self.calculateFrameHeight(list(self.frameProps.values()))
        

        for frameID, props in self.frameProps.items():
            if props["fixedHeight"]:
                calculatedHeights[frameID] = int(props["height"]) if props["open"] else 0
            else:
                calculatedHeights[frameID] = int(heightPerFrame) if props["open"] or heightPerFrame < 0 else 0 

        return calculatedHeights


    def currentFrameHeight(self,frameID, frameKey = "headerFrame"):
        ""
        return self.frameProps[frameID][frameKey].frameGeometry().height()

    def currentFrameWidth(self,frameID, frameKey = "headerFrame"):
        ""
        return self.frameProps[frameID][frameKey].frameGeometry().width()
    def addToLayout(self):
        ""

        for _,collabsabelFrames in self.frameProps.items():
            vbox = QVBoxLayout()
            vbox.addWidget(collabsabelFrames["headerFrame"])
            vbox.addWidget(collabsabelFrames["contentArea"])
            self.layout().addLayout(vbox)

    def openContextMenu(self,event=None):
        ""
        #get position of geometry
        senderGeom = self.sender().geometry()
        bottomLeft = self.sender().mapToGlobal(senderGeom.bottomLeft())
        #set sender status 
        self.sender().mouseOver = False

        typeID = self.sender().getText()
        self.buttonMenu(typeID, bottomLeft)
        
        
        