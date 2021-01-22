from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..utils import HOVER_COLOR, WIDGET_HOVER_COLOR, INSTANT_CLUE_BLUE, getStandardFont, isWindows
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap


#colorCmap = sns.choose_colorbrewer_palette("Blues",as_cmap=True)
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
colorCmap = ListedColormap(sns.color_palette("Blues").as_hex())
twColorCmap = ListedColormap(sns.color_palette("RdYlBu").as_hex())

class CollapsButton(QPushButton):
    def __init__(self,
                parent=None,
                openFrame = False, 
                text = "None", 
                strokeWidth = 0.1,
                paddingY = 0.5, 
                closeColor = "#f6f6f6",
                openColor = None,#"#C6C6B6",
                dotColor = INSTANT_CLUE_BLUE,
                hoverColor = HOVER_COLOR,
                hoverDotColor = WIDGET_HOVER_COLOR,
                font = None,
                mouseTracking = True,
                fontSize = 12,
                widgetHeight = 30):

        super(CollapsButton,self).__init__(parent)
        self.oldRect = None
        self.openFrame = openFrame
        self.paddingY = paddingY
        self.text = text
        self.closeColor = closeColor
        self.openColor = openColor
        self.dotColor = dotColor
        self.hoverColor = hoverColor
        self.hoverDotColor = hoverDotColor
        self.strokeWidth = strokeWidth
        self.setMouseTracking(mouseTracking)
        self.mouseEntered = False
        self.QFont = QFont("verdana",fontSize, weight=QFont.Light) if font is None else font
        self.QFont.setLetterSpacing(QFont.PercentageSpacing,105)
        self.QFont.setWordSpacing(3)
        self.widgetHeight = widgetHeight


    def sizeHint(self):
         ""
         return QSize(QPushButton().sizeHint().width(),self.widgetHeight)

    def enterEvent(self,event):
        ""
        self.mouseEntered = True
        self.update()

    def leaveEvent(self,event):
        self.mouseEntered = False
        self.update()

    def setMouseEntered(self,mouseEntered):
        ""
        self.mouseEntered = mouseEntered

    def setFrameOpen(self):
        self.openFrame = True
        self.update()
    
    def setFrameClose(self):
        self.openFrame = False
        self.update() 

    def setFrameState(self,openFrame):
        "Change state of frame/button"
        if openFrame:
            self.setFrameOpen()
        else:
            self.setFrameClose()

    def paintEvent(self,event):
        ""
        eventRect = event.rect()
        if self.oldRect is not None:
            if abs(self.oldRect.width() -  event.rect().width()) > 1:
                self.oldRect = eventRect
            elif abs(self.oldRect.y() -  event.rect().y()) < 5:
                
                eventRect = self.oldRect
            else:
                eventRect = event.rect()
                self.oldRect = event.rect()
        else:
            self.oldRect = event.rect()
            
       # eventRect = event.rect()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        rect = QRect(eventRect)   
       
        centerPoint = rect.center()
        centerPoint.setX(centerPoint.x() + rect.width() / 2 - 10)    
    
        pen = QPen(QColor("black"))
        pen.setWidthF(self.strokeWidth)
        
        painter.setPen(pen)
        painter.setFont(self.QFont)

        if self.mouseEntered and self.hoverColor is not None:
            brushColor = self.hoverColor 
        else:
            if self.openColor is not None:
                brushColor = self.openColor if self.openFrame else self.closeColor
            else:
                brushColor = self.closeColor
        brush = QBrush(QColor(brushColor))
        painter.setBrush(brush)
        brush.setColor(QColor(self.hoverDotColor if self.mouseEntered else self.dotColor))
        painter.drawRect(rect)
        painter.drawText(rect.x() + 10, centerPoint.y() + 6 ,self.text)
        painter.setBrush(brush)
        painter.drawEllipse(centerPoint,4,4)
        
    def getText(self):
        ""
        return self.text

    def getWidgetHeight(self):
        ""
        return self.widgetHeight

    def loseFocus(self):
        ""
        self.mouseEntered = False
        self.update()

class DataHeaderButton(CollapsButton):
    def __init__(self,openFrame,parent=None,*args,**kwargs):
        super(DataHeaderButton,self).__init__(
                    parent=parent,
                    openFrame=openFrame,
                    fontSize=8,
                    *args,
                    **kwargs)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        
    def paintEvent(self,event):
        ""
        
        eventRect = event.rect()
        if self.oldRect is not None:
            
            if abs(self.oldRect.width() -  event.rect().width()) > 1:
                self.oldRect = eventRect
            elif self.oldRect.height() > eventRect.height():
                eventRect = self.oldRect
            else:
                eventRect = event.rect()
                self.oldRect = event.rect()
        else:
            self.oldRect = event.rect()

           
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        rect = QRectF(eventRect)  

        centerPoint = rect.center()
        painter.setPen(Qt.NoPen)
        painter.setFont(self.QFont)
        
        if self.mouseEntered and self.hoverColor is not None:
            brushColor = self.hoverColor 
        else:
            if self.openColor is not None:
                brushColor = self.openColor if self.openFrame else self.closeColor
            else:
                brushColor = self.closeColor

        brush = QBrush(QColor(brushColor))
        painter.setBrush(brush)
        

        pen = QPen(QColor("black"))
        pen.setWidthF(0.1)
        painter.setPen(pen)
        painter.drawRect(rect)

        rect.adjust(18,0,0,0)
        painter.drawText(rect,
                            Qt.AlignVCenter | Qt.AlignLeft, 
                            self.text)

        yCenter = centerPoint.y() 
        xArrow = 8 
        h = rect.height()/2
        pen = QPen(QColor(self.dotColor if not self.mouseEntered else self.hoverDotColor))
        pen.setWidthF(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        p1 = QPointF(xArrow-h/3,
                    yCenter)

        p2 = QPointF(xArrow+h/3,
                    yCenter)
        painter.drawLine(p1,p2)

        if not self.openFrame:
            p3 = QPointF(xArrow,yCenter-h/3)
            p4 = QPointF(xArrow,yCenter+h/3)
            painter.drawLine(p3,p4)
        
    

 
class PushHoverButton(QPushButton):
    def __init__(self,parent = None, 
                      acceptDrops = True,
                      acceptedDragTypes = ["Categories"], 
                      getDragType = None,
                      funcKey = None, 
                      text = None,
                      callback = None, 
                      txtColor = "black",
                      tooltipStr = None):
        super(PushHoverButton,self).__init__(parent)

        self.mouseOver = False
        self.funcKey = funcKey
        self.callback = callback
        self.txtColor = txtColor
        self.getDragType = getDragType
        self.setAcceptDrops(acceptDrops)
        self.acceptedDragTypes = acceptedDragTypes
        self.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.text = text
        if tooltipStr is not None and isinstance(tooltipStr,str):
            self.setToolTip(tooltipStr)

    def paintEvent(self,event):
        
        painter = QPainter(self)
        pen = QPen(QColor(self.txtColor))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)
        b = QBrush(QColor(HOVER_COLOR if self.mouseOver else "white"))
        painter.setBrush(b)
        rect, h, w, x0, y0 = self.getRectProps(event)

        adjRect = rect
        adjRect.adjust(0.5,0.5,-0.5,-0.5)
        #painter.drawRoundedRect(adjRect,4,4)
        painter.drawRect(adjRect)
       
        if self.text is not None:
            painter.drawText(2,12,self.text)


    def getStandardFont(self, fontSize = 12):
        ""
        if isWindows():
            fontSize -= 2
        return getStandardFont(fontSize=fontSize) 

    def getMainFont(self):
        
        font = QFont("Arial")
        font.setCapitalization(QFont.SmallCaps)
        if isWindows():
            font.setPointSize(8)
        else:
            font.setPointSize(10)
        return font

    def getRectProps(self,event):
        ""
        #calculate rectangle props
        rect = event.rect()
        h = rect.height()
        w = h
        x0 = rect.center().x()
        y0 = rect.center().y()    
        return rect, h, w, x0, y0 

    def getFilterPath(self, x0, y0, w, h):

        ellipseRect = QRectF(x0-w/4,y0-h/3,w/2,h/8)
        rectCentY = ellipseRect.center().y()
        widthEnd = w/20

        polyPoints = [QPointF(x0+w/4,rectCentY), 
                      QPointF(x0+widthEnd,y0), 
                      QPointF(x0+widthEnd,y0+h/10), 

                      QPointF(x0-widthEnd,y0+h/10),
                      QPointF(x0-widthEnd,y0),
                      QPointF(x0-w/4,rectCentY),
                      QPointF(x0+w/4,rectCentY)]
        return polyPoints, ellipseRect
    
    def sizeHint(self):

        return QSize(45,45)

    def getMainLabelRect(self,x0,y0,w,h):
        ""
        return QRect(x0-w/2,y0+h/5,w,h/4)

    def dragEnterEvent(self,event):
        "Check if drag items is of correct datatype"
        if self.getDragType is None:
            self.acceptDrop  = False
        else:
            dragType = self.getDragType()
            #check if type is accpeted and check if not all items are anyway already there
            if dragType in self.acceptedDragTypes:
                self.acceptDrop = True
            else:
                self.acceptDrop = False
        
        event.accept()
       
    def dragMoveEvent(self, e):
        "Ignore/acccept drag Move Event"
        if self.acceptDrop:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self,event):
        ""
        event.accept()
        if self.callback is not None:
            self.callback() 
        elif hasattr(self.parent(),"sendRequestToThread") and self.funcKey is not None:
            funcProps = {"key":self.funcKey}
            self.parent().sendRequestToThread(funcProps)

    def enterEvent(self,event):
        ""
        self.mouseOver = True
        self.update()

    def leaveEvent(self,event):
        ""
        self.mouseOver = False
        self.update()
    
    def mouseLostFocus(self):
        """
        Function to mimic mouse left button.
        Useful when casting a menu.
        """
        self.mouseOver = False
        self.update()
    
    def setTxtColor(self,color):
        ""
        self.txtColor = color
        self.update() 


class LabelLikeButton(PushHoverButton):
    ""
    def __init__(self, text = "", fontSize = 12, itemBorder = 5, *args, **kwargs):
        super(LabelLikeButton,self).__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.MinimumExpanding ,QSizePolicy.Fixed)
        #self.buttonSize = buttonSize
        self.itemBorder = itemBorder
        self.fontSize = fontSize
        self.text = text
        self.updateSize()

    def sizeHint(self):
        ""
        return QSize(self.width,self.height)

    def paintEvent(self,event):
        ""
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)
        painter = QPainter(self)
        painter.setFont(self.getStandardFont())
        pen = QPen(QColor("black" if not self.mouseOver else INSTANT_CLUE_BLUE))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)
        rect.adjust(self.itemBorder,0,0,0)
        
        painter.drawText(rect,
                         Qt.AlignLeft | Qt.AlignVCenter, 
                         self.text)

    def updateSize(self):
        ""
        font = self.getStandardFont(self.fontSize)    
        self.width = QFontMetrics(font).width(self.text) + 2 * self.itemBorder
        self.height = QFontMetrics(font).height() + self.itemBorder
    
    def getText(self):
        ""
        return self.text

    def setText(self,text):
        ""
        self.text = text
        self.updateSize()
        self.update()
    
    

 

class ICStandardButton(PushHoverButton):

    def __init__(self,itemName = "", parent=None, itemBorder = 15, *args,**kwargs):
        super(ICStandardButton,self).__init__(parent=parent,*args,**kwargs)
        self.itemName = itemName
        self.setSizePolicy(QSizePolicy.Fixed ,QSizePolicy.Fixed)
        #get size
        font = self.getStandardFont()
        self.width = QFontMetrics(font).width(self.itemName) + int(1 * itemBorder)
        self.height = QFontMetrics(font).height() + int(0.6 * itemBorder)

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
        painter.setRenderHint(QPainter.Antialiasing,True)
        pen = QPen(QColor("black"))
        brush = QBrush(QColor("white"))
        pen.setWidthF(0.5)
        painter.setBrush(brush)
        painter.setPen(pen)

        painter.drawRect(rect)

        if self.mouseOver: 
            pen = QPen(QColor(INSTANT_CLUE_BLUE))
           
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.drawText(rect,
                         Qt.AlignCenter | Qt.AlignTop, 
                         self.itemName)


class BigArrowButton(PushHoverButton):
    """
    BigArrowButton Button. 

    Overwrite painEvent to display Size Button.
    PushHoverButton contains a main label, and 
    the circle of varying size defined by the
    objec radiiFrac list.

    The PushHoverButton Class provides Function 
    for style such as the Font for mainLabel.
    """
    def __init__(self,parent=None,direction = "down", buttonSize = None, **kwargs):
        super(BigArrowButton,self).__init__(parent, acceptDrops=False, **kwargs)
        self.direction = direction
        self.buttonSize = buttonSize
        
    def sizeHint(self):
        ""
        if self.buttonSize is not None:
            w, h  = self.buttonSize
            return QSize(w,h)
        else:
            return QSize(35,35)
            
    def paintEvent(self,event):
        
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)

        #create common background/reacts to hover
        super().paintEvent(event)
        
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        c = QColor(WIDGET_HOVER_COLOR if self.mouseOver else INSTANT_CLUE_BLUE)
        #c.setAlphaF(0.85)
        b = QBrush(c)
        painter.setBrush(b)

        try:
            polygon = QPolygonF()
            for p in self.getPointsForArrow(x0,y0,h,direction=self.direction):
                polygon.append(p)
            painter.drawPolygon(polygon)
        except Exception as e:
            print(e)



    def getPointsForArrow(self,x0,y0,h,direction="down"):
        ""
        widthEndArrow = h/7
        if direction == "down":
            #start at arrow peak
            polygonPoints = [  
                QPointF(x0,y0+h/3),
                QPointF(x0-h/3,y0),
                QPointF(x0-widthEndArrow,y0),
                QPointF(x0-widthEndArrow,y0-h/4),
                QPointF(x0+widthEndArrow,y0-h/4),
                QPointF(x0+widthEndArrow,y0),
                QPointF(x0+h/3,y0),
                QPointF(x0,y0+h/3)
            ]
        elif direction == "up":
            
            polygonPoints = [  
                QPointF(x0,y0-h/3),
                QPointF(x0+h/3,y0),
                QPointF(x0+widthEndArrow,y0),
                QPointF(x0+widthEndArrow,y0+h/4),
                QPointF(x0-widthEndArrow,y0+h/4),
                QPointF(x0-widthEndArrow,y0),
                QPointF(x0-h/3,y0),
                QPointF(x0,y0-h/3)
            ]           

        return polygonPoints


class BigPlusButton(PushHoverButton):
    """
    BigPlusButton Button. 

    Overwrite painEvent to display Size Button.
    PushHoverButton contains a main label, and 
    the circle of varying size defined by the
    objec radiiFrac list.

    The PushHoverButton Class provides Function 
    for style such as the Font for mainLabel.
    """
    def __init__(self,parent=None,buttonSize=None,**kwargs):
        super(BigPlusButton,self).__init__(parent, acceptDrops = False,**kwargs)
        self.buttonSize = buttonSize
        
    def sizeHint(self):
        ""
        if self.buttonSize is not None:
            w, h  = self.buttonSize
            return QSize(w,h)
        else:
            return QSize(35,35)

    def paintEvent(self,event):
        
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)

        #create common background/reacts to hover
        super().paintEvent(event)
        
        painter = QPainter(self)
        pen = QPen(QColor(WIDGET_HOVER_COLOR if self.mouseOver else INSTANT_CLUE_BLUE))
        pen.setWidthF(4)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        
   
        lineWidth = h/4
        painter.drawLine(x0-lineWidth,y0,x0+lineWidth,y0)
        painter.drawLine(x0,y0-lineWidth,x0,y0+lineWidth)


class SubsetDataButton(PushHoverButton):
    def __init__(self,parent=None,*args,**kwargs):
        super(SubsetDataButton,self).__init__(parent,*args,**kwargs)
        self.displayColors = ["white",INSTANT_CLUE_BLUE,"#A0D4CB"]
    
    def sizeHint(self):
        ""
        return QSize(35,35) 

    def paintEvent(self,event):
        ""
        try:
            #create common background/reacts to hover
            super().paintEvent(event)
            #crate painter
            painter = QPainter(self)
            pen = QPen(QColor("black"))
            pen.setWidthF(0.5)
            painter.setPen(pen)
            painter.setRenderHint(QPainter.Antialiasing,True)

            #calculate rectangle props
            rect, h, w, x0, y0 = self.getRectProps(event)

            polyPoints, ellipseRect = self.getFilterPath(x0,y0,w,h)
            #setup brush for filter
            c = QColor(WIDGET_HOVER_COLOR if self.mouseOver else INSTANT_CLUE_BLUE)
            #c.setAlphaF(0.75)
            b = QBrush(c)
            painter.setBrush(b)
            polygon = QPolygonF()
            for p in polyPoints:
                #append points to polygon
                polygon.append(p)
            painter.drawPolygon(polygon,2)
            
            #setup brush for white ellipse
            brush = QBrush(QColor("white"))
            painter.setBrush(brush)
            

            painter.drawEllipse(ellipseRect)

            # draw split data sets
            border = h/28
            borderBetween = h/12
            widthForSets = (w - 2*border - 2*borderBetween) / 3
            y = y0 + h/7
            colors  = self.displayColors if not self.mouseOver else self.displayColors[::-1]
            for n in range(3):
                b = QBrush(QColor(colors[n]))
                painter.setBrush(b)
                x = border + border + n * widthForSets + n * borderBetween
                painter.drawRoundedRect(QRectF(x,y,widthForSets,h/3.5),1,1)

        except Exception as e:
            print(e)

class ViewDataButton(PushHoverButton):
    def __init__(self,parent=None,*args,**kwargs):
        super(ViewDataButton,self).__init__(parent,*args,**kwargs)
        self.displayColors = ["white","#A9A8A7","#A0D4CB",INSTANT_CLUE_BLUE]
        self.hoverColors = ["white","#A9A8A7","#A0D4CB",WIDGET_HOVER_COLOR]
    def sizeHint(self):
        ""
        return QSize(35,35) 

    def paintEvent(self,event):
        ""
       
        #create common background/reacts to hover
        super().paintEvent(event)
        #crate painter
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)

        #setup brush for filter
        c = QColor("white")#WIDGET_HOVER_COLOR if self.mouseOver else INSTANT_CLUE_BLUE)
        #c.setAlphaF(0.75)
        b = QBrush(c)
        painter.setBrush(b)

        #draw rounded rect
        # h = w 
        border = h/8
        x = 0 + border
        y = 0 + border
        roundRect = QRectF(x,y,h - 1.5 * border,h - 1.5 * border)
        painter.drawRect(roundRect)


        nRows = 4
        nCols = 4
        startX = x + h/10
        startY = y + h/10
        widthTotal = h - 1.5*border - 2 * h/10
        height = width = widthTotal / nCols
        for nC in range(nCols):
            brush = QBrush(QColor(self.displayColors[nC] if not self.mouseOver else self.hoverColors[nC]))
            painter.setBrush(brush)
            for nR in range(nRows):
                rect = QRectF(startX + nC*width, startY + nR * height,width,height)
                painter.drawRect(rect)



        ellipseRect = QRectF(x0-h/10,y0-h/10,h/1.75,h/1.75)
        #draw Lupe
        c = QColor("white")
        c.setAlphaF(0.55 if self.mouseOver else 0.9)
        b=QBrush(c)
        painter.setBrush(b)
        painter.drawEllipse(ellipseRect)

class SizeButton(PushHoverButton):
    """
    Size Button. 

    Overwrite painEvent to display Size Button.
    PushHoverButton contains a main label, and 
    the circle of varying size defined by the
    objec radiiFrac list.

    The PushHoverButton Class provides Function 
    for style such as the Font for mainLabel.
    """
    def __init__(self,parent=None,*args,**kwargs):
        super(SizeButton,self).__init__(parent,*args,**kwargs)
        self.displayColors = ["#A9A8A7","#CECDCC","FAFBFB"]
        self.hoverColors = ["#397546",INSTANT_CLUE_BLUE,"#A0D4CB"]
        self.radiiFrac = [(n+1)/4.5 for n in range(3)]

    def paintEvent(self,event):
        
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)
        r = 0.3 * h

        #create common background/reacts to hover
        super().paintEvent(event)
        
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        offset = [(0,0),(h/7,h/7),(h/4,h/4)]
        yOffset = h/10
        
        for n,frac in enumerate(self.radiiFrac):
            ri = r * frac
            dx = offset[::-1][n][0]
            dy = offset[::-1][n][1]
            c = QColor(self.displayColors[n] if not self.mouseOver else self.hoverColors[n])#824329")
            c.setAlphaF(0.55 if not self.mouseOver else 0.9)
            b=QBrush(c)
            painter.setBrush(b)
            painter.drawEllipse(QPointF(x0+dx,y0-dy-yOffset),ri,ri)

        # draw main label
        painter.setFont(self.getMainFont())
        painter.drawText(self.getMainLabelRect(x0,y0,w,h),
                            Qt.AlignCenter | Qt.AlignTop, 
                            "size")
 


class FilterButton(PushHoverButton):
    """
    Filter Button. Inherits from PushHoverButton.

    Overwrite painEvent to display Color Button.
    PushHoverButton contains a main label, and 
    several color-circles.

    The PushHoverButton Class provides Function 
    for style such as the Font for mainLabel.
    """
    def __init__(self,parent=None, *args, **kwargs):
        super(FilterButton,self).__init__(parent, *args, **kwargs)
       # self.displayColors = ["#A9A8A7","#CECDCC","FAFBFB"]
        
        #self.radiiFrac = [(n+1)/4.5 for n in range(3)]

    def paintEvent(self,event):
        
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)
        r = 0.3 * h

        #create common background/reacts to hover
        super().paintEvent(event)
        
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)
        
        polyPoints, ellipseRect = self.getFilterPath(x0,y0,w,h)
        
        
        #setup brush for filter
        c = QColor(WIDGET_HOVER_COLOR if self.mouseOver else INSTANT_CLUE_BLUE)
        #c.setAlphaF(0.75)
        b = QBrush(c)
        painter.setBrush(b)
        polygon = QPolygonF()
        for p in polyPoints:
            #append points to polygon
            polygon.append(p)
        
        painter.drawPolygon(polygon,2)
        
        #setup brush for white ellipse
        brush = QBrush(QColor("white"))
        painter.setBrush(brush)
        

        painter.drawEllipse(ellipseRect)

        # draw main label
        painter.setFont(self.getMainFont())
        painter.drawText(self.getMainLabelRect(x0,y0,w,h),
                            Qt.AlignCenter | Qt.AlignTop, 
                            "filter")


class SelectButton(PushHoverButton):
    """
    Select Button. Inherits from PushHoverButton.

    Overwrite painEvent to display Color Button.
    PushHoverButton contains a main label, and 
    several color-circles.

    The PushHoverButton Class provides Function 
    for style such as the Font for mainLabel.
    """
    def __init__(self,parent=None, *args, **kwargs):
        super(SelectButton,self).__init__(parent, *args, **kwargs)
       # self.displayColors = ["#A9A8A7","#CECDCC","FAFBFB"]
        
        #self.radiiFrac = [(n+1)/4.5 for n in range(3)]

    def paintEvent(self,event):
        
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)
        r = 0.3 * h

        #create common background/reacts to hover
        super().paintEvent(event)
        
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        #setup brush for circles
        c = QColor(WIDGET_HOVER_COLOR if self.mouseOver else INSTANT_CLUE_BLUE)
        c.setAlphaF(0.75)
        b = QBrush(c)
        painter.setBrush(b)

        for p in self.getCirclePoints(x0,y0,h):
            painter.drawEllipse(p,h/10,h/10)
        
        # setup brush for white overlayed circle
        c = QColor("white")
        c.setAlphaF(0.5)
        b = QBrush(c)
        painter.setBrush(b)
        centerPoint = QPointF(x0+h/20,y0-h/20)
        painter.drawEllipse(centerPoint,h/4,h/4)

        # draw main label
        painter.setFont(self.getMainFont())
        painter.drawText(self.getMainLabelRect(x0,y0,w,h),
                            Qt.AlignCenter | Qt.AlignTop, 
                            "select")

    def getCirclePoints(self,x0,y0,h):
        ""
        centerPoints = [
                QPointF(x0,y0),
                QPointF(x0+h/7,y0-h/7),
                QPointF(x0-h/5,y0-h/5)
            ]
        return centerPoints


class ColorButton(PushHoverButton):
    """
    Color Button. 

    Overwrite painEvent to display Color Button.
    PushHoverButton contains a main label, and 
    several color-circles.

    The PushHoverButton Class provides Function 
    for style such as the Font for mainLabel.
    """

    def __init__(self,parent=None, *args, **kwargs):
        super(ColorButton,self).__init__(parent,*args, **kwargs)
        self.displayColors = ["white","#397546",INSTANT_CLUE_BLUE,"#A0D4CB"]
        

    def paintEvent(self,event):
        
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)
        r = 0.11 * h


        #create common background/reacts to hover
        super().paintEvent(event)
        
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        offset = [(-h/6,h/5),(h/6,h/6),(0,h/10),(-h/7,-h/12)]
        rScale = [1,1.2,1.3,1]
        
        for n,circleColor in enumerate(self.displayColors):
            dx = offset[n][0]
            dy = offset[n][1]
            scale = rScale[n]
            rCircle = scale * r
            c = QColor(circleColor)
            c.setAlphaF(0.55 if not self.mouseOver else 0.9)
            b=QBrush(c)
            painter.setBrush(b)
            painter.drawEllipse(QPointF(x0+dx,y0-dy),rCircle ,rCircle )
        # draw main label
        painter.setFont(self.getMainFont())
        painter.drawText(self.getMainLabelRect(x0,y0,w,h),
                         Qt.AlignCenter | Qt.AlignTop, 
                         "color")
 
class MarkerButton(PushHoverButton):
    """
    Marker Button. 

    Overwrite painEvent to display Color Button.
    PushHoverButton contains a main label, and 
    several color-circles.

    The PushHoverButton Class provides Function 
    for style such as the Font for mainLabel.
    """

    def __init__(self,parent=None, *args, **kwargs):
        super(MarkerButton,self).__init__(parent,*args, **kwargs)
        self.displayColors = ["#397546","white",INSTANT_CLUE_BLUE]#,"#A0D4CB"
        

    def paintEvent(self,event):
        
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)
        r = 0.11 * h


        #create common background/reacts to hover
        super().paintEvent(event)
        
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        offset = [(-h/6,h/5),(h/6,h/6),(-h/8,-h/7)]
        rScale = [1,1.2,1.3,1]
        
        for n,circleColor in enumerate(self.displayColors):
            dx = offset[n][0]
            dy = offset[n][1]
            scale = rScale[n]
            rCircle = scale * r
            c = QColor(circleColor)
            c.setAlphaF(0.55 if not self.mouseOver else 0.75)
            b=QBrush(c)
            painter.setBrush(b)
            x1 = x0+dx
            y1 = y0-dy
            if n == 0:
                painter.drawEllipse(QPointF(x1,y1),rCircle ,rCircle )
            elif n == 1:
                painter.drawLine(x1-+h/7,y1,x1+h/7,y1)
                painter.drawLine(x1,y1-h/7,x1,y1+h/7)
            elif n == 2:
                polyPoints = QPolygon([QPoint(x1,y1),QPoint(x1+h/9,y1-h/9*2),QPoint(x1+h/9*2,y1)])
                painter.drawPolygon(polyPoints)
        # draw main label
        painter.setFont(self.getMainFont())
        painter.drawText(self.getMainLabelRect(x0,y0,w,h),
                         Qt.AlignCenter | Qt.AlignTop, 
                         "marker")
 

class LabelButton(PushHoverButton):
    """
    Label Button. 

    Overwrite painEvent to display Label Button.
    PushHoverButton contains a main label, circle 
    and a sublabel.

    The PushHoverButton Class provides Function 
    for style such as the Font for mainLabel.
    """

    def __init__(self,parent=None, **kwargs):

        super(LabelButton,self).__init__(parent, **kwargs)
        
   
    def paintEvent(self,event):
        
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)
        r = 0.12 * h


        #create common background/reacts to hover
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        font = QFont("Arial")
        font.setPointSize(8)
        #font.setCapitalization(QFont.SmallCaps)
        painter.setFont(font)
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)
        
        b = QBrush(QColor(WIDGET_HOVER_COLOR if self.mouseOver else "#2776BC"))

        #draw sublabel
        painter.drawText(QPointF(x0+0.5*r,y0-h/5),"text")

        #draw main label
        painter.setFont(self.getMainFont())
        painter.drawText(self.getMainLabelRect(x0,y0,w,h),
                        Qt.AlignCenter | Qt.AlignTop, 
                        "labels")

        painter.setBrush(b)
        painter.drawEllipse(QPointF(x0,y0),r,r)
 

class TooltipButton(PushHoverButton):
    """
    Tooltip Button. 

    Overwrite painEvent to display Tooltip Button.

    """
    def __init__(self,parent=None, **kwargs):
        super(TooltipButton,self).__init__(parent,funcKey="data::getDataByColumnNamesForTooltip", **kwargs)
        
    def paintEvent(self,event):
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)
        r = 0.12 * h

        #create common background/reacts to hover
        super().paintEvent(event)
        painter = QPainter(self)
        #set up pen for border
        pen = QPen(QColor("black"))
        pen.setWidthF(0.2)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        #brush for rectangle
        c = QColor("#2776BC")
        c.setAlphaF(0.9)
        painter.setBrush(QBrush(c))

        #draw rectangle
        painter.drawRoundedRect(QRectF(x0-h/20,y0-h/2.5,h/2,h/2.5),2,2)

        #set up pen for tooltip lines
        pen.setColor(QColor("white"))
        pen.setWidthF(0.76)
        painter.setPen(pen)

        dy = h/2.5 / 4
        for lineIdx in range(3):
            x1 = x0 - h/20 + h/19 
            y = y0 - h /2.5 + h/2.5 - dy * (lineIdx+1)
            x2 = x0 + h/2 - h/19
            painter.drawLine(QPointF(x1,y),QPointF(x2,y))

        #set back the border pen
        pen.setColor(QColor("black"))
        pen.setWidthF(0.2)
        painter.setPen(pen)

        b = QBrush(QColor(WIDGET_HOVER_COLOR if self.mouseOver else "lightgrey"))
        painter.setBrush(b)
        painter.drawEllipse(QPointF(x0-h/5,y0+h/20),r,r)

        #draw main label
        painter.setFont(self.getMainFont())
        painter.drawText(self.getMainLabelRect(x0,y0,w,h),
                        Qt.AlignCenter | Qt.AlignTop, 
                        "tooltip")

# Buttons for Dialogs

class AcceptButton(PushHoverButton):
    def __init__(self,parent=None, **kwargs):
        super(AcceptButton,self).__init__(parent, acceptDrops=False, **kwargs)

    def sizeHint(self):
        ""
        return QSize(25,25)

    def paintEvent(self,event):
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)
        r = 0.12 * h

        #create common background/reacts to hover
        super().paintEvent(event)
        painter = QPainter(self)
        #set up pen for border
        pen = QPen(QColor(INSTANT_CLUE_BLUE))
        pen.setWidthF(2.5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        #set no brush (no filling)
        painter.setBrush(Qt.NoBrush)
        #init path
        path = QPainterPath()
        path.moveTo(x0-h/4,y0+h/12)
        path.lineTo(x0,y0+h/3)
        path.lineTo(x0+h/3,y0-h/3)
        
        #draw path
        painter.drawPath(path)



class RefreshButton(PushHoverButton):
    def __init__(self,parent=None):
        super(RefreshButton,self).__init__(parent, acceptDrops=False)
        self.setToolTip("Refresh filter.")

    def sizeHint(self):
        ""
        return QSize(25,25)
        
    def paintEvent(self,event):
        #calculate rectangle props
        rect, h, w, x0, y0 = self.getRectProps(event)

        #create common background/reacts to hover
        super().paintEvent(event)
        painter = QPainter(self)
        #set up pen for border
        pen = QPen(QColor(INSTANT_CLUE_BLUE))
        pen.setWidthF(1)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)

        #set no brush (no filling)
        painter.setBrush(Qt.NoBrush)
        #init circle
        d = h/2
        r = d/2
        #draw circle
        painter.drawEllipse(QRectF(x0-d/2,y0-d/2,d,d))

        brush = QBrush(QColor(INSTANT_CLUE_BLUE))
        painter.setBrush(brush)
        #init arrows
        arrowLength = d / 4
        arrowWidth = arrowLength * 1.1
        arrowYOffset = arrowLength / 1.5
        #init path
        path = QPainterPath()
        #right arrow
        path.moveTo(x0+r-arrowWidth,
                    y0+arrowLength)
        path.lineTo(x0+r,
                    y0-arrowYOffset)
        path.lineTo(x0+r+arrowWidth,
                    y0+arrowLength)
        
        path.lineTo(x0+r-arrowWidth,
                    y0+arrowLength)
        #left arrow
        path.moveTo(x0-r-arrowWidth,
                    y0-arrowLength)
        path.lineTo(x0-r,
                    y0+arrowYOffset)
        path.lineTo(x0-r+arrowWidth,
                    y0-arrowLength)
        path.lineTo(x0-r-arrowWidth,
                    y0-arrowLength)
        #draw complete path
        painter.drawPath(path)




#Reset Button for Quick Select
class ResetButton(PushHoverButton):
    def __init__(self, parent=None, penColor = "#B84D29", strokeWidth = 2,*args, **kwargs):
        super(ResetButton, self).__init__(parent=parent, acceptDrops = False,*args, **kwargs)

        self.strokeWidth = strokeWidth
        self.penColor = penColor
       
    
    def sizeHint(self):
        ""
        return QSize(15,15)

    def paintEvent(self, e):

        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        centerPoint = e.rect().center()
        pen = QPen(QColor(self.penColor))
        pen.setWidthF(self.strokeWidth)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        x0 = centerPoint.x()
        y0 = centerPoint.y()
        h = e.rect().height()/4
        painter.drawLine(QLineF(x0-h,y0-h,x0+h,y0+h))
        painter.drawLine(QLineF(x0+h,y0-h,x0-h,y0+h))
        

    def deltaUp(self,h):
        return [(h,h),(-h,h),(0,-h)]

    def deltaDown(self,h):
        return [(-h,-h),(h,-h),(0,h)]



#Reset Button for Quick Select
class SmallColorButton(PushHoverButton):
    def __init__(self, parent=None,*args, **kwargs):
        super(SmallColorButton, self).__init__(parent=parent, acceptDrops = False,*args, **kwargs)
        self.displayColors = [WIDGET_HOVER_COLOR,INSTANT_CLUE_BLUE,"#A0D4CB"] #"#397546"
    
    def sizeHint(self):
        ""
        return QSize(15,15)

    def paintEvent(self, e):

        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        centerPoint = e.rect().center()
        pen = QPen(QColor("darkgrey"))
        pen.setWidthF(0.5)
        #pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)


        x0 = centerPoint.x()
        y0 = centerPoint.y()
        h = e.rect().height()/4

        for n,(x,y) in enumerate([(x0-h/8,y0-h/2), (x0-h/8,y0+h/2),(x0+h,y0)]):
            p = QPointF(x,y)
            color = QColor(self.displayColors[n])
            color.setAlphaF(0.8)
            brush = QBrush(color)
            painter.setBrush(brush)
            painter.drawEllipse(p,3,3)



class ResortButton(PushHoverButton):
    def __init__(self, parent=None, strokeWidth = 0.5, **kwargs):
        super(ResortButton, self).__init__(parent=parent, acceptDrops = False, **kwargs)

        self.strokeWidth = strokeWidth
    
    def sizeHint(self):
        ""
        return QSize(15,15)

    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        centerPoint = e.rect().center()
        pen = QPen(QColor("black"))
        pen.setWidthF(self.strokeWidth)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        brush = QBrush(QColor(INSTANT_CLUE_BLUE if not self.mouseOver else WIDGET_HOVER_COLOR))
        painter.setBrush(brush)
        x0 = centerPoint.x()
        y0 = centerPoint.y()
        h = e.rect().height()

        painter.drawPolygon(self.makeArrow(QPointF(x0-h/5,y0), h , "up"))
        painter.drawPolygon(self.makeArrow(QPointF(x0+h/5,y0), h , "down"))
    

    def deltaUp(self,h):
        return [(h,h),(-h,h),(0,-h)]

    def deltaDown(self,h):
        return [(-h,-h),(h,-h),(0,h)]

    def makeArrow(self,centerPoint,height, direction = "up"):
        x0 = centerPoint.x()
        y0 = centerPoint.y()
        h = height/5
        poly = QPolygonF()
        if direction == "up":
            deltaXY = self.deltaUp(h)
        else:
            deltaXY = self.deltaDown(h)
        for dx,dy in deltaXY:
            baseP = QPointF(centerPoint)
            baseP.setX(x0 + dx)
            baseP.setY(y0 + dy)
            poly.append(baseP)
        return poly

class ArrowButton(PushHoverButton):
    def __init__(self, parent=None, direction = "up", brushColor = "#5c5f77", strokeWidth = 0.5, **kwargs):
        super(ArrowButton, self).__init__(parent=parent, acceptDrops = False, **kwargs)
        self.direction = direction
        self.strokeWidth = strokeWidth
        self.brushColor = brushColor
       

    def sizeHint(self):
        ""
        return QSize(15,15)

    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        centerPoint = e.rect().center()
        brush = QBrush(QColor(self.brushColor))
        pen = QPen(QColor("black"))
        pen.setWidthF(self.strokeWidth)
        painter.setBrush(brush)
        painter.setPen(pen)

        poly = self.makeArrow(centerPoint, e.rect().height())
        painter.drawPolygon(poly)

    def deltaUp(self,h):
        return [(h,h),(-h,h),(0,-h)]

    def deltaDown(self,h):
        return [(-h,-h),(h,-h),(0,h)]

    def makeArrow(self,centerPoint,height):
        x0 = centerPoint.x()
        y0 = centerPoint.y()
        h = height/4
        poly = QPolygonF() 
        if self.direction == "up":
            deltaXY = self.deltaUp(h)
        else:
            deltaXY = self.deltaDown(h)
        for dx,dy in deltaXY:
            baseP = QPointF(centerPoint)
            baseP.setX(x0 + dx)
            baseP.setY(y0 + dy)
            poly.append(baseP)
        return poly


class CheckButton(PushHoverButton):
    def __init__(self, parent=None, penColor = "#B84D29", strokeWidth = 0.2, **kwargs):
        super(CheckButton, self).__init__(parent=parent, acceptDrops = False, **kwargs)

        self.strokeWidth = strokeWidth
        self.penColor = penColor
        self.state = True
       
    
    def sizeHint(self):
        ""
        return QSize(15,15)

    def paintEvent(self, e):

        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        centerPoint = e.rect().center()
        pen = QPen(QColor(self.penColor))
        pen.setWidthF(self.strokeWidth)
        brush = QBrush(QColor(INSTANT_CLUE_BLUE if self.state else "white"))
        painter.setBrush(brush)
        
        painter.drawEllipse(centerPoint,4,4)
    
    def setState(self,newState):
        "Sets state"
        self.state = newState
        self.update()

    def getState(self):
        "Returns current state"
        return self.state

    def toggleState(self):
        "Toggle states"
        self.state = not self.state
        self.update()


class SaveButton(PushHoverButton):
    def __init__(self, parent=None, penColor = "#B84D29", strokeWidth = 0.1, **kwargs):
        super(SaveButton, self).__init__(parent=parent, acceptDrops = False, **kwargs)

        self.strokeWidth = strokeWidth
        self.penColor = penColor
        self.state = True
        if tooltipStr is not None and isinstance(tooltipStr,str):
            self.setToolTip(tooltipStr)
    
    def sizeHint(self):
        ""
        return QSize(15,15)

    def paintEvent(self, e):

        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        #some button calc
        centerPoint = e.rect().center()
        height = w = e.rect().height()
        x0 = centerPoint.x()
        y0 = centerPoint.y()
        h = height/3

        pen = QPen(QColor(self.penColor))
        pen.setWidthF(self.strokeWidth)
        brush = QBrush(QColor(INSTANT_CLUE_BLUE if self.state else "white"))
        painter.setBrush(brush)
        painter.setPen(pen)
        path = QPainterPath()
        topCornerOff = h/3
        path.moveTo(x0-h,y0-h)
        path.lineTo(x0+h-topCornerOff,y0-h)
        path.lineTo(x0+h,y0-h+topCornerOff*2)
        path.lineTo(x0+h,y0+h)
        path.lineTo(x0-h,y0+h)
        path.lineTo(x0-h,y0-h)

        painter.drawPath(path)
        painter.setPen(Qt.NoPen)
        brush = QBrush(QColor("white"))
        painter.setBrush(brush)
        heighRect = h / 1.5
        painter.drawRect(x0-topCornerOff,y0-heighRect,h,heighRect)
        borderBttnRect = h/3 
        painter.drawRect(QRectF(x0-h+borderBttnRect,y0,h*2-borderBttnRect*2,h-borderBttnRect))
        
       


class MaskButton(PushHoverButton):
    def __init__(self, parent=None, penColor = "#B84D29", strokeWidth = 0.05, **kwargs):
        super(MaskButton, self).__init__(parent=parent, acceptDrops = False, **kwargs)

        self.strokeWidth = strokeWidth
        self.penColor = penColor
        self.state = True
        
    
    def sizeHint(self):
        ""
        return QSize(15,15)

    def paintEvent(self, e):

        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        centerPoint = e.rect().center()
      
        painter.setPen(Qt.NoPen)
        h = w = e.rect().height()/3
        centerPoint = e.rect().center()
        x0 = centerPoint.x()
        y0 = centerPoint.y()

        # draw left (lighter) half
        path = QPainterPath()
        path.moveTo(x0-w,y0-h/1.5)
        path.quadTo(x0-w,y0-h,x0,y0-h)
        path.lineTo(x0,y0+h)
        path.quadTo(x0-w/2,y0+h,x0-w,y0)
        path.lineTo(x0-w,y0-h/1.5)
        b = QBrush(QColor(INSTANT_CLUE_BLUE if not self.state else WIDGET_HOVER_COLOR))
        painter.setBrush(b)
        painter.drawPath(path)

        #draw right (Dark) half
        path.clear()
        path.moveTo(x0+w,y0-h/1.5)
        path.quadTo(x0+w,y0-h,x0,y0-h)
        path.lineTo(x0,y0+h)
        path.quadTo(x0+w/2,y0+h,x0+w,y0)
        path.lineTo(x0+w,y0-h/1.5)
        
      
        painter.drawPath(path)
        b = QBrush(QColor("#E4DED4"))
        painter.setBrush(b)

        #draw smile
        path.clear()
        path.moveTo(x0-w/2,y0+h/3)
        path.quadTo(x0,y0+h/2,x0+w/2,y0+h/3)
        path.quadTo(x0,y0+h,x0-w/2,y0+h/3)
        painter.drawPath(path)
        

        #draw left eye
        path.clear()
        eyeHeight = h/2
        eyeWidth = h/2
        eyeLeftRect = QRectF(x0-w/4-h/2,y0-h/2,eyeWidth,eyeHeight)
        path.moveTo(eyeLeftRect.center())
        path.arcTo(eyeLeftRect,0,180)
        painter.drawPath(path)
         
        #darw right eye
        path.clear()
        eyeRightRect = QRectF(x0+w/4,y0-h/2,eyeWidth,eyeHeight)
        path.moveTo(eyeRightRect.center())
        path.arcTo(eyeRightRect,0,180)
        painter.drawPath(path)
    
    def setState(self,newState):
        ""
        self.state = newState
        self.update()

    def toggleState(self):
        ""
        self.state = not self.state
        self.update()   
 

class AnnotateButton(PushHoverButton):
    def __init__(self, parent=None, penColor = "#B84D29", strokeWidth = 0.05, numCircles = 5, **kwargs):
        super(AnnotateButton, self).__init__(parent=parent, acceptDrops = False, **kwargs)

        self.strokeWidth = strokeWidth
        self.penColor = penColor
        self.numCircles = numCircles
        self.state = False
    
    def createData(self):
        ""
        self.circleData = np.random.normal(size=(self.numCircles,2), scale = 0.7)

    def sizeHint(self):
        ""
        return QSize(15,15)

    def paintEvent(self, e):
        if not hasattr(self,"circleData") or self.mouseOver:
            self.createData()
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        centerPoint = e.rect().center()
      # pen = QPen(QColor(self.penColor))
      #  pen.setWidthF(self.strokeWidth)
        #brush = QBrush(QColor(INSTANT_CLUE_BLUE if self.state else "white"))
        #painter.setBrush(brush)
        painter.setPen(Qt.NoPen)
        h = w = e.rect().height()/4
        centerPoint = e.rect().center()
        x0 = centerPoint.x()
        y0 = centerPoint.y()

        pen = QPen(QColor("black"))
        pen.setWidthF(0.2)
        painter.setPen(pen)
        
        
        whiteBrush = QBrush(QColor("white"))
        
        for n,xy in enumerate(self.circleData):
            x = x0+xy[0]*h
            y = y0+h*xy[1]

            if n == self.numCircles - 1:
                r = 3
                c = QColor(INSTANT_CLUE_BLUE if not self.state else WIDGET_HOVER_COLOR)
                c.setAlphaF(0.9)
                brush = QBrush(c)
                painter.setBrush(brush)
            else:
                r = 3
                painter.setBrush(whiteBrush)
            borderMin = x0 - w * 2
            borderMax = x0 + w * 2
            if x + r > borderMax or x - r < borderMin:
                continue
            elif y + r > borderMax or y - r < borderMin:
                continue

            painter.drawEllipse(QPointF(x,y),r,r)

    def setState(self,newState):
        ""
        self.state = newState
        self.update()

    def toggleState(self):
        ""
        self.state = not self.state
        self.update()



#main buttons
class ViewHideIcon(PushHoverButton):

    def __init__(self,parent=None):
        super(ViewHideIcon,self).__init__(parent, acceptDrops = False)

        self.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.setState()
        self.setToolTip("Enable/Disable Shortcut Icons")

    def sizeHint(self):
        ""
        return QSize(15,15)
    
    def setState(self,state=True):
        ""
        self.state = state
    
    def stateChanged(self):
        ""
        self.state = not self.state
        self.update()

    def paintEvent(self, e):
        ""
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        rect = e.rect()
        centerPoint = rect.center()
        h = rect.height() 
        outerEllipseHeight = h/2
        outerEllipseWidth = outerEllipseHeight * 1.5
        outerEllipseRect = QRectF(
                                    centerPoint.x() - outerEllipseWidth/2,
                                    centerPoint.y() - outerEllipseHeight/2,
                                    outerEllipseWidth,
                                    outerEllipseHeight
                                )
        innerCircleRadius = outerEllipseHeight / 5
        
        brush = QBrush(QColor(INSTANT_CLUE_BLUE if self.state else WIDGET_HOVER_COLOR))
        painter.setBrush(brush)

        pen = QPen(QColor("black"))
        pen.setWidthF(0.75)
        painter.setPen(pen)
        
        painter.drawEllipse(outerEllipseRect)

        brush = QBrush(QColor("white"))
        painter.setBrush(brush)
        painter.drawEllipse(centerPoint,innerCircleRadius,innerCircleRadius)

        if not self.state:
            
            pen = QPen(QColor("black"))
            pen.setWidthF(1.5)
            painter.setPen(pen)
            lineHW = outerEllipseHeight / 2
            xy1 = QPointF(centerPoint.x() - lineHW, centerPoint.y() + lineHW)
            xy2 = QPointF(centerPoint.x() + lineHW, centerPoint.y() - lineHW)
            painter.drawLine(xy1,xy2)

class FindReplaceButton(PushHoverButton):

    def __init__(self,parent=None):
        super(FindReplaceButton,self).__init__(parent)

        self.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.setToolTip("Find & replace values/header names")

    def sizeHint(self):
        ""
        return QSize(15,15)


    def paintEvent(self, e):
        ""
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        rect = e.rect()
        centerPoint = rect.center()
        h = rect.height() 
        brush = QBrush(QColor(INSTANT_CLUE_BLUE if not self.mouseOver else WIDGET_HOVER_COLOR))
        painter.setBrush(brush)

        pen = QPen(QColor("black"))
        pen.setWidthF(0.75)
        painter.setPen(pen)

        #calculate constants for circles
        distFromCenter = h/10
        IDCircle = h/4
        x0 = centerPoint.x()
        y0 = centerPoint.y()
        
        upperRightRect = QRectF(
                                x0 + distFromCenter,
                                y0 - distFromCenter - IDCircle,
                                IDCircle,
                                IDCircle
                            )
        bottomLeftRect = QRectF(
                        x0 - distFromCenter - IDCircle,
                        y0 + distFromCenter,
                        IDCircle,
                        IDCircle
                    )
        

        #draw circles
        painter.drawEllipse(upperRightRect)
        painter.drawEllipse(bottomLeftRect)
        #add arrows
        #set no brush (no filling)
        painter.setBrush(Qt.NoBrush)
        #calculated constansts
        distFromCircle = IDCircle / 3
        distToCubicCorner = IDCircle / 2
        #init painter path
        path = QPainterPath()
        path.moveTo(x0,y0 - distFromCenter - distToCubicCorner)
        path.lineTo(x0-distFromCenter,y0 - distFromCenter - distToCubicCorner)
        path.quadTo(QPointF(x0-distFromCenter - distToCubicCorner, y0 - distFromCenter - distFromCircle),
                    QPointF(x0-distFromCenter - distToCubicCorner, y0 + distFromCenter  - distFromCircle))
        painter.drawPath(path)

        path.clear() 

        path.moveTo(x0,y0 + distFromCenter + distToCubicCorner)
        path.lineTo(x0 + distFromCenter, y0 + distFromCenter + distToCubicCorner)
        path.quadTo(QPointF(x0 + distFromCenter + distToCubicCorner, y0 + distFromCenter + distFromCircle),
                    QPointF(x0 + distFromCenter + distToCubicCorner, y0 - distFromCenter + distFromCircle))
        painter.drawPath(path)
        




def generateRandomCorrelationData(x0,x1,y0,y1,n):

    xx = np.array([x0, x1])
    yy = np.array([y0, y1])
    means = [xx.mean(), yy.mean()]
    stds = [xx.std() / 3, yy.std() / 3]
    corr = 0.8         # correlation
    covs = [[stds[0]**2          , stds[0]*stds[1]*corr],
            [stds[0]*stds[1]*corr,           stds[1]**2]]

    m = np.random.multivariate_normal(means, covs, n)
    return m

class PlotTypeButton(PushHoverButton):

    def __init__(self,parent=None, plotType = None,**kwargs):

        super(PlotTypeButton,self).__init__(parent,**kwargs)
        self.plotType = plotType
        self.funcKeyDict = {"scatter"       :   self.drawScatterPoints,
                            "boxplot"       :   self.drawBoxes,
                            "hclust"        :   self.drawHeatmap,
                            "countplot"     :   self.drawCountPlot,
                            "barplot"       :   self.drawBars,
                            "lineplot"      :   self.drawLines,
                            "pointplot"     :   self.drawPoint,
                            "histogram"     :   self.drawHistogram,
                            "swarmplot"     :   self.drawSwarm,
                            "corrmatrix"    :   self.drawCorrMatrix,
                            "countplot"     :   self.drawCountPlot,
                            "x-ys-plot"     :   self.drawXYPlot,
                            "addSwarmplot"  :   self.drawSwarmAdd,
                            "dim-red-plot"  :   self.drawDimRed}

    def paintEvent(self,event, noAxis = False):
        ""
        #create common background/reacts to hover
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)
        #get rect
        rect = event.rect()
        try:
            if not noAxis:
                rectHeight, rectWidth, rectX, rectY = self.drawAxis(painter,rect,False)
            if self.plotType is not None and self.plotType in self.funcKeyDict:
                self.funcKeyDict[self.plotType](painter,rectHeight,rectWidth, rectX, rectY)
        except Exception as e:
            print(e)


    def addMargin(self,rect):
        ""
        rect.adjust(10,10,-10,-10)
        rectHeight = rect.height()
        rectWidth = rect.width()
        rectX = rect.x()
        rectY = rect.y()
        return rectHeight, rectWidth, rectX, rectY

    def drawAxis(self,qp, rect, slopeOneLine = True, offset = 7):
        ""
        rect.adjust(offset,offset,-offset,-offset)

        rectHeight = rect.height()
        rectWidth = rect.width()
        rectX = rect.x()
        rectY = rect.y()

        topLeft = QPointF(rectX, rectY)
        topRight = QPointF(rectX + rectWidth, rectY)
        bottomLeft = QPointF(rectX, rectY + rectHeight)
        bottomRight = QPointF(rectX + rectWidth, rectY + rectHeight)

        #draw y axis
        qp.drawLine(topLeft,
                    bottomLeft + QPointF(0,4))

        #draw x axis
        qp.drawLine(bottomLeft - QPointF(4,0),
                    bottomRight)

        if slopeOneLine:
            qp.drawLine(bottomLeft,
                        topRight)

        return rectHeight, rectWidth, rectX, rectY


    def drawCorrMatrix(self,qp, height, width, rectX, rectY, X = None):
        ""
        if X is None:
            X = np.array([
                [1,	0.934801696,0.881666879,	0.308717652,	0.390402581,	0.468917592,	0.346362161	,0.040402337,	0.185281612],
                [0.934801696,   1,	0.93773992,	0.301873333,	0.419337806,	0.462718025,	0.338080257	,0.186876339,	0.2697334],
                [0.881666879,	0.93773992,	1,	0.251522608,	0.353500602,	0.407909063,	0.253350451	,0.114274697,	0.206122285],
                [0.308717652,	0.301873333,	0.251522608,	1,	0.924865715,	0.923766029,	0.91432637,	0.671117164,	0.731599326],
                [0.390402581,	0.419337806,	0.353500602,	0.924865715,	1,	0.968446562, 0.830105634,	0.644100516,	0.691674869],
                [0.468917592,	0.462718025,	0.407909063,	0.923766029,	0.968446562,	1,	0.833303147,	0.638518706,	0.710947986],
                [0.346362161,	0.338080257,	0.253350451,	0.91432637,	0.830105634,	0.833303147, 1,	0.967413652,	0.966912352],
                [0.040402337,	0.186876339,	0.114274697,	0.671117164,	0.644100516,	0.638518706,	0.967413652,	1,	0.974941771],
                [0.185281612,	0.2697334,	0.206122285,	0.731599326,	0.691674869,	0.710947986,	0.966912352,	0.974941771,	1]
                        ])

        self.drawHeatmap(qp, height, width, rectX, rectY, X =X)

    def drawXYPlot(self,qp, height, width, rectX, rectY, nLines = 3, nPoints = 5):
        ""
        try:
            defaultColors = ["#A0D4CB",INSTANT_CLUE_BLUE,"#397546"]
            margin = width / 20
            distPoints = (width - 2*margin) / nPoints
            xValues = [rectX + margin + n * distPoints for n in range(nPoints)]
            path = QPainterPath()
            for n in range(nLines):
                path.clear()
                path.moveTo(xValues[0] , rectY + height - margin)
                for xValue in xValues[1:]:
                    xyPoint = QPointF(xValue, rectY + height - np.random.random() * 3 * distPoints - margin)
                    path.lineTo(xyPoint)
                    qp.drawEllipse(xyPoint,2,2)
                pen = QPen(QColor(defaultColors[n]))
                qp.setPen(pen)
                qp.drawPath(path)
            

        except Exception as e:
            print(e)


    def drawCountPlot(self,qp, height, width, rectX, rectY):
        ""
       
        nBars = 8
        hightlighBarInt = np.random.randint(low=0,high=nBars,size=2)
        margin = width / 10
        distBars = (width - margin) / (nBars)
        xValues = [rectX + margin + n * distBars for n in range(nBars)]
        heights = np.sort(np.abs(np.random.normal(size = nBars))) + 0.1
        fracHeights = heights / np.max(heights) 
        for n in range(nBars):
            y0 = (height - height * (1 - fracHeights[n])) + margin
            if self.mouseOver:
                if n in hightlighBarInt:
                    qp.setBrush(QBrush(QColor(INSTANT_CLUE_BLUE)))
                else:
                    qp.setBrush(QBrush(QColor("white")))
            rect = QRectF(xValues[n],y0,distBars,rectX+height-y0)
            qp.drawRect(rect)
        qp.setFont(self.getStandardFont())
        qp.drawText(width,height/3,"n")


    def drawSwarm(self,qp, height, width, rectX, rectY, nSwarms = 3, allWhite=False):
        ""
        
        defaultColors = ["white",INSTANT_CLUE_BLUE,"#A0D4CB"]
        margin = width / 4
        distPoints = (width - margin) / (nSwarms)
        xValues = [rectX + margin + n * distPoints for n in range(nSwarms)]

        pen = QPen(QColor("black"))#
        pen.setWidthF(0.1)
        qp.setPen(pen)
        for n,xValue in enumerate(xValues):

            xValuesN = np.random.normal(loc=xValue,scale=0.8,size=25)
            yValuesN = np.random.normal(loc=xValue,scale=3,size=25)                         
            if allWhite:
                qp.setBrush(QBrush(QColor("white")))
            else:
                qp.setBrush(QBrush(QColor(defaultColors[n])))
            for x, y in zip(xValuesN,yValuesN):

                qp.drawEllipse(QPointF(x,y),2,2)

    def drawSwarmAdd(self,qp, height, width, rectX, rectY):
        ""
        self.drawSwarm(qp, height, width, rectX, rectY, 3, True)
        qp.setFont(self.getStandardFont())
        qp.drawText(width,height/3,"+")


    def drawHistogram(self,qp, height, width, rectX, rectY, nBars = 10):
        ""
        defaultColors = ["white",INSTANT_CLUE_BLUE,"#A0D4CB"]
        margin = width / 10
        distBars = (width - margin) / (nBars)
        xValues = [rectX + margin + n * distBars for n in range(nBars)]
        heights = np.abs(np.sort(np.random.normal(size = nBars)))
        fracHeights = heights / np.max(heights) 
        
        for n in range(nBars):
            y0 = (height - height * (1- fracHeights[n])) + margin
            rect = QRectF(xValues[n],y0,distBars,rectX+height-y0)
            qp.drawRect(rect)

    def drawPoint(self,qp, height, width, rectX, rectY, nCats = 3, nPoints = 2):
        "nCats = how many categories (e..g different colors)"
        defaultColors = ["white",INSTANT_CLUE_BLUE,"#A0D4CB"]
        margin = width / 10
        distPoints = (width - margin) / (nPoints)
        xValues = [rectX + margin + n * distPoints for n in range(nPoints)]
        yPointOffeset = [[0.2,0.5],[0.4,0.8],[0.6,0.3]]
        for n in range(nCats):
            brush = QBrush(QColor(defaultColors[n]))
            qp.setBrush(brush)
            saveY = []
            saveEllRect  = []
            for m,xValue in enumerate(xValues):
                y = rectY + height - yPointOffeset[n][m] * height - margin
                ellRect = QRectF(xValue,y,6,6)
                saveEllRect.append(ellRect)
                saveY.append(y)
            qp.drawLine(QPointF(xValues[0] + 3,saveY[0] + 3),
                        QPointF(xValues[1]+ 3,saveY[1] + 3))
            for ellRect in saveEllRect:
                qp.drawEllipse(ellRect)
    
    
    def drawLines(self, qp, height, width, rectX, rectY, nLines = 4, nPoints = 5):
        ""
        defaultColors = ["darkgrey","#A0D4CB",INSTANT_CLUE_BLUE,"#397546"]
        margin = width / 20
        distPoints = (width - 2*margin) / nPoints
        xValues = [rectX + margin + n * distPoints for n in range(nPoints)]
        path = QPainterPath()
        for n in range(nLines):
            path.clear()
            path.moveTo(xValues[0] , rectY + height - margin)
            for xValue in xValues[1:]:
                path.lineTo(QPointF(xValue, rectY + height - np.random.random() * 3 * distPoints - margin))
            pen = QPen(QColor(defaultColors[n]))
            qp.setPen(pen)
            qp.drawPath(path)

    def drawDimRed(self,qp, height, width, rectX, rectY):
        ""
        
        w8 = width/8
        points = [
                    (rectX+w8,rectY+w8),
                    (rectX+w8*1.2,rectY+w8*0.2),
                    (rectX+w8,rectY+height-w8),
                    (rectX+w8*1.2, rectY + height - w8*1.5),
                    (rectX + width/1.2, rectY+height/2 - w8*0.2),
                    (rectX+ width/1.1,rectY+height/2 + w8*0.5)

                ]
        colors = ["#A0D4CB","#A0D4CB",INSTANT_CLUE_BLUE,INSTANT_CLUE_BLUE,"white","white"]
        for n, (x,y) in enumerate(points):
            b = QBrush(QColor(colors[n]))
            qp.setBrush(b)
            xy = QPointF(x,y) 
            qp.drawEllipse(xy,3,3)


    def drawScatterPoints(self,qp, height, width, rectX, rectY):
        b = QBrush(QColor("lightgrey"))
        qp.setBrush(b)
        for x,y in generateRandomCorrelationData(0,width,0,height,n=20):
            xy = QPointF(x + 10, height - y + 10)
            qp.drawEllipse( xy,3,3)

    def drawBars(self,qp, height, width, rectX, rectY, nBars=4):
        ""
        defaultColors = ["white","#A0D4CB",INSTANT_CLUE_BLUE,"#397546"]
        nColors = len(defaultColors)
        barWidth = width *  1/(nBars+1)
        margin = barWidth / (nBars+1)


        for i in range(nBars):
            startX = rectX + (1+i) * margin + i * barWidth 
            h = height * np.random.uniform(0.3,0.6,size=1)
            b = QBrush(QColor(defaultColors[i]))
            qp.setBrush(b)
            qp.drawRect(QRectF(
                startX,
                rectY + (height - h),
                barWidth,
                h
            ))

    def drawBoxes(self,qp, height, width, rectX, rectY, nBoxes = 4):
        ""
        defaultColors = ["white","#397546",INSTANT_CLUE_BLUE,"#A0D4CB"]
        boxWidth = width * 1/(nBoxes+1)
        margin = boxWidth / (nBoxes+1)


        for i in range(nBoxes):

            minBox, maxBox = 0.4 - np.random.random() * 0.2 , 0.65 + np.random.random() * 0.3
            errorMax = maxBox + 0.1
            errorMin = minBox - 0.1
            median = 0.5 + np.random.random() * 0.15
            startX = rectX + (1+i) * margin + i * boxWidth

            self.drawBoxplot(qp,
                             height,
                             startX,rectY,
                             median,
                             minBox,
                             maxBox,
                             errorMin,
                             errorMax,
                             boxWidth,
                             margin,
                             color = defaultColors[i])

    
    def drawHeatmap(self,qp,height,width,rectX,rectY,X = None):
        "Draws a heatmap"
        cmap = twColorCmap if self.mouseOver else colorCmap
        if X is None:
            X = np.array([
                [2.5,2.4,-3.4,-3.2],
                [2.9,3.1,0.5,0.4],
                [-0.24,-0.34,0.5,0.6],
                [0.25,0.1,2.8,2.75],
                [-3.5,-3.2,2.8,2.9]
            ])
        rows, columns = X.shape
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        rectWidth = width / (columns)
        rectHeight = height / (rows)
        for nRow in range(rows):
            for nCol in range(columns):
                #get value
                colorValue = X[nRow,nCol]
                #init color
                c = QColor()
                #set color from lsitedColorMap (r,g,b,a)
                c.setRgbF(*cmap(colorValue))
                #set brush to painter
                qp.setBrush(c)
                #draw rectangle
                qp.drawRect(
                            QRectF(
                                rectX + nCol * rectWidth,
                                rectY + nRow * rectHeight,
                                rectWidth,
                                rectHeight
                                )
                            )

    def drawBoxplot(self,qp,height,startX,rectY,median,minBox,maxBox,errorMin,errorMax,boxWidth,margin,color="yellow"):
        "Draws Boxplot median, errors are given as fractions of height."
        if color is not None:
            brush = QBrush(QColor(color))
            qp.setBrush(brush)

        #draw errorLine
        #startX = startRectX + margin + boxWidth/2
        startErrorLine = startX + boxWidth/2
        minErrorHeight =  rectY + (1 - errorMax) * height
        maxErrorHeight =  rectY + (1 - errorMin) * height

        maxErrorPoint = QPointF(startErrorLine,  minErrorHeight)
        minErrorPoint = QPointF(startErrorLine,  maxErrorHeight)

        qp.drawLine(maxErrorPoint,
                    minErrorPoint)

        #draw error caps
        deltaPoint = QPointF(boxWidth/4,0)

        #top error
        qp.drawLine(maxErrorPoint - deltaPoint,
                    maxErrorPoint + deltaPoint)
        #bottom error
        qp.drawLine(minErrorPoint - deltaPoint,
                    minErrorPoint + deltaPoint)
        #draw box
        minHeight = rectY + (1 - minBox) * height
        maxHeight = rectY + (1 - maxBox) * height

        qp.drawRect(QRectF(startX,maxHeight,boxWidth,minHeight-maxHeight))
        #draw median line
        qp.drawLine(QPointF(startX,
                            rectY + median*height),
                    QPointF(startX + boxWidth,
                            rectY + median*height))

    def sizeHint(self):
        ""
        return QSize(50,50)


class MainFigureButton(PlotTypeButton):
    
    def __init__(self,parent=None,*args,**kwargs):

        super(MainFigureButton,self).__init__(parent=None,*args,**kwargs)

    def paintEvent(self,event):
        ""
       
        super().paintEvent(event,noAxis = True)
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)
        #get rect
        rect, h, w, x0, y0 = self.getRectProps(event)
        m = w/8
        w2 = w/2
        qs = [QRectF(x0 - w2, y0 - w2, w2,h/2),
            QRectF(x0, y0 - w2, w2,h/2),
            QRectF(x0 - w2, y0, w2,h/2),
            QRectF(x0, y0, w2,h/2)]
        
        for q in qs:
            #set small width of pen line
            pen.setWidthF(0.1)#
            painter.setPen(pen)
            rectHeight, rectWidth, rectX, rectY = self.drawAxis(painter, q , False, 5)
            funcKey =  ["boxplot","barplot","lineplot"]
            nFuncKeys = len(funcKey)
            randN = np.random.randint(low=0,high=nFuncKeys)
            fKey = funcKey[randN]
            self.funcKeyDict[fKey](painter,rectHeight,rectWidth, rectX, rectY)
            painter.setBrush(Qt.NoBrush)

    ## main Figure buttons


class SettingsButton(PlotTypeButton):

    def __init__(self,parent=None,*args,**kwargs):

        super(SettingsButton,self).__init__(parent=None,*args,**kwargs)

    def paintEvent(self,event):
        ""
        super().paintEvent(event,noAxis = True)
        painter = QPainter(self)
        pen = QPen(QColor("black"))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing,True)
        #get rect
        rect, h, w, x0, y0 = self.getRectProps(event)

        for n,(hFrac, xFrac) in enumerate(zip([0.25, 0.5, 0.75],[0.8,0.4,0.65])):
            x0  = 0 + w/9
            x1  = w - w/9

            y0 = y1 = 0 + h * hFrac
            painter.drawLine(x0,y0,x1,y1)
            #draw lines 
            if self.mouseOver and np.random.randint(0,4) < 2:
                painter.setBrush(QBrush(QColor(INSTANT_CLUE_BLUE)))
            else:
                painter.setBrush(QBrush(QColor("white")))
            #add circles 
            painter.drawEllipse(QPointF(x1 * xFrac , y0), 5, 5)
       
