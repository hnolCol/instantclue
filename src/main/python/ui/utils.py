from PyQt6.QtCore import *
from PyQt6.QtGui import QFont, QFontDatabase, QDoubleValidator
from PyQt6.QtWidgets import QLabel, QComboBox, QLineEdit, QMenu
import os


from numpy import short 
import hashlib
import base64 
import random
import string
import darkdetect 



CONTRAST_BG_COLOR = "#f6f6f6" if darkdetect.isDark() else "white"
#WIDGET_HOVER_COLOR = "#616161" if darkdetect.isDark() else "#B84D29"
WIDGET_HOVER_COLOR = "#B84D29" #red
TABLE_ODD_ROW_COLOR = "#E4DED4"
INSTANT_CLUE_BLUE = "#286FA4" #"lightgrey" if darkdetect.isDark() else "#286FA4" 

headerLabelStyleSheet = """ 
                    QLabel {
                        color: #4C626F;
                        font: 12px Arial;
                    }
                    """

standardFontSize = 12 
standardFontFamily = "Helvetica"


legendLocations = ["upper right","upper left","center left","center right","lower left","lower right"]

def getMainWindowBGColor():
    ""
    if darkdetect.isDark():
        return "#393939"
    else:
        return "white"
def getLargeWidgetBG():
    "Hex color used for receiver box and large frames."
    if darkdetect.isDark():
        return "#616161" 
    return "white"

def getCollapsableButtonBG():
    "Hex color for the background of buttons in the collapsable data tree view frame"
    if darkdetect.isDark():
        return "#808080" 
    else:
        return "#ECECEC"

def getDefaultWidgetBGColor():
    "HexColor to be used as bg for widgets"
    if darkdetect.isDark():
        return "#808080" 
    return "white"

def getHoverColor():
    ""
    hoverColor = "#616161" if darkdetect.isDark() else "#E4DED4"
    return hoverColor 
def getStdTextColor():
    if darkdetect.isDark():
        return "white"
    else:
        return "black"

def getStdTitleTextColor():
    ""
    if darkdetect.isDark():
        return "white"
    else:
        return "#4F6571"

def getBoolFromCheckState(checkState):
    "Returns a bool from a PyQt6 CheckState"
    return checkState == Qt.CheckState.Checked
   
def getCheckStateFromBool(checked):
    "Returns a bool from a PyQt6 CheckState"
    if checked:
        return Qt.CheckState.Checked
    
    return Qt.CheckState.Unchecked


def toggleCheckState(checkState):
    "Toggle a PyQt6 specifc State"
    if checkState == Qt.CheckState.Checked:
        return Qt.CheckState.Unchecked, False
    return Qt.CheckState.Checked, True

def getRandomString(N = 20):
    ""
    return ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=N))

def getHashedUrl(url):
    ""
    hash = hashlib.md5(url.encode()) # get the hash for the url
    hash = hash.replace('=','').replace('/','_')  # some cleaning
    shortUrl = base64.b64encode(hash)
    
    return shortUrl

def copyAttributes(obj2, obj1, attr_list):
    for i_attribute  in attr_list:
        getattr(obj2, 'set_' + i_attribute)( getattr(obj1, 'get_' + i_attribute)() )


def areFilesSuitableToLoad(filePaths):
    ""
    checkedFiles = []
    for filePath in filePaths:
        if os.path.exists(filePath):
            if any(str(filePath).endswith(endStr) for endStr in ["txt","csv","xlsx","tsv"]):
                checkedFiles.append(filePath)
    return checkedFiles

def getExtraLightFont(fontSize=12,font="Helvetica"):
    ""
    if isWindows():
        fontSize -= 2
    QFontObject = getStandardFont(fontSize,font)
    QFontObject.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
    QFontObject.setWeight(QFont.Weight.ExtraLight)
    return QFontObject

def getStandardFont(fontSize = None, font=None):
    ""
    if fontSize is None:
        fontSize = standardFontSize
    if font is None:
        font = standardFontFamily
    font = QFont(font) 
   
    if isWindows(): #ugly hack but on windows fonts appear huge
        fontSize -= 3
    font.setPointSize(fontSize)
    return font

def createLabel(text,tooltipText = None, **kwargs):
    ""
    w = QLabel()
    w.setText(text)
     #set font
    font = getStandardFont(**kwargs)
    w.setFont(font)

    if tooltipText is not None:
        w.setToolTip(tooltipText)
    return w


def createTitleLabel(text, fontSize = 18, *args, **kwargs):
    ""

    w = QLabel()
    #set font
    font = QFont(getStandardFont()) 
    if isWindows():
        fontSize -= 2
    font.setPointSize(fontSize)
    w.setFont(font)
    #set Text
    w.setText(text)
    #style
    colorString = getStdTitleTextColor()
    w.setStyleSheet("QLabel {color : "+colorString+"; }")
    return w


def createValueLineEdit(placeholderText,tooltipStr, minValue, maxValue):
    validator = QDoubleValidator()
    validator.setRange(minValue,maxValue,12)
    validator.setNotation(QDoubleValidator.Notation.StandardNotation)
    validator.setLocale(QLocale("en_US"))
    validator.setDecimals(20)
    #self.alphaLineEdit.setValidator(validator)
    
    valueEdit = QLineEdit(placeholderText = placeholderText, toolTip = tooltipStr)
    valueEdit.setStyleSheet("background: white")
    
    valueEdit.setValidator(validator)
    return valueEdit

def createLineEdit(placeHolderText="",tooltipText="",*args,**kwargs):
    """
    Creates Line Edit using the standard font.
    Additional args and kwargs will be forwarded to QLineEdit
    """
    w = QLineEdit(*args,**kwargs)
    w.setPlaceholderText(placeHolderText)
    w.setToolTip(tooltipText)
    w.setFont(getStandardFont())
    w.setStyleSheet("padding-top: 2px; padding-bottom: 2px")
    return w

def getMessageProps(title,message):
    ""
    return {"title":title,"message":message}

# string operation util functions
def removeFileExtension(fileName):
    if isinstance(fileName,str):
        rString = fileName[::-1]
        nExtension = len(rString.split(".",1)[0]) + 1 #add one for point
        return fileName[:-nExtension] #remove extension
    else:
        return fileName

def createMenu(*args,**kwargs):
    ""
    menu = QMenu(*args,**kwargs)
    menu = setStyleSheetForMenu(menu)
    
    return menu

def createMenus(menuNames = []):
    ""
    return dict([(menuName,createMenu()) for menuName in menuNames])


def createSubMenu(main = None,subMenus = []):
    "Returns a dict of menus"
    menus = dict() 
    if main is None or not isinstance(main,QMenu):
        #main menu
        main = createMenu()
    menus["main"] = main 
    for subMenu in subMenus:
        menus[subMenu] =  setStyleSheetForMenu(menus["main"].addMenu(subMenu))
    return menus

def createCombobox(parent = None, items = []):
    ""
    #set up data frame combobox and its style
    combo = QComboBox(parent)
    combo.addItems(items)
    combo.setFont(getStandardFont())
    combo.setStyleSheet("selection-background-color: white; outline: None; selection-color: {}".format(INSTANT_CLUE_BLUE)) 
    return combo


def isWindows():
    ""
    return os.name == "nt"

def setStyleSheetForMenu(menu):
    ""
    if darkdetect.isDark():

        menu.setStyleSheet("""
                QMenu 
                    {
                        background:#393939;
                        border: solid 1px black;
                    }
                QMenu::item 
                    {
                        font-family: Arial; 
                        font-size: 12px;
                        margin: 2 4 2 4;
                        padding: 0 20 0 10;
                    }
                QMenu::item:selected 
                    {
                        color:#286FA4;
                    }
                    """)
    else:
        menu.setStyleSheet("""
                QMenu 
                    {
                        background:white;
                        border: solid 1px black;
                    }
                QMenu::item 
                    {
                        font-family: Arial; 
                        font-size: 12px;
                        margin: 2 4 2 4;
                        padding: 0 20 0 10;
                    }
                QMenu::item:selected 
                    {
                        color:#286FA4;
                    }
                    """)
    return menu