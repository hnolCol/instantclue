from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
import os

from numpy import short 
import hashlib
import base64 
import random
import string

HOVER_COLOR = "#E4DED4"
WIDGET_HOVER_COLOR = "#B84D29" #red
TABLE_ODD_ROW_COLOR = "#E4DED4"
INSTANT_CLUE_BLUE = "#286FA4"

headerLabelStyleSheet = """ 
                    QLabel {
                        color: #4C626F;
                        font: 12px Arial;
                    }
                    """

standardFontSize = 12 
standardFontFamily = "Helvetica"


legendLocations = ["upper right","upper left","center left","center right","lower left","lower right"]

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
    QFontObject.setLetterSpacing(QFont.AbsoluteSpacing, 3)
    QFontObject.setWeight(QFont.ExtraLight)
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


def createTitleLabel(text, fontSize = 18, colorString = "#4F6571"):
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
    w.setStyleSheet("QLabel {color : "+colorString+"; }")
    return w

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