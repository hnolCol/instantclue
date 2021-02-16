from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..utils import createMenu, createTitleLabel, createLabel, createSubMenu, createMenus
from .buttonDesigns import BigPlusButton, ResetButton, RefreshButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.pyplot import figure

import string
from collections import OrderedDict
from matplotlib.pyplot import GridSpec

from .warnMessage import WarningMessage

COMBO_LABELS_GRID = ["Rows:","Columns:"]
COMBO_LABELS_GRID_TOOLTIP = ["Number of rows (n). The height of the a4 page is divided into equal n peices.",
                            "Number of columns (m). The width of the a4 page is divided into equal m pieces."]
DEFAULT_INDEX_GRID = [3,2]
DEFAULT_INDEX_AXIS = [0,0,0,0,0]
COMBO_LABELS = ["Position row:", "column:", "Row span:","Column span:","Axis label:"]
COMBO_LABELS_TOOLTIP = ["Defines the position of the axis within the set grid by row and column.", 
                        None,
                        "Defines height of axis: if the grid is set to 5 rows and row span is 1, then 1/5 of the a4 page is given to the axis.",
                        "Defines width of axis: if the grid is set to 3 columns and column span is 1, then 1/3 of the a4 page is given to the axis.",
                        "Label of the subplot."]
ALPHABETIC_LABEL  = list(string.ascii_lowercase)+list(string.ascii_uppercase)


class MainFigureRegistry(object):

    def __init__(self):
        self.mainFigureId = 0

        self.mainFigureTemplates = OrderedDict()
        self.mainFigures = {}
        self.exportDetails = {}
        self.figText = {}

        self.default_label_settings()


    def initiate(self, figureId = None):
        '''
        '''
        if figureId is None:
            self.mainFigureId += 1
            self.exportDetails[self.mainFigureId] = {}
            self.figText[self.mainFigureId] = {}
        else:
            ## this happens when
            self.mainFigureId = figureId


        self.mainFigureTemplates[self.mainFigureId] = {}
        self.mainFigures[self.mainFigureId] = {}


        return self.mainFigureId

    def checkID(self,figureId):
        ""
        if not hasattr(self,"mainFigures"):
            self.mainFigures = {} 
        
        if figureId not in self.mainFigures:
            self.mainFigures[figureId] = {}
            self.mainFigureTemplates[figureId] = {}


    def store_export(self,axisID,figureId,plotCount,subplotNum,exportId,boxBool,gridBool):
        '''
        '''
        limits = self.get_limits(axisID,figureId)
        self.exportDetails[figureId][axisID] = {'plotCount':plotCount,
                                                'subplotNum':subplotNum,
                                                'limits':limits,
                                                'exportId':exportId,
                                                'boxBool':boxBool,
                                                'gridBool':gridBool}

    def store_image_export(self,axisID,figureId,imagePath):
        '''
        '''
        self.exportDetails[figureId][axisID] = {'path':imagePath}

    def store_figure(self,figure,templateClass):
        '''
        Store in different dict, since we cannot pickle the
        figure in a canvas.
        '''
        if not hasattr(self,"mainFigures"):
            self.mainFigures = {}
        self.mainFigures[self.mainFigureId]['figure'] = figure
        self.mainFigures[self.mainFigureId]['template'] = templateClass


    def store_figure_text(self,figureId,id,props):
        '''
        Stores figure text added by the user. To enable session upload.
        '''
        self.figText[figureId][id] = props


    def get_limits(self,axisID,figureId):
        '''
        Return axis limits.
        '''
        ax = self.mainFigureTemplates[figureId][axisID]['ax']
        xLim = ax.get_xlim()
        yLim = ax.get_ylim()
        return [xLim,yLim]

    def getMainFigures(self):
        ""
        return [fig["figure"] for fig in self.mainFigures.values()]

    def getMainFiguresByID(self):
        ""
        return OrderedDict([(ID,fig["figure"]) for ID,fig in self.mainFigures.items()])


    def getMainFigureCurrentSettings(self):
        ""
        return OrderedDict([(ID,fig["template"].getAxisComboSettings()) for ID,fig in self.mainFigures.items()])

    def update_menus(self):
        '''
        '''
        #for figureId,props in self.mainFigures.items():
        #			props['template'].define_menu()

    def update_params(self, figureId, axisID = None, params = None, how = 'add'):
        '''
        '''
        if how == 'add':
            self.mainFigureTemplates[figureId][axisID] = params
            self.update_menu(params['ax'],axisID,figureId,params['axisLabel'])


        elif how == 'delete':
            params =  self.mainFigureTemplates[figureId][axisID]
            #self.analyze.menuCollection['main_figure_menu'].delete('Figure {} - Subplot {}'.format(figureId,params['axisLabel']))
            del self.mainFigureTemplates[figureId][axisID]
            if axisID in self.exportDetails[figureId]:
                del self.exportDetails[figureId][axisID]

        elif how == 'clear':
            params =  self.mainFigureTemplates[figureId][axisID]
            if axisID in self.exportDetails[figureId]:
                del self.exportDetails[figureId][axisID]

        elif how == 'clean_up':
            #for axisID,params in self.mainFigureTemplates[figureId].items():
            #	self.analyze.menuCollection['main_figure_menu'].delete('Figure {} - Subplot {}'.format(figureId,params['axisLabel']))
            self.mainFigureTemplates[figureId] = {}
            self.exportDetails[figureId] = {}

        elif how == 'destroy':
            for axisID,params in self.mainFigureTemplates[figureId].items():
                self.analyze.menuCollection['main_figure_menu'].delete('Figure {} - Subplot {}'.format(figureId,params['axisLabel']))
            del self.mainFigureTemplates[figureId]
            del self.exportDetails[figureId]
            del self.mainFigures[figureId]

        self.update_menus()

    def update_menu_label(self,old,new):
        '''
        Updates the menu if user re-labels.
        '''
        #self.analyze.menuCollection['main_figure_menu'].entryconfigure(old,label=new)

    def update_menu(self,ax,axisID,figureId,label):
        '''
        '''

    def deleteFigure(self,figID):
        ""
        if figID in self.mainFigureTemplates:
            del self.mainFigureTemplates[figID]

    def getMenu(self, mainMenu,actionFn):
        menus = createSubMenu(mainMenu, ["F{}".format(x) for x in self.mainFigureTemplates.keys()])
        for figID, axes in self.mainFigureTemplates.items():
            for axisID,axisProps in axes.items():
                action = menus["F{}".format(figID)].addAction("axis({}:{})".format(axisID,axisProps["axisLabel"]))
                action.triggered.connect(lambda _,ax = axisProps["ax"],figID = figID : actionFn(ax,figID)) 
        return menus
        
    def updateFigure(self, figID):
        if figID in self.mainFigures:
            self.mainFigures[figID]['template'].updateFigure.emit()

    #def getMenu(self):
            
        
    #	self.analyze.menuCollection['main_figure_menu'].add_command(label='Figure {} - Subplot {}'.format(figureId,label),
    #		command = lambda ax = ax:self.analyze.perform_export(ax,axisID,self.mainFigures[figureId]))

    def default_label_settings(self):
        '''
        '''
        self.subplotLabelStyle = {'xy':(0.0,1),'xytext':(-12,12),
                        'xycoords':('axes fraction'),'textcoords':'offset points',
                        'size':12, 'family':'Arial','ha':'left',
                        'weight':'bold','color':'black','va':'bottom',
                        'rotation':'0'}

    def __getstate__(self):
        '''
        Cant pickle menu since it is a tkinter menu object.
        We need to remove it before pickle
        '''
        for figureId, axisDict in self.mainFigureTemplates.items():
            for axisID, params in axisDict.items():
                if 'ax' in params: # if user saves session twice this will be gone already
                    del params['ax']
        state = self.__dict__.copy()
        for attr in ['analyze','mainFigures']:
            if attr in state:
                del state[attr]
        return state

class MainFigure(QDialog):
    updateFigure = pyqtSignal()

    def __init__(self,parent=None, mainController = None, mainFigureRegistry = None, figSize=(8.27,11.7), figureID = None, mainFigure = None, *args,**kwargs):
        super(MainFigure, self).__init__(parent,*args,**kwargs)
        self.setAcceptDrops(True)
        self.acceptDrop = False
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.mainFigureCollection = mainFigureRegistry
        self.updateFigure.connect(self.update)
        
        self.mainController = mainController
        self.figSize = figSize

        self.__defineVars(figureID=figureID)
        self.__controls(mainFigure)
        self.__layout()
        self.__connectEvents()
        #store figure in registry
        self.mainFigureCollection.store_figure(self.figure,self)

    def __controls(self,mainFigure=None):
        # a figure instance to plot on
        if mainFigure is None:
            self.figure = figure(figsize=self.figSize)
           
        else:
            self.figure = mainFigure
            self.figure.set_dpi(100)
            #print(self.figure.dpi)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        #set margin of figure
        self.figure.subplots_adjust(top=0.94, bottom=0.05,left=0.1,
									right=0.95, wspace = 0.32, hspace=0.32)
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.canvas)
        self.scrollArea.setWidgetResizable(False)

        self.gridLabel = createTitleLabel("Define Grid Layout", fontSize=14)
        self.gridLabel.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)

        self.rowColumnLabel = createTitleLabel("Set position and width/height of axis.", fontSize=14)
        self.rowColumnLabel.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)

        self.figureTitle = createTitleLabel("Figure {}".format(self.figureID),fontSize=14)
        self.figureTitle.setAlignment(Qt.AlignRight)

        self.gridProps = OrderedDict() 
        #add grid props
        for n,propLabel in enumerate(COMBO_LABELS_GRID):
            l = createLabel(propLabel,COMBO_LABELS_GRID_TOOLTIP[n])
            cBox = QComboBox()
            cBox.setMaximumWidth(60)
            cBox.addItems(self.gridRange)
            cBox.setCurrentIndex(DEFAULT_INDEX_GRID[n])
            self.gridProps[propLabel] = (l,cBox)
            cBox.currentTextChanged.connect(lambda _,propLabel = propLabel : self.updateAxisProps(propLabel))

        #add combobox for axis properties
        self.axisPropsLabels = OrderedDict()
        for n,propLabel in enumerate(COMBO_LABELS):
            l = createLabel(propLabel,COMBO_LABELS_TOOLTIP[n])
            cBox = QComboBox()
            cBox.setMaximumWidth(60)
            
            cBox.addItems(self.axisRanges[propLabel])
            self.axisPropsLabels[propLabel] = (l,cBox)
        #add navigation buttons
        self.addAxisButton = BigPlusButton(self, tooltipStr="Adds a new axis at given position.", buttonSize=(40,40)) #QPushButton("ADD")
        self.deleteAxisButton = ResetButton(self, tooltipStr="Deletes selected axis",buttonSize=(40,40))
        self.clearFigureButton = RefreshButton(self, tooltipStr = "Refresh figure and start new", buttonSize = (40,40))


    def __layout(self):
        layout = QGridLayout()
        layout.setVerticalSpacing(15)
        layout.addWidget(self.gridLabel,1,0,1,5)
        layout.addWidget(self.rowColumnLabel,3,0,1,5)
        layout.addWidget(self.figureTitle,1,5,1,5)
        #hbox0 = QHBoxLayout()
        n = 0
        for l,cBox in self.gridProps.values():
            layout.addWidget(l,2,n,1,1)
            n+=1
            layout.addWidget(cBox,2,n,1,1)
            n+=1
            
        n = 0
        for l,cBox in self.axisPropsLabels.values():
            layout.addWidget(l,4,n,1,1)
            n+=1
            layout.addWidget(cBox,4,n,1,1)
            n+=1
        hbox = QHBoxLayout()
        hbox.addWidget(self.addAxisButton)
        hbox.addWidget(self.deleteAxisButton)
        hbox.addStretch(1)
        hbox.addWidget(self.clearFigureButton)

        layout.addLayout(hbox,5,0,1,10)

        layout.addWidget(self.scrollArea,6,0,1,10)
        layout.addWidget(self.toolbar,7,0,1,10)
        
        self.setLayout(layout)

    def __connectEvents(self):
        ""
        self.addAxisButton.clicked.connect(self.addAxis)
        self.clearFigureButton.clicked.connect(self.clearFigure)
        self.deleteAxisButton.clicked.connect(self.openAxisMenu)
        #matplotlib eents
        self.canvas.mpl_connect('button_press_event', self.onPress)


    def __defineVars(self, figureID = None):
        ""
        #initiate MainFigure Registry
        if figureID is None:
            self.figureID = self.mainFigureCollection.initiate(figureID)
        else:
            self.mainFigureCollection.checkID(figureID)
            self.figureID = figureID
        #add axis props
        self.axisID = 0
        self.figureProps = OrderedDict() 
        self.textsAdded = {}
        self.axisLabels = {}
        self.axisItems = OrderedDict()
        self.associatedAxes = {}

        
        self.gridRange = [str(i) for i in range(1,25)]

        self.axisRanges = {
            "Axis label:":ALPHABETIC_LABEL,
            "Position row:": [str(i) for i in range(1,DEFAULT_INDEX_GRID[0]+2)],
            "column:": [str(i) for i in range(1,DEFAULT_INDEX_GRID[1]+2)],
            "Row span:": [str(i) for i in range(1,DEFAULT_INDEX_GRID[0]+2)],
            "Column span:":[str(i) for i in range(1,DEFAULT_INDEX_GRID[1]+2)]
        }


    def addAxis(self, event = None, axisParams = None, axisID = None,
            redraw = True, addToId = True):
        '''
        Ads an axis to the figure . Gets the settings from the dictionary that stores self.positionGridDict
        '''
    
        if axisParams is None:
            axisParams =  self.getAxisParams()
            #returns None upon error
            if axisParams is None:
                return
        gridRow, gridCol,posRow, posCol,rowSpan, colSpan, subplotLabel = axisParams
        
        if posRow-1 + rowSpan > gridRow or posCol -1 + colSpan > gridCol:
            warn = WarningMessage(title = "Invalid Input",infoText='Axis specification out of grid.',iconDir = self.mC.mainPath)
            warn.exec_()
            return

        grid_spec = GridSpec(gridRow,gridCol)
        subplotspec = grid_spec.new_subplotspec(loc=(posRow-1,posCol-1),
                                                rowspan=rowSpan,colspan=colSpan)

        ax = self.figure.add_subplot(subplotspec)
        self.saveAxisProps(ax,axisParams,addToId = addToId)

        if addToId:
            axisID = self.axisID
            self.addAxisLabel(ax, axisID, label = subplotLabel)
        else:
                self.addAxisLabel(ax, axisID,
                                label = self.figureProps[axisID]['axisLabel'])
        if redraw:
            self.updateFigure.emit()
            self.updateAxisParams(axisParams)
    


    def addAxisLabel(self,ax, axisID, label = None):
        '''
        Adds a subplot label to the created/updated subplot
        '''
        if label is None:
            text = self.axisPropsLabels["Axis label:"][1].currentText()
        else:
            text = label

        axesLabel = ax.annotate(text,**self.mainFigureCollection.subplotLabelStyle)
        self.axisLabels[axisID] = axesLabel
	
    def checkForAssociationsAndRemove(self,axisID):
        '''
        '''
        if axisID in self.associatedAxes:
            axes = self.associatedAxes[axisID]

            for ax in axes:
                self.figure.delaxes(ax)
            del self.associatedAxes[axisID]
    
    def checkForLegend(self,ax):
        "Checks for legend in axis"
        legend = ax.get_legend()
        if legend is None:
            warn = WarningMessage(parent=self,infoText="No legend found.",iconDir = self.mC.mainPath)
            warn.exec_()
            return 
        else:
            return legend	

    def getAxisComboSettings(self):
        ""
        #self.axisPropsLabels[propLabel] = (l,cBox)
        comboSettings = {}
        for propLabel, widgets in self.axisPropsLabels.items():
            _, cBox = widgets
            comboSettings[propLabel] = cBox.currentText()
        
        return comboSettings
    
    def updateComboSettings(self,comboSettings):
        ""
        for propLabel, widgets in self.axisPropsLabels.items():
            _, cBox = widgets
            if propLabel in comboSettings:
                cBox.setCurrentText(comboSettings[propLabel])


    def clearAxis(self,event=None):
        ""
        axisID = self.getAxisID(self.inaxes)
        self.mainFigureCollection.update_params(self.figureID,axisID = axisID, how='clear')
        self.checkForAssociationsAndRemove((axisID))
        self.figureProps[axisID]['ax'].clear()
        self.clear_axis(self.figureProps[axisID]['ax'])
        self.updateFigure.emit()

    def clearDicts(self):
        ""
        self.axisLabels.clear()
        self.figureProps.clear()
        self.textsAdded.clear() 
        self.axisItems.clear()

    def clearFigure(self,event=None):
        ""
        if len(self.figureProps) > 0:
            qm = QMessageBox(parent=self)
            ret = qm.question(self,'', "Are you sure to reset the figure?", qm.Yes | qm.No)
            if ret == qm.Yes:
                self.figure.clf()
                self.updateFigure.emit()
                self.resetAxisProps()
                self.clearDicts()
                self.mainFigureCollection.update_params(self.figureID,how='clean_up')

    def closeEvent(self,event):
        ""
        self.mainFigureCollection.deleteFigure(self.figureID)
        event.accept()

    def createMenu(self):
        ""
        actionNameFn = [("Clear",self.clearAxis),("Delete",self.deleteAxis)]
        actionNameFnLegend = [("Delete",self.removeLegend)]

        menu = createSubMenu(subMenus=["Legend","Axis Limits","Transfer to.."])
        menu["main"].addMenu(menu["Legend"])

        for name, fn in actionNameFnLegend:
            action = menu["Legend"].addAction(name)
            action.triggered.connect(fn)
        for name,fn in actionNameFn:
            action = menu["main"].addAction(name)
            action.triggered.connect(fn)

        self.menu = menu["main"]

    def deleteAxis(self,event=None, axisID = None):
        '''
        Deletes axis by an event that has the attributes inaxes. This
        will be used to find the given ID (enumerated) and it will then
        delete all entries that were done in the main Figure Collection.
        '''
        if axisID is not None and axisID in self.figureProps:
            #self.check_for_associations_and_remove(id)
            self.figure.delaxes(self.figureProps[axisID]['ax'])
            del self.figureProps[axisID]
            del self.axisItems[axisID]
            del self.axisLabels[axisID]
            self.mainFigureCollection.update_params(self.figureID,axisID = axisID,how='delete')
            self.updateFigure.emit()
            if len(self.figureProps) == 0:
                self.resetAxisProps()
                
            

    def getAxisID(self,axClicked):
        '''
        Get the id of an axes. ax is a matplotlib axis object.
        '''

        for axisID,props in self.figureProps.items():
            if props['ax'] == axClicked:
                return axisID

    def getAxisParams(self):
        ""
      
        gridRow     =   self.gridProps["Rows:"][1].currentText()
        gridCol     =   self.gridProps["Columns:"][1].currentText()
        posRow      =   self.axisPropsLabels["Position row:"][1].currentText()
        posCol      =   self.axisPropsLabels["column:"][1].currentText()
        rowSpan     =   self.axisPropsLabels["Row span:"][1].currentText()
        colSpan     =   self.axisPropsLabels["Column span:"][1].currentText()

        propsStrings = [gridRow, gridCol,posRow, posCol,rowSpan, colSpan]
        try:
            propsIntegers = [int(float(item)) for item in propsStrings]  
        except:
            w = WarningMessage(infoText="Axis properties could not be converted to integers. Invlid input.",iconDir = self.mC.mainPath)
            w.exec_()
            return None
        subplotLabel = self.axisPropsLabels["Axis label:"][1].currentText()
        #bind subplotlabel
        propsIntegers.append(subplotLabel)

        return propsIntegers

    def nextSubplotLabel(self,subplotLabel):
        '''
        '''
        if subplotLabel in ALPHABETIC_LABEL:

            idxLabel = 	ALPHABETIC_LABEL.index(subplotLabel)
            nextLabelIdx = idxLabel+1
            if nextLabelIdx == len(ALPHABETIC_LABEL):
                nextLabelIdx = 0

            nextLabel = ALPHABETIC_LABEL[nextLabelIdx]
            self.axisPropsLabels["Axis label:"][1].setCurrentText(nextLabel)
    
    def onPress(self,event):
        
        if event.inaxes is not None and event.button != 1:
            self.createMenu()
            self.inaxes = event.inaxes
            self.menu.exec_(QCursor.pos())

        self.inaxes = None

    def openAxisMenu(self,event=None):
        ""
        menu = createMenu(parent=self)
        for axisID, aixsParams in self.figureProps.items():
            action = menu.addAction("Axis id = {} label = {}".format(axisID,aixsParams['axisLabel']))
            action.triggered.connect(lambda _,axisID = axisID: self.deleteAxis(axisID = axisID))
        if hasattr(self.sender(),"mouseLostFocus"):
            self.sender().mouseLostFocus()
        senderGeom = self.sender().geometry()
        bottomLeft = self.mapToGlobal(senderGeom.bottomLeft())
        menu.popup(bottomLeft)
        


    def update(self,event=None):
        ""
        self.canvas.draw()

    def updateAxisParams(self, parametersList = None):
        '''
        Updates the comboboxes to provide convenient addition of mroe axes.
        '''
        if parametersList is None or len(parametersList) != 7:
            gridRow, gridCol,posRow, posCol,rowSpan, colSpan, subplotLabel  = self.getAxisParams()
        else:
            gridRow, gridCol,posRow, posCol,rowSpan, colSpan, subplotLabel = parametersList
        ## updating

        self.nextSubplotLabel(subplotLabel)

        # reset position in Grid..
        if posCol + colSpan > gridCol:
            posCol = 1
            posRow = posRow + rowSpan
        else:
            posCol = posCol + colSpan

        self.axisPropsLabels["Position row:"][1].setCurrentText(str(posRow))
        self.axisPropsLabels["column:"][1].setCurrentText(str(posCol))
        
    def updateAxisProps(self,propLabel):
        ""
        try:
            changedValue = self.sender().currentText()
            
            if changedValue != '':
                if propLabel == "Rows:":
                    propLabel = "Position row:"
                else:
                    propLabel = "column:"
            
                try:
                    newValue = int(float(changedValue))
                except:
                    warn = WarningMessage(infoText="Could not convert to integer : {}".format(changedValue),iconDir = self.mC.mainPath)
                    warn.exec_()
                propRange = [str(i) for i in range(1,newValue+1)]
                self.axisRanges[propLabel] = propRange
                oldValue = self.axisPropsLabels[propLabel][1].currentText() 
                self.axisPropsLabels[propLabel][1].clear() 
                self.axisPropsLabels[propLabel][1].addItems(propRange)
                if oldValue in propRange:
                    self.axisPropsLabels[propLabel][1].setCurrentText(oldValue) 
                else:
                    self.axisPropsLabels[propLabel][1].setCurrentText(propRange[0]) 
                
        except Exception as e:
            print(e)	

    def removeLegend(self,event = None):
        '''
        Removes legend from axis.
        '''
        if self.inaxes is not None:
            ax = self.inaxes
            legend = self.checkForLegend(ax)
            if legend is not None:
                legend.remove()
                self.redraw()
       
    def resetAxisProps(self):
        ""
        self.updateAxisParams(parametersList = [
                                    int(float(self.gridProps["Rows:"][1].currentText())),
                                    int(float(self.gridProps["Columns:"][1].currentText())),
                                    1,0,1,1,'Z'])


    def saveAxisProps(self,ax,axisParams,addToId = True):
        '''
        Need to save:
            - axis with id
            - label
            - subplot specs
            - associated axis (for example hclust)
        '''
        # we dont want to add 1 if we restore figures
        if addToId:
            self.axisID += 1
        axisID = self.axisID
        self.figureProps[axisID] = {}
        self.figureProps[axisID]['ax'] = ax
        self.figureProps[axisID]['axisParams'] = axisParams
        # save label for quick change
        self.figureProps[axisID]['axisLabel'] = axisParams[-1]
        self.figureProps[axisID]['addAxis'] = []
        self.mainFigureCollection.update_params(self.figureID,axisID,self.figureProps[axisID])
        self.axisItems[axisID] = {}

