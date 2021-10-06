
import matplotlib.patches as patches
import numpy as np
from operator import itemgetter
from matplotlib.font_manager import FontProperties
import time
arrow_args = dict(arrowstyle="-", color = "0.5", connectionstyle = "arc3")#"angle3,angleA=90,angleB=0")

def xLim_and_yLim_delta(ax):

        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        delta_x = xmax-xmin
        delta_y = ymax-ymin
        return delta_x,delta_y

def distance(co1, co2):
        return np.sqrt(pow(abs(co1[0] - co2[0]), 2) + pow(abs(co1[1] - co2[1]), 2))

def closest_coord_idx(list_, coord):
            if coord is not None:

            	dist_list = [distance(co,coord) for co in list_]
            	idx = min(enumerate(dist_list),key=itemgetter(1))
            	return idx

def find_nearest(array,value):
    ""
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


class ICScatterAnnotations(object):
    '''
    Adds an annotation triggered by a pick event. Deletes annotations by right-mouse click
    and moves them around.
    Big advantage over standard draggable(True) is that it takes only the closest annotation to 
    the mouse click and not all the ones below the mouse event.
    '''
    def __init__(self,
                    parent,
                    plotter,
                    ax,
                    data,
                    labelColumns,
                    numericColumns,
                   # madeAnnotations = {},
                   # selectionLabels = {},
                    scatterPlots = {},
                    annotationParams = None,
                    labelInAllPlots = False):
        self.parent = parent
        self.p = plotter
        self.ax = ax
        self.data = data
        self.numericColumns = numericColumns
        self.textAnnotationColumns = labelColumns
        self.eventOverAnnotation = False
        self.justDeletedAnnotation = False
        self.annotationParams = annotationParams
        self.scatterPlots = scatterPlots
        self.labelInAllPlots = labelInAllPlots #depracted no effect
       
        
       # self.selectionLabels = selectionLabels
        #self.madeAnnotations = madeAnnotations
        
        self.onPress =  self.p.f.canvas.mpl_connect('button_press_event', lambda event:  self.onPressMoveAndRemoveAnnotions(event))
        self.onPick =  self.p.f.canvas.mpl_connect('button_release_event', lambda event:  self.onReleaseLabelSelection(event))


    def disconnectEventBindings(self):
        ""
        if hasattr(self,"onPress"):
            self.p.f.canvas.mpl_disconnect(self.onPress)
        if hasattr(self,"onPick"):
            self.p.f.canvas.mpl_disconnect(self.onPick)
        if hasattr(self,"rectangleMoveEvent"):
            self.p.f.canvas.mpl_disconnect(self.rectangleMoveEvent)
        if hasattr(self,"releaseLabelEvent"):
            self.p.f.canvas.mpl_disconnect(self.releaseLabelEvent)
        

    def onReleaseLabelSelection(self,event):
        '''
        Drives a matplotlib pick event and labels the scatter points with the column choosen by the user
        using drag & drop onto the color button..
        '''
        try:
            ## check if the axes is used that was initiated
            ## allowing several subplots to have an annotation (PCA)
            if event.inaxes != self.ax:
                return
            

            if self.parent.getToolbarState() is not None:
                return

            if self.eventOverAnnotation:
                return
            
            if self.justDeletedAnnotation:
                #deletion is done by right click -> same as menu.
                #this prevents that menu is shown when annotationw as deleted.
                self.justDeletedAnnotation = False
                return
                        
            if event.button != 1:
                setattr(self.parent, "menuClickedInAxis",self.ax)
                self.parent.createAndShowMenu()
                return
          
            scatterPlots = self.scatterPlots
            for scatterPlot in scatterPlots.values():
                if scatterPlot.ax == self.ax:
                    if scatterPlot.idxData is None:
                        return
                    boolIdx = self.data.index.isin(scatterPlot.idxData)
                    break
            if not np.any(boolIdx):
                return
                
            selectedIdx = self.data.index[boolIdx]
            self.addAnnotations(selectedIdx)
            if self.parent.getParam("annotate.in.all.plots"):
                self.parent.annotateInAllPlots(selectedIdx,self)
            self.parent.updateFigure.emit()
            
        except Exception as e:
            print(e)

    def addAnnotations(self,idx):
        ""

        selectedData = self.data.loc[idx,:] 
        maxAnnotationLength = self.parent.getParam("annotate.max.length")
        #key = clickedData.name
        for key in selectedData.index:
            #annotations are saved by row idx
            selectedLabels = self.parent.getAnnotatedLabels(self.ax)
            if selectedLabels is not None and key in selectedLabels: ## easy way to check if that row is already annotated
                continue
            xyDataLabel = tuple(selectedData.loc[key,self.numericColumns].values)
            textLabel = "\n".join([str(x) if len(str(x)) <= maxAnnotationLength else "{}..".format(str(x)[:maxAnnotationLength]) for x in selectedData.loc[key,self.textAnnotationColumns].values.flatten()])
            
            ax = self.ax
            xLimDelta,yLimDelta = xLim_and_yLim_delta(ax)
            xyText = (xyDataLabel[0]+ xLimDelta*0.02, xyDataLabel[1]+ yLimDelta*0.02)
            textProps = dict(xy=xyDataLabel, text=textLabel, xytext = xyText)
            annotObject = ax.annotate(**textProps, ha='left', **self.getAnnotationProps())
            self.parent.saveAnnotations(self.ax,key,annotObject,textProps)
           
    

    def addAnnotationFromDf(self,dataFrame, redraw = True):
        '''
        '''
        ax = self.ax

        for rowIndex in dataFrame.index:
            if rowIndex not in self.data.index:
                continue
            textData = self.data[self.textAnnotationColumns].loc[rowIndex]
            textLabel = str(textData.iloc[0])
            
            key = rowIndex
            xyDataLabel = tuple(self.data[self.numericColumns].loc[rowIndex])
            xLimDelta,yLimDelta = xLim_and_yLim_delta(ax)
            xyText = (xyDataLabel[0]+ xLimDelta*0.02, xyDataLabel[1]+ yLimDelta*0.02)
            annotObject = ax.annotate(s=textLabel,xy=xyDataLabel,xytext= xyText, ha='left', **self.getAnnotationProps())
        
            self.selectionLabels[key] = dict(xy=xyDataLabel, s=textLabel, xytext = xyText)
            self.madeAnnotations[key] = annotObject	
        ## redraws added annotations
        if redraw:	
            self.parent.updateFigure.emit()


    def getAnnotationProps(self, fontSizeScale = 1):
        ""
        annotationParams = {
                "arrowprops" : {
                                "arrowstyle" : "-",
                                "connectionstyle" : self.parent.getParam("arrowStyle"),
                                "color" : self.parent.getParam("arrowColor")},

                "fontproperties":FontProperties(family=self.parent.getParam("annotationFontFamily"),
                                                size = self.parent.getParam("annotationFontSize") * fontSizeScale)}
        return annotationParams

    def replotAllAnnotations(self, ax):
        '''
        If user opens a new session, we need to first replot annotation and then enable
        the drag&drop events..
        '''
        
        #self.madeAnnotations.clear() 
        for key,annotationProps in self.selectionLabels.items():
            annotObject = ax.annotate(ha='left', arrowprops=arrow_args,**annotationProps)
            self.madeAnnotations[key] = annotObject
        
    def onPressMoveAndRemoveAnnotions(self,event):
        '''
        Depending on which button used by the user, it trigger either moving around (button-1 press)
        or remove the label.
        '''
        if self.parent.getToolbarState() is not None:
            return
        if event.inaxes is None and event.inaxes != self.ax:
            return 
        if event.button in [2,3]: ## mac is 2 and windows 3..
            self.remove_clicked_annotation(event)
        elif event.button == 1:
            self.moveAnnotations(event)

    def annotateData(self):
        '''
        '''
        self.addAnnotationFromDf(self.data)

    def eventInBBox(self,bboxbounds,eventXY):
        ""
        x0,y0,w,h = bboxbounds
        xE,yE = eventXY
        
        if xE > x0 and xE < x0+w and yE > y0 and yE < y0+h:
            return True
        return False

    def remove_clicked_annotation(self,event):
        '''
        Removes annotations upon click from dicts and figure
        does not redraw canvas
        '''
      #  self.plotter.castMenu = True
        if self.ax != event.inaxes:
            return
        toDelete = None
        annotationObjects = self.parent.getAnnotationTextObjs(self.ax)
        annotationBbox = self.parent.getAnnotationBbox(self.ax)
        xyE = (event.x,event.y)
        if annotationObjects is not None:
            for key,madeAnnotation  in annotationObjects.items():
                bboxBounds = annotationBbox[key]#madeAnnotation.get_window_extent().bounds
                if self.eventInBBox(bboxBounds,xyE):
                    madeAnnotation.remove()
                    toDelete = key
                    break
                
            if toDelete is not None:
                self.parent.deleteAnnotation(self.ax,toDelete)
  
                self.eventOverAnnotation = False
                self.justDeletedAnnotation = True
                
                self.parent.updateFigure.emit()		
            
            
    def removeAnnotations(self):
        '''
        Removes all annotations. Might be called from outside to let 
        the user delete all annotations added
        '''
        annotationObjects = self.parent.getAnnotationTextObjs(self.ax)
        if annotationObjects is not None:
            for madeAnnotation  in annotationObjects.values():
                    madeAnnotation.remove()
    
    def findClosestAnnotation(self,xyEvent,event):
        '''
        '''
        selectionLabels = self.parent.getAnnotatedLabels(self.ax)

        if selectionLabels is None or len(selectionLabels) == 0 or event.inaxes is None:
            return
        annotationsKeysAndPositions = [(key,annotationDict['xytext']) for key,annotationDict in self.parent.getAnnotationTextProps(self.ax).items()][::-1]
        
        
        keys, xyPositions = zip(*annotationsKeysAndPositions)
        idxClosest = closest_coord_idx(xyPositions,xyEvent)[0]
        keyClosest = keys[idxClosest]
        annotationClostest = self.parent.getAnnotationTextObjs(self.ax)[keyClosest]
        self.eventOverAnnotation = annotationClostest.contains(event)[0]
        
        return annotationClostest,xyPositions,idxClosest,keyClosest
            
    def moveAnnotations(self,event):
        '''
        wrapper to move around labels. We did not use the annotation.draggable(True) option
        because this moves all artists around that are under the mouseevent.
        '''
        selectionLabels = self.parent.getAnnotatedLabels(self.ax)
        if selectionLabels is None or len(selectionLabels) == 0 or event.inaxes is None:
            return 
        self.eventOverAnnotation = False	
        
        xyEvent =  (event.xdata,event.ydata)
        
        annotationClostest,xyPositions,idxClosest,keyClosest = \
        self.findClosestAnnotation(xyEvent,event)	
                    
        
        if self.eventOverAnnotation and event.inaxes == self.ax:
            ax = self.ax
            inv = ax.transData.inverted()
            
            #renderer = self.figure.canvas.renderer() 
            xyPositionOfLabelToMove = xyPositions[idxClosest] 
            background = self.p.f.canvas.copy_from_bbox(ax.bbox)
            widthRect, heightRect = self.getRectangleSizeOfText(ax,annotationClostest.get_text(),inv)
            recetangleToMimicMove = patches.Rectangle(xyPositionOfLabelToMove,width=widthRect,height=heightRect,
                                                    fill=False, linewidth=0.6, edgecolor="darkgrey",
                                                    animated = True,linestyle = 'dashed', clip_on = False)
            
            ax.add_patch(recetangleToMimicMove)
            
            self.rectangleMoveEvent = self.p.f.canvas.mpl_connect('motion_notify_event', 
                                        lambda event,
                                                rect = recetangleToMimicMove,
                                                b = background,
                                                inv = inv,
                                                ax  = ax : self.moveRectangle(event,rect,b,inv,ax))
                                                
                                                                
                                                                
            self.releaseLabelEvent = self.p.f.canvas.mpl_connect('button_release_event', 
                                        lambda event,
                                        rect = recetangleToMimicMove,
                                        a = annotationClostest,
                                        k = keyClosest: self.disconnectLabelAndUpdateAnnotation(event,
                                                                                    rect,
                                                                                    a,
                                                                                    k))
        
    def moveRectangle(self,event,rectangle,background,inv, ax):
        '''
        actually moves the rectangle
        '''
        x_s,y_s = event.x, event.y
        x,y= list(inv.transform((x_s,y_s)))
        self.p.f.canvas.restore_region(background)
        rectangle.set_xy((x,y))  
        ax.draw_artist(rectangle)
        self.p.f.canvas.blit(ax.bbox)    
            
    def disconnectLabelAndUpdateAnnotation(self,event,rectangle,annotation,keyClosest):
        '''
        Mouse release event. disconnects event handles and updates the annotation dict to
        keep track for export
        '''

        if hasattr(self,"rectangleMoveEvent"):
            
            self.p.f.canvas.mpl_disconnect(self.rectangleMoveEvent)
            del self.rectangleMoveEvent

        if hasattr(self,"releaseLabelEvent"):

            self.p.f.canvas.mpl_disconnect(self.releaseLabelEvent)
            del self.releaseLabelEvent
        
        xyRectangle = rectangle.get_xy()
        annotation.set_position(xyRectangle)
        
        rectangle.remove()
        self.eventOverAnnotation = False
        self.parent.updateFigure.emit()
        self.parent.updateAnnotationPosition(self.ax,keyClosest,xyRectangle)

    def getRectangleSizeOfText(self,ax,text,inv):
        '''
        Returns rectangle to mimic the position
        '''	
        renderer = self.p.f.canvas.get_renderer() 
        fakeText = ax.text(0,0,s=text, fontproperties =self.getAnnotationProps()["fontproperties"])
        patch = fakeText.get_window_extent(renderer)
        xy0 = list(inv.transform((patch.x0,patch.y0)))
        xy1 = list(inv.transform((patch.x1,patch.y1)))
        fakeText.remove()
        widthText = xy1[0]-xy0[0]
        heightText = xy1[1]-xy0[1]
        return widthText, heightText
                                
    def updateData(self,data):
        '''
        Updates data to be used. This is needed if the order of the data 
        changed.
        '''
        self.data = data		


    def setLabelInAllPlots(self):
        ""
        self.labelInAllPlots = not self.labelInAllPlots
    
    def getLabelInAllPlots(self):
        ""
        return self.labelInAllPlots

    def mirrorAnnotationsToTargetAxis(self,sourceAx,targetAx,scaleFactor = 1):
        ""
        if sourceAx == self.ax:
            #get annotation props (these are the only ones that are being updated)
            annotationProps = self.parent.getAnnotationTextProps(self.ax)
            for textProps in annotationProps.values():
                #textProps = dict(xy=xyDataLabel, s=textLabel, xytext = xyText)

                targetAx.annotate(**textProps, ha='left', **self.getAnnotationProps(fontSizeScale=scaleFactor))
                