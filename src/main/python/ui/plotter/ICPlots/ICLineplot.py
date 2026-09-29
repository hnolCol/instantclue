

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

class ICLineplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICLineplot,self).__init__(*args,**kwargs)

        self.quickSelectLineKwargs = {}
        self.quickSelectPolygonKwargs = {}
        self.quickSelectPolygon = {}

    def addHoverLine(self):
        ""
        self.hoverLines = {}
        for ax in self.axisDict.values():
            hoverLine = ax.plot([],[],
                            linewidth=self.getParam("linewidth.median"), 
                            marker= None if  self.getParam("marker.median")  == "none" else self.getParam("marker.median"), 
                            color = self.getParam("scatter.hover.color"),
                            markeredgecolor = "black", 
                            linestyle = "-",
                            markeredgewidth = self.getParam("markeredgewidth.median"))
            self.hoverLines[ax] = hoverLine[0]

        self.setHoverLinesInivisble()

    def addHoverArea(self):
        ""
        self.hoverAreas = {}
        for ax in self.axisDict.values():
            hoverArea  = Polygon(
                            [[0,0],[1,1]],
                            visible=False,
                            alpha=self.getParam("alpha.IQR"),
                            facecolor=self.getParam("scatter.hover.color"),
                            edgecolor="black",
                            linewidth=0.1,
                            fill=True, 
                            closed=True)
            
            ax.add_patch(hoverArea)
            self.hoverAreas[ax] = hoverArea
        
        
    def setHoverLinesInivisble(self):
        ""
        for l in self.hoverLines.values():
            l.set_visible(False)

    def plotQuantiles(self,x,data,ax,color = "black",single=True, saveLines = True):
        ""
        if not data.shape[0] == 5:
            return
        q25 = data[1,:]
        q50 = data[2,:]
        q75 = data[3,:]
        #plot min and max quantiles
        #plot 25 and 75 quantiles
        qArea = ax.fill_between(x,q25,q75,alpha=self.getParam("alpha.IQR"),facecolor=color,edgecolors="black",linewidth=0.1)
        #plot median
        line = ax.plot(
                    x,
                    q50,
                    linestyle="-",
                    color="black" if single else color,
                    linewidth=self.getParam("linewidth.median"), 
                    marker= None if self.getParam("marker.median") == "none" else self.getParam("marker.median"),
                    markerfacecolor = color, 
                    markeredgecolor = "black", 
                    markeredgewidth = self.getParam("markeredgewidth.median"))
        if saveLines:
            internalID = self.getInternalIDByColor(color)
            if not internalID in self.lineplotItems:
                self.lineplotItems[internalID] = []
            self.lineplotItems[internalID].append(qArea)
            self.lineplotItems[internalID].append(line[0])


    def initLineplot(self, onlyForID = None, targetAx = None):
        ""

        try:
            if not hasattr(self,"lineplotItems"):
                self.lineplotItems = dict() 
            else:
                self.lineplotItems.clear() 

            for n, lineData in self.data["plotData"].items():                
                if n in self.axisDict:

                    if onlyForID is not None and n != onlyForID:
                        continue
                    elif onlyForID is not None and n == onlyForID and targetAx is not None:
                        ax = targetAx
                    else:
                        ax = self.axisDict[n]

                    singleLine = len(lineData) == 1 #will cause colorling the line in black
                    for q in lineData:
                        self.plotQuantiles(
                                        ax = ax, 
                                        data = q["quantiles"],
                                        x = q["xValues"],
                                        color= q["color"],
                                        single=singleLine,
                                        saveLines= onlyForID is None) #if this is None, we are not exporting to the main figure

                    
        except Exception as e:
            print(e)

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
           
            self.initAxes(data["axisPositions"])
            self.initLineplot()
            for n,ax in self.axisDict.items():
                if n in self.data["axisLimits"]:
                    self.setAxisLimits(ax,
                            self.data["axisLimits"][n]["xLimit"],
                            self.data["axisLimits"][n]["yLimit"])

            if self.interactive:
               self.addHoverLine()
               self.addHoverArea()
               self.addHoverBinding() 

            self.addTitles()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.setXTicksForAxes(self.axisDict,
                        data["tickPositions"],
                        data["tickLabels"],
                        rotation=90)
            self.checkForQuickSelectDataAndUpdateFigure()
           
           
        except Exception as e:
            print(e)
        

    def setHoverData(self,dataIndex, showText = False):
        ""
        
        #dataIndex = dataIndex[0]
        for axB in self.axBackground.keys():
            self.p.f.canvas.restore_region(self.axBackground[axB])

        for n,ax in self.axisDict.items():

            idx = self.data["hoverData"][n].index.intersection(dataIndex)
            if not idx.empty:
                
                ys = self.data["hoverData"][n].loc[idx,self.data["numericColumns"]].values
                if ys.shape[0] > 1:
                    
                    columnQuantiles = np.nanquantile(ys, q = [0.25,0.5,0.75],axis=0)
                    ys = columnQuantiles[1,:]
                    q25 = columnQuantiles[0,:]
                    q75 = columnQuantiles[2,:]
                    xs = np.arange(ys.size)
                    polygonXY = np.array([(x,y) for x,y in zip(xs,q25)] + list(reversed([(x,y) for x,y in zip(xs,q75)])))
                    
                    self.hoverAreas[ax].set_xy(polygonXY)
                    self.hoverAreas[ax].set_visible(True)
                    ax.draw_artist(self.hoverAreas[ax])
                else:
                    xs = np.arange(ys.size)

                self.hoverLines[ax].set_visible(True)
                self.hoverLines[ax].set_data(xs,ys)
                

                ax.draw_artist(self.hoverLines[ax])
               
            
        #blit canvas
        self.p.f.canvas.blit(ax.bbox)


    def setHoverObjectsInvisible(self):
        ""
        if hasattr(self,"hoverLines"):
            for l in self.hoverLines.values():
                l.set_visible(False)
        if hasattr(self,"hoverAreas"):
            for area in self.hoverAreas.values():
                area.set_visible(False)

    def getInternalIDByColor(self, color):
        ""
        colorGroupData = self.data["dataColorGroups"]
        boolIdx = colorGroupData["color"].values ==  color
        if np.any(boolIdx):
            return colorGroupData.loc[boolIdx,"internalID"].values[0]

    def updateGroupColors(self,colorGroup,changedCategory=None):
        "changed category is encoded in a internalID"
      
        if self.colorCategoryIndexMatch is not None:
            
            if changedCategory in self.colorCategoryIndexMatch:
                indices = self.colorCategoryIndexMatch[changedCategory]

                for idx in indices:
                    if idx in self.quickSelectLines:
                        dataBool = colorGroup["internalID"] == changedCategory
                        c = colorGroup.loc[dataBool,"color"].values[0]
                        self.quickSelectLines[idx].set_color(c)
                        self.quickSelectLines[idx].set_markerfacecolor(c)
        else:
            # Bug fix (round 1, item 32): this loop only ever mutated the LIVE
            # matplotlib artists (set_color/set_facecolor) that are already on
            # screen. It never touched self.data["plotData"][n][i]["color"],
            # which is the color initLineplot() reads when it (re)builds the
            # lines from scratch. "To main figure" (mirrorAxis ->
            # mirrorAxisContent -> initLineplot(axisID, targetAx)) does
            # exactly that rebuild, so a manually chosen color was silently
            # discarded in favor of the original color - the panel came out
            # looking "reset to default". The color LEGEND transferred
            # correctly because addColorLegendToGraph() below is rebuilt
            # directly from the current colorGroup every time, not from
            # plotData, so it never went stale like the lines did.
            #
            # Bug fix (round 2, item 35): the round-1 fix persisted the new
            # color by looking up the *old* color per internalID from
            # self.data["dataColorGroups"] and finding plotData entries whose
            # stored color still equalled it. That lookup is unusable in
            # practice: ICColorTable's ColorTableModel.initData() keeps
            # self._labels as the SAME DataFrame object passed in (no
            # .copy()), and this plot hands it exactly self.data["dataColorGroups"]
            # via setDataInColorTable(). ColorTableModel.setColor() then edits
            # that shared object in place ("self._labels.loc[idx,'color'] =
            # hexColor") the moment the user picks a color - before
            # updateGroupColors() is even called. So by the time this method
            # runs, self.data["dataColorGroups"] already reads the NEW color
            # for the very entry that changed, "oldColor" was in fact already
            # the new color, the "q.get('color') == oldColor" match against
            # plotData's still-old stored color could never succeed, and the
            # color was silently never persisted - only ever changed live
            # on-screen. Matching by position instead of by color value
            # sidesteps the aliasing entirely: colorGroup's rows and each
            # self.data["plotData"][n] list are built in the same fixed group
            # order at plot creation (see plotterCalculations.py), so row i of
            # colorGroup always corresponds to element i of every plotData
            # facet list, regardless of what color value is stored where.
            positionByInternalID = {internalID: pos for pos, internalID in enumerate(colorGroup["internalID"])}

            for color, _ , internalID in colorGroup.values:
                if internalID in self.lineplotItems:
                    artits = self.lineplotItems[internalID]
                    for l in artits:
                        if hasattr(l,"set_facecolor"):
                            l.set_facecolor(color)
                        elif hasattr(l,"set_markerfacecolor"):
                            l.set_markerfacecolor(color)
                        if isinstance(l,Line2D): #2D line has marker colors and line colors
                            l.set_color(color)

                # persist the new color into the stored plot data (by
                # position, not by matching a now-unrecoverable old color) so
                # a later rebuild (main-figure export, session save/restore)
                # uses it
                pos = positionByInternalID.get(internalID)
                if pos is not None:
                    for lineData in self.data["plotData"].values():
                        if pos < len(lineData):
                            lineData[pos]["color"] = color

            self.data["dataColorGroups"] = colorGroup

        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)
        self.updateFigure.emit()

    def updateBackgrounds(self):
        "Update Background for blitting"
        self.axBackground = dict()
        for ax in self.axisDict.values():
            self.axBackground[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)	
    
    def updateQuickSelectItems(self,propsData=None):
        "Saves lines by idx id"

        colorData = self.getQuickSelectData()
        
        if not hasattr(self,"quickSelectLines"):
            self.quickSelectLines = dict() 

        if not hasattr(self,"quickSelectAreas"):
            self.quickSelectAreas = dict() 

        if self.isQuickSelectModeUnique() and hasattr(self,"quickSelectCategoryIndexMatch"):

            dataIndex = np.concatenate([idx for idx in self.quickSelectCategoryIndexMatch.values()])

        else:

            dataIndex = self.getDataIndexOfQuickSelectSelection() 

        for intID, indics in self.quickSelectCategoryIndexMatch.items():
            for n,ax in self.axisDict.items():

                if ax not in self.quickSelectLines:
                    self.quickSelectLines[ax] = {}
                    self.quickSelectLineKwargs[ax] = {}
                if ax not in self.quickSelectPolygon:
                    self.quickSelectPolygonKwargs[ax] = {}
                    self.quickSelectPolygon[ax] = {}
                
                if intID in self.quickSelectLines[ax]:
                    continue

                c = propsData.loc[indics,"color"].values[0]

                idxInAxSet = self.data["hoverData"][n].index.intersection(indics)
                if idxInAxSet.size == 1:
                    y = self.data["hoverData"][n].loc[idxInAxSet,self.data["numericColumns"]].values.flatten()
                    x = np.arange(y.size)
                    lineKwargs = dict(
                                    xdata = x,
                                    ydata = y, 
                                    marker = self.getParam("marker.quickSelect"), 
                                    markerfacecolor = c, 
                                    color = c, 
                                    linewidth = self.getParam("linewidth.quickSelect"), 
                                    markeredgecolor = "black", 
                                    markeredgewidth = self.getParam("markeredgewidth.quickSelect")
                                    )
                    line = Line2D(**lineKwargs)
                    ax.add_line(line)
                    
                    self.quickSelectLineKwargs[ax][intID] = lineKwargs
                    self.quickSelectLines[ax][intID] = line
                
                elif idxInAxSet.size > 1:

                    ys = self.data["hoverData"][n].loc[idxInAxSet,self.data["numericColumns"]].values
                    columnQuantiles = np.nanquantile(ys, q = [0.25,0.5,0.75],axis=0)

                    ys = columnQuantiles[1,:]
                    q25 = columnQuantiles[0,:]
                    q75 = columnQuantiles[2,:]
                    xs = np.arange(ys.size)
                    polygonXY = np.array([(x,y) for x,y in zip(xs,q25)] + list(reversed([(x,y) for x,y in zip(xs,q75)])))
                    
                    
                    x = np.arange(ys.size)
                    lineKwargs = dict(
                                    xdata = xs,
                                    ydata = ys, 
                                    marker = self.getParam("marker.quickSelect"), 
                                    markerfacecolor = c, 
                                    color = c, 
                                    linewidth = self.getParam("linewidth.quickSelect"), 
                                    markeredgecolor = "black", 
                                    markeredgewidth = self.getParam("markeredgewidth.quickSelect")
                                    )
                    poylgonKwargs = dict(xy = polygonXY,
                                        visible=True,
                                        alpha=self.getParam("alpha.IQR"),
                                        facecolor=c,
                                        edgecolor="black",
                                        linewidth=0.1,
                                        fill=True, 
                                        closed=True)

                    line = Line2D(**lineKwargs)
                    ax.add_line(line)

                    poly = Polygon(**poylgonKwargs)
                    ax.add_patch(poly)
                    
                    self.quickSelectPolygonKwargs[ax][intID] = poylgonKwargs
                    self.quickSelectPolygon[ax][intID] = poly
                    self.quickSelectLineKwargs[ax][intID] = lineKwargs
                    self.quickSelectLines[ax][intID] = line


                else:
                    continue

                self.quickSelectScatterDataIdx[ax] = idxInAxSet.values
    
    def updateQuickSelectData(self,quickSelectGroup,changedCategory=None):
        ""
        c = quickSelectGroup.loc[quickSelectGroup["internalID"] == changedCategory]["color"].values[0]
        for ax in self.axisDict.values():
            if hasattr(self,"quickSelectLines") and ax in self.quickSelectLines and changedCategory in self.quickSelectLines[ax]:
                qSLine = self.quickSelectLines[ax][changedCategory]
                qSLine.set_color(c)
                qSLine.set_markerfacecolor(c)

            if hasattr(self,"quickSelectPolygon") and ax in self.quickSelectPolygon and changedCategory in self.quickSelectPolygon[ax]:

                    poly = self.quickSelectPolygon[ax][changedCategory]
                    poly.set_facecolor(c)

        self.updateFigure.emit()
    
    def mirrorQuickSelectArtists(self,axisID,targetAx):
        ""
        if axisID in self.axisDict:
            sourceAx = self.axisDict[axisID]
            if sourceAx in self.quickSelectLineKwargs:
                for lineKwargs in self.quickSelectLineKwargs[sourceAx].values():
                    targetAx.add_line(Line2D(**lineKwargs))
                for poylgonKwargs in self.quickSelectPolygonKwargs[sourceAx].values():
                    targetAx.add_patch(Polygon(**poylgonKwargs))

    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        
        data = self.data
        self.initLineplot(axisID,targetAx)
        for n,_ in self.axisDict.items():
            if axisID == n and axisID in data["axisLimits"]:
                # Bug fix: this used to pass `ax` (self.axisDict[n], the SOURCE
                # plot's own live axis) instead of `targetAx` (the main-figure
                # export axis). That mutated the source plot's own live zoom/pan
                # back to the limits stored at plot-creation time every time
                # "To main figure" was used, and - because mirrorLimits() later
                # reads sourceAx.get_xlim()/get_ylim() to copy the live view onto
                # the target - it also meant the target panel got those same
                # stale, pre-zoom limits instead of the user's actual current
                # zoom ("axis scaling is not transferred"). ICHistogram.py's
                # mirrorAxisContent already uses the correct target here.
                self.setAxisLimits(targetAx,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
    
        self.setXTicksForAxes({axisID:targetAx},data["tickPositions"],data["tickLabels"], onlyForID = axisID, rotation=90)                         
           

    def resetQuickSelectArtists(self):
        ""
        for ax in self.axisDict.values():
            if hasattr(self,"quickSelectPolygon") and ax in self.quickSelectPolygon:
                for poly in self.quickSelectPolygon[ax].values():
                    poly.set_visible(False)
            if hasattr(self,"quickSelectLines") and ax in self.quickSelectLines:
                for line in self.quickSelectLines[ax].values():
                    line.set_visible(False)

        self.quickSelectPolygonKwargs.clear()
        self.quickSelectLineKwargs.clear()

        if hasattr(self,"quickSelectPolygon"):
            self.quickSelectPolygon.clear()
        if hasattr(self,"quickSelectLines"):
            self.quickSelectLines.clear()
            
        self.updateFigure.emit()