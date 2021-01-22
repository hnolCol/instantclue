

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
import numpy as np

class ICDimReductionplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICDimReductionplot,self).__init__(*args,**kwargs)


    

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
           
            self.initAxes(data["axisPositions"])

            for n,ax in self.axisDict.items():
                if n in self.data["axisLimits"]:
                    self.setAxisLimits(ax,
                            self.data["axisLimits"][n]["xLimit"],
                            self.data["axisLimits"][n]["yLimit"])

            if self.interactive:
               pass 

            self.addTitles()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.setXTicksForAxes(self.axisDict,
                        data["tickPositions"],
                        data["tickLabels"],
                        rotation=90)
           
           
        except Exception as e:
            print(e)
        

    def setHoverData(self,dataIndex, showText = False):
        ""
       # print(dataIndex)
       # if dataIndex in self.data["plotData"].index:
        
        dataIndex = dataIndex[0]
        for n,ax in self.axisDict.items():
            if not dataIndex in self.data["hoverData"][n].index:
                continue
            else:
                
                for axB in self.axBackground.keys():
                    self.p.f.canvas.restore_region(self.axBackground[axB])
                    
                y = self.data["hoverData"][n].loc[dataIndex,self.data["numericColumns"]].values
                x = np.arange(y.size)
                self.hoverLines[ax].set_visible(True)
                self.hoverLines[ax].set_data(x,y)
                ax.draw_artist(self.hoverLines[ax])
                break 
            
        #blit canvas
        self.p.f.canvas.blit(ax.bbox)

    def updateGroupColors(self,colorGroup,changedCategory=None):
        "changed category is encoded in a internalID"
      
       
        self.updateFigure.emit()

    def updateBackgrounds(self):
        "Update Background for blitting"
        self.axBackground = dict()
        for ax in self.axisDict.values():
            self.axBackground[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)	
    
    def updateQuickSelectItems(self,propsData=None):
        "Saves lines by idx id"

        
    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
    
           
         