from collections import OrderedDict
import pandas as pd
from matplotlib.colors import Normalize, to_hex
from matplotlib import cm 
from itertools import combinations

from .ICExclusiveGroup import ICExlcusiveGroup
from ..utils.stringOperations import getMessageProps

class ICGrouping(object):
    
    def __init__(self,sourceData):

        self.exclusivesMinNonNaN = 1 
        self.sourceData = sourceData
        self.currentGrouping = None
        self.groups = OrderedDict()
        self.groupCmaps = OrderedDict()
        self.exclusiveGrouping = ICExlcusiveGroup(self,sourceData)

    def addCmap(self, groupingName,groupedItems):
        ""
        
        norm = Normalize(vmin=-0.5, vmax=len(groupedItems)+0.5)
        twoColorMap = self.sourceData.colorManager.colorMap
        cmap = self.sourceData.colorManager.get_max_colors_from_pallete(twoColorMap)
        cmap.set_bad(self.sourceData.colorManager.nanColor)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        self.groupCmaps[groupingName] = m
        

    def addGrouping(self,groupingName,groupedItems, setToCurrent = True):
        "Add grouping to collection"
        if groupingName in self.groups:
            del self.groups["groupingName"]
        
        self.groups[groupingName] = groupedItems
        self.addCmap(groupingName,groupedItems)

        if setToCurrent:
            self.currentGrouping = groupingName
    
    def setCurrentGrouping(self,groupingName):
        ""
        if groupingName in self.groups:
            self.currentGrouping = groupingName

    def groupingExists(self):
        "Check if any grouping exists"
        return len(self.groups) > 0

    def findGroupNameOfColumn(self,colName,grouping):
        ""
        for groupName, columnNames in grouping.items():
            if colName in columnNames.values:
                return groupName

    def getColumnNames(self):
        ""
        cNames = pd.Series()
        cNames = cNames.append(list(self.groups[self.currentGrouping].values()))
        return cNames
    
    def getColorsForGroupMembers(self):
        ""
        if not self.currentGrouping in self.groups:
            return None
        groupColors = self.getGroupColors()
        colorMaps = []
        for groupName, columnNames in self.groups[self.currentGrouping].items():
            color = groupColors[groupName]
            colorMaps.extend([(columnName,color) for columnName in columnNames.values])
        return dict(colorMaps)

    def getCurrentCmap(self):
        ""
        if self.currentGrouping in self.groupCmaps:
            return self.groupCmaps[self.currentGrouping]

    def getCurrentGroupingName(self):
        ""
        return self.currentGrouping

    def getCurrentGrouping(self):
        ""
        if self.currentGrouping in self.groups:
            return self.groups[self.currentGrouping]
    
    def getCurrentGroupNames(self):
        ""
        if self.currentGrouping in self.groups:
            return list(self.groups[self.currentGrouping].keys())


    def getPositiveExclusives(self,dataID = None, columnNames = None):
        ""
        if self.groupingExists():
            currentGroupName = self.getCurrentGroupingName() 
            results = self.exclusiveGrouping.findExclusivePositives(dataID)
            results.columns = ["ExclusivePos:({})".format(currentGroupName)]
            return self.sourceData.joinDataFrame(dataID,results)
        else:
            return getMessageProps("Error","No Grouping found.")

    def getNegativeExclusives(self,dataID, columnNames = None):
        ""
       
        if self.groupingExists():
            currentGroupName = self.getCurrentGroupingName() 
            results = self.exclusiveGrouping.findExclusiveNegatives(dataID)
            results.columns = ["ExclusiveNegs:({})".format(currentGroupName)]
            return self.sourceData.joinDataFrame(dataID,results)
        else:
            return getMessageProps("Error","No Grouping found.")

    def getGroupPairs(self,referenceGroup = None):
        ""
        grouping = self.getCurrentGrouping()
        groupNames = self.getCurrentGroupNames()
        if referenceGroup is not None and referenceGroup in grouping:

            return [(referenceGroup,groupName) for groupName in groupNames if groupName != referenceGroup]
        
        else:
            print(list(combinations(groupNames,r=2)))
            return list(combinations(groupNames,r=2))


    def getGroupings(self):
        "Get Grouping Names"
        return list(self.groups.keys())
    
    def getGrouping(self,groupingName):
        ""
        if groupingName in self.groups:
            return self.groups[groupingName]

    def getGroupColors(self):
        ""
        if self.currentGrouping in self.groupCmaps:
            
            mapper = self.groupCmaps[self.currentGrouping]
            colors = dict([(groupName,to_hex(mapper.to_rgba(x))) for x,groupName in enumerate(self.groups[self.currentGrouping].keys())])
            return colors

    def getFactorizedColumns(self):
        ""
        factorizedColumns = {}
        grouping = self.groups[self.currentGrouping]
        factors = dict([(groupName,n) for n,groupName in enumerate(grouping.keys())])
        for colName in self.getColumnNames():

            factorizedColumns[colName] = factors[self.findGroupNameOfColumn(colName,grouping)]

        return factorizedColumns 

    def getGroupItems(self,groupingName):
        ""
        if groupingName in self.groups:
            return self.groups[groupingName] 

    def getNames(self):
        ""
        return list(self.groups.keys())

    def getSizes(self):
        ""
        return dict([(k,len(v)) for k,v in self.groups.items()])

    def nameExists(self,groupingName):
        ""
        return groupingName in self.groups