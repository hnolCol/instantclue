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
    
    def checkGroupsForExistingColumnNames(self,columnNames):
        ""
        deletedColumnNames = columnNames
        deleteGroups = []
        alteredGroups = []
        if len(self.groups) > 0:
            for groupingName, groupedItems in self.groups.items():
                columnsInGrouping = self.getColumnNamesFromGroup(groupingName)
               
                if any(colName in columnsInGrouping.values for colName in deletedColumnNames.values):
                    updatedGrouping = OrderedDict()
                    for groupName , columnNames in groupedItems.items():
                        
                        updatedColumnNames = pd.Series([colName for colName in columnNames if colName not in deletedColumnNames.values])
                       
                        if len(updatedColumnNames) < 2 and groupingName not in deleteGroups:
                            deleteGroups.append(groupingName)
                        else:
                            updatedGrouping[groupName] = updatedColumnNames
                       
                    alteredGroups.append((groupingName,updatedGrouping))
                    
           
            for groupingName, updatedGrouping in alteredGroups:
                self.groups[groupingName] = updatedGrouping   
            
            for groupingNameToDelete in deleteGroups:
                #delete groups
                del self.groups[groupingNameToDelete]   
                if groupingNameToDelete == self.currentGrouping:
                    self.currentGrouping = None

            

    
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

    def getColumnNames(self,groupingName = None):
        ""
        if groupingName is None:
            groupingName = self.currentGrouping
        cNames = pd.Series()
        cNames = cNames.append(list(self.groups[self.currentGrouping].values()))
        return cNames
    
    def getColumnNamesFromGroup(self, groupName):
        ""
        if groupName in self.groups:
            cNames = pd.Series()
            cNames = cNames.append(list(self.groups[groupName].values()))
            return cNames
        return pd.Series()

    def getColorsForGroupMembers(self, groupingName = None):
        ""
        
        if not self.currentGrouping in self.groups:
            return None
        if groupingName is None:
            groupingName = self.currentGrouping
        groupColors = self.getGroupColors()
        colorMaps = []
        for groupName, columnNames in self.groups[groupingName].items():
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

    def getGroupNameByColumn(self, groupingName):
        ""
        if groupingName in self.groups:
            return OrderedDict([(colName,k) for k,v in self.groups[groupingName].items() for colName in v.values])

    def getGroupingsByColumnNames(self,columnNames, currentGroupingOnly = False):
        ""
        if not self.groupingExists():
            return OrderedDict()

        annotatedGroupins = OrderedDict() 
        if not currentGroupingOnly:
            for groupingName, groupedItems in self.groups.items():

                groupingColumnNames = self.getColumnNamesFromGroup(groupingName)
                if any(colName in groupingColumnNames.values for colName in columnNames.values):
                    columnNameGroupMatches = self.getGroupNameByColumn(groupingName)
                    annotatedGroupins[groupingName] = columnNames.map(columnNameGroupMatches)
        elif self.currentGrouping is not None:
            groupingColumnNames = self.getColumnNamesFromGroup(self.currentGrouping)
            if any(colName in groupingColumnNames.values for colName in columnNames.values):
                    columnNameGroupMatches = self.getGroupNameByColumn(self.currentGrouping)
                    annotatedGroupins[self.currentGrouping] = columnNames.map(columnNameGroupMatches)

        return annotatedGroupins


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
           # print(list(combinations(groupNames,r=2)))
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

    def getFactorizedColumns(self, groupingName = None):
        ""
        groupingName = self.currentGrouping if groupingName is None else groupingName
        factorizedColumns = {}
        if groupingName in self.groups:
        
            grouping = self.groups[groupingName]
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

    def getGroupNamesByColumnNames(self,columnNames):
        ""
        columnNameMapper = self.getGroupingsByColumnNames(columnNames,True)
        return columnNameMapper[self.currentGrouping]
        
    def nameExists(self,groupingName):
        ""
        return groupingName in self.groups
    
    def updateGroupingsByColumnNameMapper(self,columnNameMapper):
        "Updates items in groups based on a columnNameMapper dict (oldName,newName)"
        prevColumnNames = list(columnNameMapper.keys())
        newColumnNames = list(columnNameMapper.values())
        for groupingName, groups in self.groups.items():
            for groupName, items in groups.items():
                items.replace(to_replace = prevColumnNames,value = newColumnNames,inplace=True)

    
