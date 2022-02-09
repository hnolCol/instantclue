from collections import OrderedDict
import pandas as pd
from matplotlib.colors import Normalize, to_hex
from matplotlib import cm 
from itertools import combinations
import json
from datetime import datetime
import socket

#from main.python.ui.custom.ICQuickSelect import FavoriteSelectionCollection


from .ICExclusiveGroup import ICExlcusiveGroup
from ..utils.stringOperations import getMessageProps

class ICGrouping(object):
    
    def __init__(self,sourceData):

        self.exclusivesMinNonNaN = 1 
        self.sourceData = sourceData
        self.currentGrouping = None
        self.groups = OrderedDict()
        self.groupCmaps = OrderedDict()
        self.groupColorMap = OrderedDict()
        self.exclusiveGrouping = ICExlcusiveGroup(self,sourceData)

    def addCmap(self, groupingName,groupedItems, colorMap = None):
        ""
        m = self.getCmapMapper(groupedItems,colorMap)
        colorMapper, colorMap = m
  
        self.groupCmaps[groupingName] = colorMapper
        self.groupColorMap[groupingName] = colorMap

    def getCmapMapper(self,groupedItems, colorMap = None):
        ""
        norm = Normalize(vmin=-0.2, vmax=len(groupedItems)-0.8)
        if colorMap is None:
            colorMap = self.sourceData.colorManager.colorMap
        
        cmap = self.sourceData.colorManager.get_max_colors_from_pallete(colorMap)
        cmap.set_bad(self.sourceData.colorManager.nanColor)
        return cm.ScalarMappable(norm=norm, cmap=cmap), colorMap

    def getColorMap(self,groupingName):
        ""
        if groupingName in self.groupColorMap:
            return self.groupColorMap[groupingName]

    def getTheroeticalColorsForGroupedItems(self,groupedItems,colorMap=None):
        ""
        cmapMapper,_ = self.getCmapMapper(groupedItems,colorMap)
        colors = dict([(groupName,to_hex(cmapMapper.to_rgba(x))) for x,groupName in enumerate(groupedItems.keys())])
        return colors

    def addGrouping(self,groupingName,groupedItems, setToCurrent = True, colorMap = None):
        "Add grouping to collection"
        if groupingName in self.groups:
            self.deleteGrouping(groupingName)
        
        self.groups[groupingName] = groupedItems
        self.addCmap(groupingName,groupedItems,colorMap)

        if setToCurrent:
            self.currentGrouping = groupingName
    
    def checkGroupsForExistingColumnNames(self,columnNames):
        "Removes groupings if columnNames are removed from the data"
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

    def deleteGrouping(self,groupingName):
        "Deletes a grouping"
        if groupingName in self.groups:
            del self.groups[groupingName]
            del self.groupCmaps[groupingName]
            del self.groupColorMap[groupingName]
            if len(self.groups) == 0:
                self.currentGrouping = None
                funcProps = getMessageProps("Done..","Grouping {} deleted. No groupig found to set as current.".format(groupingName))
            elif self.currentGrouping == groupingName:
                newGrouping = list(self.groups.keys())[-1]
                self.setCurrentGrouping(newGrouping)
                funcProps = getMessageProps("Done..","Grouping {} deleted. New grouping defined: {}".format(groupingName,newGrouping))
            else:
                funcProps = getMessageProps("Done..","Grouping {} deleted.".format(groupingName))
            return funcProps
        else:
            return getMessageProps("Error..","Grouping not found.")

    def renameGrouping(self,groupingName, newGroupingName):
        "Renames a grouping"
        if groupingName in self.groups:
            if newGroupingName is self.groups:
                return getMessageProps("Error..","Grouping name already exists.")
            self.groups[newGroupingName] = self.groups[groupingName].copy()
            self.addCmap(newGroupingName,self.groups[groupingName],colorMap=self.groupColorMap[groupingName])

            if self.currentGrouping == groupingName:
                self.setCurrentGrouping(newGroupingName)
            
            self.deleteGrouping(groupingName)

            return getMessageProps("Done..","Grouping renamed.")
        else:
            return getMessageProps("Error..","Grouping not found.")

    def getAllGroupings(self):
        ""
        groupingKwargs = {"currentGrouping":self.currentGrouping, 
                        "groups" : self.groups,
                        "groupColorMap" : self.groupColorMap,
                        "groupCmaps" : self.groupCmaps}

        return groupingKwargs

    def setGroupinsFromSavedSesssion(self,groupingState):
        ""
        
        for attrName, attrValue in groupingState.items():
            
            setattr(self,attrName,attrValue)
    
        

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
        return self.getColumnNamesFromGroup(groupingName)
    
    def getColumnNamesFromGroup(self, groupingName):
        ""
        if groupingName in self.groups:
            cNames = pd.Series()
            cNames = cNames.append(list(self.groups[groupingName].values()))
            return cNames
        return pd.Series()

    def getColorsForGroupMembers(self, groupingName = None):
        ""
        
        if not self.currentGrouping in self.groups:
            return None
        if groupingName is None:
            groupingName = self.currentGrouping
        groupColors = self.getGroupColors(groupingName)
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
        return self.getGroupNames(self.currentGrouping)

    def getGroupNames(self,grouping):
        ""
        if grouping in self.groups:
            return list(self.groups[grouping].keys())

    def getGroupNameByColumn(self, groupingName):
        ""
        if groupingName in self.groups:
            return OrderedDict([(colName,k) for k,v in self.groups[groupingName].items() for colName in v.values])

    def getGroupingsByColumnNames(self,columnNames, currentGroupingOnly = False):
        ""
        if not self.groupingExists():
            return OrderedDict()
        if isinstance(columnNames,list):
            columnNames = pd.Series(columnNames)

        annotatedGroupings = OrderedDict() 
        if not currentGroupingOnly:
            for groupingName in self.groups.keys():

                groupingColumnNames = self.getColumnNamesFromGroup(groupingName)
                if any(colName in groupingColumnNames.values for colName in columnNames.values):
                    columnNameGroupMatches = self.getGroupNameByColumn(groupingName)
                    annotatedGroupings[groupingName] = columnNames.map(columnNameGroupMatches)
        elif self.currentGrouping is not None:
            groupingColumnNames = self.getColumnNamesFromGroup(self.currentGrouping)
            if any(colName in groupingColumnNames.values for colName in columnNames.values):
                    columnNameGroupMatches = self.getGroupNameByColumn(self.currentGrouping)
                    annotatedGroupings[self.currentGrouping] = columnNames.map(columnNameGroupMatches)

        return annotatedGroupings


    def getNumberOfGroups(self):
        ""
        return len(self.groups)

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
    
    def getGroupPairsOfGrouping(self,grouping):
        ""
        groupNames = self.getGroupNames(grouping)
        return list(combinations(groupNames,r=2))


    def getGroupings(self):
        "Get Grouping Names"
        return list(self.groups.keys())
    
    def getGrouping(self,groupingName):
        ""
        if groupingName in self.groups:
            return self.groups[groupingName]

    def getGroupingsByList(self,groupinNames=[]):
        ""
        if isinstance(groupinNames,list):
            return OrderedDict([(groupingName,self.groups[groupingName]) for groupingName in groupinNames if groupingName in self.groups])
        else:
            return {}

    def getGroupColorsByGroupingList(self,groupingNames):
        ""
        if isinstance(groupingNames,list):
            return dict([(groupingName,self.getGroupColors(groupingName)) for groupingName in groupingNames if groupingName in self.groups])
        else:
            return {}
    def getGroupColors(self, groupingName = None):
        ""
        if groupingName is None:
            groupingName = self.currentGrouping
        if groupingName in self.groupCmaps:
            
            mapper = self.groupCmaps[groupingName]
            colors = dict([(groupName,to_hex(mapper.to_rgba(x))) for x,groupName in enumerate(self.groups[groupingName].keys())])
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

    def getUniqueGroupItemsByGroupingList(self,groupingNames):
        ""
        columnNames = [self.getColumnNamesFromGroup(groupingName) for groupingName in groupingNames]
        return pd.concat(columnNames,ignore_index=True).unique()

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

    def exportGroupingToJson(self,groupingNames, filePath):
        ""
        computerName = socket.gethostname()
        jsonOut = OrderedDict()
        jsonOut["Creation Date"] = datetime.now().strftime("%Y%m%d %H:%M:%S")
        jsonOut["Software"] = "Instant Clue"
        jsonOut["Version"] = self.sourceData.parent.version
        jsonOut["Computer"] = computerName
        jsonOut["grouping"] = OrderedDict()
        jsonOut["groupingCmap"] = dict()
        jsonOut["groupingNames"] = groupingNames.tolist()
        for groupingName in groupingNames:
            if groupingName in self.groups:
                jsonOut["grouping"][groupingName] = dict([(k,v.values.tolist()) for k,v in self.groups[groupingName].items()])
                jsonOut["groupingCmap"][groupingName] = self.getColorMap(groupingName)
       
        with open(filePath, 'w', encoding='utf-8') as f:
            json.dump(jsonOut, f, ensure_ascii=False, indent=4)
        
        return getMessageProps("Done","Grouping saved to json file. You can upload groupings from this file into Instant Clue.")
        
        
    def loadGroupingFromJson(self,filePath):
        ""
        try:
            with open(filePath, 'r', encoding='utf-8') as f:
                jsonLoaded = json.load(f)
                for groupingName, groups in jsonLoaded["grouping"].items():
                    groupedItems = dict([(k,pd.Series(v)) for k,v in groups.items()])
                    if groupingName in jsonLoaded["groupingCmap"]:
                        cmapName = jsonLoaded["groupingCmap"][groupingName]
                    else:
                        cmapName = None
                    self.addGrouping(groupingName, groupedItems,colorMap=cmapName)
        except:
            return getMessageProps("Error ..","There was an error reading the selected json file.")
        return getMessageProps("Done..","Grouping loaded.")
