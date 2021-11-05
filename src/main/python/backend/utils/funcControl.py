



updateTreeView = {"obj":"self","fn":"updateDataInTreeView","objKey":"data","objName":"mainFrames","requiredKwargs":["columnNamesByType"],"optionalKwargs":["dataID"]}
sendMessageProps = {"obj":"self","fn":"sendMessageRequest","requiredKwargs":["messageProps"]}
updateGrouping = {"obj":"self","fn":"updateGroupingInTreeView","objKey":"data","objName":"mainFrames","requiredKwargs":[]}
refreshColumnView = [
                updateTreeView,
                sendMessageProps]   

addDataAndRefresh = [
                {"obj":"self","fn":"resetReceiverBoxes","objKey":"middle","objName":"mainFrames","requiredKwargs":[]},
                {"obj":"self","fn":"updateDataFrames","objKey":"data","objName":"mainFrames","requiredKwargs":["dfs"]},
                updateTreeView,
                sendMessageProps]

addDataOrShow = [
                {"obj":"self","fn":"updateDataFrames","objKey":"data","objName":"mainFrames","requiredKwargs":["dfs"]},
                {"obj":"self","fn":"openDataFrameinDialog","objKey":"data","objName":"mainFrames","requiredKwargs":["dataFrame"]},
                sendMessageProps]
#addDataAndRefresh = [
 #               {"obj":"self","fn":"updateDataFrames","objKey":"data","objName":"mainFrames","requiredKwargs":["dfs"]},
 #               sendMessageProps]
#
funcPropControl = {
    
    "addDataFrame": #will be removed soon.
        {
            "threadRequest":{"obj":"data","fn":"addDataFrame","requiredKwargs":["dataFrame"]},
            "completedRequest": 
                            addDataAndRefresh
        },

    "data::renameDataFrame":
        {
            "threadRequest":{"obj":"data","fn":"setFileNameByID","requiredKwargs":["dataID","fileName"]},
            "completedRequest": 
                            [{"obj":"self","fn":"updateDataFrames","objKey":"data","objName":"mainFrames","requiredKwargs":["dfs"]},
                            sendMessageProps]
        }, 
    "data::addDataFrame":
        {
            "threadRequest":{"obj":"data","fn":"addDataFrame","requiredKwargs":["dataFrame"]},
            "completedRequest": 
                            addDataAndRefresh
        },
    "data::addDataFrameFromTxtFile":
        {
            "threadRequest":{"obj":"data","fn":"addDataFrameFromTxtFile","requiredKwargs":["pathToFile","fileName"]},
            "completedRequest":
                            addDataAndRefresh
                   
        },
    "data::addDataFrameFromExcelFile":
        {
            "threadRequest":{"obj":"data","fn":"addDataFrameFromExcelFile","requiredKwargs":["pathToFile","fileName"]},
            "completedRequest":
                            addDataAndRefresh      
        },
    "data::addDataFrameFromClipboard":
        {
            "threadRequest":{"obj":"data","fn":"readDataFromClipboard","requiredKwargs":[]},
            "completedRequest":[
                {"obj":"self","fn":"resetReceiverBoxes","objKey":"middle","objName":"mainFrames","requiredKwargs":[]},
                {"obj":"self","fn":"updateDataFrames","objKey":"data","objName":"mainFrames","requiredKwargs":["dfs"]},
                updateTreeView,
                sendMessageProps]       
        },
        
    "data::aggregateNRows":
        {
            "threadRequest":{"obj":"data","fn":"aggregateNRows","requiredKwargs":["dataID","columnNames","metric"]},
            "completedRequest": addDataAndRefresh
        },
   
    "data::getColumnNamesByDataID":
        {
            "threadRequest":{"obj":"data","fn":"getColumnNamesByDataID","requiredKwargs":["dataID"]},
            "completedRequest":[
                {"obj":"self","fn":"resetReceiverBoxes","objKey":"middle","objName":"mainFrames","requiredKwargs":[]},
                updateTreeView,
                sendMessageProps]     
        },
    "data::getColumnNamesByDataIDSilently":
        {
            "threadRequest":{"obj":"data","fn":"getColumnNamesByDataID","requiredKwargs":["dataID"]},
            "completedRequest":[
                updateTreeView,
                sendMessageProps]     
        },
        
    "data::getDataByColumnNamesForPlotter":
        {
            "threadRequest":{"obj":"data","fn":"getDataByColumnNames","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
               # #{"obj":"self","fn":"updateActivePlotterFn","objKey":"middle","objName":"mainFrames","requiredKwargs":["fnName","fnKwargs"]},
               # {"obj":"self","fn":"setData","objKey":"middle","objName":"mainFrames","requiredKwargs":["fnName","fnKwargs"]}#
                {"obj":"self","fn":"updateActivePlotterFn","objKey":"middle","objName":"mainFrames","requiredKwargs":["fnName","fnKwargs"]},
                {"obj":"self","fn":"updateFigure","objKey":"middle","objName":"mainFrames","requiredKwargs":[],"optionalKwargs":["newPlot"]}
                ]     
        },
    "data::getDataByColumnNamesForTooltip":
        {
            "threadRequest":{"obj":"data","fn":"getDataByColumnNames","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                {"obj":"self","fn":"setDataToolTipData","objKey":"middle","objName":"mainFrames","requiredKwargs":["data"]}
                ]     
        },
     "data::copyDataFrameSelection":
        {
            "threadRequest":{"obj":"data","fn":"copyDataFrameSelection","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                sendMessageProps]   
        },
    "data::dropColumns":
        {
            "threadRequest":{"obj":"data","fn":"dropColumns","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                {"obj":"self","fn":"removeItemsFromReceiverBox","objKey":"middle","objName":"mainFrames","requiredKwargs":["columnNames"]},
                {"obj":"grouping","fn":"checkGroupsForExistingColumnNames","requiredKwargs":["columnNames"]},
                updateTreeView,
                sendMessageProps]   
        },
    "data::deleteData":
        {
            "threadRequest":{"obj":"data","fn":"deleteData","requiredKwargs":["dataID"]},
            "completedRequest": addDataAndRefresh
        },
    "data::exportData":
        {
            "threadRequest":{"obj":"data","fn":"exportData","requiredKwargs":["dataID","path"]},
            "completedRequest":[
                sendMessageProps]
        },
     "data::explodeDataByColumn":
        {
            "threadRequest":{"obj":"data","fn":"explodeDataByColumn","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":addDataAndRefresh
        },   
     "data::transpose":
        {
            "threadRequest":{"obj":"data","fn":"transposeDataFrame","requiredKwargs":["dataID"]},
            "completedRequest":addDataAndRefresh
        }, 
    "data::transposeSelection":
        {
            "threadRequest":{"obj":"data","fn":"transposeDataFrame","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":addDataAndRefresh
        }, 
    "data::renameColumns":
        {
            "threadRequest":{"obj":"data","fn":"renameColumns","requiredKwargs":["dataID","columnNameMapper"]},
            "completedRequest":[
                {"obj":"self","fn":"updateGroupingsByColumnNameMapper","objName":"grouping","requiredKwargs":["columnNameMapper"]},
                updateTreeView,
                {"obj":"self","fn":"renameColumns","objKey":"middle","objName":"mainFrames","requiredKwargs":["columnNameMapper"]},
                sendMessageProps]   
        },
    "data::changeDataType":
        {
            "threadRequest":{"obj":"data","fn":"changeDataType","requiredKwargs":["dataID","columnNames","newDataType"]},
            "completedRequest":refreshColumnView
        },
    "data::combineColumns":
        {
            "threadRequest":{"obj":"data","fn":"combineColumns","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":refreshColumnView
        },
    "data::duplicateColumns":
        {
            "threadRequest":{"obj":"data","fn":"duplicateColumns","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":refreshColumnView
        },
    "data::addIndexColumn":
        {
            "threadRequest":{"obj":"data","fn":"addIndexColumn","requiredKwargs":["dataID"]},
            "completedRequest":refreshColumnView
        },
    "data::addGroupIndexColumn":
        {
            "threadRequest":{"obj":"data","fn":"addGroupIndexColumn","requiredKwargs":["dataID"]},
            "completedRequest":refreshColumnView
        },

    "data::filterFasta":
        {
            "threadRequest":{"obj":"data","fn":"filterFastaFileByColumnIDs","requiredKwargs":["dataID","columnNames","fastaFile"]},
            "completedRequest":[sendMessageProps]
        },
    "data::factorizeColumns":
        {
            "threadRequest":{"obj":"data","fn":"factorizeColumns","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":refreshColumnView
        },
    "data::filterDataByVariance":
        {
            "threadRequest":{"obj":"data","fn":"filterDataByVariance","requiredKwargs":["dataID","columnNames","varThresh"]},
            "completedRequest":addDataAndRefresh
        },
    
    "data::setClippingMask":
        {
            "threadRequest":{"obj":"data","fn":"setClippingByFilter","requiredKwargs":["dataID","filterProps","checkedLabels","columnName"]},
            "completedRequest":[
                {"obj":"self","fn":"setMask","objKey":"middle","objName":"mainFrames","requiredKwargs":["maskIndex"]},
                sendMessageProps]   
        },
    "data::resetClipping":
        {
            "threadRequest":{"obj":"data","fn":"resetClipping","requiredKwargs":["dataID"]},
            "completedRequest":[
                {"obj":"self","fn":"resetMask","objKey":"middle","objName":"mainFrames","requiredKwargs":[]},
                {"obj":"self","fn":"updateFigure","objKey":"middle","objName":"mainFrames","requiredKwargs":[],"optionalKwargs":["newPlot"]},
                sendMessageProps]   
        },
    "data::groupbyAndAggregate":
        {
            "threadRequest":{"obj":"data","fn":"groupbyAndAggregate","requiredKwargs":["dataID","columnNames","groupbyColumn"]},
            "completedRequest": addDataAndRefresh  
        },

        
    "receiverBox:addItems":
        {
            "threadRequest":{"obj":"data","fn":"evaluateColumnsForPlot","requiredKwargs":["dataID","columnNames","dataType"]},
            "completedRequest":[
                {"obj":"self","fn":"addItemsToReceiverBox","objKey":"middle","objName":"mainFrames","requiredKwargs":["columnNamesByType"],"optionalKwargs":["numUniqueValues"]},
                sendMessageProps]   
        },

    "receiverBox:removeItems":
        {
            "threadRequest":{"obj":"self","fn":"removeItemsFromReceiverBox","objKey":"middle","objName":"mainFrames","requiredKwargs":[]},
            "completedRequest":[]
        },
    "data::getColorDictsByFilter":
        {
            "threadRequest":{"obj":"data","fn":"getColorDictsByFilter","requiredKwargs":["dataID","filterProps","checkedLabels","columnName"]},
            "completedRequest":[
                    {"obj":"self","fn":"setCategoryIndexMatch","objKey":"middle","objName":"mainFrames","requiredKwargs":["categoryIndexMatch"],"optionalKwargs":["categoryEncoded"]},
                    {"obj":"self","fn":"updateColorAndSizeInQuickSelect","objKey":"data","objName":"mainFrames","requiredKwargs":[],"optionalKwargs":["checkedColors","checkedSizes"]},
                    {"obj":"self","fn":"updateQuickSelectSelectionInGraph","objKey":"middle","objName":"mainFrames","requiredKwargs":["propsData"]},
                    {"obj":"self","fn":"updateFigure","objKey":"middle","objName":"mainFrames","requiredKwargs":[],"optionalKwargs":["newPlot","ommitRedraw"]},
                    {"obj":"self","fn":"setQuickSelectData","objKey":"sliceMarks","objName":"mainFrames","requiredKwargs":["quickSelectData"],"optionalKwargs":["title"]},
                    sendMessageProps]   
        },
    "data::getColorMapByCategoricalColumn":
        {
            "threadRequest":{"obj":"data","fn":"ggetCategoricalColorMap","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                    {"obj":"self","fn":"updateActivePlotterFn","objKey":"middle","objName":"mainFrames","requiredKwargs":["fnName","fnKwargs"]},
                    sendMessageProps]   
        },
    "data::transformData":
        {
            "threadRequest":{"obj":"data","fn":"transformData","requiredKwargs":["dataID","columnNames","transformation"]},
            "completedRequest":[
                    sendMessageProps]   
        },
    
    
    "data::joinDataFrame":
        {
            "threadRequest":{"obj":"data","fn":"joinDataFrame","requiredKwargs":["dataID","dataFrame"]},
            "completedRequest":
                    refreshColumnView
        },


    "data::kernelDensity":
        {
            "threadRequest":{"obj":"data","fn":"getKernelDensity","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    refreshColumnView
                    
        },

    "data::meltData":
        {
            "threadRequest":{"obj":"data","fn":"meltData","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    addDataAndRefresh
        },
    "data::unstackColumn":
        {
            "threadRequest":{"obj":"data","fn":"unstackColumn","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    addDataAndRefresh
        },

    "data::pivotTable":
        {
            "threadRequest":{"obj":"data","fn":"pivotTable","requiredKwargs":["dataID","indexColumn","columnNames"]},
            "completedRequest":
                    addDataAndRefresh
        },
    "data::updateData":
        {
            "threadRequest":{"obj":"data","fn":"updateData","requiredKwargs":["dataID","data"]},
            "completedRequest":
                    [sendMessageProps]
        },
    "data::fillNa":
        {
            "threadRequest":{"obj":"data","fn":"fillNaNBy","requiredKwargs":["dataID","columnNames","fillBy"]},
            "completedRequest":
                    [sendMessageProps]
        },

    "copyDataFrameByIdToClipboard":
        {
            "threadRequest":{"obj":"data","fn":"copyDataFrameToClipboard","requiredKwargs":["dataID"]},
            "completedRequest":[sendMessageProps]
        },
    
    "copyDataFrameToClipboard":
        {
            "threadRequest":{"obj":"data","fn":"copyDataFrameToClipboard","requiredKwargs":["data"]},
            "completedRequest":[sendMessageProps]
        },
    "data:copyDataToClipboard":
        {
            "threadRequest":{"obj":"data","fn":"copyDataFrameToClipboard","requiredKwargs":[]},
            "completedRequest":[sendMessageProps]
        },
    "data:copyDataFromQuickSelectToClipboard":
        {
            "threadRequest":{"obj":"data","fn":"joinAndCopyDataForQuickSelect","requiredKwargs":["dataID","columnName","selectionData"]},
            "completedRequest":[sendMessageProps]
        },
    "data::updateQuickSelectData":
        {
        "threadRequest":{"obj":"data","fn":"getQuickSelectData","requiredKwargs":["dataID","filterProps"]},
        "completedRequest":
            [{"obj":"self","fn":"updateDataInQuickSelect","objKey":"data","objName":"mainFrames","requiredKwargs":["data"]},
            sendMessageProps]
        },
    "data::removeNaN":
        {
            "threadRequest":{"obj":"data","fn":"removeNaN","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    [
                    {"obj":"grouping","fn":"checkGroupsForExistingColumnNames","requiredKwargs":["columnNames"]},
                    updateTreeView,
                    sendMessageProps]
        },
    "data::rowWiseCalculations":
        {
            "threadRequest":{"obj":"data","fn":"rowWiseCalculations","requiredKwargs":["dataID","calculationProps"]},
            "completedRequest":
                    refreshColumnView
        },
        
    "data::countNaN":
        {
            "threadRequest":{"obj":"data","fn":"countNaN","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    refreshColumnView
        },
     "data::correlateDataFrames":
        {
            "threadRequest":{"obj":"data","fn":"correlateDfs","requiredKwargs":[]},
            "completedRequest":
                    addDataAndRefresh
        },   


     "data::correlateFeaturesOfDataFrames":
        {
            "threadRequest":{"obj":"data","fn":"correlateEachFeatureOfTwoDfs","requiredKwargs":[]},#correlateFeaturesDfs
            "completedRequest":
                    addDataAndRefresh
        },  
        
     "data::mergeDataFrames":
        {
            "threadRequest":{"obj":"data","fn":"mergeDfs","requiredKwargs":["mergeParams"]},
            "completedRequest":
                    addDataAndRefresh
        },   
        
    "data::sortData":
        {
            "threadRequest":{"obj":"data","fn":"sortData","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    [sendMessageProps]
        },
    
    "data::sortColumns":
        {
            "threadRequest":{"obj":"data","fn":"sortColumns","requiredKwargs":["dataID","sortedColumnDict"]},
            "completedRequest":
                    [sendMessageProps]
        },

    "data::sortDataByValues":
        {
            "threadRequest":{"obj":"data","fn":"sortDataByValues","requiredKwargs":["dataID","columnName","values"]},
            "completedRequest":
                    [sendMessageProps]
        },

    "data::splitColumnsByString":
        {
            "threadRequest":{"obj":"data","fn":"splitColumnsByString","requiredKwargs":["dataID","columnNames","splitString"]},
            "completedRequest":refreshColumnView
        },
    "data::summarizeGroups":
        {
            "threadRequest":{"obj":"data","fn":"summarizeGroups","requiredKwargs":["dataID","grouping","metric"]},
            "completedRequest":refreshColumnView
        },
    
    "data::exportHClustToExcel":
        {
            "threadRequest":{"obj":"data","fn":"exportHClustToExcel","requiredKwargs":["dataID","pathToExcel","clusteredData","colorArray","totalRows","quickSelectData"]},
            "completedRequest":
                [sendMessageProps]
        },
    "data::smartReplace":
        {
            "threadRequest":{"obj":"data","fn":"fillNaNBySmartReplace","requiredKwargs":["dataID","columnNames","grouping"]},
            "completedRequest":
                [sendMessageProps]
        },
    "data::subsetDataByIndex":
        {
            "threadRequest":{"obj":"data","fn":"subsetDataByIndex","requiredKwargs":["dataID", "filterIdx", "subsetName"]},
            "completedRequest":
                [{"obj":"self","fn":"updateDataFrames","objKey":"data","objName":"mainFrames","requiredKwargs":["dfs"]},
                sendMessageProps]
        },

    "data::replaceNaNByGroupMean":
        {
            "threadRequest":{"obj":"data","fn":"fillNaNByGroupMean","requiredKwargs":["dataID"]},
            "completedRequest":
                refreshColumnView
        },


    "data::replaceGroupOutlierWithNaN":
        {
            "threadRequest":{"obj":"data","fn":"replaceGroupOutlierWithNaN","requiredKwargs":["dataID","grouping"]},
            "completedRequest":
                refreshColumnView
        },
    "data::replaceOutlierWithNaN":
        {
            "threadRequest":{"obj":"data","fn":"replaceSelectionOutlierWithNaN","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                refreshColumnView
        },

    "data::imputeByModel":
        {
            "threadRequest":{"obj":"data","fn":"imputeNanByModel","requiredKwargs":["dataID","columnNames","estimator"]},
            "completedRequest":
                refreshColumnView
        },
    "data::replace":
        {
            "threadRequest":{"obj":"data","fn":"replaceInColumns","requiredKwargs":["dataID","findStrings","replaceStrings"]},
            "completedRequest":
                [{"obj":"self","fn":"updateGroupingsByColumnNameMapper","objName":"grouping","requiredKwargs":["columnNameMapper"]}] + 
                refreshColumnView
        },
    "data::setNaNBasedOnCondition":
        {
            "threadRequest":{"obj":"data","fn":"setNaNBasedOnCondition","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                refreshColumnView
        },        
    

    "filter::subsetShortcut":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"subsetDataOnShortcut","requiredKwargs":["dataID","columnNames","how","stringValue"]},
            "completedRequest":[
                {"obj":"self","fn":"updateDataFrames","objKey":"data","objName":"mainFrames","requiredKwargs":["dfs"],"optionalKwargs":["selectLastDf"]},
                sendMessageProps]
        },

    "filter::splitDataFrame":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"splitDataFrame","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                {"obj":"self","fn":"updateDataFrames","objKey":"data","objName":"mainFrames","requiredKwargs":["dfs"],"optionalKwargs":["selectLastDf"]},
                sendMessageProps]
        },
    "filter::subsetData":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"subsetData","requiredKwargs":[]},
            "completedRequest":addDataAndRefresh
        },
    "filter::annotateCategory":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"annotateCategory","requiredKwargs":["dataID","columnName","searchString"]},
            "completedRequest":[sendMessageProps,
            updateTreeView]
        },
    "filter::searchString":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"searchString","requiredKwargs":["dataID","columnNames","searchString"]},
            "completedRequest":[
                sendMessageProps,
                updateTreeView]
        },
    "filter::uniqueCategories":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"getUniqueCategories","requiredKwargs":["dataID","columnName"]},
            "completedRequest":[
                sendMessageProps]
        },
    "filter::setupLiveStringFilter":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"setupLiveStringFilter","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                sendMessageProps]
        },
    "filter::liveStringSearch":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"liveStringSearch","requiredKwargs":["searchString"]},#"optionalKwargs":["updatedData","forceSearch","inputIsRegEx","caseSensitive"]
            "completedRequest":[
                {"obj":"self","fn":"updateFilter","objKey":"sliceMarks","objName":"mainFrames","requiredKwargs":["boolIndicator"],"optionalKwargs":["resetData"]}]
        },
    "filter::applyLiveFilter":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"applyLiveFilter","requiredKwargs":["searchString"]},#"optionalKwargs":["updatedData","forceSearch","inputIsRegEx","caseSensitive"]
            "completedRequest":[
                updateTreeView,
                sendMessageProps]
        },
    "filter::stopLiveStringSearch":
        {
            "threadRequest":{"obj":"categoricalFilter","fn":"stopLiverFilter","requiredKwargs":[]},
            "completedRequest":[]
        },

    "filter::filterFromMenu":
        {
            "threadRequest":
                {"obj":"self","fn":"applyFilter","objKey":"sliceMarks","objName":"mainFrames","requiredKwargs":["columnNames"]},
            "completedRequest":[]
        },
    "filter::numericFilter":
        {
            "threadRequest":{"obj":"numericFilter","fn":"applyFilter","requiredKwargs":["dataID","filterProps"]},
            "completedRequest":refreshColumnView               
        },

    "filter::subsetNumericFilter":
        {
            "threadRequest":{"obj":"numericFilter","fn":"applyFilter","requiredKwargs":["dataID","filterProps"]},
            "completedRequest": addDataAndRefresh
                    
        },
    "filter::consecutiveValues":
        {
            "threadRequest":{"obj":"numericFilter","fn":"findConsecutiveValues","requiredKwargs":["dataID","columnNames","increasing"]},
            "completedRequest":refreshColumnView               
        },
    "filter::consecutiveValuesInGrouping":
        {
            "threadRequest":{"obj":"numericFilter","fn":"findConsecutiveValuesInGrouping","requiredKwargs":["dataID","groupingName","increasing"]},
            "completedRequest":refreshColumnView               
        },
        

    "data::removeDuplicates":
        {
            "threadRequest":{"obj":"data","fn":"removeDuplicates","requiredKwargs":["dataID","columnNames"]},
            "completedRequest": addDataAndRefresh      
        },
    
    "normalizer::normalizeGroupMedian":
        {
            "threadRequest":{"obj":"normalizer","fn":"normalizeGroupMedian","requiredKwargs":["dataID","normKey"]},
            "completedRequest": [
                updateTreeView,
                sendMessageProps]
                    
        },
    "normalizer::normalizeData":
        {
            "threadRequest":{"obj":"normalizer","fn":"normalizeData","requiredKwargs":["dataID","columnNames","normKey"]},
            "completedRequest": [
                updateTreeView,
                sendMessageProps]
                    
        },
    "transformer::transformData":
        {
            "threadRequest":{"obj":"transformer","fn":"transformData","requiredKwargs":["dataID","columnNames","transformKey"]},
            "completedRequest": [
                updateTreeView,
                sendMessageProps]       
        },
    "statistic::calculateAUC":
        {
            "threadRequest":{"obj":"statCenter","fn":"calculateAUCFromGraph","requiredKwargs":["dataID","numericColumnPairs","chartData"]},
            "completedRequest":
                addDataOrShow
        },

        
    "dimReduction::LDA":
        {
            "threadRequest":{"obj":"statCenter","fn":"runLDA","requiredKwargs":["dataID","groupingName"]},
            "completedRequest":
                addDataAndRefresh
        },

    "dimReduction::TSNE":
        {
            "threadRequest":{"obj":"statCenter","fn":"runTSNE","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                updateTreeView,
                sendMessageProps]  
        },
    "dimReduction::PCA":
        {
            "threadRequest":{"obj":"statCenter","fn":"runPCA","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                addDataAndRefresh
        },
    "dimReduction::CVAE":
        {
            "threadRequest":{"obj":"statCenter","fn":"runCVAE","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                    updateTreeView,
                    sendMessageProps]           
        },
    "dimReduction::ManifoldEmbedding":
        {
            "threadRequest":{"obj":"statCenter","fn":"runManifoldEmbedding","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                    updateTreeView,
                    sendMessageProps]           
        },      
    "dimReduction::UMAP":
        {
            "threadRequest":{"obj":"statCenter","fn":"runUMAP","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                addDataAndRefresh
        },
    "dimReduction::PCAForPlot":
        {
            "threadRequest":{"obj":"statCenter","fn":"runPCA","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[
                {"obj":"self","fn":"updateDataInPlotter","objKey":"middle","objName":"mainFrames","requiredKwargs":["data"]},
                sendMessageProps
            ]
        },
    "stats::fitModel": 
        {
            "threadRequest":{"obj":"statCenter","fn":"fitModel","requiredKwargs":["dataID", "timeGrouping", "compGrouping","columnNames"]},
            "completedRequest":
                    refreshColumnView          
        },
    "stats::runCombat": 
        {
            "threadRequest":{"obj":"statCenter","fn":"runBatchCorrection","requiredKwargs":["dataID", "groupingName","grouping"]},
            "completedRequest":
                    addDataAndRefresh        
        },
    "stats::runNWayANOVA": 
        {
            "threadRequest":{"obj":"statCenter","fn":"runNWayANOVA","requiredKwargs":["dataID", "groupings"]},
            "completedRequest":
                    addDataAndRefresh        
        },  
    "stats::runRMTwoWayANOVA": 
        {
            "threadRequest":{"obj":"statCenter","fn":"runRMOneTwoWayANOVA","requiredKwargs":["dataID", "withinGrouping1","withinGrouping2","subjectGrouping"]},
            "completedRequest":
                    addDataAndRefresh        
        },    

       
     "stats::runMixedTwoWayANOVA": 
        {
            "threadRequest":{"obj":"statCenter","fn":"runMixedTwoWayANOVA","requiredKwargs":["dataID", "groupingWithin","groupingBetween","groupingSubject"]},
            "completedRequest":
                    addDataAndRefresh        
        },   

    "stats::runFisherEnrichment":
        {
            "threadRequest":{"obj":"statCenter","fn":"runCategoricalFisherEnrichment","requiredKwargs":["data", "categoricalColumn","testColumns"]},
            "completedRequest":
                    addDataAndRefresh        
        },   
          

    "stats::compareGroups": 
        {
            "threadRequest":{"obj":"statCenter","fn":"runComparison","requiredKwargs":["dataID","grouping","test"]},
            "completedRequest":
                    refreshColumnView          
        },

    "stats::oneDEnrichment":
        {
            "threadRequest":{"obj":"statCenter","fn":"runOneDEnrichment","requiredKwargs":["dataID","columnNames","categoricalColumns","alternative"]},
            "completedRequest":
                    addDataAndRefresh        
        },

    "proteomics::matchModSequenceToSites": 
        {
            "threadRequest":{"obj":"data","fn":"matchModSequenceToSites","requiredKwargs":["dataID","fastaFilePath","proteinGroupColumn","modifiedPeptideColumn"]},
            "completedRequest":
                    refreshColumnView        
        },
    
    
    "transform::TSNE": 
        {
            "threadRequest":{"obj":"statCenter","fn":"runTSNE","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":[]           
        },
    "plotter:addLinearFit": 
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getLinearRegression","requiredKwargs":["dataID","numericColumnPairs"]},
            "completedRequest":[
                {"obj":"self","fn":"addLine","objKey":"middle","objName":"mainFrames","requiredKwargs":["lineData"]},
                {"obj":"self","fn":"updateFigure","objKey":"middle","objName":"mainFrames","requiredKwargs":[],"optionalKwargs":["newPlot"]},
                sendMessageProps
            ]           
        },  

    
    "plotter:addLowessFit": 
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getLowessLine","requiredKwargs":["dataID","numericColumnPairs"]},
            "completedRequest":[
                {"obj":"self","fn":"addLine","objKey":"middle","objName":"mainFrames","requiredKwargs":["lineData"]},
                {"obj":"self","fn":"addArea","objKey":"middle","objName":"mainFrames","requiredKwargs":["areaData"]},
                {"obj":"self","fn":"updateFigure","objKey":"middle","objName":"mainFrames","requiredKwargs":[],"optionalKwargs":["newPlot"]},
                sendMessageProps
            ]           
        }, 
    "plotter:getData": 
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getPlotProps","requiredKwargs":["dataID","numericColumns","categoricalColumns","plotType"]},
            "completedRequest":[
                {"obj":"self","fn":"setGraph","objKey":"middle","objName":"mainFrames","requiredKwargs":["plotType"]},
                {"obj":"self","fn":"setData","objKey":"middle","objName":"mainFrames","requiredKwargs":["data"]},
                sendMessageProps
            ]           
        },
    "plotter:getScatterColorGroups": 
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getColorGroupsDataForScatter","requiredKwargs":["dataID"]},
            "completedRequest":[
                {"obj":"self","fn":"setColorGroupData","objKey":"sliceMarks","objName":"mainFrames","requiredKwargs":["colorGroupData"],"optionalKwargs":["title","isEditable"]},
                {"obj":"self","fn":"updateScatterProps","objKey":"middle","objName":"mainFrames","requiredKwargs":["propsData"]},
                {"obj":"self","fn":"setCategoryIndexMatch","objKey":"middle","objName":"mainFrames","requiredKwargs":["categoryIndexMatch"],"optionalKwargs":["categoryEncoded"]},
                sendMessageProps
            ]           
        },
    "plotter:getSwarmColorGroups": 
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getColorGroupsDataForScatter","requiredKwargs":["dataID"]},
            "completedRequest":[
                {"obj":"self","fn":"setColorGroupData","objKey":"sliceMarks","objName":"mainFrames","requiredKwargs":["colorGroupData"],"optionalKwargs":["title","isEditable"]},
                {"obj":"self","fn":"updateScatterProps","objKey":"middle","objName":"mainFrames","requiredKwargs":["propsData"]},
                {"obj":"self","fn":"setCategoryIndexMatch","objKey":"middle","objName":"mainFrames","requiredKwargs":["categoryIndexMatch"],"optionalKwargs":["categoryEncoded"]},
                sendMessageProps
            ]           
        },
    "plotter:getScatterSizeGroups": 
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getSizeGroupsForScatter","requiredKwargs":["dataID"]},
            "completedRequest":[
                {"obj":"self","fn":"setSizeGroupData","objKey":"sliceMarks","objName":"mainFrames","requiredKwargs":["sizeGroupData"],"optionalKwargs":["title","isEditable"]},
                {"obj":"self","fn":"updateScatterProps","objKey":"middle","objName":"mainFrames","requiredKwargs":["propsData"]},
                {"obj":"self","fn":"setCategoryIndexMatch","objKey":"middle","objName":"mainFrames","requiredKwargs":["categoryIndexMatch"],"optionalKwargs":["categoryEncoded"]},
                sendMessageProps
            ]           
        },
    "plotter:getNearestNeighbors": 
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getNearestNeighborConnections","requiredKwargs":["dataID","numericColumnPairs"]},
            "completedRequest":[
                {"obj":"self","fn":"addLineCollections","objKey":"middle","objName":"mainFrames","requiredKwargs":["lineCollections"]},
                {"obj":"self","fn":"updateFigure","objKey":"middle","objName":"mainFrames","requiredKwargs":[],"optionalKwargs":["newPlot"]},
                sendMessageProps
            ]           
        },


        

    "plotter:getScatterMarkerGroups":
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getMarkerGroupsForScatter","requiredKwargs":["dataID"]},
            "completedRequest":[
                {"obj":"self","fn":"updateScatterProps","objKey":"middle","objName":"mainFrames","requiredKwargs":["propsData"]},
                {"obj":"self","fn":"setMarkerGroupData","objKey":"sliceMarks","objName":"mainFrames","requiredKwargs":["markerGroupData"],"optionalKwargs":["title"]},
                sendMessageProps
            ]           
        },
    
    "plotter:getHclustSizeGroups": 
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getSizeQuadMeshForHeatmap","requiredKwargs":["dataID"]},
            "completedRequest":[
                #{"obj":"self","fn":"setSizeGroupData","objKey":"sliceMarks","objName":"mainFrames","requiredKwargs":["sizeGroupData"],"optionalKwargs":["title"]},
                {"obj":"self","fn":"updateHclustSize","objKey":"middle","objName":"mainFrames","requiredKwargs":["sizeData"]},
                #{"obj":"self","fn":"setCategoryIndexMatch","objKey":"middle","objName":"mainFrames","requiredKwargs":["categoryIndexMatch"],"optionalKwargs":["categoryEncoded"]},
                #sendMessageProps
            ]           
        },  
    "plotter:getHclustColorGroups": 
        {
            "threadRequest":{"obj":"plotterBrain","fn":"getColorQuadMeshForHeatmap","requiredKwargs":["dataID"]},
            "completedRequest":[
                #{"obj":"self","fn":"setSizeGroupData","objKey":"sliceMarks","objName":"mainFrames","requiredKwargs":["sizeGroupData"],"optionalKwargs":["title"]},
                {"obj":"self","fn":"updateHclustColor","objKey":"middle","objName":"mainFrames","requiredKwargs":["colorData","colorGroupData","cmap"],"optionalKwargs":["title"]},
                #{"obj":"self","fn":"setCategoryIndexMatch","objKey":"middle","objName":"mainFrames","requiredKwargs":["categoryIndexMatch"],"optionalKwargs":["categoryEncoded"]},
                #sendMessageProps
            ]           
        },  
    "stats::kmeansElbow":
        {
            "threadRequest":{"obj":"statCenter","fn":"runKMeansElbowMethod","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    addDataAndRefresh
        },
    "stats::runHDBSCAN":
        {
            "threadRequest":{"obj":"statCenter","fn":"runHDBSCAN","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    refreshColumnView 
        },
     "stats::multipleTesting":
        {
            "threadRequest":{"obj":"statCenter","fn":"runMultipleTestingCorrection","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    refreshColumnView 
        },   
    "stats::runKMeans":
        {
            "threadRequest":{"obj":"statCenter","fn":"runKMeans","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    refreshColumnView 
        },

    "stats::rowCorrelation":
        {
            "threadRequest":{"obj":"statCenter","fn":"runRowCorrelation","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    addDataAndRefresh
        },
    "stats::featureSelection":
        {
            "threadRequest":{"obj":"statCenter","fn":"runFeatureSelection","requiredKwargs":["dataID","columnNames","grouping","groupFactors"]},
            "completedRequest":
                    addDataAndRefresh
        },
    "grouping::exclusivePositives":
        {
            "threadRequest":{"obj":"grouping","fn":"getPositiveExclusives","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    refreshColumnView
        },
    "grouping::exclusiveNegative":
        {
            "threadRequest":{"obj":"grouping","fn":"getNegativeExclusives","requiredKwargs":["dataID","columnNames"]},
            "completedRequest":
                    refreshColumnView
        },
    "grouping::deleteGrouping":
        {
            "threadRequest":{"obj":"grouping","fn":"deleteGrouping","requiredKwargs":["groupingName"]},
            "completedRequest":
                    [updateGrouping,sendMessageProps]
        },
    "grouping::renameGrouping":
        {
            "threadRequest":{"obj":"grouping","fn":"renameGrouping","requiredKwargs":["groupingName"]},
            "completedRequest":
                    [updateGrouping,sendMessageProps]
        },
    "webApp::getChartData":
        {
            "threadRequest":{"obj":"webAppComm","fn":"getChartData","requiredKwargs":["graphID"]},
            "completedRequest":
                    addDataAndRefresh
        },
    "session::load":
        {
            "threadRequest":{"obj":"sessionManager","fn":"openSession","requiredKwargs":["sessionPath"]},
            "completedRequest":
                    [{"obj":"self","fn":"updateDataFrames","objKey":"data","objName":"mainFrames","requiredKwargs":["dfs","sessionIsBeeingLoaded"],"optionalKwargs":["dataComboboxIndex"]}] + \
                    [{"obj":"self","fn":"updateReceiverBoxItemsSilently","objKey":"middle","objName":"mainFrames","requiredKwargs":["receiverBoxItems"]}] + \
                    [{"obj":"grouping","fn":"setGroupinsFromSavedSesssion","requiredKwargs":["groupingState"]}]    + \
                    [updateTreeView] + \
                    [{"obj":"self","fn":"openMainFiguresForSession","objKey":"right","objName":"mainFrames","requiredKwargs":["mainFigures","mainFigureRegistry","mainFigureComboSettings"]}] + \
                    [{"obj":"self","fn":"restoreGraph","objKey":"middle","objName":"mainFrames","requiredKwargs":["graphData","plotType"]}] + \
                    [{"obj":"self","fn":"addTooltip","objKey":"middle","objName":"mainFrames","requiredKwargs":["tooltipColumnNames","dataID"]}]
        },
    
    }

#plotter:getHclustColorGroup