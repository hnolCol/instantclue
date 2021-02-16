import xlsxwriter
import numpy as np 
import pandas as pd

from matplotlib.colors import to_hex

baseNumFormat =   {'align': 'center',
                    'valign': 'vcenter',
                    'border':1}

class ICHClustExporter(object):
    ""
    def __init__(self,pathToExcel,clusteredData,columnHeaders,colorArray,totalRows,extraData,clusterLabels,clusterColors):
        ""
        self.pathToExcel = pathToExcel
        self.clusteredData = clusteredData
        self.columnHeaders = columnHeaders
        self.colorArray = colorArray
        self.totalRows = totalRows
        self.extraData = extraData
        self.clusterLabels = clusterLabels
        self.clusterColors = clusterColors

    def export(self):

        try:
       
            self.clusteredData = self.clusteredData.iloc[::-1]
            self.extraData = self.extraData.iloc[::-1]
            self.columnHeaders.extend(["IC Cluster Index","IC Data Index"])
            if self.clusterLabels is not None:
                self.clusterLabels = self.clusterLabels.loc[self.clusteredData.index]
            #reshape color array to fit.
            self.colorArray = self.colorArray.reshape(self.clusteredData.index.size,-1,4)
           
            workbook = xlsxwriter.Workbook(self.pathToExcel, {'constant_memory': True, "nan_inf_to_errors":True} )
            worksheet = workbook.add_worksheet()
            #start with headers
            writeRow = 0
            for n,columnHeader in enumerate(self.columnHeaders):
                worksheet.write_string(writeRow,n,columnHeader)
            
            for nRow in range(self.totalRows):
                for nCol in range(len(self.columnHeaders)):
                    
                    if self.columnHeaders[nCol] == "Cluster ID" and self.clusterLabels is not None:
                        #find cluster ID format
                        clustID = self.clusterLabels.iloc[nRow].values[0]
                        formatDict = baseNumFormat.copy()
                        formatDict["bg_color"] = self.clusterColors[clustID] 
                        cell_format = workbook.add_format(formatDict)
                        worksheet.write_string(nRow+1,nCol,clustID,cell_format)

                    elif nCol < self.clusteredData.columns.size + 1:#indlucate all columns (first one is alsoway "Cluster ID") 
                        c = self.colorArray[self.totalRows-nRow-1,nCol - 1].tolist()
                        formatDict = baseNumFormat.copy()
                        formatDict["bg_color"] = to_hex(c)
                        cell_format = workbook.add_format(formatDict)
                        worksheet.write_number(nRow + 1 ,nCol,self.clusteredData.iloc[nRow,nCol - 1], cell_format) #-1 to account for first Cluster ID column
                    elif self.columnHeaders[nCol] == "IC Data Index":
                        worksheet.write_number(nRow+1,nCol, self.clusteredData.index.values[nRow])
                    elif self.columnHeaders[nCol] == "IC Cluster Index":
                        worksheet.write_number(nRow+1,nCol,nRow)
                    else:
                        dtype = self.extraData[self.columnHeaders[nCol]].dtype
                        if dtype == np.float64 or dtype == np.int64:
                            worksheet.write_number(nRow+1,nCol,self.extraData[self.columnHeaders[nCol]].iloc[nRow])
                        else:
                            worksheet.write_string(nRow+1,nCol,str(self.extraData[self.columnHeaders[nCol]].iloc[nRow]))

            
            workbook.close()
        except Exception as e:
            print("error here")
            print(e)
                    


        #totalRows = int(colorArray.shape[0]/len(self.numericColumns))
        #write data row by row (needed due to "constant_memory" : True)



# 		totalRows = int(colorArray.shape[0]/len(self.numericColumns))









# pathSave = tf.asksaveasfilename(initialdir=path_file,
#                                         title="Choose File",
#                                         filetypes = (("Excel files","*.xlsx"),),
#                                         defaultextension = '.xlsx',
#                                         initialfile='hClust_export')
# 		if pathSave == '' or pathSave is None:
# 			return
       
# 		selectableColumns = self.dfClass.get_columns_of_df_by_id(self.dataID)
# 		columnsNotUsed = [col for col in selectableColumns if col not in self.df.columns]
# 		selection = []
# 		if len(columnsNotUsed) != 0:
# 			dialog = simpleListboxSelection('Select column to add from the source file',
#          		data = columnsNotUsed)   		
# 			selection = dialog.selection
		
# 		workbook = xlsxwriter.Workbook(pathSave)
# 		worksheet = workbook.add_worksheet()
# 		nColor = 0
# 		currColumn = 0
# 		colorSave = {}
# 		clustRow = 0
		
# 		progBar = Progressbar(title='Excel export')
		
# 		colorsCluster = sns.color_palette(self.cmapRowDendrogram,self.uniqueCluster.size)[::-1]
# 		countClust_r = self.countsClust[::-1]
# 		uniqueClust_r = self.uniqueCluster[::-1]
# 		progBar.update_progressbar_and_label(10,'Writing clusters ..')
# 		for clustId, clustSize in enumerate(countClust_r):
# 			for n in range(clustSize):
# 				cell_format = workbook.add_format() 
# 				cell_format.set_bg_color(col_c(colorsCluster[clustId]))
# 				worksheet.write_string(clustRow + 1,
# 					0,'Cluster_{}'.format(uniqueClust_r[clustId]), 
# 					cell_format)
# 				clustRow += 1
		
# 		progBar.update_progressbar_and_label(20,'Writing column headers ..')
		
# 		for n ,colHead in enumerate(['Clust_#'] +\
# 			 self.numericColumns + self.colorData.columns.tolist()  + \
# 			 ['Cluster Index','Data Index'] +\
# 			 self.labelColumnList + selection):
			 
# 			worksheet.write_string(0, n, colHead)		 
		
# 		colorArray = self.colorMesh.get_facecolors()#np.flip(,axis=0)	
# 		totalRows = int(colorArray.shape[0]/len(self.numericColumns))
# 		progBar.update_progressbar_and_label(22,'Writing cluster map data ..')

# 		for nRow in range(totalRows):
# 			for nCol in range(len(self.numericColumns)):
# 				c = colorArray[nColor].tolist()
# 				if str(c) not in colorSave:
# 					colorSave[str(c)] = col_c(c)
# 				cell_format = workbook.add_format({'align': 'center',
#                                      			   'valign': 'vcenter',
#                                      			   'border':1,
#                                      			   'bg_color':colorSave[str(c)]}) 
# 				worksheet.write_number(totalRows - nRow ,nCol + 1,self.df.iloc[nRow,nCol], cell_format)
# 				nColor += 1
				
# 		worksheet.set_column(1,len(self.numericColumns),3)
# 		worksheet.freeze_panes(1, 0)
# 		progBar.update_progressbar_and_label(37,'Writing color data ..')

# 		if len(self.colorData.columns) != 0:
# 			currColumn = nCol + 1
# 			colorFac_r = dict((v,k) for k,v in self.factorDict.items())
# 			colorArray = self.colorDataMesh.get_facecolors()
# 			nColor = 0		
# 			totalRows = int(colorArray.shape[0]/len(self.colorData.columns))	
# 			for nRow in range(totalRows):
# 				for nCol in range(len(self.colorData.columns)):
# 					c = colorArray[nColor].tolist()
# 					if str(c) not in colorSave:
# 						colorSave[str(c)] = col_c(c)
						
# 					cellInt = self.colorData.iloc[nRow,nCol]
# 					cellStr = str(colorFac_r[cellInt])
					
# 					cell_format = workbook.add_format({
#                                      			   'border':1,
#                                      			   'bg_color':colorSave[str(c)]}) 
                                     			   
# 					worksheet.write_string(totalRows - nRow, nCol + 1 + currColumn , cellStr, cell_format)
# 					nColor += 1
					
# 		currColumn = nCol + 1 + currColumn	
						
# 		for n,idx in enumerate(np.flip(self.df.index,axis=0)):
# 			worksheet.write_number(n+1,currColumn+1,n+1)
# 			worksheet.write_number(n+1,currColumn+2,idx + 1)	
			
# 		progBar.update_progressbar_and_label(66,'Writing label data ..')
# 		if len(self.labelColumnList) != 0:
# 			for nRow, labelStr in enumerate(self.labelColumn):
# 				worksheet.write_string(totalRows-nRow,currColumn+3,str(labelStr))
			
# 		progBar.update_progressbar_and_label(77,'Writing additional data ..')
		
# 		df = self.dfClass.join_missing_columns_to_other_df(self.df, self.dataID, definedColumnsList = selection) 
# 		df = df[selection]
# 		dataTypes = dict([(col,df[col].dtype) for col in selection])
# 		if len(selection) != 0:
# 			for nRow in range(totalRows):
# 				data = df.iloc[nRow,:].values
# 				for nCol in range(len(selection)):
# 					cellContent = data[nCol]
# 					if dataTypes[selection[nCol]] == 'object':
# 						worksheet.write_string(totalRows-nRow, currColumn+3+nCol,str(cellContent))
# 					else:
# 						try:
# 							worksheet.write_number(totalRows-nRow, currColumn+3+nCol,cellContent)
# 						except:
# 							#ignoring nans
# 							pass

# 		workbook.close()
# 		progBar.update_progressbar_and_label(100,'Done..')
# 		progBar.close()
	