


import pandas as pd 
import numpy as np 
import re
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

lastChange = "20211024"

def buildRegex(categoriesList, withSeparator = True, splitString = None):
		'''
		Build regular expression that will search for the selected category. Importantly it will prevent 
		cross findings with equal substring
		=====
		Input:
			List of strings (categories) - userinput
        Returns:
            Regular expression that can be used. 
		====

		'''
		regExp = r'' #init reg ex
		for category in categoriesList:
			category = re.escape(category) #escapes all special characters
			if withSeparator and splitString is not None:
				regExp = regExp + r'({}{})|(^{}$)|({}{}$)|'.format(category,splitString,category,splitString,category)
			else:
				regExp = regExp + r'({})|'.format(category)
				
		regExp = regExp[:-1] #strip of last |
		return regExp


class PTMProteinNormalizer(object):
    
    def __init__(self, 
                ptmData, 
                proteinData, 
                itensityColumns = [], #can be list or dict
                identifierColumnPTM = "UniprotID",
                identifierColumnProtein = "UniprotID",
                preferExactMatches = True,
                splitString = ";",
                minNonNaNForMultipleMatches = 2,
                normalizeProteinDataByMean = True,
                *args,**kwargs):
        """
        Class to normalize PTM (peptide) data by proteme 
        data (e.g. intensity value of proteins).

        Challenge
        ==============
        
                A PTM can be detected on peptides that are
        present in multiple proteins. Hence, the 
        multiple-protein values have to be combined. 
        The metric defaults to 'mean'. 
        Additionally, it is not clear what do to
        when a protein group is made by proteins that
        do not contain the p-peptide.    


        Parameter
        ==============

        ptmData(pd.DataFrame)  : ptm peptide data

        proteinData(pd.DataFrame)  : protein data

        identifierColumnPTM(str) : identifier column in ptmData (e.g. Uniprot Identifier)

        dentifierColumnProtein(str) : identifier column in proteinData (e.g. Uniprot Identifier)

        preferExactMatches(bool) : if enabled exact matches (e.g. id of ptm and id of protein). Other protein groups that might match
                                   are ingored. Example ptmID = abc2 , proteinIDs [abc2,abc2;abc8]. The protein intensity will only be calculated
                                   form the first match (abc2 == abc2). This is recommended, and setting this to false has no effect at the moment.

        minNonNaNForMultipleMatches(int) : Number of non NaN Values required if multiple protein groups match ptm peptide

        normalizeProteinDataByMean(bool) : If enabled, protein data will be normalized (subtracted) by mean value across all experiments. 
                                           Hence the ptm-data will be normalize using the deviation from the mean rather the actual intensity.
                                           This will retain the abundance information compared to other p-peptides

        """
        self.checked = False
        self.ptmData = ptmData
        self.proteinData = proteinData
        self.intensityColumns = itensityColumns
        self.identifierColumnPTM = identifierColumnPTM
        self.identifierColumnProtein = identifierColumnProtein
        self.splitString = splitString
        self.minNonNaNForMultipleMatches = minNonNaNForMultipleMatches
        self.preferExactMatches = preferExactMatches
        self.normalizeProteinDataByMean = normalizeProteinDataByMean
        self.checked = self._checkInput() 

    def _checkInput(self):

        if not all(isinstance(x,pd.DataFrame) for x in [self.ptmData, self.proteinData]):
            raise TypeError("'ptmData' and 'proteinData' must be of type pd.DataFrame.")

        if self.identifierColumnPTM not in self.ptmData.columns:
            raise ValueError("'identifierColumnPTM' not in ptmData")

        if self.identifierColumnProtein not in self.proteinData.columns:
            raise ValueError("'identifierColumnProtein' not in proteinData")
        if isinstance(self.intensityColumns,list):
            if len(self.intensityColumns) == 0:
                print("Warning :: no intensity column provided. All columns in ptmData except the identifier will be considered intensity column.")
                self.intensityColumns = [colName for colName in self.ptmData.columns if colName != self.identifierColumnPTM]
                print("Detected columns:")
                print(self.intensityColumns)

            if not all(colName in self.proteinData.columns for colName in self.intensityColumns):
                raise ValueError("Not all intensity columns detected in proteinData. ")
        elif isinstance(self.intensityColumns,dict):
            if len(self.intensityColumns) != 2:
                raise ValueError("If intensityCoumns is dict, it must be of length 2 and the keys must be 'ptm' and 'protein'")
            if not all(k in self.intensityColumns for k in ["ptm","protein"]):
                raise ValueError("If intensityColumns is dict, it must contain ptm and protein as keys.")

        if not self.preferExactMatches:
            print("Warning :: setting preferExactMatches to False is currently not supported. Will be set to True")
            self.preferExactMatches = True

        return True 

    def transform(self):
        "Returns transformed/normalized PTM data"

        #first clear off the exact matches (e.g exact identifier of ptm found in protein data)
        self.ptmData = self.ptmData.set_index(self.identifierColumnPTM, drop=False)
        self.proteinData = self.proteinData.set_index(self.identifierColumnProtein) 
        protIntensityColumns = self.intensityColumns["protein"] if isinstance(self.intensityColumns,dict) else self.intensityColumns
        ptmIntensityColumns = self.intensityColumns["protein"] if isinstance(self.intensityColumns,dict) else self.intensityColumns
        firstPass = self.ptmData.join(self.proteinData[protIntensityColumns],how="inner",rsuffix="_prot",lsuffix="_ptm")
        firstPass["protein.groups.for.norm"] = 1
        firstPass["match.found"] = True
        firstPass["filtered.by.nan.thresh"] = False
        firstPass["ID_ptm"] = firstPass.index
        firstPass["ID_prot"] = firstPass.index
        
        matchedCasesBoolIdx = self.ptmData.index.isin(firstPass.index)
        print("Info :: Number of exact matched cases",np.sum(matchedCasesBoolIdx))

        #reset first passt index
        firstPass = firstPass.reset_index(drop=True)
        #reset index to iterate through ptms one by one.
        self.ptmData = self.ptmData.reset_index(drop=True)

        diffMatches = []
        #create column names matching the pd Data Frame first pass (suffxied added due to joining)
        protIntensityIndexedColumns = ["{}_prot".format(colName) for colName in protIntensityColumns]
        ptmIntensityIndexedColumns = ["{}_ptm".format(colName) for colName in ptmIntensityColumns]
        
        for idx in self.ptmData.loc[~matchedCasesBoolIdx].index:
            proteinID  = self.ptmData.loc[idx,self.identifierColumnPTM]
            r = pd.Series(index = firstPass.columns.values, name=idx, dtype="object")
            r.loc["filtered.by.nan.thresh"] = False
            r.loc["match.found"] = True
            r.loc["ID_ptm"] = proteinID
           
            r.loc[ptmIntensityIndexedColumns] = self.ptmData.loc[idx,ptmIntensityColumns].values.flatten()
            
            if self.splitString in proteinID:
                ids = proteinID.split(self.splitString)
                regEx = buildRegex(ids,True,self.splitString)
                boolIdx = self.proteinData.index.str.contains(regEx)

            else:
                boolIdx = self.proteinData.index.str.contains(proteinID)

            if np.any(boolIdx):
                r.loc["ID_prot"] = "_".join(self.proteinData.index[boolIdx])
                r = self._aggreateIntensityData(self.proteinData,boolIdx,r,protIntensityColumns,protIntensityIndexedColumns)
                
            else:
                r.loc["ID_prot"] = ""
                r.loc[protIntensityIndexedColumns] = np.nan
                r.loc["match.found"] = False
                r.loc["protein.groups.for.norm"] = 0

            remainingColumns = [colName for colName in self.ptmData.columns if colName not in ptmIntensityColumns]
            r.loc[remainingColumns] = self.ptmData.loc[idx,remainingColumns].values.flatten()
            
            diffMatches.append(r) 

        Y = firstPass.append(diffMatches)
        proteinNormPTM = ["ProteinNormPTM_{}".format(colName.replace("_ptm","")) for colName in ptmIntensityIndexedColumns]
        if self.normalizeProteinDataByMean:
            ##normalize protein intensities by mean, this will retain the intensity information of the p-peptide
            normalizedColumns = ["meanNorm_{}".format(colName) for colName in protIntensityIndexedColumns]
            Y.loc[:,normalizedColumns] = Y.loc[:,protIntensityIndexedColumns].values - Y.loc[:,protIntensityIndexedColumns].mean(axis=1).values.reshape(-1,1)
            Y.loc[:,proteinNormPTM] = Y[ptmIntensityIndexedColumns].values - Y[normalizedColumns].values
        else:
            Y.loc[:,proteinNormPTM] = Y[ptmIntensityIndexedColumns].values - Y[protIntensityIndexedColumns].values
        return Y


    def _aggreateIntensityData(self,X,boolIdx,r,intensityColumns,indexedIntensityColumn):
        ""
        proteinValuesFound = True
        nMatches = np.sum(boolIdx)
        if self.minNonNaNForMultipleMatches > 0 and nMatches > 1:
            filteredX = X.loc[boolIdx,intensityColumns].dropna(thresh=self.minNonNaNForMultipleMatches)
            if filteredX.empty:
                r.loc["filtered.by.nan.thresh"]  = True
                r.loc["protein.groups.for.norm"] = 0
                r.loc[indexedIntensityColumn] = np.nan
                proteinValuesFound = False
                            
            else:         
                X = filteredX.mean().values.flatten()   
        elif nMatches == 1:
            X = self.proteinData.loc[boolIdx,intensityColumns].values.flatten()
        else:
            X = self.proteinData.loc[boolIdx,intensityColumns].mean().values.flatten()
        #add data to results
        if proteinValuesFound:
            r[indexedIntensityColumn] = X
            r.loc["protein.groups.for.norm"] = nMatches

        return r



if __name__ == "__main__":
    #load example data
    ptm= pd.read_csv("phosphoFile.txt", sep="\t")#
    protein= pd.read_csv("proteinFile.txt", sep="\t")
    #init normalizer 
    #be aware that with real data you will have to specify the columns that should be used 
    #they must be in the correct order (experiment wise)
    columns = ptm.columns[0:23]
    intensityColumns = {"ptm":columns.values.tolist(),
                        "protein":columns.values.tolist()}
    print(intensityColumns)           
    n = PTMProteinNormalizer(
                    ptm,
                    protein,
                    itensityColumns = intensityColumns,
                    identifierColumnProtein="Protein.Group",
                    identifierColumnPTM="Protein.Group")

    transformedPTMS = n.transform()
    print(transformedPTMS)
    transformedPTMS.to_csv("r.txt",sep="\t")




