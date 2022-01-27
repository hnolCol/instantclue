import pandas as pd
import numpy as np
import re

from .utils import fasta_iter

class ICModPeptidePositionFinder(object):
    ""
    def __init__(self,sourceData,dataID,*args,**kwargs):
        ""
        self.dataID = dataID
        self.sourceData = sourceData

        self.regEx = "\|(.*)\|"   

    def _replace_all(self,text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    def _cleanSequence(self,modString,modPeptideSequences,allMods):
        ""
        replaceDict = dict([(modS,"") for modS in allMods if modS != modString])
        return [self._replace_all(modSequence,replaceDict) for modSequence in modPeptideSequences]

    def loadFasta(self,fastaFile):
        ""
        self.sequenceByID = dict() 
        for ff in fasta_iter(fastaFile):
            headerStr, seq = ff
            match = re.search(self.regEx, headerStr)
            self.sequenceByID[match.group(1)] = seq 

    def findPeptidePostionByID(self,proteinID,peptideSequence):
        ""

        if proteinID in self.sequenceByID:
            peptideLength = len(peptideSequence)
            idxStart = self.sequenceByID[proteinID].find(peptideSequence)
            idxEnd = idxStart+peptideLength
            return idxStart,idxEnd
        else:
            return None, None
    def _joinStrings(self,l):
        ""
        return ";".join([str(x) for x in l])

    def findPeptideAndModPosition(self,proteinID,modPeptideSequence,modString,idx): #"(UniMod:21)"
        if hasattr(self,"sequenceByID"):
            outputIndex = ["Amino acid {}".format(modString),"Amino acid position {} [peptide]".format(modString),"Amino acid position {} [protein]".format(modString)]
            strippedSequence = re.sub(r"\([^()]*\)", "", modPeptideSequence)
            if ";" in proteinID:
                proteinID = proteinID.split(";")[0]
            if proteinID in self.sequenceByID:
                idxStart, idxEnd = self.findPeptidePostionByID(proteinID,strippedSequence)
                inProteinStart = str(idxStart)
                inProteinEnd = str(idxEnd)
                print(idxStart,idxEnd)
                if idxStart is not None and idxEnd is not None:
                    if modString in modPeptideSequence:
                        modPositions = [(m.start(), m.end()) for m in re.finditer(modString, modPeptideSequence)]
                        aaModified = [modPeptideSequence[modIdxStart-2] for modIdxStart, _ in modPositions]
                        aaPositionInPeptide = [modIdxStart-1-(len(modString)*n) for n,(modIdxStart, _) in enumerate(modPositions)]
                        aaPositionInProtein = [idxStart + modAAPosInPeptide  for modAAPosInPeptide in aaPositionInPeptide ]
                        
                        aasMod = self._joinStrings(aaModified)
                        aaModProt = self._joinStrings(aaPositionInProtein)
                        aaModPept = self._joinStrings(aaPositionInPeptide)
                        return pd.Series([aasMod,aaModPept,aaModProt], index=outputIndex, name=idx)
        print("R?")
        print(hasattr(self,"sequenceByID"))
        print(proteinID in self.sequenceByID)
        print("==")
        return pd.Series([""]*len(outputIndex), index=outputIndex, name = idx)

    
    def matchModPeptides(self,proteinIDs,modPeptideSequences,index, ignoredMods = ["(UniMod:4)"]):
        ""
        allModifications = pd.Series(np.hstack([re.findall('\(.*?\)',modPeptideSequence) for modPeptideSequence in modPeptideSequences])).unique()
        
        modSpecSequences = dict([(modString,self._cleanSequence(modString,modPeptideSequences,allModifications)) for modString in allModifications if modString not in ignoredMods])        
        data = []
        for modString in modSpecSequences.keys():
            X = [self.findPeptideAndModPosition(proteinID,modPeptideSequence,modString,index[n]) for n, (proteinID, modPeptideSequence) in enumerate(zip(proteinIDs,modSpecSequences[modString]))]
            #print(X)
            XX = pd.concat(X,axis=1).T
            data.append(XX)
        if len(data) > 1:
            return pd.concat(data,axis=1)
        else:
            return data[0]

if __name__ == "__main__":
    data = pd.read_csv("wojtekData.txt", sep="\t")
    seqs = data["Modified.Sequence"]
    uniprots = data["Protein.Group"]
    index = data.index.values.tolist() 
    modFinder = ICModPeptidePositionFinder(None,None)
    modFinder.loadFasta("CEEL.fasta")
    d = modFinder.matchModPeptides(uniprots,seqs,index)#MAS(UniMod:21)RKTVNRRQRP(UniMod:21)QR
    data = data.join(d)
    print(data)
    data.to_csv("wojtekDataMod.txt",sep="\t")