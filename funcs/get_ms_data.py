import itertools
import os
import sys
import numpy as np
def running_mean(x,N):
     cumsum=np.cumsum(np.insert(x,0,0))
     return (cumsum[N:] - cumsum[:-N])/N
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
def find_nearest_index(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
def find_isotopes(mass, intens, thres):
     mass_list_with_isotopes = []
     intens_list_with_isotopes = [] 
     charge_list_big = []
     for i in range(len(mass)):
         for n in range(5):
                 #try:
                     try:
                         distance_to_next_peak =mass[i+n] - mass[i]
                     except:
                         pass#nr isotopes
                     for charge in range(1,6,1):
                         offset = 1/charge
                         if  offset - thres <= distance_to_next_peak <= offset + thres:

                             isotopes_list = []
                             intens_list = []
                             charge_list = []

                             for isotopes in range(6):
                                 try:
                                     distance_to_next_peak = mass[i+isotopes] - mass[i]
                                 except:
                                     pass
                                 if (offset*isotopes)-thres <= distance_to_next_peak <= (offset*isotopes)+thres:
                                 
                                     isotopes_list.append(mass[i+isotopes])
                                     intens_list.append(intens[i+isotopes])
                                     charge_list.append([mass[i+isotopes],offset]) 
                                   


                             mass_list_with_isotopes.append(isotopes_list)
                             intens_list_with_isotopes.append(intens_list)
                             charge_list_big.append(charge_list)
     return mass_list_with_isotopes, intens_list_with_isotopes, charge_list_big

def filter_isotopes_list(mass_list_with_isotopes):
     index = []
     for i in range(len(mass_list_with_isotopes)):
         try:
             list_to_check = mass_list_with_isotopes[i]
             if len(list_to_check) > 0:
                 for n in range(i,len(mass_list_with_isotopes)):
                     try:
                          list_to_compare = mass_list_with_isotopes[n+1]
                          if len([m for m in list_to_check if m in list_to_compare]) > 0:
                              index.append(n+1)
                     except:
                          pass
         except:
             pass
     return index

def deconvolute_msms_spectra(mass_list_extracted, thres = 0.025):

     mass, intens = extract_from_MsFileReader_tuple(mass_list_extracted,True)

     mass_list_with_isotopes, intens_list_with_isotopes, charge_list_big = find_isotopes(mass, intens, thres)

     index = filter_isotopes_list(mass_list_with_isotopes)
     mass_list_with_isotopes = [x for x in mass_list_with_isotopes if mass_list_with_isotopes.index(x) not in index]
     charge_list = [x for x in charge_list_big if charge_list_big.index(x) not in index]
     intens_list_with_isotopes = [x for x in intens_list_with_isotopes if intens_list_with_isotopes.index(x) not in index]

     masses_w_iso = list(itertools.chain.from_iterable(mass_list_with_isotopes))
     intens_w_iso = list(itertools.chain.from_iterable(intens_list_with_isotopes))


     no_iso_found_masses = [i for i in mass if i not in masses_w_iso]
     no_iso_found_intens = [i for i in intens if i not in intens_w_iso]

     mass_first = [i[0] for i in mass_list_with_isotopes]
     charge_first = [i[0] for i in charge_list]
     l1 = list(itertools.chain.from_iterable(charge_first))
     adjusted_mass = [] 
     for i in mass_first:
          offset = l1[l1.index(i)+1]
     
          adjusted_mass.append(float(i)*(1/offset))

     intens_summed = [sum(i) for i in intens_list_with_isotopes]

     complete_spec_deconv = [intens_summed + no_iso_found_intens, adjusted_mass + no_iso_found_masses]

     return complete_spec_deconv

def extract_mass_errors(scanNumberList, rawFileList, msms_data, folder_path):
     grouped_msms_data = msms_data.groupby('Raw file')
     matches, masses, errors = [], [], [] 
     for ScanNum in scanNumberList:
          try:
               ScanNum = int(ScanNum)
          except:
               messagebox.showinfo('Error...','No Scan Number in evidence file found...')
          rawFile = rawFileList[scanNumberList.index(ScanNum)]
          msms_data = grouped_msms_data.get_group(rawFile[:-4])
          msms_data = msms_data.loc[msms_data['Scan number'] == ScanNum]
          matches.append(list(msms_data['Matches'].str.split(";")))
          masses.append(list(msms_data['Masses'].str.split(";")))
          errors.append(list(msms_data['Mass Deviations [ppm]'].str.split(";"))) 
     return matches, masses, errors 
          
         
          
def get_ms2_spectra_over_scanNumbers(scanNumberList,rawFileList,msms_data, folder_path):

     MS_spectra_label_data, masses, BasePeakIntens, peptide_charge, mass_list, intens_mz_MS2, matches, mono_isotp_mass, intens, LowMass, HighMass, deconv = [],[],[] ,[],[],[],[], [], [], [], [], [] 
     grouped_msms_data = msms_data.groupby('Raw file')
     
     for ScanNum in scanNumberList:
          ScanNum = int(ScanNum)
          rawFile = rawFileList[scanNumberList.index(ScanNum)]
          msms_data = grouped_msms_data.get_group(rawFile[:-4])
          msms_data = msms_data.loc[msms_data['Scan number'] == ScanNum]
         
          with cd(folder_path):
               rawfile = ThermoRawfile(rawFile)          
               order = rawfile.GetMSOrderForScanNum(ScanNum)
          
               name = rawfile.GetFileName()
           
               
               
               output = rawfile.GetTrailerExtraForScanNum(ScanNum)
               mass_list_extracted = rawfile.GetMassListFromScanNum(ScanNum, centroidResult=True, centroidPeakWidth=0.2) ### TO DO: make this editable 
               
               ScanHeaderInfo = rawfile.GetScanHeaderInfoForScanNum(ScanNum)

               complete_spec_deconv = deconvolute_msms_spectra(mass_list_extracted, thres = 0.04)
               deconv.append(complete_spec_deconv)
          names_properties,data_properties = extract_from_MsFileReader_tuple(output,False) 
          msms_mz = data_properties[names_properties.index('Monoisotopic M/Z')]
          matches.append(list(msms_data['Matches'].str.split(";")))
          intens.append(list(msms_data['Intensities'].str.split(";")))
          masses.append(list(msms_data['Masses'].str.split(";")))
          mono_isotp_mass.append(msms_mz)

          mass_list_scan, intens_mz_MS2_scan = extract_from_MsFileReader_tuple(mass_list_extracted,True)
          mass_list.append(mass_list_scan)
          intens_mz_MS2.append(intens_mz_MS2_scan)
              
          ScanHeaderInfo_ScanNumber = extract_from_MsFileReader_tuple(ScanHeaderInfo,False)

          LowMass_scan , HighMass_scan = ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('LowMass')], ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('HighMass')]
          LowMass.append(LowMass_scan)
          HighMass.append(HighMass_scan) 
          BasePeakIntens_scan = ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('BasePeakIntensity')]
          BasePeakMass = ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('BasePeakMass')]

          peptide_seq, peptide_score,peptide_mass,peptide_charge_scan = str(msms_data['Sequence'].iloc[0]),str(msms_data['Score'].iloc[0]),str(msms_data['Mass'].iloc[0]),str(msms_data['Charge'].iloc[0])
          MS_spectra_label_data_string = '\nMass error [ppm]: '+str(msms_data['Mass Error [ppm]'].iloc[0])+'\nPeptide mass: '+str(peptide_mass)+'\nMS/MS mz: '+ str(msms_data['m/z'].iloc[0])+'\nPrecursor: '+str(msms_mz)+'\nScore: '+str(peptide_score)
          MS_spectra_label_data.append(MS_spectra_label_data_string)
          peptide_charge.append(peptide_charge_scan)
          BasePeakIntens.append(BasePeakIntens_scan)
          rawfile.Close() 
     return MS_spectra_label_data, masses, matches, intens, BasePeakIntens, peptide_charge, mass_list, intens_mz_MS2, mono_isotp_mass, LowMass, HighMass, deconv

def get_information_over_time_from_raw_file(rawfile, MSOrder,end_time):
     
     
     end_scan_num = rawfile.ScanNumFromRT(end_time)

     output_list = []
     rawfile_name = rawfile.GetFileName()
     for ScanNum in range(1,end_scan_num):
          if rawfile.GetMSOrderForScanNum(ScanNum) == MSOrder:
              
              ScanHeaderInfo = rawfile.GetScanHeaderInfoForScanNum(ScanNum)
              ScanHeaderInfo_ScanNumber = extract_from_MsFileReader_tuple(ScanHeaderInfo,False)
              
              output = rawfile.GetTrailerExtraForScanNum(ScanNum)
              names_properties,data_properties = extract_from_MsFileReader_tuple(output,False)
              output_list.append(data_properties+ScanHeaderInfo_ScanNumber[1])      
              
          else:
              pass
     
     return names_properties + ScanHeaderInfo_ScanNumber[0], output_list 



def get_tic_both_levels(rawfile,cutoff = 2):
     TIC_MS1 = []
     TIC_MS2 = []
     RT_MS1 = []
     RT_MS2 = []
     num_of_drops = 0
     end_scan_num = rawfile.GetNumSpectra()
     for ScanNum in range(1,end_scan_num):
          if rawfile.GetMSOrderForScanNum(ScanNum) == 1:
              
              ScanHeaderInfo = rawfile.GetScanHeaderInfoForScanNum(ScanNum)
              ScanHeaderInfo_ScanNumber = extract_from_MsFileReader_tuple(ScanHeaderInfo,False)
              tic = ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('TIC')]

              if ScanNum > 1:
                   if (TIC_MS1[-1]/tic > cutoff or TIC_MS1[-1]/tic < 1/cutoff):
                        num_of_drops += 1
              RT_MS1.append(ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('StartTime')])
              TIC_MS1.append(tic)
          else:
              ScanHeaderInfo = rawfile.GetScanHeaderInfoForScanNum(ScanNum)
              ScanHeaderInfo_ScanNumber = extract_from_MsFileReader_tuple(ScanHeaderInfo,False)
              TIC_MS2.append(ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('TIC')])
              RT_MS2.append(ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('StartTime')])
     return TIC_MS1,TIC_MS2,RT_MS1,RT_MS2,num_of_drops 
def getRawFileNames(path):
     allFiles = os.listdir(path)
     rawFiles = [file for file in allFiles if '.raw' in file[-4:] or '.RAW' in file[-4:]]
     
     return rawFiles

def get_mass_low_and_high_MS1(rawfile):
    for ScanNum in range(1,50): 
          if rawfile.GetMSOrderForScanNum(ScanNum) == 1:
              
              ScanHeaderInfo = rawfile.GetScanHeaderInfoForScanNum(ScanNum)
              ScanHeaderInfo_ScanNumber = extract_from_MsFileReader_tuple(ScanHeaderInfo,False)

              return [ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('LowMass')],ScanHeaderInfo_ScanNumber[1][ScanHeaderInfo_ScanNumber[0].index('HighMass')]]

              break
               
     


     
def extract_number_of_MS1_and_MS2_spectra(rawfile, give_time_back=False, give_real_time_back = False):
     start_time = rawfile.GetStartTime()
     end_time = rawfile.GetEndTime()
     end_scan_num = rawfile.GetNumSpectra()
     MS1 = 0
     MS2 = 0
     rt_1 = []
     rt_2 = []
     MS1_list = []
     MS2_list = []
     mono_mass = [] 
     for i in range(1,end_scan_num):
          if rawfile.GetMSOrderForScanNum(i) == 1:
                 MS1 += 1
                 if give_real_time_back:
                      rt_1.append(rawfile.RTFromScanNum(i))
                      MS1_list.append(MS1)
          else:
                 MS2 += 1
                 if give_real_time_back:
                      rt_2.append(rawfile.RTFromScanNum(i))
                      MS2_list.append(MS2) 
     if give_time_back:
          return MS1, MS2, start_time, end_time
     elif give_real_time_back:
          return MS1, MS2, rt_1, rt_2, MS1_list, MS2_list 
     else:
          return MS1, MS2 
def extract_only_real_isotopes(rawfile, param_list, SILAC_offset, dem = False, n_K = 0):

     rt_start = param_list[0]
     rt_end = param_list[1]
     
     charge_state = param_list[3]
     
     mz_value = param_list[2]
     if dem == True:
          
          mz_value += (28.031300/charge_state)*(n_K+1)
     else:
          pass
     
     mz_value += (SILAC_offset/charge_state)
     collect_sum = []
     collect_rt = []
     range_for_extract = 0.03
     if rt_end > rawfile.GetEndTime():
          rt_end = rawfile.GetEndTime()
     if rt_start > rawfile.GetEndTime():
          rt_start = 0.0
     try:     
          ScanNum_start = rawfile.ScanNumFromRT(rt_start)
     except:
          ScanNum_start = 1 
     try:
          ScanNum_end = rawfile.ScanNumFromRT(rt_end)
     except:
          ScanNum_end = rawfile.GetNumSpectra()
     for ScanNum in range(ScanNum_start,ScanNum_end,1):
          
          if rawfile.GetMSOrderForScanNum(ScanNum)== 1:
               try:
                    MS_spectra_data = rawfile.GetMassListRangeFromScanNum(ScanNum,massRange=str(mz_value-3 )+'-'+str(mz_value+8))

               
                    mz_data, intens_data = extract_from_MsFileReader_tuple(MS_spectra_data,True)
                    list_nearest_start = []
                    list_nearest_end = [] 

                    for i in range(0,int(4)):
                         list_nearest_start.append(find_nearest(np.array(mz_data),mz_value+i/charge_state-range_for_extract))
                         list_nearest_end.append(find_nearest(np.array(mz_data),mz_value+i/charge_state+range_for_extract))   
                    sum_fade = 0 
                    for i in range(len(list_nearest_start)):
                         sum_fade = sum_fade + sum(intens_data[mz_data.index(list_nearest_start[i]):mz_data.index(list_nearest_end[i])])
                    collect_sum.append(sum_fade) 
                    
                    collect_rt.append(rawfile.RTFromScanNum(ScanNum))
   
               except:
                    pass
     return [collect_rt, collect_sum]
     
def extract_only_real_isotopes_raw(rawfile, param_list):
     rt_start = param_list[0]
     rt_end = param_list[1]
     
     charge_state = param_list[3]
     mz_value = param_list[2] 
     collect_sum = []
     collect_rt = []
     range_for_extract = 0.03
     try:     
          ScanNum_start = rawfile.ScanNumFromRT(rt_start)
     except:
          ScanNum_start = 1 
     try:
          ScanNum_end = rawfile.ScanNumFromRT(rt_end)
     except:
          ScanNum_end = rawfile.GetNumSpectra()
     
     for ScanNum in range(ScanNum_start,ScanNum_end,1):
          try:
               if str(rawfile.GetMSOrderForScanNum(ScanNum)) == str(param_list[5][-1]):
                    MS_spectra_data = rawfile.GetMassListRangeFromScanNum(ScanNum,massRange=str(mz_value-3 )+'-'+str(mz_value+8))

               
                    mz_data, intens_data = extract_from_MsFileReader_tuple(MS_spectra_data,True)
                    
                    list_nearest_start = []
                    list_nearest_end = [] 

                    for i in range(0,int(param_list[4])):
                         
                         list_nearest_start.append(find_nearest(np.array(mz_data),mz_value+i/charge_state-range_for_extract))
                         list_nearest_end.append(find_nearest(np.array(mz_data),mz_value+i/charge_state+range_for_extract)  )   
                    sum_fade = 0 
                    for i in range(len(list_nearest_start)):
                         sum_fade = sum_fade + sum(intens_data[mz_data.index(list_nearest_start[i]):mz_data.index(list_nearest_end[i])])
                    collect_sum.append(sum_fade) 
                    
                    collect_rt.append(rawfile.RTFromScanNum(ScanNum))
          except:
               pass 
                
     return [collect_rt, collect_sum]

def extract_MassLists_for_MS1_scans(rawfile, massListRange, ScanNumMSMS, type_of_extraction = 'Before MSMS', scan_off = 0):

     count_for_MS1_hits = int() 
     if scan_off > 0:
          type_of_extraction = "After"
     elif scan_off < 0:
         type_of_extraction = "Before MSMS"
     
     if type_of_extraction == "Before MSMS":
          change_log = 1000
          steps = -1
     if type_of_extraction == 'After':
          change_log = -1000
          steps = 1 
     if True:

          if scan_off < 0:
          
                              count_for_MS1_hits = 1

          else:
                              count_for_MS1_hits = 0
                              
          for ScanNum in range(ScanNumMSMS,ScanNumMSMS-change_log,steps):
                    
                    if rawfile.GetMSOrderForScanNum(ScanNum) == 1:
                         
                         if scan_off < 0:
                              count_for_MS1_hits -=1
                         else:
                              count_for_MS1_hits += 1

                         if count_for_MS1_hits == scan_off or scan_off == 0:
                              MS_spectra_data = rawfile.GetMassListRangeFromScanNum(ScanNum, massRange = massListRange ) 
                              mz_data, intens_data = extract_from_MsFileReader_tuple(MS_spectra_data,True)
                              break
     
          
     return mz_data, intens_data


        
def extract_from_MsFileReader_tuple(output,double_tuple):
     """
     This function returns two lists from a tuple , tuple list output that we recieve from
     MS file reader functions.
     Double tuple give gives you the option to extract data structures like
     [((RT,intens),'None')] if True
     and
     [(Property,Value)] if False 
     """
     if double_tuple:
          list1 = list(zip(*output[0]))
          data = list(map(list,zip(*list1)))
     else:
          out_list = list(output.items())
          data = list(map(list, zip(*out_list)))

     return data[0], data[1]
    
def get_3D_peak(rawfile,startTime, endTime, mzStart, mzEnd, iterated=False):
     """
     Returns mz Data, RT data, and intensity data in a given range in three big lists [all data of a certain type
     are combined in one]
     """
     scan_start = rawfile.ScanNumFromRT(startTime)
     scan_end = rawfile.ScanNumFromRT(endTime)
     save_mz_data = []
     save_intens_data = []
     save_RT_data = []
  
     
     for i in range(scan_start,scan_end,1):

          if rawfile.GetMSOrderForScanNum(i) == 1:
                  
               chrom_data = rawfile.GetMassListRangeFromScanNum( i , 
                                                                 intensityCutoffType = 0,
                                                                 intensityCutoffValue = 0 ,
                                                                 massRange = str(mzStart)+'-'+str(mzEnd))#)"{}-{}".format(mzStart, mzEnd))

               try:  ## to prevent an error if first Scan number is actually a MS2 !! 
                    mz, intens = extract_from_MsFileReader_tuple(chrom_data,True)
                    save_RT_data.append( [rawfile.RTFromScanNum(i)] *len(mz))
                    save_mz_data.append(mz)
                    save_intens_data.append(intens) 
               except:
                    pass 
          else:
               pass
     if iterated:
          big_Rt = save_RT_data
          big_mz = save_mz_data
          big_intens = save_intens_data
     else:
          big_Rt = list(itertools.chain(*save_RT_data))
          big_mz = list(itertools.chain(*save_mz_data))
          big_intens = list(itertools.chain(*save_intens_data) )
     rawfile.Close() 
     return [big_Rt, big_intens, big_mz] 


def get_trailer_properties(rawfile, ScanNum):
     
    output = rawfile.GetTrailerExtraForScanNum(ScanNum)
    names_properties,data_properties = extract_from_MsFileReader_tuple(output,False)
    return names_properties,data_properties

def return_trailer_param_and_RT_time_for_ScanNum_Range(rawfile,
                                            Param ='Ion Injection Time (ms)' ,       
                                            ScanNumFirst = 0,
                                            ScanNumLast  = 0,
                                            startTime = 0.0,
                                            endTime = 0.0,
                                            MSOrder = 2,
                                            round_digit_output = 2,
                                                       Param2 = False):
     save_param = []
     save_RT = []
 
     if endTime != 0:
          try:
               scan_start = rawfile.ScanNumFromRT(startTime)
          except:
               scan_start = 1
               
          try:
               scan_end = rawfile.ScanNumFromRT(endTime)
          except:
               scan_end = rawfile.GetNumSpectra()
              
     else:
          scan_start = ScanNumFirst
          scan_end = ScanNumLast
     for i in range(scan_start,scan_end,1):
          if rawfile.GetMSOrderForScanNum(i) == MSOrder:
               output = rawfile.GetTrailerExtraForScanNum(i)
               names_properties,data_properties = extract_from_MsFileReader_tuple(output,False)
               #print(names_properties,data_properties)
               if Param2 == False:
                    save_param.append(round(data_properties[names_properties.index(Param)],2))
               else:
                    save_param.append([round(data_properties[names_properties.index(Param)],2),round(data_properties[names_properties.index(Param2)],2)])
               save_RT.append(rawfile.RTFromScanNum(i)) 
          else:
               pass
     return save_RT, save_param
     
     
     
def get_2D_peak(rawfile, startTime, endTime, mzStart, mzEnd):
     try:
          chrom_data = rawfile.GetChroData(startTime= startTime, endTime= endTime, massRange1="{}-{}".format(mzStart, mzEnd), filter="Full ms ")

          RT_data , intens_data = extract_from_MsFileReader_tuple(chrom_data,True)
          rawfile.Close()
     except:
          return 
     return [RT_data, intens_data]

import time
import logging
log = logging.getLogger(os.path.basename(__file__))
from collections import namedtuple
from collections import OrderedDict
from ctypes import *
import copy
__version__ = "Bindings_MSFileReader 3.0 SP3 (3.0.31.0), Apr 30, 2015"
# XRawfile2(_x64).dll 3.0.29.0
# fregistry(_x64).dll 3.0.0.0
# Fileio(_x64).dll 3.0.0.0

try :
    import comtypes
    from comtypes.client import GetModule, CreateObject
except (ImportError,NameError) as e:
    sys.exit('Please install comtypes >= 0.6.2 : http://pypi.python.org/pypi/comtypes/') 
    
try:
    from comtypes.gen import MSFileReaderLib
except ImportError:
    XRawfile2_dll_loaded = False
    XRawfile2_dll_path = []
    XRawfile2_dll_path.append( os.path.dirname(os.path.abspath(__file__) ) + os.sep + 'XRawfile2_x64.dll' )    # 64bits msFileReader aside raw.py
    XRawfile2_dll_path.append( u'C:\\Program Files\\Thermo\\MSFileReader\\XRawfile2_x64.dll' )    # 64bits msFileReader with 64bits system 
    XRawfile2_dll_path.append( os.path.dirname(os.path.abspath(__file__) ) + os.sep + 'XRawfile2.dll' )    # 32bits msFileReader aside raw.py
    XRawfile2_dll_path.append( u'C:\\Program Files (x86)\\Thermo\\MSFileReader\\XRawfile2.dll' )  # 32bits msFileReader with 64bits system 
    XRawfile2_dll_path.append( u'C:\\Program Files\\Thermo\\MSFileReader\\XRawfile2.dll' )        # 32bits msFileReader with 32bits system
    XRawfile2_dll_path_0 = copy.deepcopy(XRawfile2_dll_path)
    while not XRawfile2_dll_loaded:
        try :
            # TODO ? version with XRawfile2.dll integrated = no need to install MSFileReader, dll not registered to the COM server
            # problem : IXRawfile4 not found
            #  -> http://osdir.com/ml/python.comtypes.user/2008-07/msg00045.html messages 42-46 talk about it
            XRawfile2_dll_filename = XRawfile2_dll_path.pop(0)
            log.debug("Trying comtypes.client.GetModule " + XRawfile2_dll_filename + " ...")
            GetModule(XRawfile2_dll_filename) # -> generation
        except IndexError:
            msg = '1) The msFileReader DLL (XRawfile2.dll or XRawfile2_x64.dll) may not be installed and therefore not registered to the COM server' \
            '2) the msFileReader DLL may not be a these paths :\n' + '\n'.join(XRawfile2_dll_path_0)
            sys.exit(msg)
        except Exception as e:
            log.debug(e)
        else:
            log.debug('DLL path : ' + XRawfile2_dll_filename)
            XRawfile2_dll_loaded = True


def _to_float(x):
    try :
        out = float(x)
    except ValueError :
        out = str(x)
    return out

try:
    basestring
except NameError:
    basestring = str
  
class ThermoRawfile(object):
    
    # static class members
    
    sampleType = {0: 'Unknown',
    1: 'Blank',
    2: 'QC',
    3: 'Standard Clear (None)',
    4: 'Standard Update (None)',
    5: 'Standard Bracket (Open)',
    6: 'Standard Bracket Start (multiple brackets)',
    7: 'Standard Bracket End (multiple brackets)'}
    
    controllerType = { -1: 'No device',
                    0: 'MS',
                    1: 'Analog',
                    2: 'A/D card',
                    3: 'PDA',
                    4: 'UV',
                    'No device':-1,
                    'MS':0,
                    'Analog':1,
                    'A/D card':2,
                    'PDA':3,
                    'UV':4 }
    
    massAnalyzerType = {'ITMS': 0,
                            'TQMS': 1,
                            'SQMS': 2,
                            'TOFMS': 3,
                            'FTMS': 4,
                            'Sector': 5,
                            0: 'ITMS',
                            1: 'TQMS',
                            2: 'SQMS',
                            3: 'TOFMS',
                            4: 'FTMS',
                            5: 'Sector'}
    activationType = { 'CID':  0,
                            'MPD': 1,
                            'ECD':  2,
                            'PQD': 3,
                            'ETD': 4,
                            'HCD': 5,
                            'Any activation type': 6,
                            'SA': 7,
                            'PTR': 8,
                            'NETD': 9,
                            'NPTR': 10,
                            0: 'CID',
                            1: 'MPD',
                            2: 'ECD',
                            3: 'PQD',
                            4: 'ETD',
                            5: 'HCD',
                            6: 'Any activation type',
                            7: 'SA',
                            8: 'PTR',
                            9: 'NETD',
                            10: 'NPTR'}
                            
    detectorType = {'CID': 0,
                            'PQD': 1,
                            'ETD': 2,
                            'HCD': 3,
                            0: 'CID',
                            1: 'PQD',
                            2: 'ETD',
                            3: 'HCD'}
                            
    scanType = { 'ScanTypeFull': 0,
                        'ScanTypeSIM': 1,
                        'ScanTypeZoom': 2,
                        'ScanTypeSRM': 3,
                        0: 'ScanTypeFull',
                        1: 'ScanTypeSIM',
                        2: 'ScanTypeZoom',
                        3: 'ScanTypeSRM'}
                            
    def __init__(self, filename, **kwargs):   ## FILE NAME=?? 
        
        self.filename = filename
        self.source = None
        
        try:
            log.debug("obj = CreateObject('MSFileReader.XRawfile')")
            obj = CreateObject('MSFileReader.XRawfile')
        except Exception as e:
            log.debug(e)
            try:
                log.debug("obj = CreateObject('XRawfile.XRawfile')")
                obj = CreateObject('XRawfile.XRawfile')
            except Exception as e: 
                log.debug(e)
                sys.exit('Please install the appropriate Thermo MSFileReader version depending of your Python version (32bits or 64bits)')

        self.source = obj
        
        try:
            error = obj.Open(filename)
        except WindowsError: 
            raise WindowsError(  "RAWfile {0} could not be opened, is the file accessible and not opened in Xcalibur/QualBrowser ?".format(self.filename) )
        if error: raise IOError( "RAWfile {0} could not be opened, is the file accessible ?".format(self.filename) )

        error = obj.SetCurrentController(c_long(0),c_long(1))
        if error:
            obj.Close()
            raise IOError( "Open error {} : {}".format(self.filename,error))

        self.StartTime = self.GetStartTime()
        self.EndTime = self.GetEndTime()
        self.FirstSpectrumNumber = self.GetFirstSpectrumNumber()
        self.LastSpectrumNumber = self.GetLastSpectrumNumber()
        self.LowMass = self.GetLowMass()
        self.HighMass = self.GetHighMass()
        self.MassResolution = self.GetMassResolution()
        self.NumSpectra = self.GetNumSpectra()
        self.InstMethodNames = self.GetInstMethodNames()
        self.NumInstMethods = self.GetNumInstMethods()
        self.NumStatusLog = self.GetNumStatusLog()
        self.NumErrorLog = self.GetNumErrorLog()
        self.NumTuneData = self.GetNumTuneData()
        self.NumTrailerExtra = self.GetNumTrailerExtra()
        self.dll_version = self.Version()
        self.FileName = self.GetFileName()
        self.InstrumentDescription = self.GetInstrumentDescription()
        self.AcquisitionDate = self.GetAcquisitionDate()
        self.InstName = self.GetInstName()
        self.InstModel = self.GetInstModel()
        self.InstSoftwareVersion = self.GetInstSoftwareVersion()
        self.InstHardwareVersion = self.GetInstHardwareVersion()
        
    def Close(self):
        """Closes a raw file and frees the associated memory."""
        self.source.Close()
        
    def Version(self): # MSFileReader DLL version
        """This function returns the version number for the DLL."""
        MajorVersion, MinorVersion, SubMinorVersion, BuildNumber =  c_long(), c_long(), c_long(), c_long()
        error = self.source.Version(byref(MajorVersion), byref(MinorVersion), byref(SubMinorVersion), byref(BuildNumber))
        if error : raise IOError("Version error :", error)
        return '{}.{}.{}.{}'.format(MajorVersion.value, MinorVersion.value, SubMinorVersion.value, BuildNumber.value)
        
    def GetFileName(self):
        """Returns the fully qualified path name of an open raw file."""
        pbstrFileName = comtypes.automation.BSTR()
        error = self.source.GetFileName( byref(pbstrFileName) )
        if error : raise IOError("GetFileName error :", error)
        return pbstrFileName.value
        
    def GetCreatorID(self):
        """Returns the creator ID. The creator ID is the logon name of the user when the raw file was acquired."""
        pbstrCreatorID = comtypes.automation.BSTR()
        error = self.source.GetCreatorID( byref(pbstrCreatorID) )
        if error : raise IOError("GetCreatorID error :", error)
        return pbstrCreatorID.value

    def GetVersionNumber(self):
        '''Returns the file format version number'''
        versionNumber =  c_long()
        error = self.source.GetVersionNumber(byref(versionNumber))
        if error : raise IOError("GetVersionNumber error :", error)
        return versionNumber.value
        
    def GetCreationDate(self):
        """Returns the file creation date in DATE format."""
        # https://msdn.microsoft.com/en-us/library/82ab7w69.aspx
        # The DATE type is implemented using an 8-byte floating-point number
        pCreationDate =  c_double()
        error = self.source.GetCreationDate(byref(pCreationDate))
        if error : raise IOError("GetCreationDate error :", error)
        return pCreationDate.value
        
    def IsError(self):
        """Returns the error state flag of the raw file. A return value of TRUE indicates that an error has
        occurred. For information about the error, call the GetErrorCode or GetErrorMessage
        functions."""
        pbIsError =  c_long()
        error = self.source.IsError(byref(pbIsError))
        if error : raise IOError("IsError error :", error)
        return bool(pbIsError.value)
        
    def IsNewFile(self):
        """Returns the creation state flag of the raw file. A return value of TRUE indicates that the file
        has not previously been saved."""
        bNewFile =  c_long()
        error = self.source.IsNewFile(byref(bNewFile))
        if error : raise IOError("IsNewFile error :", error)
        return bool(bNewFile.value)
        
    def IsThereMSData(self):
        """This function checks to see if there is MS data in the raw file. A return value of TRUE means
        that the raw file contains MS data. You must open the raw file before performing this check."""
        pbMSData =  c_long()
        error = self.source.IsThereMSData(byref(pbMSData))
        if error : raise IOError("IsThereMSData error :", error)
        return bool(pbMSData.value)

    def HasExpMethod(self):
        """This function checks to see if the raw file contains an experimental method. A return value of
        TRUE indicates that the raw file contains the method. You must open the raw file before
        performing this check."""
        bHasMethod =  c_long()
        error = self.source.HasExpMethod(byref(bHasMethod))
        if error : raise IOError("HasExpMethod error :", error)
        return bool(bHasMethod.value)
        
    def InAcquisition(self):
        """Returns the acquisition state flag of the raw file. A return value of TRUE indicates that the
        raw file is being acquired or that all open handles to the file during acquisition have not been
        closed."""
        pbInAcquisition =  c_long()
        error = self.source.InAcquisition(byref(pbInAcquisition))
        if error : raise IOError("InAcquisition error :", error)
        return bool(pbInAcquisition.value)
        
    def GetErrorCode(self):
        """Returns the error code of the raw file. A return value of 0 indicates that there is no error."""
        nErrorCode =  c_long()
        error = self.source.GetErrorCode(byref(nErrorCode))
        if error : raise IOError("GetErrorCode error :", error)
        return nErrorCode.value
        
    def GetErrorMessage(self):
        """Returns error information for the raw file as a descriptive string. If there is no error, the
        returned string is empty."""
        pbstrErrorMessage = comtypes.automation.BSTR()
        error = self.source.GetErrorMessage(byref(pbstrErrorMessage))
        if error : raise IOError ("GetErrorMessage error : ", error)
        return pbstrErrorMessage.value
        
    def GetWarningMessage(self):
        """Returns warning information for the raw file as a descriptive string. If there is no warning, the
        returned string is empty."""
        pbstrWarningMessage = comtypes.automation.BSTR()
        error = self.source.GetWarningMessage(byref(pbstrWarningMessage))
        if error : raise IOError ("GetWarningMessage error : ", error)
        return pbstrWarningMessage.value
        
    def RefreshViewOfFile(self):
        """Refreshes the view of a file currently being acquired. This function provides a more efficient
        mechanism for gaining access to new data in a raw file during acquisition without closing and
        reopening the raw file. This function has no effect with files that are not being acquired."""
        error = self.source.RefreshViewOfFile()
        if error : raise IOError("RefreshViewOfFile error :", error)
        return
        
    def GetNumberOfControllers(self):
        """Returns the number of registered device controllers in the raw file. A device controller
        represents an acquisition stream such as MS data, UV data, and so on. Devices that do not
        acquire data, such as autosamplers, are not registered with the raw file during acquisition."""
        pnNumControllers =  c_long()
        error = self.source.GetNumberOfControllers(byref(pnNumControllers))
        if error : raise IOError("GetNumberOfControllers error :", error)
        return pnNumControllers.value
        
    def GetNumberOfControllersOfType(self, controllerType = 'MS'):
        """Returns the number of registered device controllers of a particular type in the raw file. See
        Controller Type in the Enumerated Types section for a list of the available controller types
        and their respective values."""
        if isinstance(controllerType, basestring):
            controllerType = ThermoRawfile.controllerType[controllerType]
        pnNumControllersOfType = c_long()
        error = self.source.GetNumberOfControllersOfType(c_long(controllerType), byref(pnNumControllersOfType) )
        if error : raise IOError("GetNumberOfControllersOfType error :", error)
        return pnNumControllersOfType.value
        
    def GetControllerType(self, index):
        """Returns the type of the device controller registered at the specified index position in the raw
        file. Index values start at 0. See Controller Type in the Enumerated Types section for a list of
        the available controller types and their respective values."""
        controllerType = c_long()
        error = self.source.GetControllerType(index, byref(controllerType) )
        if error : raise IOError("GetControllerType error :", error)
        return ThermoRawfile.controllerType[controllerType.value]
        
    def SetCurrentController(self, controllerType, controllerNumber):
        """Sets the current controller in the raw file. This function must be called before subsequent calls
        to access data specific to a device controller (for example, MS or UV data) may be made. All
        requests for data specific to a device controller are forwarded to the current controller until the
        current controller is changed. The controller number is used to indicate which device
        controller to use if there is more than one registered device controller of the same type (for
        example, multiple UV detectors). Controller numbers for each type are numbered starting
        at 1. See Controller Type in the Enumerated Types section for a list of the available controller
        types and their respective values."""
        if isinstance(controllerType, basestring):
            controllerType = ThermoRawfile.controllerType[controllerType]
        error = self.source.SetCurrentController(c_long(controllerType), c_long(controllerNumber))
        if error : raise IOError("SetCurrentController error :", error)
        return
        
    def GetCurrentController(self):
        """Gets the current controller type and number for the raw file. The controller number is used to
        indicate which device controller to use if there is more than one registered device controller of
        the same type (for example, multiple UV detectors). Controller numbers for each type are
        numbered starting at 1. See Controller Type in the Enumerated Types section for a list of the
        available controller types and their respective values."""
        pnControllerType = c_long()
        pnControllerNumber = c_long()
        error = self.source.GetCurrentController(byref(pnControllerType), byref(pnControllerNumber) )
        if error : raise IOError("GetCurrentController error :", error)
        return pnControllerType.value, pnControllerNumber.value
        
    def GetExpectedRunTime(self):
        """Gets the expected acquisition run time for the current controller. The actual acquisition may
        be longer or shorter than this value. This value is intended to allow displays to show the
        expected run time on chromatograms. To obtain an accurate run time value during or after
        acquisition, use the GetEndTime function."""
        pdExpectedRunTime = c_double()
        error = self.source.GetExpectedRunTime(byref(pdExpectedRunTime) )
        if error : raise IOError("GetExpectedRunTime error :", error)
        return pdExpectedRunTime.value
        
    def GetNumTrailerExtra(self):
        """Gets the trailer extra entries recorded for the current controller. Trailer extra entries are only
        supported for MS device controllers and are used to store instrument specific information for
        each scan if used."""
        pnNumberOfTrailerExtraEntries = c_long()
        error = self.source.GetNumTrailerExtra(byref(pnNumberOfTrailerExtraEntries) )
        if error : raise IOError("GetNumTrailerExtra error :", error)
        return pnNumberOfTrailerExtraEntries.value
    
    def GetMaxIntegratedIntensity(self):
        """Gets the highest integrated intensity of all the scans for the current controller. This value is
        only relevant to MS device controllers."""
        pdMaxIntegIntensity = c_double()
        error = self.source.GetMaxIntegratedIntensity(byref(pdMaxIntegIntensity) )
        if error : raise IOError("GetMaxIntegratedIntensity error :", error)
        return pdMaxIntegIntensity.value
        
    def GetMaxIntensity(self):
        """Gets the highest base peak of all the scans for the current controller. This value is only relevant
        to MS device controllers."""
        dMaxInt = c_long()
        error = self.source.GetMaxIntensity(byref(dMaxInt) )
        if error : raise IOError("GetMaxIntensity error :", error)
        return dMaxInt.value

    def GetInletID(self):
        """Gets the inlet ID number for the current controller. This value is typically only set for raw
        files converted from other file formats."""
        nInletID = c_long()
        error = self.source.GetInletID(byref(nInletID) )
        if error : raise IOError("GetInletID error :", error)
        return nInletID.value
        
    def GetErrorFlag(self):
        """Gets the error flag value for the current controller. This value is typically only set for raw files
        converted from other file formats."""
        pnErrorFlag = c_long()
        error = self.source.GetErrorFlag(byref(pnErrorFlag) )
        if error : raise IOError("GetErrorFlag error :", error)
        return pnErrorFlag.value
        
    def GetFlags(self):
        """Returns the acquisition flags field for the current controller. This value is typically only set for
        raw files converted from other file formats."""
        pbstrFlags = comtypes.automation.BSTR(None)
        error = self.source.GetFlags(byref(pbstrFlags))
        if error: raise IOError("GetFlags error :", error)
        return pbstrFlags.value
        
    def GetAcquisitionFileName(self):
        """Returns the acquisition file name for the current controller. This value is typically only set for
        raw files converted from other file formats."""
        pbstrFileName = comtypes.automation.BSTR(None)
        error = self.source.GetAcquisitionFileName(byref(pbstrFileName))
        if error: raise IOError("GetAcquisitionFileName error :", error)
        return pbstrFileName.value
        
    def GetAcquisitionDate(self):
        """Returns the acquisition date for the current controller. This value is typically only set for raw
        files converted from other file formats."""
        pbstrAcquisitionDate = comtypes.automation.BSTR(None)
        error = self.source.GetAcquisitionDate(byref(pbstrAcquisitionDate))
        if error: raise IOError("GetAcquisitionDate error :", error)
        return pbstrAcquisitionDate.value
        
    def GetOperator(self):
        """Returns the operator name for the current controller. This value is typically only set for raw
        files converted from other file formats."""
        pbstrOperator = comtypes.automation.BSTR(None)
        error = self.source.GetOperator(byref(pbstrOperator))
        if error: raise IOError("GetOperator error :", error)
        return pbstrOperator.value
    
    def GetComment1(self):
        """Returns the first comment for the current controller. This value is typically only set for raw
        files converted from other file formats."""
        pbstrComment1 = comtypes.automation.BSTR(None)
        error = self.source.GetComment1(byref(pbstrComment1))
        if error: raise IOError("GetComment1 error :", error)
        return pbstrComment1.value
        
    def GetComment2(self):
        """Returns the second comment for the current controller. This value is typically only set for raw
        files converted from other file formats."""
        pbstrComment2 = comtypes.automation.BSTR(None)
        error = self.source.GetComment2(byref(pbstrComment2))
        if error: raise IOError("GetComment2 error :", error)
        return pbstrComment2.value
        
    def GetFilters(self):
        """Returns the list of unique scan filters for the raw file. This function is only supported for MS
        device controllers. If the function succeeds, pvarFilterArray points to an array of BSTR fields,
        each containing a unique scan filter, and pnArraySize contains the number of scan filters in the
        pvarFilterArray."""
        pvarFilterArray = comtypes.automation.VARIANT()
        pnArraySize = c_long()
        error = self.source.GetFilters(byref(pvarFilterArray), byref(pnArraySize))
        if error: raise IOError("GetFilters error :", error)
        return pvarFilterArray.value  
        
        
    def SetMassTolerance(self, userDefined = True, massTolerance = 500.0, units = 0):
        """This function sets the mass tolerance that will be used with the raw file.
        
        userDefined        A boolean indicating whether the mass tolerance is user-defined (True) or
                            based on the values in the raw file (False).
        massTolerance      The mass tolerance value.
        units              The type of tolerance value (amu = 2, mmu = 0, or ppm = 1).
        """
        error = self.source.SetMassTolerance(userDefined, c_double(massTolerance), c_long(units))
        if error: raise IOError("SetMassTolerance error :", error)
        return  
        
    def GetMassTolerance(self):
        """This function gets the mass tolerance that is being used with the raw file. To set these values,
        use the SetMassTolerance() function.
        
        bUserDefined        A flag indicating whether the mass tolerance is user-defined (TRUE) or
                            based on the values in the raw file (FALSE).
        dMassTolerance      The mass tolerance value.
        nUnits              The type of tolerance value (amu = 2, mmu = 0, or ppm = 1).
        """
        bUserDefined = c_long()
        dMassTolerance = c_double()
        nUnits = c_long()
        error = self.source.GetMassTolerance(byref(bUserDefined), byref(dMassTolerance), byref(nUnits))
        if error: raise IOError("GetMassTolerance error :", error)
        return bool(bUserDefined.value), dMassTolerance.value, nUnits.value
    
        
    ######## INSTRUMENT BEGIN
    def GetInstrumentDescription(self):
        """Returns the instrument description field for the current controller. This value is typically only
        set for raw files converted from other file formats."""
        pbstrInstrumentDescription = comtypes.automation.BSTR(None)
        error = self.source.GetInstrumentDescription(byref(pbstrInstrumentDescription))
        if error: raise IOError("GetInstrumentDescription error :", error)
        return pbstrInstrumentDescription.value
        
    def GetInstrumentID(self):
        """Gets the instrument ID number for the current controller. This value is typically only set for
        raw files converted from other file formats."""
        pnInstrumentID = c_long()
        error = self.source.GetInstrumentID(byref(pnInstrumentID) )
        if error : raise IOError("GetInstrumentID error :", error)
        return pnInstrumentID.value
        
    def GetInstName(self):
        """Returns the instrument name, if available, for the current controller."""
        pbstrInstName = comtypes.automation.BSTR(None)
        error = self.source.GetInstName(byref(pbstrInstName))
        if error: raise IOError("GetInstName error :", error)
        return pbstrInstName.value
    
    def GetInstModel(self):
        """Returns the instrument model, if available, for the current controller."""
        pbstrInstModel = comtypes.automation.BSTR(None)
        error = self.source.GetInstModel(byref(pbstrInstModel))
        if error: raise IOError("GetInstModel error :", error)
        return pbstrInstModel.value
    
    def GetInstSerialNumber(self):
        """Returns the serial number, if available, for the current controller."""
        pbstrInstSerialNumber = comtypes.automation.BSTR(None)
        error = self.source.GetInstSerialNumber(byref(pbstrInstSerialNumber))
        if error: raise IOError("GetInstSerialNumber error :", error)
        return pbstrInstSerialNumber.value
    
    def GetInstSoftwareVersion(self):
        '''Returns revision information for the current controller software, if available.'''
        InstSoftwareVersion = comtypes.automation.BSTR()
        error = self.source.GetInstSoftwareVersion(byref(InstSoftwareVersion) )
        if error : raise IOError("GetInstSoftwareVersion error :", error)
        return InstSoftwareVersion.value
                
    def GetInstHardwareVersion(self):
        '''Returns revision information for the current controller software, if available.'''
        InstHardwareVersion = comtypes.automation.BSTR()
        error = self.source.GetInstHardwareVersion(byref(InstHardwareVersion) )
        if error : raise IOError("GetInstHardwareVersion error :", error)
        return InstHardwareVersion.value
    
    def GetInstFlags(self):
        """Returns the experiment flags, if available, for the current controller. The returned string may
        contain one or more fields denoting information about the type of experiment performed.
        These are the currently defined experiment fields:
        TIM - total ion map
        NLM - neutral loss map
        PIM - parent ion map
        DDZMap - data-dependent ZoomScan map"""
        pbstrInstFlags = comtypes.automation.BSTR()
        error = self.source.GetInstFlags(byref(pbstrInstFlags) )
        if error : raise IOError("GetInstFlags error :", error)
        return pbstrInstFlags.value
    
    def GetInstNumChannelLabels(self):
        """Returns the number of channel labels specified for the current controller. This field is only
        relevant to channel devices such as UV detectors, A/D cards, and Analog inputs. Typically, the
        number of channel labels, if labels are available, is the same as the number of configured
        channels for the current controller."""
        pnInstNumChannelLabels = c_long()
        error = self.source.GetInstNumChannelLabels(byref(pnInstNumChannelLabels) )
        if error : raise IOError("GetInstNumChannelLabels error :", error)
        return pnInstNumChannelLabels.value
    
    def GetInstChannelLabel(self, channelLabelNumber = 0):
        """Returns the channel label, if available, at the specified index for the current controller. This
        field is only relevant to channel devices such as UV detectors, A/D cards, and Analog inputs.
        Channel label indices are numbered starting at 0."""
        pbstrFlags = comtypes.automation.BSTR()
        error = self.source.GetInstChannelLabel(c_long(channelLabelNumber), byref(pbstrFlags) )
        if error : raise IOError("GetInstChannelLabel error :", error)
        return pbstrFlags.value
         
    ######## INSTRUMENT END
            
    def GetScanEventForScanNum(self, scanNumber):
        """This function returns scan event information as a string for the specified scan number."""
        pbstrScanEvent = comtypes.automation.BSTR()
        error = self.source.GetScanEventForScanNum(c_long(scanNumber), byref(pbstrScanEvent) )
        if error : raise IOError("GetScanEventForScanNum error :", error)
        return pbstrScanEvent.value
    
    def GetSegmentAndEventForScanNum(self, scanNumber): # NOT GetSegmentAndScanEventForScanNum
        """Returns the segment and scan event indexes for the specified scan."""
        pbstrScanEvent = comtypes.automation.BSTR()
        pnSegment = c_long(0)
        pnScanEvent = c_long(0)
        error = self.source.GetSegmentAndEventForScanNum(c_long(scanNumber), byref(pnSegment), byref(pnScanEvent) )
        if error : raise IOError("GetSegmentAndEventForScanNum error :", error)
        return pbstrScanEvent.value
    
    def GetSegmentAndScanEventForScanNum(self, scanNumber):
        return self.GetSegmentAndEventForScanNum(scanNumber)

    def GetCycleNumberFromScanNumber(self, scanNumber):
        """This function returns the cycle number for the scan specified by scanNumber from the scan
        index structure in the raw file."""
        pbstrScanEvent = comtypes.automation.BSTR()
        pnCycleNumber = c_long()
        error = self.source.GetCycleNumberFromScanNumber(c_long(scanNumber), byref(pnCycleNumber) )
        if error : raise IOError("GetCycleNumberFromScanNumber error :", error)
        return pnCycleNumber.value
    
    def GetAValueFromScanNum(self, scanNumber):
        """This function gets the A parameter value in the scan event. The value returned is either 0, 1,
        or 2 for parameter A off, parameter A on, or accept any parameter A, respectively."""
        pnXValue = c_long()
        error = self.source.GetAValueFromScanNum(c_long(scanNumber), byref(pnXValue) )
        if error : raise IOError("GetAValueFromScanNum error :", error)
        return pnXValue.value
    
    def GetBValueFromScanNum(self, scanNumber):
        """This function gets the B parameter value in the scan event. The value returned will be either
        0, 1, or 2 for parameter B off, parameter B on, or accept any parameter B, respectively."""
        pnXValue = c_long()
        error = self.source.GetBValueFromScanNum(c_long(scanNumber), byref(pnXValue) )
        if error : raise IOError("GetBValueFromScanNum error :", error)
        return pnXValue.value
    
    def GetFValueFromScanNum(self, scanNumber):
        """This function gets the F parameter value in the scan event. The value returned is either 0, 1,
        or 2 for parameter F off, parameter F on, or accept any parameter F, respectively."""
        pnXValue = c_long()
        error = self.source.GetFValueFromScanNum(c_long(scanNumber), byref(pnXValue) )
        if error : raise IOError("GetFValueFromScanNum error :", error)
        return pnXValue.value
    
    def GetKValueFromScanNum(self, scanNumber):
        """This function gets the K parameter value in the scan event. The value returned is either 0, 1,
        or 2 for parameter K off, parameter K on, or accept any parameter K, respectively."""
        pnXValue = c_long()
        error = self.source.GetKValueFromScanNum(c_long(scanNumber), byref(pnXValue) )
        if error : raise IOError("GetKValueFromScanNum error :", error)
        return pnXValue.value
    
    def GetRValueFromScanNum(self, scanNumber):
        """This function gets the R parameter value in the scan event. The value returned is either 0, 1,
        or 2 for parameter R off, parameter R on, or accept any parameter R, respectively."""
        pnXValue = c_long()
        error = self.source.GetRValueFromScanNum(c_long(scanNumber), byref(pnXValue) )
        if error : raise IOError("GetRValueFromScanNum error :", error)
        return pnXValue.value
    
    def GetVValueFromScanNum(self, scanNumber):
        """This function gets the V parameter value in the scan event. The value returned is either 0, 1,
        or 2 for parameter V off, parameter V on, or accept any parameter V, respectively."""
        pnXValue = c_long()
        error = self.source.GetVValueFromScanNum(c_long(scanNumber), byref(pnXValue) )
        if error : raise IOError("GetVValueFromScanNum error :", error)
        return pnXValue.value
    
    def GetMSXMultiplexValueFromScanNum(self, scanNumber):
        """This function gets the msx-multiplex parameter value in the scan event. The value returned is
        either 0, 1, or 2 for multiplex off, multiplex on, or accept any multiplex, respectively."""
        pnMSXValue = c_long()
        error = self.source.GetMSXMultiplexValueFromScanNum(c_long(scanNumber), byref(pnMSXValue) )
        if error : raise IOError("GetMSXMultiplexValueFromScanNum error :", error)
        return pnMSXValue.value

    def GetNumberOfMassRangesFromScanNum(self, scanNumber):
        """This function gets the number of MassRange data items in the scan."""
        result = c_long()
        error = self.source.GetNumberOfMassRangesFromScanNum(c_long(scanNumber), byref(result) )
        if error : raise IOError("GetNumberOfMassRangesFromScanNum error :", error)
        return result.value
        
    def GetMassRangeFromScanNum(self, scanNumber, massRangeIndex):
        """This function retrieves information about the mass range data of a scan (high and low
        masses). You can find the count of mass ranges for the scan by calling
        GetNumberOfMassRangesFromScanNum()."""
        pdMassRangeLowValue = c_double()
        pdMassRangeHighValue = c_double()
        error = self.source.GetMassRangeFromScanNum(c_long(scanNumber), c_long(massRangeIndex), byref(pdMassRangeLowValue), byref(pdMassRangeHighValue) )
        if error : raise IOError("GetMassRangeFromScanNum error :", error)
        return pdMassRangeLowValue.value, pdMassRangeHighValue.value
    
    def GetNumberOfSourceFragmentsFromScanNum(self, scanNumber):
        """This function gets the number of source fragments (or compensation voltages) in the scan."""
        result = c_long()
        error = self.source.GetNumberOfSourceFragmentsFromScanNum(c_long(scanNumber), byref(result) )
        if error : raise IOError("GetNumberOfSourceFragmentsFromScanNum error :", error)
        return result.value
    
    def GetSourceFragmentValueFromScanNum(self, scanNumber, sourceFragmentIndex):
        """This function retrieves information about one of the source fragment values of a scan. It is
        also the same value as the compensation voltage. You can find the count of source fragments
        for the scan by calling GetNumberOfSourceFragmentsFromScanNum ()."""
        pdSourceFragmentValue = c_double()
        error = self.source.GetSourceFragmentValueFromScanNum(c_long(scanNumber), c_long(sourceFragmentIndex), byref(pdSourceFragmentValue) )
        if error : raise IOError("GetSourceFragmentValueFromScanNum error :", error)
        return pdSourceFragmentValue.value
        
    def GetNumberOfSourceFragmentationMassRangesFromScanNum(self, scanNumber):
        """This function gets the number of source fragmentation mass ranges in the scan."""
        result = c_long()
        error = self.source.GetNumberOfSourceFragmentationMassRangesFromScanNum(c_long(scanNumber), byref(result) )
        if error : raise IOError("GetNumberOfSourceFragmentationMassRangesFromScanNum error :", error)
        return result.value
    
    def GetSourceFragmentationMassRangeFromScanNum(self, scanNumber, sourceFragmentIndex):
        """This function retrieves information about the source fragment mass range data of a scan (high
        and low source fragment masses). You can find the count of mass ranges for the scan by calling
        GetNumberOfSourceFragmentationMassRangesFromScanNum ()."""
        pdSourceFragmentRangeLowValue = c_double()
        pdSourceFragmentRangeHighValue = c_double()
        error = self.source.GetSourceFragmentationMassRangeFromScanNum(c_long(scanNumber), c_long(sourceFragmentIndex), byref(pdSourceFragmentRangeLowValue), byref(pdSourceFragmentRangeHighValue) )
        if error : raise IOError("GetSourceFragmentationMassRangeFromScanNum error :", error)
        return pdSourceFragmentRangeLowValue.value, pdSourceFragmentRangeHighValue.value
    
    def GetIsolationWidthForScanNum(self, scanNumber, MSOrder):
        """This function returns the isolation width for the scan specified by scanNumber and the
        transition specified by MSOrder from the scan event structure in the raw file."""
        result = c_double()
        error = self.source.GetIsolationWidthForScanNum(c_long(scanNumber), c_long(MSOrder), byref(result) )
        if error : raise IOError("GetIsolationWidthForScanNum error :", error)
        return result.value
    
    def GetCollisionEnergyForScanNum(self, scanNumber, MSOrder):
        """This function returns the collision energy for the scan specified by scanNumber and the
        transition specified by MSOrder from the scan event structure in the raw file. """
        result = c_double()
        error = self.source.GetCollisionEnergyForScanNum(c_long(scanNumber), c_long(MSOrder), byref(result) )
        if error : raise IOError("GetCollisionEnergyForScanNum error :", error)
        return result.value
     
        
    def GetActivationTypeForScanNum(self, scanNumber, MSOrder):
        """This function returns the activation type for the scan specified by scanNumber and the
        transition specified by MSOrder from the scan event structure in the RAW file.
        The value returned in the pnActivationType variable is one of the following:
        CID  0
        MPD 1
        ECD  2
        PQD 3
        ETD 4
        HCD 5
        Any activation type 6
        SA 7
        PTR 8
        NETD 9
        NPTR 10"""
        result = c_long()
        error = self.source.GetActivationTypeForScanNum(c_long(scanNumber), c_long(MSOrder), byref(result) )
        if error : raise IOError("GetActivationTypeForScanNum error :", error)
        return ThermoRawfile.activationType[result.value]
        
    def GetMassAnalyzerTypeForScanNum(self, scanNumber):
        """This function returns the mass analyzer type for the scan specified by scanNumber from the
        scan event structure in the RAW file. The value of scanNumber must be within the range of
        scans or readings for the current controller. The range of scans or readings for the current
        controller may be obtained by calling GetFirstSpectrumNumber and
        GetLastSpectrumNumber.
        The value returned in the pnMassAnalyzerType variable is one of the following:
        ITMS  0
        TQMS 1
        SQMS  2
        TOFMS 3
        FTMS 4
        Sector 5"""
        result = c_long()
        error = self.source.GetMassAnalyzerTypeForScanNum(c_long(scanNumber), byref(result) )
        if error : raise IOError("GetMassAnalyzerTypeForScanNum error :", error)
        return ThermoRawfile.massAnalyzerType[result.value]
    
    def GetDetectorTypeForScanNum(self, scanNumber):
        """This function returns the detector type for the scan specified by scanNumber from the scan
        event structure in the RAW file.
        The value returned in the pnDetectorType variable is one of the following:
        CID  0
        PQD  1
        ETD  2
        HCD  3"""
        result = c_long()
        error = self.source.GetDetectorTypeForScanNum(c_long(scanNumber), byref(result) )
        if error : raise IOError("GetDetectorTypeForScanNum error :", error)
        return ThermoRawfile.detectorType[result.value]
        
    def GetScanTypeForScanNum(self, scanNumber):
        """This function returns the scan type for the scan specified by scanNumber from the scan
        event structure in the RAW file.
        The value returned in the pnScanType variable is one of the following:
        ScanTypeFull  0
        ScanTypeSIM  1
        ScanTypeZoom  2
        ScanTypeSRM  3"""
        result = c_long()
        error = self.source.GetScanTypeForScanNum(c_long(scanNumber), byref(result) )
        if error : raise IOError("GetScanTypeForScanNum error :", error)
        return ThermoRawfile.scanType[result.value]
        

     
    def GetNumberOfMassCalibratorsFromScanNum(self, scanNumber):
        """This function gets the number of mass calibrators (each of which is a double) in the scan."""
        result = c_long()
        error = self.source.GetNumberOfMassCalibratorsFromScanNum(c_long(scanNumber), byref(result) )
        if error : raise IOError("GetNumberOfMassCalibratorsFromScanNum error :", error)
        return result.value
        
    def GetMassCalibrationValueFromScanNum(self, scanNumber, massCalibrationIndex):
        """This function retrieves information about one of the mass calibration data values of a scan.
        You can find the count of mass calibrations for the scan by calling
        GetNumberOfMassCalibratorsFromScanNum()."""
        result = c_double()
        error = self.source.GetMassCalibrationValueFromScanNum(c_long(scanNumber), c_long(massCalibrationIndex), byref(result) )
        if error : raise IOError("GetMassCalibrationValueFromScanNum error :", error)
        return result.value
     
    def GetFilterMassPrecision(self):
        """This function gets the mass precision for the filter associated with an MS scan."""
        result = c_long()
        error = self.source.GetFilterMassPrecision(byref(result))
        if error : raise IOError("GetFilterMassPrecision error :", error)
        return result.value
        
    def GetMassPrecisionEstimate(self, scanNumber):
        """This function is only applicable to scanning devices such as MS. It gets the mass precision
        information for an accurate mass spectrum (that is, acquired on an FTMS- or Orbitrap-class
        instrument).
        
        If no scan filter is supplied, the scan corresponding to pnScanNumber is returned. If a scan
        filter is provided, the closest matching scan to pnScanNumber that matches the scan filter is
        returned.

        The format of the mass list returned is an array of double-precision
        values in the order of intensity, mass, accuracy in MMU, accuracy in PPM, and resolution."""
        pnArraySize = c_long()
        result = comtypes.automation.VARIANT()
        error = self.source.GetMassPrecisionEstimate( c_long(scanNumber), byref(result), byref(pnArraySize) )
        if error : raise IOError("GetMassPrecisionEstimate error :", error)
        return result.value
     
    def IsQExactive(self): 
        """Checks the instrument name by calling GetInstName() and comparing the result to Q
        Exactive Orbitrap. If it matches, IsQExactive pVal is set to TRUE. Otherwise, pVal is set to
        FALSE.
        
        NOTE : not implemented in MSFileReader 3.0 SP2 (3.0.29.0)
        """
        result = c_long()
        error = self.source.IsQExactive( byref(result) )
        if error : raise IOError("IsQExactive error :", error)
        return bool(result.value)
    
    def GetMassResolution(self):
        """Gets the mass resolution value recorded for the current controller. The value is returned as one
        half of the mass resolution. For example, a unit resolution controller returns a value of 0.5.
        This value is only relevant to scanning controllers such as MS."""
        result = c_double()
        error = self.source.GetMassResolution(byref(result))
        if error : raise IOError("GetMassResolution error :", error)
        return result.value
        
    def GetLowMass(self):
        """Gets the lowest mass or wavelength recorded for the current controller. This value is only
        relevant to scanning devices such as MS or PDA."""
        pdLowMass = c_double()
        error = self.source.GetLowMass(byref(pdLowMass))
        if error : raise IOError("GetLowMass error :", error)
        return pdLowMass.value
        
    def GetHighMass(self):
        """Gets the highest mass or wavelength recorded for the current controller. This value is only
        relevant to scanning devices such as MS or PDA."""
        pdHighMass = c_double()
        error = self.source.GetHighMass(byref(pdHighMass))
        if error : raise IOError("GetHighMass error :", error)
        return pdHighMass.value
    
    def GetStartTime(self):
        """Gets the start time of the first scan or reading for the current controller. This value is typically
        close to zero unless the device method contains a start delay."""
        pdStartTime = c_double()
        error = self.source.GetStartTime(byref(pdStartTime))
        if error : raise IOError("GetStartTime error :", error)
        return pdStartTime.value
    
    def GetEndTime(self):
        pdEndTime = c_double()
        error = self.source.GetEndTime(byref(pdEndTime))
        if error : raise IOError("GetEndTime error :", error)
        return pdEndTime.value
    
    def GetNumSpectra(self):
        """Gets the number of spectra acquired by the current controller. For non-scanning devices like 
        UV detectors, the number of readings per channel is returned."""
        numSpectra = c_long()
        error = self.source.GetNumSpectra(byref(numSpectra))
        if error : raise IOError("GetNumSpectra error :", error)
        return numSpectra.value
        
    def GetFirstSpectrumNumber(self):
        """Gets the first scan or reading number for the current controller. If data has been acquired, this
        value is always one."""
        pnFirstSpectrum = c_long()
        error = self.source.GetFirstSpectrumNumber(byref(pnFirstSpectrum))
        if error : raise IOError("GetFirstSpectrumNumber error :", error)
        return pnFirstSpectrum.value
        
    def GetLastSpectrumNumber(self):
        """Gets the last scan or reading number for the current controller."""
        pnLastSpectrum = c_long()
        error = self.source.GetLastSpectrumNumber(byref(pnLastSpectrum))
        if error : raise IOError("GetLastSpectrumNumber error :", error)
        return pnLastSpectrum.value
        
    def ScanNumFromRT(self, RT):
        """Returns the closest matching scan number that corresponds to RT for the current controller.
        For non-scanning devices, such as UV, the closest reading number is returned. The value of
        RT must be within the acquisition run time for the current controller. The acquisition run
        time for the current controller may be obtained by calling GetStartTime and GetEndTime."""
        pnScanNumber = c_long()
        error = self.source.ScanNumFromRT(c_double(RT),byref(pnScanNumber))
        if error : raise IOError( "scan {}, ScanNumFromRT error : {}".format(pnScanNumber,error) )
        else: return pnScanNumber.value

    def RTFromScanNum(self, scanNumber):
        """Returns the closest matching run time or retention time that corresponds to scanNumber for
        the current controller. For non-scanning devices, such as UV, the scanNumber is the reading
        number."""
        pdRT = c_double()
        error = self.source.RTFromScanNum(c_long(scanNumber),byref(pdRT))
        if error : raise IOError( "scan {}, RTFromScanNum error : {}".format(scanNumber,str(error)) )
        else: return pdRT.value
        
    def IsProfileScanForScanNum(self, scanNumber):
        """Returns TRUE if the scan specified by scanNumber is a profile scan, FALSE if the scan is a
        centroid scan."""
        pbIsProfileScan = c_long()
        error = self.source.IsProfileScanForScanNum(c_long(scanNumber), byref(pbIsProfileScan) )
        if error : raise IOError("IsProfileScanForScanNum error :", error)
        return bool(pbIsProfileScan.value)
        
    def IsCentroidScanForScanNum(self, scanNumber):
        """Returns TRUE if the scan specified by scanNumber is a centroid scan, FALSE if the scan is a
        profile scan."""
        pbIsCentroidScan = c_long()
        error = self.source.IsCentroidScanForScanNum(c_long(scanNumber), byref(pbIsCentroidScan) )
        if error : raise IOError("IsCentroidScanForScanNum error :", error)
        return bool(pbIsCentroidScan.value)
        
    def GetFilterForScanNum(self, scanNumber):
        """Returns the closest matching run time that corresponds to scanNumber for the current
        controller. This function is only supported for MS device controllers. 
        e.g. "FTMS + c NSI Full ms [300.00-1800.00]"
        """
        pbstrFilter = comtypes.automation.BSTR(None)
        error = self.source.GetFilterForScanNum(scanNumber,byref(pbstrFilter))
        if error: raise IOError( "scan {}, GetFilterForScanNum error : {}".format(scanNumber,str(error)) )
        else: return pbstrFilter.value
        
    
    def GetMassListFromScanNum(self, scanNumber,
                                    filter="",
                                    intensityCutoffType = 0,
                                    intensityCutoffValue = 0,
                                    maxNumberOfPeaks = 0,
                                    centroidResult = False,
                                    centroidPeakWidth = 0.0):
        """This function is only applicable to scanning devices such as MS and PDA.
        
        If no scan filter is supplied, the scan corresponding to pnScanNumber is returned. If a scan
        filter is provided, the closest matching scan to pnScanNumber that matches the scan filter is
        returned.
        Scan filters must match the Xcalibur scan filter format (e.g. "FTMS + c NSI Full ms [300.00-1800.00]"). 
        
        To reduce the number of low intensity data peaks returned, an intensity cutoff,
        nIntensityCutoffType, may be applied. The available types of cutoff are 
        0   None (all values returned)
        1   Absolute (in intensity units)
        2   Relative (to base peak)
        
        To limit the total number of data peaks that are returned in the mass list, set
        nMaxNumberOfPeaks to a value greater than zero. To have all data peaks returned, set
        nMaxNumberOfPeaks to zero.

        To have profile scans centroided, set bCentroidResult to TRUE. This parameter is ignored for
        centroid scans.

        The pvarPeakFlags variable is currently not used. This variable is reserved for future use to
        return flag information, such as saturation, about each mass intensity pair.
        """
        peakList = comtypes.automation.VARIANT()
        peakFlags = comtypes.automation.VARIANT()
        pnArraySize = c_long()
        error = self.source.GetMassListFromScanNum(c_long(scanNumber), filter, intensityCutoffType, 
            intensityCutoffValue, maxNumberOfPeaks, centroidResult, c_double(centroidPeakWidth), peakList, peakFlags, byref(pnArraySize))
        if error : raise IOError ("GetMassListFromScanNum error : ",error)
        return peakList.value, peakFlags.value
        
    def GetMassListRangeFromScanNum(self, scanNumber,
                                            massRange = "",
                                            filter = "",
                                            intensityCutoffType = 0,
                                            intensityCutoffValue = 0,
                                            maxNumberOfPeaks = 0,
                                            centroidResult = False,
                                            centroidPeakWidth = 0.0):
        """This function is only applicable to scanning devices such as MS and PDA.
        
        If no scan filter is supplied, the scan corresponding to pnScanNumber is returned. If a scan
        filter is provided, the closest matching scan to pnScanNumber that matches the scan filter is
        returned.
        Scan filters must match the Xcalibur scan filter format (e.g. "FTMS + c NSI Full ms [300.00-1800.00]"). 
        
        To reduce the number of low intensity data peaks returned, an intensity cutoff,
        nIntensityCutoffType, may be applied. The available types of cutoff are 
        0   None (all values returned)
        1   Absolute (in intensity units)
        2   Relative (to base peak)
        
        To limit the total number of data peaks that are returned in the mass list, set
        nMaxNumberOfPeaks to a value greater than zero. To have all data peaks returned, set
        nMaxNumberOfPeaks to zero.

        To have profile scans centroided, set bCentroidResult to True. This parameter is ignored for
        centroid scans.

        To get a range of masses between two points that are returned in the mass list, set the string of
        szMassRange1 to a valid range (e.g. "450.00-640.00").

        The pvarPeakFlags variable is currently not used. This variable is reserved for future use to
        return flag information, such as saturation, about each mass intensity pair.

        NOTE: same as GetMassListFromScanNum but is able to filter on a m/z range."""
        if not massRange: 
            massRange = "{}-{}".format(self.LowMass, self.HighMass)
        peakList = comtypes.automation.VARIANT()
        peakFlags = comtypes.automation.VARIANT() 
        pnArraySize = c_long()
        error = self.source.GetMassListRangeFromScanNum(c_long(scanNumber), filter, intensityCutoffType, 
            intensityCutoffValue, maxNumberOfPeaks, centroidResult  ,c_double(centroidPeakWidth) ,peakList, peakFlags, massRange, byref(pnArraySize))
        if error : raise IOError("GetMassListRangeFromScanNum error :", error)
        return peakList.value, peakFlags.value

    def GetSegmentedMassListFromScanNum(self, scanNumber,
                                            filter = "",
                                            intensityCutoffType = 0,
                                            intensityCutoffValue = 0,
                                            maxNumberOfPeaks = 0,
                                            centroidResult = False,
                                            centroidPeakWidth = 0.0):
        """This function is only applicable to scanning devices such as MS.
        
        If no scan filter is supplied, the scan corresponding to pnScanNumber is returned. If a scan
        filter is provided, the closest matching scan to pnScanNumber that matches the scan filter is
        returned.
        Scan filters must match the Xcalibur scan filter format (e.g. "FTMS + c NSI Full ms [300.00-1800.00]"). 
        
        To reduce the number of low intensity data peaks returned, an intensity cutoff,
        nIntensityCutoffType, may be applied. The available types of cutoff are 
        0   None (all values returned)
        1   Absolute (in intensity units)
        2   Relative (to base peak)
        
        To limit the total number of data peaks that are returned in the mass list, set
        nMaxNumberOfPeaks to a value greater than zero. To have all data peaks returned, set
        nMaxNumberOfPeaks to zero.

        To have profile scans centroided, set bCentroidResult to True. This parameter is ignored for
        centroid scans.

        The pvarPeakFlags variable is currently not used. This variable is reserved for future use to
        return flag information, such as saturation, about each mass intensity pair.
        
        The varSegments array contains information about the segments, and the varMassRange array
        contains the mass range for each segment. The nSegments variable contains the number of
        segments.
        
        NOTE: same as GetMassListFromScanNum but is able to filter on a m/z range."""                    
        peakList = comtypes.automation.VARIANT()
        peakFlags = comtypes.automation.VARIANT() 
        pnArraySize = c_long()
        pvarSegments = comtypes.automation.VARIANT() 
        pnNumSegments = c_long()
        massRange = comtypes.automation.VARIANT() 
        error = self.source.GetSegmentedMassListFromScanNum(byref(c_long(scanNumber)), 
                                                            filter, 
                                                            intensityCutoffType, 
                                                            intensityCutoffValue, 
                                                            maxNumberOfPeaks, 
                                                            centroidResult,
                                                            c_double(centroidPeakWidth),
                                                            peakList, 
                                                            peakFlags, 
                                                            byref(pnArraySize),
                                                            byref(pvarSegments),
                                                            byref(pnNumSegments),
                                                            byref(massRange))
        if error : raise IOError("GetSegmentedMassListFromScanNum error :", error)
        return peakList.value, peakFlags.value, pvarSegments.value, pnNumSegments.value
        
    def GetAverageMassList(self, firstAvgScanNumber,
                                lastAvgScanNumber,
                                firstBkg1ScanNumber = 0,
                                lastBkg1ScanNumber = 0,
                                firstBkg2ScanNumber = 0,
                                lastBkg2ScanNumber = 0,
                                filter = "",
                                intensityCutoffType = 0,
                                intensityCutoffValue = 0,
                                maxNumberOfPeaks = 0,
                                centroidResult = False,
                                centroidPeakWidth = 0.0):
        """
        This function is only applicable to scanning devices such as MS and PDA.
        
        If no scan filter is supplied, the scans between firstAvgScanNumber and
        lastAvgScanNumber that match the filter of the firstAvgScanNumber, inclusive, are
        returned. Likewise, all the scans between firstBkg1ScanNumber and
        lastBkg1ScanNumber and firstBkg2ScanNumber and lastBkg2ScanNumber, inclusive,
        are averaged and subtracted from the firstAvgScanNumber to lastAvgScanNumber
        averaged scans. 
        If a scan filter is provided, the scans in the preceding scan number ranges that
        match the scan filter are utilized in obtaining the background subtracted mass list. The
        specified scan numbers must be valid for the current controller. If no background subtraction
        is performed, the background scan numbers should be set to zero. On return, the scan
        number variables contain the actual first and last scan numbers, respectively, for the scans
        used.
        
        Scan filters must match the Xcalibur scan filter format (e.g. "FTMS + c NSI Full ms [300.00-1800.00]"). 
        
        To reduce the number of low intensity data peaks returned, an intensity cutoff,
        nIntensityCutoffType, may be applied. The available types of cutoff are 
        0   None (all values returned)
        1   Absolute (in intensity units)
        2   Relative (to base peak)
        
        To limit the total number of data peaks that are returned in the mass list, set
        nMaxNumberOfPeaks to a value greater than zero. To have all data peaks returned, set
        nMaxNumberOfPeaks to zero.

        To have profile scans centroided, set bCentroidResult to True. This parameter is ignored for
        centroid scans.

        The format of the mass list returned is an array of double precision values in 
        mass intensity pairs in ascending mass order, 
        for example, mass 1, intensity 1, mass 2, intensity 2, mass 3, intensity 3.
        
        The pvarPeakFlags variable is currently not used. This variable is reserved for future use to
        return flag information, such as saturation, about each mass intensity pair.
        """
        peakList = comtypes.automation.VARIANT()
        peakFlags = comtypes.automation.VARIANT() 
        pnArraySize = c_long()
        error = self.source.GetAverageMassList( byref(c_long(firstAvgScanNumber)),
                                                byref(c_long(lastAvgScanNumber)),
                                                byref(c_long(firstBkg1ScanNumber)),
                                                byref(c_long(lastBkg1ScanNumber)),
                                                byref(c_long(firstBkg2ScanNumber)),
                                                byref(c_long(lastBkg2ScanNumber)),
                                                filter,
                                                intensityCutoffType,
                                                intensityCutoffValue,
                                                maxNumberOfPeaks,
                                                centroidResult,
                                                c_double(centroidPeakWidth),
                                                byref(peakList),
                                                byref(peakFlags),
                                                byref(pnArraySize))
        if error : raise IOError("GetAverageMassList error :", error)
        return peakList.value, peakFlags.value
        
    def GetAveragedMassSpectrum(self, listOfScanNumbers, centroidResult = False):
        """
        This function is only applicable to scanning devices such as MS.
        
        Returns the average spectrum for the list of scans that are supplied
        to the function in listOfScanNumbers. If no scans are provided in listOfScanNumbers,
        then the function returns an error code.
        
        If the bCentroidData value is true, profile data is centroided before it is returned by this
        routine.
        
        The format of the mass list returned is an array of double precision values in 
        mass intensity pairs in ascending mass order, 
        for example, mass 1, intensity 1, mass 2, intensity 2, mass 3, intensity 3.
        
        The pvarPeakFlags variable is currently not used. This variable is reserved for future use to 
        return flag information, such as saturation, about each mass intensity pair
        """
        # http://stackoverflow.com/questions/1363163/pointers-and-arrays-in-python-ctypes
        x = (c_long*len(listOfScanNumbers))()
        cast(x, POINTER(c_long))
        for i,scanNumber in enumerate(listOfScanNumbers):
            x[i] = scanNumber

        peakList = comtypes.automation.VARIANT()
        peakFlags = comtypes.automation.VARIANT() 
        pnArraySize = c_long()
        error = self.source.GetAveragedMassSpectrum( x,
                                                len(x),
                                                centroidResult,
                                                byref(peakList),
                                                byref(peakFlags),
                                                byref(pnArraySize))
        if error : raise IOError("GetAveragedMassSpectrum error :", error)
        return peakList.value, peakFlags.value
        
    def GetSummedMassSpectrum(self, listOfScanNumbers, centroidResult = False):
        """
        This function is only applicable to scanning devices such as MS.
        
        Returns the summed spectrum for the list of scans that are supplied
        to the function in listOfScanNumbers. If no scans are provided in listOfScanNumbers,
        then the function returns an error code.
        
        If the bCentroidData value is true, profile data is centroided before it is returned by this
        routine.

        The format of the mass list returned is an array of double precision values in 
        mass intensity pairs in ascending mass order, 
        for example, mass 1, intensity 1, mass 2, intensity 2, mass 3, intensity 3.
        
        The pvarPeakFlags variable is currently not used. This variable is reserved for future use to 
        return flag information, such as saturation, about each mass intensity pair
        
        NOTE: seems to return same output that GetAveragedMassSpectrum...
        """
        # http://stackoverflow.com/questions/1363163/pointers-and-arrays-in-python-ctypes
        x = (c_long*len(listOfScanNumbers))()
        cast(x, POINTER(c_long))
        for i,scanNumber in enumerate(listOfScanNumbers):
            x[i] = scanNumber

        peakList = comtypes.automation.VARIANT()
        peakFlags = comtypes.automation.VARIANT() 
        pnArraySize = c_long()
        error = self.source.GetSummedMassSpectrum( x,
                                                len(x),
                                                centroidResult,
                                                byref(peakList),
                                                byref(peakFlags),
                                                byref(pnArraySize))
        if error : raise IOError("GetSummedMassSpectrum error :", error)
        return peakList.value, peakFlags.value
        
        
    def IncludeReferenceAndExceptionData(self, boolean):
        """Controls whether the reference and exception data is included in the spectral data when using
        the GetLabelData method. Reference and exception peaks are only present on instruments
        that can collect FTMS data. A value of TRUE causes the reference and exception data to be
        included in the spectrum, and a value of FALSE excludes this data.
        """
        error = self.source.IncludeReferenceAndExceptionData(c_bool(boolean))
        if error : raise IOError("IncludeReferenceAndExceptionData error :", error)
        return
        
    def GetLabelData(self, scanNumber): # This is a higher level function than GetMassListRangeFromScanNum, only for profile data (MS1).
        """This method enables you to read the FT-PROFILE labels of a scan represented by the scanNumber.
        The label data contains values of :
        mass (double), 
        intensity (double), 
        resolution (float), 
        baseline (float), 
        noise (float) 
        and charge (int).
        
        The flags are returned as unsigned char values. The flags are :
        saturated, 
        fragmented, 
        merged, 
        exception, 
        reference, 
        and modified.
        
        NOTE : This is a higher level function than GetMassListRangeFromScanNum, only for profile data (MS1).
        """
        pvarLabels = comtypes.automation.VARIANT()
        pvarFlags = comtypes.automation.VARIANT()
        error = self.source.GetLabelData(pvarLabels,pvarFlags,c_long(scanNumber))
        if error : raise IOError("GetLabelData error :", error)
        Labels = namedtuple('Labels', 'mass intensity resolution baseline noise charge')
        Flags = namedtuple('Flags', 'saturated fragmented merged exception reference modified')
        return Labels(*pvarLabels.value) , Flags(*pvarFlags.value)
        
    def GetAveragedLabelData(self, listOfScanNumbers):
        """This method enables you to read the averaged FT-PROFILE labels for the list of scans 
        represented by the listOfScanNumbers. If no scans are provided in listOfScanNumbers,
        the function returns an error code.
        
        The format of the mass list returned is an array of double precision values in 
        mass intensity pairs in ascending mass order, 
        for example, mass 1, intensity 1, mass 2, intensity 2, mass 3, intensity 3.
         
        The flags are returned as unsigned char values. 
        These flags are saturated, fragmented, merged, exception, reference, and modified.
        """
        x = (c_long*len(listOfScanNumbers))()
        cast(x, POINTER(c_long))
        for i,scanNumber in enumerate(listOfScanNumbers):
            x[i] = scanNumber

        peakList = comtypes.automation.VARIANT()
        peakFlags = comtypes.automation.VARIANT() 
        pnArraySize = c_long()
        error = self.source.GetAveragedLabelData( x,
                                                len(x),
                                                byref(peakList),
                                                byref(peakFlags),
                                                byref(pnArraySize))
        if error : raise IOError("GetAveragedLabelData error :", error)
        Flags = namedtuple('Flags', 'saturated fragmented merged exception reference modified')
        return peakList.value, Flags(*peakFlags.value)
        
    def GetNoiseData(self, scanNumber): # already included in GetLabelData ?
        """This method enables you to read the FT-PROFILE noise packets of a scan represented by the scanNumber.
        The noise packets contain values of mass (double), noise (float) and baseline (float).
        NOTE: already included in GetLabelData ?
        """
        NoisePacket = comtypes.automation.VARIANT()
        error = self.source.GetNoiseData(NoisePacket,c_long(scanNumber))
        if error : raise IOError("GetNoiseData error :", error)
        return NoisePacket.value
        
    def GetAllMSOrderData(self, scanNumber): # same as GetLabelData ?
        """This method enables you to obtain all of the precursor information from the scan (event).
        
        The FT-PROFILE labels of a scan are represented by scanNumber. PvarFlags can be NULL
        if you do not want to receive the flags. The label data contains values of mass (double),
        intensity (double), resolution (float), baseline (float), noise (float), and charge (int). 
        The flags are returned as unsigned character values. The flags are saturated, fragmented, 
        merged, exception, reference, and modified."""
        pvarLabels = comtypes.automation.VARIANT()
        pvarFlags = comtypes.automation.VARIANT()
        pnNumberOfMSOrders = c_long()
        error = self.source.GetAllMSOrderData(scanNumber,pvarLabels, pvarFlags, byref(pnNumberOfMSOrders) )
        Labels = namedtuple('Labels', 'mass intensity resolution baseline noise charge')
        Flags = namedtuple('Flags', 'activation_type is_precursor_range_valid')
        return Labels(*pvarLabels.value) , Flags(*pvarFlags.value), pnNumberOfMSOrders.value
    
    def GetFullMSOrderPrecursorDataFromScanNum(self, scanNumber, MSOrder):
        """This function retrieves information about the reaction data of a data-dependent MSn for the
        scan specified by scanNumber and the transition specified by MSOrder from the scan event
        structure in the raw file.
        
        - Reaction data refers to precursor mass, isolation width, collision energy, whether the
        collision energy is valid, whether the precursor mass is valid, the first precursor mass, the
        last precursor mass, and the isolation width offset.
        
        - Specify the data-dependent MSn through the MSOrder input. You can find the count of
        MS orders by calling GetNumberOfMSOrdersFromScanNum.
        
        - Specify the scan through the scanNumber input. The value of scanNumber must be
        within the range of scans or readings for the current controller. You can obtain the range
        of scans or readings for the current controller by calling GetFirstSpectrumNumber and
        GetLastSpectrumNumber.
        """
        pvarFullMSOrderPrecursorInfo = comtypes.automation.VARIANT()
        error = self.source.GetFullMSOrderPrecursorDataFromScanNum(scanNumber, MSOrder, byref(pvarFullMSOrderPrecursorInfo) )
        if error : raise IOError("GetFullMSOrderPrecursorDataFromScanNum error :", error)
        FullMSOrderPrecursorData = namedtuple('FullMSOrderPrecursorData', 'precursorMass isolationWidth collisionEnergy collisionEnergyValid rangeIsValid firstPrecursorMass lastPrecursorMass isolationWidthOffset')
        return FullMSOrderPrecursorData(*pvarFullMSOrderPrecursorInfo.value[:8])
            
    def GetMSOrderForScanNum(self, scanNumber):
        """This function returns the MS order for the scan specified by scanNumber from the scan
        event structure in the raw file.
        The value returned in the pnScanType variable is one of the following:
        Neutral gain -3
        Neutral loss -2
        Parent scan -1
        Any scan order 0
        MS  1
        MS2  2
        MS3  3
        MS4  4
        MS5  5
        MS6  6
        MS7  7
        MS8  8
        MS9  9
        """
        MSOrder = c_long()
        error = self.source.GetMSOrderForScanNum(c_long(scanNumber),byref(MSOrder))
        if error : raise IOError( "scan {} : GetMSOrderForScanNum error : {}".format(scanNumber,error) )
        return MSOrder.value
        
    def GetNumberOfMSOrdersFromScanNum(self, scanNumber):
        """This function gets the number of MS reaction data items in the scan event for the scan
        specified by scanNumber and the transition specified by MSOrder from the scan event
        structure in the raw file."""
        result = c_long()
        error = self.source.GetNumberOfMSOrdersFromScanNum(c_long(scanNumber), byref(result) )
        if error : raise IOError("GetNumberOfMSOrdersFromScanNum error :", error)
        return result.value
        
        
    # # # # # # # # # # # # # PRECURSOR BEGIN
    def GetPrecursorInfoFromScanNum(self, scanNumber):
        """This function is used to retrieve information about the parent scans of a data-dependent MS n
        scan.
        You retrieve the scan number of the parent scan, the isolation mass used, the charge state, and
        the monoisotopic mass as determined by the instrument firmware. You will obtain access to
        the scan data of the parent scan in the form of a XSpectrumRead object.
        Further refine the charge state and the monoisotopic mass values from the actual parent scan
        data.
        NOTE : !!! VARIANT conversion to struct does not work : we only retrieve dIsolationMass and dMonoIsoMass AND this does not work with Qexactive files...
        """
        # struct PrecursorInfo
        # {
        #       double dIsolationMass;
        #       double dMonoIsoMass;
        #       long nChargeState;
        #       long scanNumber;
        # }
        # http://www.codeproject.com/Articles/6462/A-simple-class-to-encapsulate-VARIANTs
        # http://digital.ni.com/public.nsf/3efedde4322fef19862567740067f3cc/c5834a84795dfa86862565fc0079baf5/$FILE/Information%20on%20Variants%20and%20SafeArrays.doc
        # http://www.roblocher.com/whitepapers/oletypes.html
        variant = comtypes.automation.VARIANT()
        pnArraySize = c_long()
        error = self.source.GetPrecursorInfoFromScanNum(scanNumber, variant, byref(pnArraySize) )
        if error : raise IOError("GetPrecursorInfoFromScanNum error :", error)
        if variant.value:
            variant.vt = comtypes.automation.VT_ARRAY | comtypes.automation.VT_R8 # SAFEARRAY of double
            dMonoIsoMass, dIsolationMass = variant.value[:2]
            variant.vt = comtypes.automation.VT_ARRAY | comtypes.automation.VT_I4 # SAFEARRAY of long
            nChargeState, nScanNumber = variant.value[4:6]
            PrecursorInfo = namedtuple('PrecursorInfo', 'isolationMass monoIsoMass chargeState scanNumber')
            return PrecursorInfo(isolationMass=dIsolationMass, monoIsoMass=dMonoIsoMass, chargeState=nChargeState, scanNumber=nScanNumber)
        else:
            return
        
    def GetPrecursorMassForScanNum(self, scanNumber, MSOrder):
        """This function returns the precursor mass for the scan specified by scanNumber and the
        transition specified by MSOrder from the scan event structure in the RAW file."""
        precursorMass = c_double()
        error = self.source.GetPrecursorMassForScanNum(scanNumber, MSOrder, byref(precursorMass))
        if error : raise IOError( "scan {} : GetPrecursorMassForScanNum error : {}".format(scanNumber,error) )
        return precursorMass.value
        
    def GetPrecursorRangeForScanNum(self, scanNumber, MSOrder):
        """This function returns the first and last precursor mass values of the range and whether they are valid for the scan specified by scanNumber and the transition specified by MSOrder from the scan event structure in the raw file."""
        pdFirstPrecursorMass = c_double()
        pdLastPrecursorMass = c_double()
        pbIsValid = c_long()
        error = self.source.GetPrecursorRangeForScanNum(c_long(scanNumber), c_long(MSOrder), byref(pdFirstPrecursorMass), byref(pdLastPrecursorMass), byref(pbIsValid) )
        if error : raise IOError("GetPrecursorRangeForScanNum error :", error)
        return pdFirstPrecursorMass.value, pdLastPrecursorMass.value, bool(pbIsValid.value)
    # # # # # # # # # # # # # PRECURSOR END


    ############################################### XCALIBUR INTERFACE BEGIN
    def GetScanHeaderInfoForScanNum(self, scanNumber): # "View/Scan header", upper part
        """For a given scan number, this function returns information from the scan header for the
        current controller.

        The validity of these parameters depends on the current controller. For example, pdLowMass,
        pdHighMass, pdTIC, pdBasePeakMass, and pdBasePeakIntensity are only likely to be set on
        return for MS or PDA controllers. PnNumChannels is only likely to be set on return for
        Analog, UV, and A/D Card controllers. PdUniformTime, and pdFrequency are only likely to be
        set on return for UV, and A/D Card controllers and may be valid for Analog controllers. In
        cases where the value is not set, a value of zero is returned.
        
        NOTE : XCALIBUR INTERFACE "View/Scan header", upper part
        """
        header = OrderedDict()
        header['numPackets'] = c_long()
        header['StartTime'] = c_double()
        header['LowMass'] = c_double()
        header['HighMass'] = c_double()
        header['TIC'] = c_double()
        header['BasePeakMass'] = c_double()
        header['BasePeakIntensity'] = c_double()
        header['numChannels'] = c_long()
        header['uniformTime'] = c_long()
        header['Frequency'] = c_double()
        error = self.source.GetScanHeaderInfoForScanNum(c_long(scanNumber), header['numPackets'], header['StartTime'], header['LowMass'], header['HighMass'], 
                        header['TIC'], header['BasePeakMass'], header['BasePeakIntensity'], header['numChannels'], header['uniformTime'], header['Frequency'])
        if error : raise IOError("GetScanHeaderInfoForScanNum error :", error)
        for k in header:
            header[k] = header[k].value
        return header
        
    def GetTrailerExtraForScanNum(self, scanNumber): # "View/Scan header", lower part
        """Returns the recorded trailer extra entry labels and values for the current controller. This
        function is only valid for MS controllers.
        
        NOTE : XCALIBUR INTERFACE "View/Scan header", lower part
        """
        labels = comtypes.automation.VARIANT()
        values = comtypes.automation.VARIANT()
        val_num = c_long()
        error = self.source.GetTrailerExtraForScanNum(c_long(scanNumber),labels,values,val_num)
        if error : raise IOError("GetTrailerExtraForScanNum error :", error)
        return OrderedDict(zip( map(lambda x: str(x[:-1]), labels.value) , map(_to_float, values.value) ))
        
    def GetNumTuneData(self):
        """Gets the number of tune data entries recorded for the current controller. Tune Data is only
        supported by MS controllers. Typically, if there is more than one tune data entry, each tune
        data entry corresponds to a particular acquisition segment."""
        pnNumTuneData = c_long()
        error = self.source.GetNumTuneData(byref(pnNumTuneData) )
        if error : raise IOError("GetNumTuneData error :", error)
        return pnNumTuneData.value
        
    def GetTuneData(self, segmentNumber = 1): # "View/Report/Tune Method"
        """Returns the recorded tune parameter labels and values for the current controller. This
        function is only valid for MS controllers. The value of segmentNumber must be within the
        range of one to the number of tune data items recorded for the current controller. The
        number of tune data items for the current controller may be obtained by calling
        GetNumTuneData.
        
        NOTE : XCALIBUR INTERFACE "View/Report/Tune Method"
        """
        pvarLabels = comtypes.automation.VARIANT() 
        pvarValues = comtypes.automation.VARIANT() 
        pnArraySize = c_long()
        error = self.source.GetTuneData(c_long(segmentNumber),pvarLabels,pvarValues,byref(pnArraySize) )
        if error : raise IOError("GetTuneData error :", error)
        # return dict(zip(pvarLabels.value,pvarValues.value))
        # return dict(zip([label.rstrip(':') for label in pvarLabels.value],pvarValues.value))
        result = []
        for label, value in zip(pvarLabels.value,pvarValues.value):
            result.append(label)
            result.append(value)
            result.append('\n')
        return ''.join(result)
        
    def GetNumInstMethods(self):
        """Returns the number of instrument methods contained in the raw file. Each instrument used
        in the acquisition with a method that was created in Instrument Setup (for example,
        autosampler, LC, MS, PDA) has its instrument method contained in the raw file."""
        pnNumInstMethods = c_long()
        error = self.source.GetNumInstMethods(byref(pnNumInstMethods) )
        if error : raise IOError("GetNumInstMethods error : ", error) 
        return pnNumInstMethods.value
        
    def GetInstMethod(self, instMethodItem = 0): # "View/Report/Instrument Method"
        """Returns the channel label, if available, at the specified index for the current controller. This
        field is only relevant to channel devices such as UV detectors, A/D cards, and Analog inputs.
        Channel labels indices are numbered starting at 0.
        Returns the instrument method, if available, at the index specified in instMethodItem. The
        instrument method indices are numbered starting at 0. The number of instrument methods
        are obtained by calling GetNumInstMethods.
        
        instMethodItem     A long variable containing the index value of the instrument method to
                            be returned.
        
        NOTE : XCALIBUR INTERFACE "View/Report/Instrument Method"
        """
        strInstMethod = comtypes.automation.BSTR()
        error = self.source.GetInstMethod(c_long(instMethodItem), byref(strInstMethod) )
        if error : raise IOError ("GetInstMethod error :", error)
        if sys.version_info.major == 2:
            return strInstMethod.value.encode('utf-8').replace('\r\n','\n')
        elif sys.version_info.major == 3:
            return strInstMethod.value.replace('\r\n','\n')
        
    def GetInstMethodNames(self):
        """This function returns the recorded names of the instrument methods for the current
        controller."""
        pnArraySize = c_long(0)
        pvarNames = comtypes.automation.VARIANT()
        error = self.source.GetInstMethodNames( byref(pnArraySize), byref(pvarNames) )
        if error : raise IOError("GetInstMethodNames error :", error)
        return pvarNames.value
        
    def ExtractInstMethodFromRaw(self,instMethodFileName):
        """This function enables you to save the embedded instrument method in the raw file in a
        separated method (.meth)) file. It overwrites any pre-existing method file in the same path
        with the same name."""
        error = self.source.ExtractInstMethodFromRaw(comtypes.automation.BSTR(instMethodFileName))
        if error : raise IOError("ExtractInstMethodFromRaw error :", error)
        return
    
    # # # # # # # "View/Report/Sample Information" BEGIN
    def GetVialNumber(self):
        """Gets the vial number for the current controller. This value is typically only set for raw files
        converted from other file formats.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_long()
        error = self.source.GetVialNumber(byref(result) )
        if error : raise IOError("GetVialNumber error : ", error) 
        return result.value
        
    def GetInjectionVolume(self):
        """Gets the injection volume for the current controller. This value is typically only set for raw
        files converted from other file formats.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_double()
        error = self.source.GetInjectionVolume(byref(result) )
        if error : raise IOError("GetInjectionVolume error : ", error) 
        return result.value
        
    def GetInjectionAmountUnits(self):
        """Returns the injection amount units for the current controller. This value is typically only set
        for raw files converted from other file formats.  
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetInjectionAmountUnits(byref(result) )
        if error : raise IOError("GetInjectionAmountUnits error : ", error) 
        return result.value
        
    def GetSampleVolume(self):
        """Gets the sample volume value for the current controller. This value is typically only set for raw files converted from other file formats.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_double()
        error = self.source.GetSampleVolume(byref(result) )
        if error : raise IOError("GetSampleVolume error : ", error) 
        return result.value
        
    def GetSampleVolumeUnits(self):
        """Returns the sample volume units for the current controller. This value is typically only set for raw files converted from other file formats.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSampleVolumeUnits(byref(result) )
        if error : raise IOError("GetSampleVolumeUnits error : ", error) 
        return result.value
        
    def GetSampleWeight(self):
        """Gets the sample weight value for the current controller. This value is typically only set for raw files converted from other file formats.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_double()
        error = self.source.GetSampleWeight(byref(result) )
        if error : raise IOError("GetSampleWeight error : ", error) 
        return result.value
        
    def GetSampleAmountUnits(self):
        """Returns the sample amount units for the current controller. This value is typically only set for raw files converted from other file formats.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSampleAmountUnits(byref(result) )
        if error : raise IOError("GetSampleAmountUnits error : ", error) 
        return result.value
    
    def GetSeqRowNumber(self):
        """Returns the sequence row number for this sample in an acquired sequence. The numbering
        starts at 1.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_long()
        error = self.source.GetSeqRowNumber(byref(result) )
        if error : raise IOError("GetSeqRowNumber error : ", error) 
        return result.value
        
    def GetSeqRowSampleType(self):
        """Returns the sequence row sample type for this sample. See Sample Type in the Enumerated
        Types section for the possible sample type values.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_long()
        error = self.source.GetSeqRowSampleType(byref(result) )
        if error : raise IOError("GetSeqRowSampleType error : ", error) 
        return ThermoRawfile.sampleType[result.value]
        
    def GetSeqRowDataPath(self):
        """Returns the path of the directory where this raw file was acquired.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSampleAmountUnits(byref(result) )
        if error : raise IOError("GetSampleAmountUnits error : ", error) 
        return result.value
        
    def GetSeqRowRawFileName(self):
        """Returns the file name of the raw file when the raw file was acquired. This value is typically
        used in conjunction with GetSeqRowDataPath to obtain the fully qualified path name of the
        raw file when it was acquired.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowRawFileName(byref(result) )
        if error : raise IOError("GetSeqRowRawFileName error : ", error) 
        return result.value
        
    def GetSeqRowSampleName(self):
        """Returns the sample name value from the sequence row of the raw file.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowSampleName(byref(result) )
        if error : raise IOError("GetSeqRowSampleName error : ", error) 
        return result.value
        
    def GetSeqRowSampleID(self):
        """Returns the sample ID value from the sequence row of the raw file.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowSampleID(byref(result) )
        if error : raise IOError("GetSeqRowSampleID error : ", error) 
        return result.value
        
    def GetSeqRowComment(self):
        """Returns the comment field from the sequence row of the raw file.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowComment(byref(result) )
        if error : raise IOError("GetSeqRowComment error : ", error) 
        return result.value
        
    def GetSeqRowLevelName(self):
        """Returns the level name from the sequence row of the raw file. This field is empty except for
        standard and QC sample types, which may contain a value if a processing method was
        specified in the sequence at the time of acquisition.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowLevelName(byref(result) )
        if error : raise IOError("GetSeqRowLevelName error : ", error) 
        return result.value
        
    def GetSeqRowUserText(self, index = 0):
        """Returns a user text field from the sequence row of the raw file. There are five user text fields in
        the sequence row that are indexed 0 through 4.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowUserText(c_long(index), byref(result) )
        if error : raise IOError("GetSeqRowUserText error : ", error) 
        return result.value
        
    def GetSeqRowInstrumentMethod(self):
        """Returns the fully qualified path name of the instrument method used to acquire the raw file.
        If the raw file is created by file format conversion or acquired from a tuning program, this
        field is empty.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowInstrumentMethod(byref(result) )
        if error : raise IOError("GetSeqRowInstrumentMethod error : ", error) 
        return result.value
        
    def GetSeqRowProcessingMethod(self):
        """Returns the fully qualified path name of the processing method specified in the sequence used
        to acquire the raw file. If no processing method is specified at the time of acquisition, this field is empty.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowProcessingMethod(byref(result) )
        if error : raise IOError("GetSeqRowProcessingMethod error : ", error) 
        return result.value
        
    def GetSeqRowCalibrationFile(self):
        """Returns the fully qualified path name of the calibration file specified in the sequence used to
        acquire the raw file. If no calibration file is specified at the time of acquisition, this field is empty.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowCalibrationFile(byref(result) )
        if error : raise IOError("GetSeqRowCalibrationFile error : ", error) 
        return result.value
        
    def GetSeqRowVial(self):
        """Returns the vial or well number of the sample when it was acquired. If the raw file is not
        acquired using an autosampler, this value should be ignored.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowVial(byref(result) )
        if error : raise IOError("GetSeqRowVial error : ", error) 
        return result.value
        
    def GetSeqRowInjectionVolume(self):
        """Returns the autosampler injection volume from the sequence row for this sample.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_double()
        error = self.source.GetSeqRowInjectionVolume(byref(result) )
        if error : raise IOError("GetSeqRowInjectionVolume error : ", error) 
        return result.value
        
    def GetSeqRowSampleWeight(self):
        """Returns the sample weight from the sequence row for this sample.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_double()
        error = self.source.GetSeqRowSampleWeight(byref(result) )
        if error : raise IOError("GetSeqRowSampleWeight error : ", error) 
        return result.value
        
    def GetSeqRowSampleVolume(self):
        """Returns the sample volume from the sequence row for this sample.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_double()
        error = self.source.GetSeqRowSampleVolume(byref(result) )
        if error : raise IOError("GetSeqRowSampleVolume error : ", error) 
        return result.value
        
    def GetSeqRowISTDAmount(self):
        """Returns the bulk ISTD correction amount from the sequence row for this sample.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_double()
        error = self.source.GetSeqRowISTDAmount(byref(result) )
        if error : raise IOError("GetSeqRowISTDAmount error : ", error) 
        return result.value
        
    def GetSeqRowDilutionFactor(self):
        """Returns the bulk dilution factor (volume correction) from the sequence row for this sample.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_double()
        error = self.source.GetSeqRowDilutionFactor(byref(result) )
        if error : raise IOError("GetSeqRowDilutionFactor error : ", error) 
        return result.value
        
    def GetSeqRowUserLabel(self, index = 0):
        """Returns a user label field from the sequence row of the raw file. 
        There are five user label fields in the sequence row that are indexed 0 through 4. 
        The user label fields correspond one-to-one with the user text fields.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowUserLabel(c_long(index), byref(result) )
        if error : raise IOError("GetSeqRowUserLabel error : ", error) 
        return result.value
        
    def GetSeqRowUserTextEx(self, index = 0):
        """This function returns a user text field from the sequence row of the raw file. 
        There are five user text fields in the sequence row that are indexed 0 through 4.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowUserTextEx(c_long(index), byref(result) )
        if error : raise IOError("GetSeqRowUserTextEx error : ", error) 
        return result.value
    
    def GetSeqRowBarcode(self):
        """This function returns the barcode used to acquire the raw file. 
        This field is empty if the raw file was created by file format 
        conversion or acquired from a tuning program.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = comtypes.automation.BSTR()
        error = self.source.GetSeqRowBarcode(byref(result) )
        if error : raise IOError("GetSeqRowBarcode error : ", error) 
        return result.value
    
    def GetSeqRowBarcodeStatus(self):
        """This function returns the barcode status from the raw file. 
        This field is empty if the raw file was created by file format 
        conversion or acquired from a tuning program.
        NOTE : XCALIBUR INTERFACE "View/Report/Sample Information" part
        """
        result = c_long()
        error = self.source.GetSeqRowBarcodeStatus(byref(result) )
        if error : raise IOError("GetSeqRowBarcodeStatus error : ", error) 
        return result.value
    # # # # # # # "View/Report/Sample Information" END
    
    def GetNumStatusLog(self):
        """Gets the number of status log entries recorded for the current controller."""
        pnNumberOfStatusLogEntries = c_long()
        error = self.source.GetNumStatusLog(byref(pnNumberOfStatusLogEntries))
        if error : raise IOError("GetNumStatusLog error :", error)
        return pnNumberOfStatusLogEntries.value
        
    def GetStatusLogForScanNum(self, scanNumber): # "View/Report/Status Log"
        """Returns the recorded status log entry labels and values for the current controller.
        On return, pdStatusLogRT contains the retention time when the status log entry was recorded.
        This time may not be the same as the retention time corresponding to the specified scan
        number but is the closest status log entry to the scan time.
        NOTE : XCALIBUR INTERFACE "View/Report/Status Log"
        """
        pdStatusLogRT = c_double()
        pvarLabels = comtypes.automation.VARIANT()
        pvarValues = comtypes.automation.VARIANT()
        pnArraySize = c_long()
        error = self.source.GetStatusLogForScanNum(c_long(scanNumber), byref(pdStatusLogRT), pvarLabels, pvarValues, byref(pnArraySize) ) 
        if error : raise IOError("GetStatusLogForScanNum error :", error)
        # return pdStatusLogRT.value, pvarLabels.value, pvarValues.value
        return pdStatusLogRT.value, list(zip([label.rstrip(':') for label in pvarLabels.value],pvarValues.value))
        
    def GetStatusLogForPos(self, position = 0):
        """This function returns the recorded status log entry labels and values for the current controller.
        position    The position that the status log information is to be returned for.
        """
        pvarRT = comtypes.automation.VARIANT()
        pvarValues = comtypes.automation.VARIANT()
        pnArraySize = c_long()
        error = self.source.GetStatusLogForPos(c_long(position), byref(pvarRT), byref(pvarValues), byref(pnArraySize) ) 
        if error : raise IOError("GetStatusLogForPos error :", error)
        return pvarRT.value, pvarValues.value
        
    def GetStatusLogPlottableIndex(self):
        """This function returns the recorded status log entry labels and values for the current controller."""
        pvarIndex = comtypes.automation.VARIANT()
        pvarValues = comtypes.automation.VARIANT()
        pnArraySize = c_long()
        error = self.source.GetStatusLogPlottableIndex(byref(pvarIndex), byref(pvarValues), byref(pnArraySize) ) 
        if error : raise IOError("GetStatusLogPlottableIndex error :", error)
        return pvarIndex.value, pvarValues.value
        
    def GetNumErrorLog(self):
        """Gets the number of error log entries recorded for the current controller."""
        pnNumberOfErrorLogEntries = c_long()
        error = self.source.GetNumErrorLog(byref(pnNumberOfErrorLogEntries))
        if error : raise IOError("GetNumErrorLog error :", error)
        return pnNumberOfErrorLogEntries.value
        
    def GetErrorLogItem(self, itemNumber = 1): # "View/Report/Error Log"
        """Returns the specified error log item information and the retention time when the error
        occurred. The value of itemNumber must be within the range of one to the number of error
        log items recorded for the current controller. The number of error log items for the current
        controller may be obtained by calling GetNumErrorLog.
        
        NOTE : XCALIBUR INTERFACE "View/Report/Error Log"
        """
        pdRT = c_double() # A valid pointer to a variable of type double to receive the retention time when the error occurred. This variable must exist.
        pbstrErrorMessage = comtypes.automation.BSTR()
        error = self.source.GetErrorLogItem(c_long(itemNumber), byref(pdRT), byref(pbstrErrorMessage) )
        if error : raise IOError ("GetErrorLogItem error :", error)
        return pbstrErrorMessage.value, pdRT.value
    ############################################### XCALIBUR INTERFACE END
        
   
    def GetChroData(self, startTime = 0.0, 
                        endTime = 0.0, 
                        massRange1 = "", 
                        massRange2 = "", 
                        filter = "", 
                        chroType1 = 0, 
                        chroOperator = 0, 
                        chroType2 = 0, 
                        delay = 0.0 , 
                        smoothingType = 0, 
                        smoothingValue = 0):
        """Returns the requested chromatogram data as an array of double precision time intensity pairs
        in pvarChroData.

        The start and end times, pdStartTime and pdEndTime, may be used to return a portion of the
        chromatogram. The start time and end time must be within the acquisition time range of the
        current controller which may be obtained by calling GetStartTime and GetEndTime,
        respectively. Or, if the entire chromatogram is returned, pdStartTime and pdEndTime may be
        set to zero. On return, pdStartTime and pdEndTime contain the actual time range of the
        returned chromatographic data.
        
        The mass ranges are only valid for MS or PDA controllers. For all other controller types, these
        fields must be NULL or empty strings. For MS controllers, the mass ranges must be correctly
        formatted mass ranges and are only valid for Mass Range and Base Peak chromatogram trace
        types. For PDA controllers, the mass ranges must be correctly formatted wavelength ranges
        and are only valid for Wavelength Range and Spectrum Maximum chromatogram trace types.
        These values may be left empty for Base Peak or Spectrum Maximum trace types but must be
        specified for Mass Range or Wavelength Range trace types. For information on how to format
        mass ranges, go to the Mass1 (m/z) text box topic in the Xcalibur Help.
        
        The scan filter field is only valid for MS controllers (e.g. "Full ms ").
        
        The chromatogram trace types and operator values of chroType1, chroOperator, and
        chroType2 depend on the current controller. See Chromatogram Type and Chromatogram
        Operator in the Enumerated Types section for a list of the valid values for the different
        controller types.
        
        The delay value contains the retention time offset to add to the returned chromatogram
        times. The value may be set to 0.0 if no offset is desired. This value must be 0.0 for MS
        controllers. It must be greater than or equal to 0.0 for all other controller types.
        
        The smoothingType :
          0 None (no smoothing)
          1 Boxcar http://en.wikipedia.org/wiki/Boxcar_function
          2 Gaussian http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
        The value of smoothingValue must be an odd number in the range of 3-15 if smoothing is desired.
        
        The pvarPeakFlags variable is currently not used. This variable is reserved for future use to
        return flag information, such as saturation, about each time intensity pair.
        
        NOTE: Generates Total Ion Chromatogram (TIC) and eXtracted Ion Chromatogram (XIC) for given time range and mz range.
        """
        pvarChroData = comtypes.automation.VARIANT()
        pvarPeakFlags = comtypes.automation.VARIANT() 
        pnArraySize = c_long()
        error = self.source.GetChroData(chroType1,
                                        chroOperator,
                                        chroType2,
                                        filter,
                                        massRange1,
                                        massRange2,
                                        c_double(delay),
                                        byref(c_double(startTime)),
                                        byref(c_double(endTime)),
                                        smoothingType,
                                        smoothingValue,
                                        pvarChroData,
                                        pvarPeakFlags,
                                        byref(pnArraySize))
        if error : raise IOError("GetChroData error :", error)
        # scan2tic = dict( zip( [self.ScanNumFromRT(rt) for rt in pvarChroData.value[0]], pvarChroData.value[1] ) )
        return pvarChroData.value, pvarPeakFlags.value

    def _GetChros(self): # REDUNDANT with GetChroData
        pass
    
    def GetChroByCompoundName(self, compoundNamesList = None,
                                    startTime = 0.0, 
                                    endTime = 0.0, 
                                    massRanges1 = "", 
                                    massRanges2 = "", 
                                    chroType1 = 0, 
                                    chroOperator = 0, 
                                    chroType2 = 0, 
                                    delay = 0.0, 
                                    smoothingType = 0, 
                                    smoothingValue = 0):
        """
        chroType1      A long variable containing the first chromatogram trace type of interest
        chroOperator   A long variable containing the chromatogram trace operator.
        chroType2      A long variable containing the second chromatogram trace type of interest.
        compoundNamesList (Input) An array of strings containing the compounds to filter the chromatogram with.
        massRanges1   A string containing the formatted mass ranges for the first chromatogram trace type.
        massRanges2   A string containing the formatted mass ranges for the second chromatogram trace type.
        delay          A double-precision variable containing the chromatogram delay in minutes.
        startTime     A pointer to a double-precision variable containing the start time of the chromatogram time range to return.
        endTime       A pointer to a double-precision variable containing the end time of the chromatogram time range to return.
        smoothingType  A long variable containing the type of chromatogram smoothing to be performed.
        smoothingValue A long variable containing the chromatogram smoothing value.
        """
        if not compoundNamesList: compoundNamesList = []
        pvarChroData = comtypes.automation.VARIANT()
        pvarPeakFlags = comtypes.automation.VARIANT()
        pnArraySize = c_long()
        error = self.source.GetChroByCompoundName(   c_long(chroType1), 
                                        c_long(chroOperator), 
                                        c_long(chroType2), 
                                        comtypes.automation.VARIANT(compoundNamesList),
                                        massRanges1,
                                        massRanges2,
                                        c_double(delay),
                                        byref(c_double(startTime)),
                                        byref(c_double(endTime)),
                                        c_long(smoothingType),
                                        c_long(smoothingValue),
                                        pvarChroData, 
                                        pvarPeakFlags, 
                                        byref(pnArraySize) )
        if error : raise IOError("GetChroByCompoundName error :", error)
        return pvarChroDataArray.value, pvarChroParamsArray.value, pvarPeakFlagsArray.value
        
    def GetUniqueCompoundNames(self):
        """This function returns the list of unique compound names for the raw file."""
        pvarCompoundNamesArray = comtypes.automation.VARIANT()
        pnArraySize = c_long()
        error = self.source.GetUniqueCompoundNames(byref(pvarCompoundNamesArray), byref(pnArraySize) )
        if error : raise IOError ("GetUniqueCompoundNames error :", error)
        return pvarCompoundNamesArray.value
    
    def GetCompoundNameFromScanNum(self, scanNumber):
        """This function returns a compound name as a string for the specified scan number."""
        pvarCompoundName = comtypes.automation.BSTR()
        error = self.source.GetCompoundNameFromScanNum(c_long(scanNumber), byref(pvarCompoundName) )
        if error : raise IOError ("GetCompoundNameFromScanNum error :", error)
        return pvarCompoundName.value
        
        
    def _GetStatusLogForRT(self): # REDUNDANT with GetStatusLogForScanNum
        pass
    
    def _GetStatusLogLabelsForScanNum(self): # REDUNDANT with GetStatusLogForScanNum
        pass
        
    def _GetStatusLogLabelsForRT(self): # REDUNDANT with GetStatusLogForScanNum
        pass
        
    def _GetStatusLogValueForScanNum(self): # REDUNDANT with GetStatusLogForScanNum
        pass
        
    def _GetStatusLogValueForRT(self): # REDUNDANT with GetStatusLogForScanNum
        pass
        
    def _GetTrailerExtraForRT(self): # REDUNDANT with GetTrailerExtraForScanNum
        pass
        
    def _GetTrailerExtraLabelsForScanNum(self): # REDUNDANT with GetTrailerExtraForScanNum
        pass
        
    def _GetTrailerExtraLabelsForRT(self): # REDUNDANT with GetTrailerExtraForScanNum
        pass
        
    def _GetTrailerExtraValueForScanNum(self): # REDUNDANT with GetTrailerExtraForScanNum
        pass
        
    def _GetTrailerExtraValueForRT(self): # REDUNDANT with GetTrailerExtraForScanNum
        pass
        
    def _GetTuneDataValue(self): # REDUNDANT with GetTuneData
        pass
        
    def _GetTuneDataLabels(self): # REDUNDANT with GetTuneData
        pass
        
    def _GetPrevMassListRangeFromScanNum(self): # REDUNDANT with GetMassListRangeFromScanNum
        pass
        
    def _GetMassListRangeFromRT(self):  # REDUNDANT with GetMassListRangeFromScanNum
        pass

    def _GetNextMassListRangeFromScanNum(self): # REDUNDANT with GetMassListRangeFromScanNum
        pass

    def _GetMassListFromRT(self): # REDUNDANT with GetMassListFromScanNum
        pass
        
    def _GetNextMassListFromScanNum(self): # REDUNDANT with GetMassListFromScanNum
        pass
        
    def _GetPrevMassListFromScanNum(self): # REDUNDANT with GetMassListFromScanNum
        pass
        
    def _GetFilterForScanRT(self): # REDUNDANT with GetFilterForScanNum
        pass
        
    def _GetSegmentedMassListFromRT(self): # REDUNDANT with GetSegmentedMassListFromScanNum
        pass

