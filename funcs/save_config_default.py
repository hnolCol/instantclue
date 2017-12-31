import configparser
import os 

def set_to_default(path="D:\MS_PYStasi\settings"):
     """ This function restores the default settings."""

     config = configparser.ConfigParser()


     config['DataPath'] = {'PathToRawFiles':'C:/Xcalibur/data'}

     config['ImportSettings'] = {'ShortenName':'True',
                                 'MethodExtraction':'True',
                                 'RawFileProps':'True',
                                 'SpectraProps':'False'}

     config['MetricExtract']= {'Metric':'Ion Injection Time (ms)',
                               'DefaultPlot':'Line plot',
                               'ms_level':'MS1'}

     config['GraphOptions'] = {'FontsizeTicks':'8',
                            'FontsizeAxis':'9',
                            '2DLinewidth': '0.5',
                            '3DLinewidth':'0.1',
                            'NumOfTickLabels(x)':'4',
                            'NumOfTickLabels(y)':'4',
                            'NumOfTickLabels(z)':'5',
                            'FontsizeLegend':'8',
                            'CutXChar':'0',
                            'Grid':'True',
                            'cmColor':'Blues',
                            'MatplotStyle':'classic'}
     
     config['msExtractDataDefault'] = {'mzMin':'450.00',
                                    'mzMax':'450.29',
                                    'RetTimeStart':'33.00',
                                    'RetTimeEnd':'33.21',
                                    'ExactMass':'450.34',
                                    'PeakWidth':'0.5',
                                    'IsotopePattern':'3',
                                    'MSLevel':'MS1',
                                    'Rollmean_data':'5',
                                    'Charge':'2'}



     config['MSMSdeconvolution'] = {'ThresholdMZisotopes':'0.04',
                                    'possibleisotopepeaks':'6',
                                    'centroidpeakwidth':'0.2'}



     with open(os.path.join(path,'default_settings.ini'), 'w') as configfile:
           config.write(configfile)





