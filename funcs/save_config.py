import configparser
import os

def save_now_config(list_with_attribs, path, home_path):
          """Function to save new setting profile
          *args - list of settings to update
          *path - name and path to save settings"""
          
          load_path_default = os.path.join(os.path.dirname(home_path),'settings','default_settings.ini')
          
          list_with_attribs = [str(i) for i in list_with_attribs]
          config = configparser.SafeConfigParser()
    
          config.read(load_path_default)
              
               
          config.set('DataPath','PathToRawFiles',list_with_attribs[0])
          
          config.set('ImportSettings','ShortenName',list_with_attribs[1])
          config.set('ImportSettings','MethodExtraction',list_with_attribs[2])
          config.set('ImportSettings','SpectraProps',list_with_attribs[3])

          config.set('MetricExtract','Metric',list_with_attribs[4])
          config.set('MetricExtract','DefaultPlot',list_with_attribs[5])
          config.set('MetricExtract','metric_level',list_with_attribs[6])

          config.set('GraphOptions','fontsizeticks',list_with_attribs[16])
          config.set('GraphOptions','FontsizeAxis',list_with_attribs[17])
          config.set('GraphOptions','2DLinewidth', list_with_attribs[18])
          config.set('GraphOptions','3DLinewidth',list_with_attribs[19])
          config.set('GraphOptions','NumOfTickLabels(x)',list_with_attribs[20])
          config.set('GraphOptions','NumOfTickLabels(y)',list_with_attribs[21])
          config.set('GraphOptions','NumOfTickLabels(z)',list_with_attribs[22])
          config.set('GraphOptions','FontsizeLegend',list_with_attribs[23])
        
          config.set('GraphOptions','Grid',list_with_attribs[25])
          config.set('GraphOptions','cmColor',list_with_attribs[26])
          config.set('GraphOptions','MatplotStyle',list_with_attribs[24])
     
          config.set('msExtractDataDefault','mzmin',list_with_attribs[7])
          config.set('msExtractDataDefault','mzMax',list_with_attribs[8])
          config.set('msExtractDataDefault','rettimestart',list_with_attribs[9])
          config.set('msExtractDataDefault','rettimeend',list_with_attribs[10])
          config.set('msExtractDataDefault','exactmass',list_with_attribs[11])
          config.set('msExtractDataDefault','PeakWidth','0.5')
          config.set('msExtractDataDefault','IsotopePattern',list_with_attribs[13])
          config.set('msExtractDataDefault','MSLevel',list_with_attribs[14])
          config.set('msExtractDataDefault','Rollmean_data',list_with_attribs[15])
          config.set('msExtractDataDefault','Charge',list_with_attribs[12])



          config.set('MSMSdeconvolution','ThresholdMZisotopes',list_with_attribs[27])
          config.set('MSMSdeconvolution','possibleisotopepeaks',list_with_attribs[28])
          config.set('MSMSdeconvolution','centroidpeakwidth',list_with_attribs[29])

      
          with open(path, 'w') as configfile:
                          config.write(configfile)

     
