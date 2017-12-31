style_choices = [ 'bmh','classic','dark_background','fivethirtyeight','ggplot','grayscale','seaborn-pastel','seaborn-darkgrid','seaborn-bright','seaborn-colorblind','seaborn-dark']

color_map_choices =['Blues', 'BuGn', 'BuPu','GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                    'PuBu', 'PuBuGn', 'Purples', 'RdPu',
                    'Reds', 'YlGn', 'YlGnBu', 'YlOrBr',
                    'YlOrRd','BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn',
                    'PuOr','RdBu', 'RdGy', 'RdYlBu', 'RdYlGn','seismic',
                    'terrain', 'ocean']
info_for_label_in_graph_options = ['\nThe Fontsize of Tick Labels in pts. You can only modify all ticks no axis-wise.',
                                   '\nThe Fontsize of Axis Labels in pts. Note for better look, the Z-Axis in 3D Plots will be 1 pt smaller.',
                                   '\nLinewidth in pt for 2D plots.','\nLinewidth in pt for 3D plots.\nWe do not recommand higher than 0.5 especially in Subplots',
                                   '\nMaximum number of labels on specified label. You can set this\nto increase or decrease the number of labels at each axis. For example in 3D Plots preventing overlapping labels.',
                                   '\nMaximum number of labels on specified label. You can set this\nto increase or decrease the number of labels at each axis. For example in 3D Plots preventing overlapping labels.',
                                   '\nMaximum number of labels on specified label. You can set this\nto increase or decrease the number of labels at each axis. For example in 3D Plots preventing overlapping labels.',
                                   '\nFontsize of the Legend. This is powerful if in big Subplots the Legends are overlapping. Note that this is also the Fontsize of titles in 3D Plots',
                                   '\nThis Parameter allows to subset the rawfile name by X Character from the left. Even though STASI tries to find the longest match of rawfiles to prevent very long names.',
                                   '\nColorMap look for 3D Plots and Hexbin Plots. Explanation: BuPu means from Blue to Purple.',
                                   '\nStyle available in Matplotlib\nPython library. Just try!',
                                   '\nCheckbutton to set grid on or\noff plots.',
                                   '\nRefresh graph and apply changed settings. You can also simply use <Enter> to refresh the graph after adjusting a graph parameter.']

Qc_metrices_spectra = ['Multiple Injection', 'Multi Inject Info', 'Micro Scan Count', 'Scan Segment',
                       'Scan Event', 'Master Index', 'Charge State', 'Monoisotopic M/Z', 'Ion Injection Time (ms)',
                       'Max. Ion Time (ms)', 'FT Resolution', 'MS2 Isolation Width', 'MS2 Isolation Offset', 'AGC Target',
                       'HCD Energy', 'Analyzer Temperature', '=== Mass Calibration: ===', 'Conversion Parameter B',
                       'Conversion Parameter C', 'Temperature Comp. (ppm)', 'RF Comp. (ppm)', 'Space Charge Comp. (ppm)',
                       'Resolution Comp. (ppm)', 'Number of Lock Masses', 'Lock Mass #1 (m/z)', 'Lock Mass #2 (m/z)', 'Lock Mass #3 (m/z)',
                       'LM Search Window (ppm)', 'LM Search Window (mmu)', 'Number of LM Found', 'Last Locking (sec)', 'LM m/z-Correction (ppm)',
                       '=== Ion Optics Settings: ===', 'S-Lens RF Level', 'S-Lens Voltage (V)', 'Skimmer Voltage (V)', 'Inject Flatopole Offset (V)',
                       'Bent Flatapole DC (V)', 'MP2 and MP3 RF (V)', 'Gate Lens Voltage (V)', 'C-Trap RF (V)', '====  Diagnostic Data:  ====',
                       'Intens Comp Factor', 'Res. Dep. Intens', 'CTCD NumF', 'CTCD Comp', 'CTCD ScScr', 'RawOvFtT', 'LC FWHM parameter',
                       'Rod', 'PS Inj. Time (ms)', 'AGC PS Mode', 'AGC PS Diag', 'HCD Energy eV', 'AGC Fill', 'Injection t0',
                       't0 FLP', 'Access Id', 'Analog Input 1 (V)', 'Analog Input 2 (V)','==== Scan Header Info ====',
                       'numPackets', 'StartTime', 'LowMass', 'HighMass', 'TIC', 'BasePeakMass', 'BasePeakIntensity', 'numChannels', 'uniformTime', 'Frequency']


label_Qc_metric_and_figure_nav = ['\nQC metric that is stored in a raw file. Choose any to be plotted against the retention time.','\nChoose level of QC metric. So far: MS1 and MS2 or both.',
                                  '\nDefine number of data points to be used to calculate the rollmean to smooth data.','\nChoose plot style to be used. A Description of Boxplots, Violinplots and Histograms can be found on Wikipedia.',
                                  '\nPress the here to start the extraction from selected raw files.','\nUse scale bar to increase the number of figures that are stored in memory. Note that a significant high number of for example 3D plots will slow down the GUI.',
                                  '\nFigure navigation - Use left and right arrow, and home botton to navigate through the generated figures. Disk will save all figures to one pdf file']


plot_metric_level = ['MS1','MS2','Both']

qc_plot_option = ['Line plot','Histogram','Boxplot','Violin plot']

info_for_extract_peptide_profiles = ['\nUpload evidence file for choosen\nraw files ...','\nSearch in uploaded evidence files for proteins using Gene names or Uniprot IDs ...\nIf you enter an Uniprot ID before uploading the evindence file, only the targeted peptides will be loaded',
                                     '\nShow MSMS spectra with\nannotations ...\nRight click allows you to set settings for MSMS spectra deconvolution',
                                     '\nExtract peptide 2D elution\nprofile ....','\nExtract peptide 3D elution\nprofile ....',
                                     '\nSet number of data points for\nrollmean calculation ...',
                                     '\nDelete current session...\nNote that figures will be saved\nand be exported later.']


columns_to_load_in_evidence_file = ['MS/MS Scan Number','Modified sequence','Score','Intensity','Reverse','Potential contaminant','Retention time','Retention length', 'Proteins','Leading proteins','Leading razor protein','Gene names','Protein names',
                                    'Type','Raw file','Experiment','MS/MS m/z','Charge','m/z','Mass','Resolution','Match time difference','Mass Error [ppm]','Calibrated retention time start','Calibrated retention time finish','Retention time calibration',
                                    'Match m/z difference','Match q-value','Match score','PEP','MS/MS Count']

feature_sel = ['log2','sqrt','auto']

rss_feeds = ['http://feeds.nature.com/NatureLatestResearch.rss',
             'http://feeds.nature.com/ncb/rss/current.rss',
             'http://feeds.nature.com/ng/rss/aop.rss',
             'http://pubs.acs.org/action/showFeed?ui=0&mi=0&ai=52c&jc=jprobs&type=etoc&feed=rss.rss',
             'http://pubs.acs.org/action/showFeed?ui=0&mi=0&ai=54k&jc=ancham&type=etoc&feed=rss.rss']




