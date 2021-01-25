

<img src="/img/logo.png" height="15%" width="15%">

# Instant Clue

## About

Instant Clue is a Python (>3.7) based desktop application (GUI) using the PyQt5 library for data visualization and analysis.
The tool was developed to equip everyone with a tool that enables analysis and visualization of high dimensional data sets in an easy and playful way.

## New Features (v. 0.9.2.240)

Novel features incorporated into Instant Clue are listed below. Features that are under development are indicated.

- [x] QuickSelect and LiveGraph Widgets (interact with the main graph and with each other)
- [x] User definable settings
- [x] Responsive User Interface
- [x] Computational expensive task on Thread 
- [x] Improved saving of session issues: [12](https://github.com/hnolCol/instantclue/issues/12)][5](https://github.com/hnolCol/instantclue/issues/5)]
- [x] Categorica values can now be encoded by different markers in settings
- [x] Categorical countplot. 
- [x] Grouping of columns to perform row-wise statistical tests. Groups are highligted in dimensional reduction plots as well as correlation matrix plots.
- [ ] Creating log (saving version and processing of data as well as creation of charts)


### Quick Select and Live Graph Widget

The two newly implemented widgets "QuickSelect" and "LiveGraph" intend to accelarete visual anlytics in InstantClue. An illustration of the functionality is shown below and [this video]() demonstrates usage. 

<img src="/img/QuickSelectLiveGraph.png">


## Issues

Please report Issues and Bugs using the GitHub issue functionality.

[![GitHub issues](https://img.shields.io/github/issues-closed/Naereen/StrapDown.js.svg)](https://github.com/hnolCol/instantclue/issues)

Issues that are current taken care of:
- [ ] 
- [ ] 
- [ ]


## Tutorials

Please visit https://www.instantclue.uni-koeln.de for video tutorials. 

## Requested Features and their Progress

- [x] QuickSelect sorting using color and size values
- [ ] Add MainFigure Icons
- 

## Get Started 

Executable binary files are available at the [website](http://www.instantclue.uni-koeln.de) for Mac OS and Windows.
We recommend using the development snapshots as they are equipped with more features and recent bug reports, but may contain non-functional widgets or context menu functions.

### Source Code
First download the code and extract it. Open terminal (mac) or command line tool (windows) and navigate to instantclue/src/main/python
Then create a virtual environment using 

```
python3 -m venv env #mac 
source env/bin/activate # activate env
```
or 
```
py -m venv env #windows
.\env\Scripts\activate
```
Then use the requirements.txt file to install packages and finally start InstantClue.

```
pip install -r requirements.txt #install packages

python3 main.py #starts InstantClue
```

## Builds 

From now on we are publishing development version builds to give faster acess to the users. Builds are available at [GitHub](https://github.com/hnolCol/instantclue/releases). The official website will only host the latest version. 

## Citation

If you found usage of Instant Clue useful within your scientific publication, please consider to cite the original article was published in [Scientific reports](https://www.nature.com/articles/s41598-018-31154-6).

Nolte et al. Instant Clue: A Software Suite for Interactive Data Visualization and Analysis, Scientific reports 8, 12648 (2018)

Any acknowledgement of Instant Clue is highly appreciated. 


