

<img src="/img/logo.png" height="15%" width="15%">

# Instant Clue

## About

Instant Clue is a Python (>3.7) based desktop application (GUI) using the PyQt5 library for data visualization and analysis.
The tool was developed to equip everyone with a tool that enables analysis and visualization of high dimensional data sets in an easy and playful way.

### Tutorials

A written tutorial is available at the [Wiki Page](https://github.com/hnolCol/instantclue/wiki). Please let us now in the discussions for urgent topics to cover in the tutorial. We are now focusing on providing much better introduction material to get you started swiftly.

### Status

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

## Executables 

Please find the executables for Windows and Mac in the [Releases](https://github.com/hnolCol/instantclue/releases).

### New Features (v. >= 0.10,0) (Worklist)

Novel features incorporated into Instant Clue are listed below. Features that are under development are indicated.
This list provides an overview about upcoming features. Release specific features are document for each release separately (GitHub Release Page).
Please use the [discussion forum](https://github.com/hnolCol/instantclue/discussions) to suggest new features and/or discuss new plot types.

- [ ] Grouping: Allow export to excel. Add to hierarchical clustering. 
- [ ] Kinetic - Model fitting (first/second order). Available but experimental.
- [ ] Dimensional reduction plot
- [x] Add dimensional reduction techniques.
- [x] Clusterplot. Visualize detected clusters by several methods.
- [x] QuickSelect and LiveGraph Widgets (interact with the main graph and with each other)
- [x] User definable settings
- [x] Responsive and modern User Interface (UI)
- [x] Computational expensive tasks are computed on Thread 
- [x] Improved saving of session issues: [12](https://github.com/hnolCol/instantclue/issues/12),[5](https://github.com/hnolCol/instantclue/issues/5)
- [x] Categorical values can now be encoded by different markers
- [x] Categorical countplot. 
- [x] Grouping of columns to perform row-wise statistical tests. Groups are highligted in dimensional reduction plots as well as correlation matrix plots.
- [x] Export to markdown friendly string format and json file format of datasets.
- [ ] Creating log (saving version and processing of data as well as creation of charts) to ensure tracability of performed functions.
- [x] MainFigure Icons Update (responsive)

## Quick Select and Live Graph Widget

The two newly implemented widgets "QuickSelect" and "LiveGraph" intend to accelarete visual anlytics in InstantClue. An illustration of the functionality is shown below and [this video]() demonstrates usage. 

<img src="/img/QuickSelectLiveGraph.png" width="60%">

## New plot types

There are several new plot types included in the new version of Instant Clue.

#### Categorical Countplot

The countplot can only be used with categorical columns (anythin that is not an integer or float). In case of a single categorical column, the countplot displays the occurance of each unique value. The countplot is particular useful, when using multiple categorical colums, each combination of unique values in all columns is considered and displayed. The combinations are indicated by connecting categorical values below the barplot (see example below). Of note, the QuickSelect (see below) works with the countplot in such way, that hovering over the connection lines, will show the underlying rows in the QuickSelect widget.
An example is shown below using the TutorialData02.txt finding the overlap of mitochondrial proteins and proteins that were found to be significantly regulated. To find the mitochondrial proteins, drag & drop the column header "GOCC name" to the filter icon, and type "mitochondrion" into the search field. Select the GO term and click the check button. A new columns is created assigning rows containing the category "mitochondrion" by a "+" sign. Drag & Drop the columns "t-test SignificantS277A_CTRL" and "mitochondrion:GOCC name" to the reviever box (Catgerories) and the countplot shown below appears.

<img src="/img/countplot.png" width="50%">

#### WordCloud

The word cloud is generate based on the [wordcloud package](https://github.com/amueller/word_cloud). If you are using this type of chart, please acknowledge amueller's (Andreas Mueller) work. WordClouds have become less on-vogue but many users requested this type of chart. Input is a simple categorical column. You can find unique values by spliting the text in each row first using a specific string (customizable in the settings) or just merge the text to each other. An idea would be to visualize occurance of GO terms (but any other text like data can be used) as shown below. 

<img src="/img/wordcloud.png" width="50%">

#### Fosterplot 

The foster plot (blobbogram) is a widely used plot type in metanalysis. The [wikipedia](https://en.wikipedia.org/wiki/Forest_plot)] website hosts useful information.


#### Dimensional Reduction Plot

Coming soon - under development.


#### Cluster plot 

Available but experimental. At the moment, only boxplots are used for visualization. Numerous cluster methods from scikit-learn library are available.


## Issues

Please report Issues and Bugs using the GitHub issue functionality.

[![GitHub issues](https://img.shields.io/github/issues-closed/Naereen/StrapDown.js.svg)](https://github.com/hnolCol/instantclue/issues)

Issues that are currently taken care of:
- []


## Tutorials

Please visit https://www.instantclue.uni-koeln.de for video tutorials. 

## Get Started 

Executable binary files are available at the [website](http://www.instantclue.uni-koeln.de) for Mac OS and Windows.
We recommend using the development snapshots as they are equipped with more features and recent bug reports, but may contain non-functional widgets or context menu functions.

### Source Code
First download the code and extract it. Open terminal (mac) or command line tool (windows) and navigate to instantclue/src/main/python
Then create a virtual environment to not mix up required package versions with your python installation using 

```
python3 -m venv env #mac 
source env/bin/activate # activate env
```
or 
```
py -m venv env #windows
.\env\Scripts\activate
```
Then use the requirements.txt file to install required packages and finally start InstantClue.

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


