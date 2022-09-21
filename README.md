

<img src="/img/logo.png" height="15%" width="15%">

# Instant Clue

## About

Instant Clue is a Python based desktop application (GUI) using the PyQt5 library for data visualization and analysis. The software was developed to equip everyone with a tool that enables analysis and visualization of high dimensional data sets in an easy and playful way.

### Download (Executables) 

Please find the executables for Windows and Mac in the [Releases](https://github.com/hnolCol/instantclue/releases).

### Tutorials

A written tutorial is available at the [Wiki Page](https://github.com/hnolCol/instantclue/wiki). Please let us now in the discussions for urgent topics to cover in the tutorial. We are now focusing on providing much better introduction material to get you started swiftly. We are working on video tutorials which will be available here https://www.instantclue.de.

### Share Web Application 

We are establishing a web application to share app graphs with collaborators. The website is available at [https://app.instantclue.de](https://app.instantclue.de) and will be avaiable for everyone soon.

### Status

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

The master branch is the developing branch. For each release (not dev. builds) a new branch with the version number is created. 
Please put pull requests to the master branch.


### Issues / Bugs

Please report Issues and Bugs using the [GitHub issue](https://github.com/hnolCol/instantclue/issues) functionality.

### Suggestions? Tutorial topics missing?

Please open a [discussion](https://github.com/hnolCol/instantclue/discussions) to suggest new features and/or discuss new plot types. But also urgent tutorial topics. (We are at the moment working intensively on the [Wiki Page](https://github.com/hnolCol/instantclue/wiki) so please take a look.)


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
Then use the requirements.txt file to install required packages and finally start InstantClue. Please be aware that the package [combat](https://github.com/epigenelabs/pyComBat/tree/master/combat) is not in the requirements table because it has very specific versioning requirements of pandas and numpy. Therefore we recommend to install combat first and then install the remaining packages and ingore the warning about the numpy/pandas version mismatch (so far it works in the tests). Please note that using Pyinstaller requires some optimization. If you want to create your own executable please contact us for custimized hooks.  

```
pip install -r requirements.txt #install packages

python3 main.py #starts InstantClue
```

## Citation

If you found usage of Instant Clue useful within your scientific publication, please consider to cite the original article was published in [Scientific reports](https://www.nature.com/articles/s41598-018-31154-6). However, really any acknowledgement of Instant Clue is highly appreciated. 

Nolte et al. Instant Clue: A Software Suite for Interactive Data Visualization and Analysis, Scientific reports 8, 12648 (2018)




