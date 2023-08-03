from collections import OrderedDict

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 



from .resortableTable import ResortableTable
from .warnMessage import WarningMessage, AskQuestionMessage, AskForFile

from ..delegates.ICDataTreeView import *
from ..utils import createSubMenu, createMenu, createMenus, getStandardFont, getStdTextColor
from ..dialogs.Selections.ICSelectionDialog import SelectionDialog
from ..dialogs.Groupings.ICGrouper import ICGrouper
from ..dialogs.Groupings.ICCompareGroups import ICCompareGroups
from ..dialogs.ICModel import ICModelBase, ICLinearFitModel
from ..dialogs.Transformations.ICBasicOperationDialog import BasicOperationDialog
from ..dialogs.Selections.ICDSelectItems import ICDSelectItems
from ..dialogs.Filter.ICNumericFilter import ICNumericFilterForSelection 
from .utils import dataFileExport 
#data import 
from backend.config.data.params import MTMethods

import pandas as pd
import numpy as np
import webbrowser


MT_MENUS = [{
        "subM":"Multiple testing corrections",
        "name":methodName,
        "funcKey": "stats::multipleTesting",
        "dataType": "Numeric Floats",
        "fnKwargs":{"method":methodName}
    } for methodName in MTMethods]


EXPORT_MENU = [
    {
        "subM":"Export Data",
        "name":actionName,
        "funcKey": "exportData",
        "dataType": "All",
        "fnKwargs" : {"txtFileFormat":fileFormat}
    } for fileFormat, actionName in dataFileExport + [("clipboard","To Clipboard")]]

dataTypeSubMenu = {
    "Numeric Floats": [
        ("main",["Column operation ..",
                "Sorting",
                "Value Transformation",
                "Data Format Transformation", 
                "Feature Selection", 
                "Filter",
                "Clustering",
                "Model Fitting",
                "(Prote-)omics-toolkit",
                "Groupings",
                "Export Data" 
                ]),
        ("Value Transformation",["Logarithmic","Normalization (row)","Normalization (column)","Smoothing","Density Estimation","Dimensional Reduction","Summarize","Multiple testing corrections"]),
        ("Data Format Transformation",["Group by and Aggregate .."]),
        ("Filter",["NaN Filter (rows)","NaN Filter (columns)","Outlier","Set NaN if..","Consecutive .."]),
        ("Clustering",["k-means"]),
        ("Smoothing",["Aggregate rows ..","Rolling window .."]),
        ("Density Estimation", ["Kernel Density"]),
        ("Column operation ..", ["Change data type to ..","Missing values (NaN)","Counting"]),
        ("Feature Selection", ["Model ..","Recursive Elimination .."]),
        ("Model Fitting",["Kinetic"]),
        ("Missing values (NaN)",["Replace NaN by .."]),
        ("Replace NaN by ..",["Iterative Imputer"]),
        ("Groupings",["Pairwise Tests","Multiple Groupings","Summarize Groups"]),
        ("(Prote-)omics-toolkit", ["pulse-SILAC"])
        ],
    "Integers" : [
        ("main",["Column operation ..","Sorting","Filter","Export Data"]),
        ("Column operation ..", ["Change data type to .."])
        ],
    "Categories" : [
        ("main",["Column operation ..",
            "Sorting",
            "Data Format Transformation", 
            "Filter",
            "(Prote-)omics-toolkit",
            "Export Data"]), #
        ("Column operation ..", ["Change data type to ..","String operation"]),
        ("Data Format Transformation",["Group by and Aggregate .."]),
        ("String operation",["Split on ..","Format"]),
        ("Filter",["Subset Shortcuts","To QuickSelect .."]),
        ("Subset Shortcuts",["Keep","Remove"]),
        # ("(Prote-)omics-toolkit", ["Annotations"]),
        ]
}


menuBarItems = [
    {
        "subM":"Pairwise Tests",
        "name":"t-test",
        "funcKey": "compareGroups",
        "dataType": "Numeric Floats",
        "fnKwargs":{"test":"t-test"}
    },
    {
        "subM":"Pairwise Tests",
        "name":"Welch-test",
        "funcKey": "compareGroups",
        "dataType": "Numeric Floats",
        "fnKwargs":{"test":"welch-test"}
    },
    {
        "subM":"Groupings",
        "name":"Within/Between correlation",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs": {"funcKey":"groupings:runGroupCorrelation",
                    "selectFromGroupings": "all",
                    }
    },
     {
        "subM":"Groupings",
        "name":"Annotate Groups",
        "funcKey": "createGroups",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Groupings",
        "name":"Help",
        "funcKey": "openWebsite",
        "dataType": "Numeric Floats",
        "fnKwargs": {
                    "link" : "https://github.com/hnolCol/instantclue/wiki/Groupings"
                    }
    },
    
    {
        "subM":"(Prote-)omics-toolkit",
        "name":"Filter fasta by ids",
        "funcKey": "filterFasta",
        "dataType": "Categories",
    },
    {
        "subM":"(Prote-)omics-toolkit",
        "name":"Categorical Enrichment Test",
        "funcKey": "fisherCategoricalEnrichmentTest",
        "dataType": "Categories",
    },
    # {
    #     "subM":"Annotations",
    #     "name":"MitoCarta 3.0 (member)",
    #     "funcKey": "annotate::",
    #     "dataType": "Categories",
    # },
    # {
    #     "subM":"Annotations",
    #     "name":"MitoCarta 3.0 (full)",
    #     "funcKey": "annotate::",
    #     "dataType": "Categories",
    # },
    # {
    #     "subM":"Annotations",
    #     "name":"Human MitoCoP (member)",
    #     "funcKey": "annotate::",
    #     "dataType": "Categories",
    # },
    # {
    #     "subM":"Annotations",
    #     "name":"Human MitoCoP (full)",
    #     "funcKey": "annotate::",
    #     "dataType": "Categories",
    # },
    {
        "subM":"(Prote-)omics-toolkit",
        "name":"1D-Enrichment",
        "funcKey": "run1DEnrichment",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"(Prote-)omics-toolkit",
        "name":"Match mod. pept. sequence to sites",
        "funcKey": "matchModPeptideSequenceToSites",
        "dataType": "Categories",
    },
    # {
    #     "subM":"(Prote-)omics-toolkit",
    #     "name":"Protein-Peptide Profile View",
    #     "funcKey": "openProteinPeptideView",
    #     "dataType": "Categories",
    # },
    # {
    #     "subM":"(Prote-)omics-toolkit",
    #     "name":"Intra Batch correction (lowess)",
    #     "funcKey": "openBatchCorrectionDialog",
    #     "dataType": "Numeric Floats",
    # },
    {
        "subM":"pulse-SILAC",
        "name":"A * exp(-k*t) + b (Exp. Degradation)",
        "funcKey": "runExponentialFit",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"fitType":"decrease"}
    },
    {
        "subM":"pulse-SILAC",
        "name":" 1- (A * exp(-k*t) + b) (Exp. Synthesis)",
        "funcKey": "runExponentialFit",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"fitType":"increase"}
    },
        {
       "subM":"pulse-SILAC",
        "name":"Two-compartment model (in-vivo)",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs": {"funcKey":"stats::fitTwoThreeCompModel",
                    "requiredGrouping": ["timeGroupingName"],
                    "otherKwargs": {"compartments":2}}
    },
    {
       "subM":"pulse-SILAC",
        "name":"Three-compartment model (in-vivo)",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs": {"funcKey":"stats::fitTwoThreeCompModel",
                    "requiredGrouping": ["timeGroupingName"],
                    "otherKwargs": {"compartments":3}}
    },
    # {
    #     "subM":"Organization",
    #     "name":"Create Sample List",
    #     "funcKey": "createSampleList",
    #     "dataType": "Categories",
    # },
    {
        "subM":"Multiple Groupings",
        "name":"N-W-ANOVA",
        "funcKey": "runNWayANOVA",
        "dataType": "Numeric Floats",
        "fnKwargs":{}
    },
    # {
    #     "subM":"Multiple Groupings",
    #     "name":"Repeated Measures 1/2 W ANOVA",
    #     "funcKey": "runRMOneTwoWayANOVA",
    #     "dataType": "Numeric Floats",
    #     "fnKwargs":{}
    # },
    # {
    #     "subM":"Multiple Groupings",
    #     "name":"Mixed Two-W-ANOVA",
    #     "funcKey": "runMixedANOVA",
    #     "dataType": "Numeric Floats",
    #     "fnKwargs":{}
    # },
    {
        "subM":"Pairwise Tests",
        "name":"Euclidean distance",
        "funcKey": "compareGroups",
        "dataType": "Numeric Floats",
        "fnKwargs":{"test":"euclidean"}
    },
    {
        "subM":"Summarize Groups",
        "name":"min",
        "funcKey": "summarizeGroups",
        "dataType": "Numeric Floats",
        "fnKwargs":{"metric":"min"}
    },
    {
        "subM":"Summarize Groups",
        "name":"mean",
        "funcKey": "summarizeGroups",
        "dataType": "Numeric Floats",
        "fnKwargs":{"metric":"mean"}
    },
    {
        "subM":"Summarize Groups",
        "name":"median",
        "funcKey": "summarizeGroups",
        "dataType": "Numeric Floats",
        "fnKwargs":{"metric":"median"}
    },
    {
        "subM":"Summarize Groups",
        "name":"max",
        "funcKey": "summarizeGroups",
        "dataType": "Numeric Floats",
        "fnKwargs":{"metric":"max"}
    },
    {
        "subM":"Summarize Groups",
        "name":"stdev",
        "funcKey": "summarizeGroups",
        "dataType": "Numeric Floats",
        "fnKwargs":{"metric":"std"}
    },
    {
        "subM":"Summarize Groups",
        "name":"variance",
        "funcKey": "summarizeGroups",
        "dataType": "Numeric Floats",
        "fnKwargs":{"metric":"var"}
    },

    {
        "subM":"Logarithmic",
        "name":"ln",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"logarithmic","base":"ln"}
    },
    {
        "subM":"Logarithmic",
        "name":"log2",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"logarithmic","base":"log2"}
    },
    {
        "subM":"Logarithmic",
        "name":"log10",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"logarithmic","base":"log10"}
    },
    {
        "subM":"Logarithmic",
        "name":"-log2",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"logarithmic","base":"-log2"}
    },
    {
        "subM":"Logarithmic",
        "name":"-log10",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"logarithmic","base":"-log10"}
    },
    
    {
        "subM":"Summarize",
        "name":"max",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"summarize","metric":"max"}
    },
    {
        "subM":"Summarize",
        "name":"75% quantile",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"summarize","metric":"quantile","q":0.75}
    },
    {
        "subM":"Summarize",
        "name":"mean",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"summarize","metric":"mean"}
    },
    {
        "subM":"Summarize",
        "name":"50% quantile (median)",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"summarize","metric":"median"}
    },
    {
        "subM":"Summarize",
        "name":"25% quantile",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"summarize","metric":"quantile","q":0.25}
    },
    {
        "subM":"Summarize",
        "name":"min",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"summarize","metric":"min"}
    },
    {
        "subM":"Summarize",
        "name":"stdev",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"summarize","metric":"std"}
    },
        {
        "subM":"Summarize",
        "name":"variance",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs":{"transformKey":"summarize","metric":"var"}
    },
    {
        "subM":"Kernel Density",
        "name":"gaussian",
        "funcKey": "data::kernelDensity",
        "dataType": "Numeric Floats",
        "fnKwargs":{"kernel":"gaussian"}
    },
    {
        "subM":"Kernel Density",
        "name":"tophat",
        "funcKey": "data::kernelDensity",
        "dataType": "Numeric Floats",
        "fnKwargs":{"kernel":"tophat"}
    },

    #‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine
    {
        "subM":"Sorting",
        "name":"Value",
        "funcKey": "data::sortData",
        "dataType": "All"
    },
    {
        "subM":"Sorting",
        "name":"Custom",
        "funcKey": "customSorting",
        "dataType": ["Integers","Categories"]
    },
    {
        "subM":"Filtr",
        "name":"Numeric/Categorical Filtering",
        "funcKey": "applyFilter",
        "dataType": ["Integers"]
    },
    {
        "subM":"Filter",
        "name":"Numeric Filtering",
        "funcKey": "applyFilter",
        "dataType": ["Numeric Floats"]
    },
    {
        "subM":"Filter",
        "name":"Numeric Filtering (Selection)",
        "funcKey": "applyNumericFilterForSelection",
        "dataType": "Numeric Floats"
    },
    {
        "subM":"Filter",
        "name":"Random Row Selection",
        "funcKey": "applyNumericFilterForSelection",
        "funcKey": "getUserInput",
        "dataType" : "All",
        "fnKwargs" : {"funcKey":"data::randomSelection",
                      "info":"Number of rows that should be randomly selected..",
                      "min": 1,
                      "max": "nDataRows",
                      "requiredInt":"N",
                      }
    },
    
    {
        "subM":"Smoothing",
        "name":"IIR Filter",
        "funcKey": "data::transformData",
        "dataType": "Numeric Floats"
    },
    {
        "subM":"Data Format Transformation",
        "name":"To long format (melt)",
        "funcKey": "data::meltData",
        "dataType": "All"
    },
    {
        "subM":"Data Format Transformation",
        "name":"Explode",
        "funcKey": "data::explodeDataByColumn",
        "dataType": "Categories"
    },
    {
        "subM":"Data Format Transformation",
        "name":"Transpose",
        "funcKey": "getUserInput",
        "dataType" : "All",
        "fnKwargs": {"funcKey":"data::transpose",
                    "requiredColumns": ["columnLabel"],
                    }
    },
    {
        "subM":"Data Format Transformation",
        "name":"Transpose (Selection)",
        "funcKey": "getUserInput",
        "dataType" : "All",
        "fnKwargs": {"funcKey":"data::transposeSelection",
                    "requiredColumns": ["columnLabel"],
                    "addColumns" : True
                    }
    },  
    {
        "subM":"Data Format Transformation",
        "name":"Row Correlation Matrix",
        "dataType": "Numeric Floats",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"stats::rowCorrelation",
                    "requiredColumns": ["indexColumn"],
                    "addColumns" : True}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"mean",
        "dataType": "Numeric Floats",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "title" : "Please select columns to be used for grouping",
                    "addColumns" : True,
                    "otherKwargs": {"metric":"mean"}}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"mean & stdev",
        "dataType": "Numeric Floats",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "title" : "Please select columns to be used for grouping",
                    "addColumns" : True,
                    "otherKwargs": {"metric":["mean",np.std]}}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"sum",
        "dataType": "Numeric Floats",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "title" : "Please select columns to be used for grouping",
                    "addColumns" : True,
                    "otherKwargs": {"metric":"sum"}}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"median",
        "dataType": "Numeric Floats",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "title" : "Please select columns to be used for grouping",
                    "addColumns" : True,
                    "otherKwargs": {"metric":"median"}}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"min",
        "dataType": "Numeric Floats",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "title" : "Please select columns to be used for grouping",
                    "addColumns" : True,
                    "otherKwargs": {"metric":"min"}}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"max",
        "dataType": "Numeric Floats",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "title" : "Please select columns to be used for grouping",
                    "addColumns" : True,
                    "otherKwargs": {"metric":"max"}}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"count valid values",
        "dataType": "Numeric Floats",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "title" : "Please select columns to be used for grouping",
                    "addColumns" : True,
                    "otherKwargs": {"metric":"count"}}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"count valid values (+total size)",
        "dataType": "Numeric Floats",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "title" : "Please select columns to be used for grouping",
                    "addColumns" : True,
                    "otherKwargs": {"metric":["count" ,"size"]}}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"custom combination",
        "dataType": "Numeric Floats",
        "funcKey": "getCustomGroupByInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate"}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"count unique values",
        "dataType": "Categories",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "addColumns" : True,
                    "otherKwargs": {"metric":"nunique"}}
    },
    {
        "subM":"Group by and Aggregate ..",
        "name":"join categories",
        "dataType": "Categories",
        "funcKey": "getUserInput",
        "fnKwargs": {"funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "addColumns" : True,
                    "otherKwargs": {"metric":"text-merge"}}
    },
    {
        "subM":"Value Transformation",
        "name":"Combat (Batch correction)",
        "funcKey": "runCombat",
        "dataType": "Numeric Floats",
        "fnKwargs":{"funcKey":"stats::runCombat"}
    },
    {
        "subM":"Value Transformation",
        "name":"Absolute values",
        "funcKey": "transformer::transformData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"transformKey":"absolute"}
    },
    {
        "subM":"Value Transformation",
        "name":"In place transformation",
        "funcKey": "toggleParam",
        "checkable" : True,
        "dataType": "Numeric Floats",
        "fnKwargs":{"paramName":"perform.transformation.in.place"}
    },
    
    {
        "subM":"Dimensional Reduction",
        "name":"PCA (Loadings)",
        "funcKey": "dimReduction::PCA",
        "dataType": "Numeric Floats"
    },
    {
        "subM":"Dimensional Reduction",
        "name":"PCA (Projection)",
        "funcKey": "dimReduction::PCA",
        "dataType": "Numeric Floats",
        "fnKwargs":{"returnProjections":True}
    },
    {
        "subM":"Dimensional Reduction",
        "name":"Linear Discriminant Analysis",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs": {"funcKey":"dimReduction::LDA",
                    "requiredGrouping": ["groupingName"],
                    "otherKwargs": {}}
    },
    # {
    #     "subM":"Dimensional Reduction",
    #     "name":"CVAE",
    #     "funcKey": "dimReduction::CVAE",
    #     "dataType": "Numeric Floats"
    # },
    {
        "subM":"Dimensional Reduction",
        "name":"UMAP",
        "funcKey": "dimReduction::UMAP",
        "dataType": "Numeric Floats"
    },
    # {
    #     "subM":"Dimensional Reduction",
    #     "name":"UMAP.T",
    #     "funcKey": "dimReduction::UMAP",
    #     "dataType": "Numeric Floats",
    #     "fnKwargs":{"transpose":True}
    # }, ######### DIM red

    {
        "subM":"Dimensional Reduction",
        "name":"Isomap",
        "funcKey": "dimReduction::ManifoldEmbedding",
        "dataType": "Numeric Floats",
        "fnKwargs":{"manifoldName":"Isomap"}
    },
 #######
    {
        "subM":"Dimensional Reduction",
        "name":"t-SNE",
        "funcKey": "dimReduction::ManifoldEmbedding",
        "dataType": "Numeric Floats",
        "fnKwargs":{"manifoldName":"TSNE"}
    },
        {
        "subM":"Dimensional Reduction",
        "name":"SpectralEmbedding",
        "funcKey": "dimReduction::ManifoldEmbedding",
        "dataType": "Numeric Floats",
        "fnKwargs":{"manifoldName":"SpecEmb"}
    },
    {
        "subM":"Dimensional Reduction",
        "name":"Multidimensional scaling (MDS)",
        "funcKey": "dimReduction::ManifoldEmbedding",
        "dataType": "Numeric Floats",
        "fnKwargs":{"manifoldName":"MDS"}
    },
    {
        "subM":"Dimensional Reduction",
        "name":"Locally Linear Embedding (LLE)",
        "funcKey": "dimReduction::ManifoldEmbedding",
        "dataType": "Numeric Floats",
        "fnKwargs":{"manifoldName":"LLE"}
    },
    {
        "subM":"Data Format Transformation",
        "name":"Pivot Table",
        "funcKey": "getUserInput",
        "dataType": "Categories",
        "fnKwargs": {"funcKey":"data::pivotTable",
                    "requiredColumns": ["indexColumn","columnNames"]}
    },
    {
        "subM":"Normalization (row)",
        "name":"Standardize (Z-Score)",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "Standardize (Z-Score)"}
    },
    {
        "subM":"Normalization (row)",
        "name":"Scale (0 - 1)",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "Scale (0 - 1)"}
    },
    {
        "subM":"Normalization (row)",
        "name":"Quantile (25 - 75)",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "Quantile (25 - 75)"}
    },
    {
        "subM":"Normalization (row)",
        "name":"Row Loess Fit Correction",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "loessRowNorm"}
    },
    {
        "subM":"Normalization (row)",
        "name":"To specific group",
        "funcKey": "normalizeToSpecificGroup",
        "dataType": "Numeric Floats",
        #"fnKwargs": {"normKey": "loessRowNorm"}
    },
    # {
    #     "subM":"Normalization (row)",
    #     "name":"Relative within Group",
    #     "funcKey": "normalizer::relativeWithinGroup",
    #     "dataType": "Numeric Floats",
    #     "fnKwargs": {"normKey": "loessRowNorm"}
    # },
    {
        "subM":"Normalization (column)",
        "name":"Cumulative Sum",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "cumSum","axis": 0}
    },
    {
        "subM":"Normalization (column)",
        "name":"Standardize (Z-Score)",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "Standardize (Z-Score)","axis": 0}
    },
    {
        "subM":"Normalization (column)",
        "name":"Scale (0 - 1)",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "Scale (0 - 1)","axis": 0}
    },
    {
        "subM":"Normalization (column)",
        "name":"Quantile (25 - 75)",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "Quantile (25 - 75)","axis": 0}
    },
    {
        "subM":"Normalization (column)",
        "name":"Divide by max",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "DivideByMax","axis": 0}
    },
    {
        "subM":"Normalization (column)",
        "name":"Divide by column sum",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "DivideByColSum","axis": 0}
    },
    {
        "subM":"Normalization (column)",
        "name":"Adjust Median",
        "funcKey": "normalizer::normalizeData",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "globalMedian"}
    },
    {
        "subM":"Normalization (column)",
        "name":"Adjust Median By Subset",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs": {"funcKey":"normalizer::adjustMedianBySubset",
                    "requiredColumns": ["subsetColumn"],
                    "addColumns" : True
                    }
    },
    {
        "subM":"Normalization (column)",
        "name":"Adjust Group Median",
        "funcKey": "normalizer::normalizeGroupMedian",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "normGroupColumnMedian"}
    },
    {
        "subM":"Normalization (column)",
        "name":"Adjust Group Quantiles",
        "funcKey": "normalizer::normalizeGroupQuantile",
        "dataType": "Numeric Floats",
        "fnKwargs": {"normKey": "normGroupColumnQuantile"}
    },
    {
        "subM":"Change data type to ..",
        "name":"Numeric Floats",
        "funcKey": "data::changeDataType",
        "fnKwargs": {"newDataType":"float64"},
        "dataType": ["Integers","Categories"]
    },
    {
        "subM":"Change data type to ..",
        "name":"Numeric Floats (flex - copy)",
        "funcKey": "data::changeDataType",
        "fnKwargs": {"newDataType":"float64","flex":True},
        "dataType": ["Integers","Categories"]
    },
    {
        "subM":"Change data type to ..",
        "name":"Integers",
        "funcKey": "data::changeDataType",
        "fnKwargs": {"newDataType":"int64"},
        "dataType": ["Numeric Floats","Categories"]
    },
    {
        "subM":"Change data type to ..",
        "name":"Categories",
        "funcKey": "data::changeDataType",
        "fnKwargs": {"newDataType":"str"},
        "dataType": ["Numeric Floats","Integers"]
    },
    {
        "subM":"Filter",
        "name":"Find string(s)",
        "funcKey": "applyFilter",
        "dataType": "Categories",
        "fnKwargs": {"filterType":"string","calledFromMenu":True}
    },
    {
        "subM":"Filter",
        "name":"Categorical Filter",
        "funcKey": "applyFilter",
        "dataType": "Categories",
        "fnKwargs": {"calledFromMenu":True}
    },
    {
        "subM":"Filter",
        "name":"Custom Categorical Filter",
        "funcKey": "applyFilter",
        "dataType": "Categories",
        "fnKwargs": {"filterType":"multiColumnCategory","calledFromMenu":True}
    },
    {
        "subM":"Keep",
        "name":"+",
        "funcKey": "filter::subsetShortcut",
        "dataType": "Categories",
        "fnKwargs": {"how":"keep","stringValue":"+"}
    },
    {
        "subM":"Keep",
        "name":"-",
        "funcKey": "filter::subsetShortcut",
        "dataType": "Categories",
        "fnKwargs": {"how":"keep","stringValue":"-"}
    },
    {
        "subM":"Filter",
        "name":"Drop duplicates",
        "funcKey": "data::removeDuplicates",
        "dataType": "Categories",
    },
    {
        "subM":"Remove",
        "name":"+",
        "funcKey": "filter::subsetShortcut",
        "dataType": "Categories",
        "fnKwargs": {"how":"remove","stringValue":"+"}
    },
    {
        "subM":"Remove",
        "name":"-",
        "funcKey": "filter::subsetShortcut",
        "dataType": "Categories",
        "fnKwargs": {"how":"remove","stringValue":"-"}
    },
    {
        "subM":"Column operation ..",
        "name":"Combine columns",
        "funcKey": "data::combineColumns",
        "dataType": "All",
    },
    {
        "subM":"Column operation ..",
        "name":"Duplicate column(s)",
        "funcKey": "data::duplicateColumns",
        "dataType": "All",
    },
    {
        "subM":"Column operation ..",
        "name":"Add index column",
        "funcKey": "data::addIndexColumn",
        "dataType": "All",
    },
    {
        "subM":"Column operation ..",
        "name":"Add group index column",
        "funcKey": "data::addGroupIndexColumn",
        "dataType": "Categories",
    },
    {
        "subM":"Column operation ..",
        "name":"Factorize column(s)",
        "funcKey": "data::factorizeColumns",
        "dataType": "Categories",
    },
    {
        "subM":"Value Transformation",
        "name":"Row wise calculations",
        "funcKey": "rowWiseCalculations",
        "dataType": "Numeric Floats"
    },
    {
        "subM":"Counting",
        "name":"Count NaN in row (Selection)",
        "funcKey": "data::countNaN",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Counting",
        "name":"Count NaN in row (Grouping)",
        "funcKey": "data::countNaN",
        "dataType": "Numeric Floats",
        "fnKwargs": {"grouping":True}
    },
    {
        "subM":"Counting",
        "name":"Count valid values in columns",
        "funcKey": "data::countValidValuesInColumns",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Counting",
        "name":"Count valid values in row (Frequencies)",
        "funcKey": "data::countValidProfiles",
        "dataType": "Numeric Floats",
    },

    
    {
        "subM":"Counting",
        "name":"Count valid values in row (Selection)",
        "funcKey": "data::countValidValues",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Counting",
        "name":"Count valid values in row (Grouping)",
        "funcKey": "data::countValidValues",
        "dataType": "Numeric Floats",
        "fnKwargs": {"grouping":True}
    },
    {
        "subM":"Counting",
        "name":"Count valid values in subsets",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs": {
                    "funcKey":"data::groupbyAndAggregate",
                    "requireMultipleColumns" : "groupbyColumn",
                    "addColumns" : True,
                    "otherKwargs": {"metric":"count"}}
    },
    {
        "subM":"NaN Filter (rows)",
        "name":"Any == NaN",
        "funcKey": "data::removeNaN",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Set NaN if..",
        "name":"Value below X",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"data::setNaNBasedOnCondition", 
                      "info":"Values below the given value will be set to NaN.\nA filtered set columns will be added.",
                      "min": -np.inf,
                      "max": np.inf,
                      "default" : 1,
                      "requiredFloat":"belowThreshold"}
    },
    {
        "subM":"Set NaN if..",
        "name":"Value above X",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"data::setNaNBasedOnCondition", 
                      "info":"Values above the given value will be set to NaN.\nA filtered set columns will be added.",
                      "min": -np.inf,
                      "max": np.inf,
                      "default" : 1,
                      "requiredFloat":"aboveThreshold"}
    },
    # {
    #     "subM":"Set NaN if..",
    #     "name":"Not decreasing (Group)",
    #     "funcKey": ",
    #     "dataType": "Numeric Floats",
    #     "fnKwargs": {"increasing":True}
    # },
    {
        "subM":"Set NaN if..",
        "name":"Not decreasing (Group)",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs": {"funcKey":"filter::NaNconsecutiveDecreasing",
                    "requiredGrouping": ["Grouping"],
                    "otherKwargs": {"increasing":False}
        }
    },
    {
        "subM":"Consecutive ..",
        "name":"Increasing (Selection)",
        "funcKey": "filter::consecutiveValues",
        "dataType": "Numeric Floats",
        "fnKwargs": {"increasing":True}
    },
    {
        "subM":"Consecutive ..",
        "name":"Decreasing (Selection)",
        "funcKey": "filter::consecutiveValues",
        "dataType": "Numeric Floats",
        "fnKwargs": {"increasing":False}
    },
    {
        "subM":"Consecutive ..",
        "name":"Increasing (Grouping)",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs": {"funcKey":"filter::consecutiveValuesInGrouping",
                    "requiredGrouping": ["Grouping"],
                    "otherKwargs": {"increasing":True}
        }
    },
    {
        "subM":"Consecutive ..",
        "name":"Decreasing (Grouping)",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs": {"funcKey":"filter::consecutiveValuesInGrouping",
                    "requiredGrouping": ["Grouping"],
                    "otherKwargs": {"increasing":False}
        }
    },
    {
        "subM":"Filter",
        "name":"Variance Filter (rows)",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"data::filterDataByVariance", 
                      "info":"Please provide a variance threshold.",
                      "min": 0.0,
                      "max": np.inf,
                      "default" : 0.2,
                      "requiredFloat":"varThresh"}
    },
    {
        "subM":"Filter",
        "name":"Variance Filter (columns)",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"data::filterDataByVariance", 
                      "info":"Please provide a variance threshold.",
                      "min": 0.0,
                      "max": np.inf,
                      "default" : 0.2,
                      "requiredFloat":"varThresh",
                      "otherKwargs":{"direction":"columns"}}
    },
    {
        "subM":"NaN Filter (rows)",
        "name":"All == NaN",
        "funcKey": "data::removeNaN",
        "dataType": "Numeric Floats",
        "fnKwargs": {"how":"all"}
    },
    {
        "subM":"NaN Filter (rows)",
        "name":"Threshold",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"data::removeNaN", 
                      "info":"Provide the number of non-NaN values.\nIf value < 1, the fraction of selected columns is considered.",
                      "min": 0.0,
                      "max": "nColumns",
                      "requiredFloat":"thresh"}
    },
    {
        "subM":"NaN Filter (columns)",
        "name":"Any == NaN",
        "funcKey": "data::removeNaN",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"axis":1}
    },
    {
        "subM":"NaN Filter (columns)",
        "name":"All == NaN",
        "funcKey": "data::removeNaN",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"axis":1,"how":"all"}
    },
    {
        "subM":"NaN Filter (rows)",
        "name":"Group positives",
        "funcKey": "grouping::exclusivePositives",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"NaN Filter (rows)",
        "name":"Group negatives",
        "funcKey": "grouping::exclusiveNegative",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Split on ..",
        "name":"Space ( )",
        "funcKey": "data::splitColumnsByString",
        "dataType": "Categories",
        "fnKwargs": {"splitString":" "}
    },
    {
        "subM":"Split on ..",
        "name":"Semicolon (;)",
        "funcKey": "data::splitColumnsByString",
        "dataType": "Categories",
        "fnKwargs": {"splitString":";"}
    },
    {
        "subM":"Split on ..",
        "name":"Underscore (_)",
        "funcKey": "data::splitColumnsByString",
        "dataType": "Categories",
        "fnKwargs": {"splitString":"_"}
    },
    {
        "subM":"Split on ..",
        "name":"Comma (,)",
        "funcKey": "data::splitColumnsByString",
        "dataType": "Categories",
        "fnKwargs": {"splitString":","}
    },
    {
        "subM":"Split on ..",
        "name":"Custom string",
        "funcKey": "getUserInput",
        "dataType": "Categories",
        "fnKwargs" : {"funcKey":"data::splitColumnsByString", 
                      "info":"Provide split string to separate strings in a column.",
                      "requiredStr":"splitString",
                      }
    },
    {
        "subM":"Format",
        "name":"UPPER",
        "funcKey": "data::formatString",
        "dataType": "Categories",
        "fnKwargs": {"formatFn":"upper"}
    },
    {
        "subM":"Format",
        "name":"lower",
        "funcKey": "data::formatString",
        "dataType": "Categories",
        "fnKwargs": {"formatFn":"lower"}
    },
    {
        "subM":"Format",
        "name":"Capitilize",
        "funcKey": "data::formatString",
        "dataType": "Categories",
        "fnKwargs": {"formatFn":"capitilize"}
    },
    {
        "subM":"To QuickSelect ..",
        "name":"Raw values",
        "funcKey": "sendSelectionToQuickSelect",
        "dataType": "Categories",
        "fnKwargs" : {"mode" : "raw"}
    },
    {
        "subM":"To QuickSelect ..",
        "name":"Unique Categories",
        "funcKey": "sendSelectionToQuickSelect",
        "dataType": "Categories",
    },
    {
        "subM":"Recursive Elimination ..",
        "name":"Random Forest",
        "funcKey": "featureSelection",
        "dataType": "Numeric Floats",
        "fnKwargs": {"RFEVC":True}
    },
    {
        "subM":"Recursive Elimination ..",
        "name":"SVM (linear)",
        "funcKey": "featureSelection",
        "dataType": "Numeric Floats",
        "fnKwargs": {"RFEVC":True}
    },
    # {
    #     "subM":"Recursive Elimination ..",
    #     "name":"SVM (rbf)",
    #     "funcKey": "featureSelection",
    #     "dataType": "Numeric Floats",
    #     "fnKwargs": {"RFEVC":True}
    # },
    # {
    #     "subM":"Recursive Elimination ..",
    #     "name":"SVM (poly)",
    #     "funcKey": "featureSelection",
    #     "dataType": "Numeric Floats",
    #     "fnKwargs": {"RFEVC":True}
    # },
    {
        "subM":"Model ..",
        "name":"Random Forest",
        "funcKey": "featureSelection",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Model ..",
        "name":"SVM (linear)",
        "funcKey": "featureSelection",
        "dataType": "Numeric Floats",
    },
#    {
#        "subM":"Model ..",
#        "name":"SVM (rbf)",
#        "funcKey": "featureSelection",
#        "dataType": "Numeric Floats",
#    },
#    {
#        "subM":"Model ..",
#        "name":"SVM (poly)",
#        "funcKey": "featureSelection",
#        "dataType": "Numeric Floats",
#    },
    {
        "subM":"Model ..",
        "name":"False Positive Rate",
        "funcKey": "featureSelection",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Model ..",
        "name":"False Discovery Rate",
        "funcKey": "featureSelection",
        "dataType": "Numeric Floats",
    },  
    {
        "subM":"Aggregate rows ..",
        "name":"Mean",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"data::aggregateNRows", 
                      "info":"Provide window size (n) for aggregation.\nA window of 50 rows will be aggregated using selected metric.",
                      "min": 2,
                      "max": "nDataRows",
                      "requiredInt":"n",
                      "otherKwargs":{"metric":"mean"}}
    },
    {
        "subM":"Rolling window ..",
        "name":"mean",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"transformer::transformData", 
                      "info":"Provide rolling window size.\nA rolling window of 50 rows will be used.",
                      "min": 2,
                      "max": "nDataRows",
                      "requiredInt":"windowSize",
                      "otherKwargs":{"transformKey":"rolling","metric":"mean"}}
    },
    {
        "subM":"Rolling window ..",
        "name":"median",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"transformer::transformData", 
                      "info":"Provide rolling window size.\nA rolling window of 50 rows will be used.",
                      "min": 2,
                      "max": "nDataRows",
                      "requiredInt":"windowSize",
                      "otherKwargs":{"transformKey":"rolling","metric":"median"}}
    },
    {
        "subM":"Rolling window ..",
        "name":"sum",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"transformer::transformData", 
                      "info":"Provide rolling window size.\nA rolling window of 50 rows will be used.",
                      "min": 2,
                      "max": "nDataRows",
                      "requiredInt":"windowSize",
                      "otherKwargs":{"transformKey":"rolling","metric":"sum"}}
    },
    {
        "subM":"Rolling window ..",
        "name":"standard deviation",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"transformer::transformData", 
                      "info":"Provide rolling window size.\nA rolling window of 50 rows will be used.",
                      "min": 2,
                      "max": "nDataRows",
                      "requiredInt":"windowSize",
                      "otherKwargs":{"transformKey":"rolling","metric":"std"}}
    },
    {
        "subM":"Rolling window ..",
        "name":"max",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"transformer::transformData", 
                      "info":"Provide rolling window size.\nA rolling window of 50 rows will be used.",
                      "min": 2,
                      "max": "nDataRows",
                      "requiredInt":"windowSize",
                      "otherKwargs":{"transformKey":"rolling","metric":"max"}}
    },
    {
        "subM":"Rolling window ..",
        "name":"min",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"transformer::transformData", 
                      "info":"Provide rolling window size.\nA rolling window of 50 rows will be used.",
                      "min": 2,
                      "max": "nDataRows",
                      "requiredInt":"windowSize",
                      "otherKwargs":{"transformKey":"rolling","metric":"min"}}
    },
    {   
        "subM":"Replace NaN by ..",
        "name":"Row mean",
        "funcKey": "data::fillNa",
        "dataType": "Numeric Floats",
        "fnKwargs": {"fillBy":"Row mean"}
    },
    {
        "subM":"Replace NaN by ..",
        "name":"Row median",
        "funcKey": "data::fillNa",
        "dataType": "Numeric Floats",
        "fnKwargs": {"fillBy":"Row median"}
    },
    {
        "subM":"Replace NaN by ..",
        "name":"Column mean",
        "funcKey": "data::fillNa",
        "dataType": "Numeric Floats",
        "fnKwargs": {"fillBy":"Column mean"}
    },
    {
        "subM":"Replace NaN by ..",
        "name":"Column median",
        "funcKey": "data::fillNa",
        "dataType": "Numeric Floats",
        "fnKwargs": {"fillBy":"Column median"}
    },
    {
        "subM":"Replace NaN by ..",
        "name":"Gaussian distribution",
        "funcKey": "data::fillNa",
        "dataType": "Numeric Floats",
        "fnKwargs": {"fillBy":"Gaussian distribution"}
    },
    {
        "subM":"Replace NaN by ..",
        "name":"Group Mean",
        "funcKey": "data::replaceNaNByGroupMean",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Replace NaN by ..",
        "name":"Constant value",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"data::fillNa", 
                      "info":"Provide constant value to fill nans.",
                      "min": -np.inf,
                      "default": 0.0,
                      "max": np.inf,
                      "requiredFloat":"fillBy"}
    },  
    {
        "subM":"Replace NaN by ..",
        "name":"Smart Group Replace",
        "funcKey": "smartReplace",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Kinetic",
        "name":"First Order",
        "funcKey": "fitModel",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Model Fitting",
        "name":"Linear fit",
        "funcKey": "fitModel",
        "dataType": "Numeric Floats",
    },

    {
        "subM":"Outlier",
        "name":"Remove outliers (Group)",
        "funcKey": "removeOutliersFromGroup",
        "dataType": "Numeric Floats",
    },
    {
        "subM":"Outlier",
        "name":"Remove outliers (Selection)",
        "funcKey": "data::replaceOutlierWithNaN",
        "dataType": "Numeric Floats"
    },
     {
        "subM":"Clustering",
        "name":"HDBSCAN",
        "funcKey": "stats::runHDBSCAN",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"attachToSource":True}
    },
    {
        "subM":"k-means",
        "name":"Elbow method",
        "funcKey": "stats::kmeansElbow",
        "dataType": "Numeric Floats"
    },
    {
        "subM":"k-means",
        "name":"K-Means",
        "funcKey": "getUserInput",
        "dataType": "Numeric Floats",
        "fnKwargs" : {"funcKey":"stats::runKMeans", 
                      "info":"Provide number of clusters (k).",
                      "min": 2,
                      "default": "kmeans.default.number.clusters",
                      "max": "nDataRows",
                      "requiredInt":"k"}
    },
    {
        "subM":"Iterative Imputer",
        "name":"BayesianRidge",
        "funcKey": "data::imputeByModel",
        "dataType": "Numeric Floats",
        "fnKwargs": {"estimator":"BayesianRidge"}
    },
    {
        "subM":"Iterative Imputer",
        "name":"DecisionTreeRegressor",
        "funcKey": "data::imputeByModel",
        "dataType": "Numeric Floats",
        "fnKwargs": {"estimator":"DecisionTreeRegressor"}
    },
    {
        "subM":"Iterative Imputer",
        "name":"ExtraTreesRegressor",
        "funcKey": "data::imputeByModel",
        "dataType": "Numeric Floats",
        "fnKwargs": {"estimator":"ExtraTreesRegressor"}
    },
    {
        "subM":"Iterative Imputer",
        "name":"KNeighborsRegressor",
        "funcKey": "data::imputeByModel",
        "dataType": "Numeric Floats",
        "fnKwargs": {"estimator":"KNeighborsRegressor"}
    }
] + MT_MENUS + EXPORT_MENU

class DataTreeView(QWidget):
     
    updateData = pyqtSignal(pd.Series,dict)

    def __init__(self,parent=None, mainController = None, dataID = None, tableID = None):
        super(DataTreeView, self).__init__(parent)
        self.tableID = tableID

        self.mC = mainController

        self.__controls()
        self.__layout()
        self.__connectEvents()
        self.__connectSignals()

        self.showShortcuts = True
        self.dataID = dataID
        self.groupingName = ""
        

    def __controls(self):

        self.__setupTable()

    def __connectEvents(self):
        ""

    def __connectSignals(self):
        "Connect the signals to safely update"
        self.updateData.connect(self.addData)

    def __setupTable(self):
        ""
        
        self.table = DataTreeViewTable(parent = self, 
                                    tableID= self.tableID, 
                                    mainController=self.mC,
                                    sendToThread= self.sendToThread)
        self.table.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.model = DataTreeModel(parent=self.table)
        self.table.setItemDelegateForColumn(0,ItemDelegate(self.table))
        self.table.setItemDelegateForColumn(1,AddDelegate(self.table,highLightColumn=1))
        self.table.setItemDelegateForColumn(3,GroupDelegate(self.table,highLightColumn=3))#CopyDelegateCopyDelegate
        self.table.setItemDelegateForColumn(2,FilterDelegate(self.table,highLightColumn=2))
        self.table.setItemDelegateForColumn(4,DeleteDelegate(self.table,highLightColumn=4))

        self.table.setModel(self.model)
        

        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3,QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4,QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        #self.table.resizeColumns()
        
    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0,0,0,0)
        self.layout().addWidget(self.table)
        
    def getData(self) -> pd.DataFrame:
        ""
        return self.table.model().getLabels()

    def getDataID(self) -> str:
        ""
        return self.dataID

    def getSelectedData(self):
        "Returns Selected Data."
        return self.table.getSelectedData()

    def sendToThread(self, funcProps = {}, addSelectionOfAllDataTypes = False, addDataID = False):
        ""
        if addDataID:
            if not "kwargs" in funcProps:
                funcProps["kwargs"] = {}
            funcProps["kwargs"]["dataID"] = self.getDataID() 
        if addSelectionOfAllDataTypes:
            funcProps = self.mC.mainFrames["data"].dataTreeView.addSelectionOfAllDataTypes(funcProps)
        
        self.mC.sendRequestToThread(funcProps)

    def setDataID(self,dataID : str) -> None:
        "Save the dataID"
        self.dataID = dataID 

    def sortLabels(self):
        "Sorts label, first ascending, then descending, then raw order"
        if hasattr(self.table.model(),"lastSearchType"):
            lastSerachType = getattr(self.table.model(),"lastSearchType")
            if lastSerachType is None:
                how = "ascending"
            elif lastSerachType == "ascending":
                how = "descending"
            elif lastSerachType == "descending":
                #reset view
                setattr(self.table.model(),"lastSearchType",None)
            
        self.table.model().sort(how=how)
        columnDict = self.mC.mainFrames["data"].dataTreeView.getColumns("all")
       
        funcProps = {"key":"data::sortColumns",
                    "kwargs":{"sortedColumnDict":columnDict,"dataID":self.mC.getDataID()}}
        self.mC.sendRequestToThread(funcProps)

    def customSortLabels(self):
        ""
        resortLabels = ResortableTable(self.table.model()._labels)
        if resortLabels.exec():
            sortedLabels = resortLabels.savedData
            self.addData(sortedLabels)

    def hideShowShortCuts(self):
        "Show / Hide Shortcuts"
        for i in range(1,5):
            if self.showShortcuts:
                self.table.hideColumn(i)
            else:
                self.table.showColumn(i)

        self.showShortcuts = not self.showShortcuts

    def addData(self, X : pd.Series, tooltipData : dict = {} ,dataID = None) -> None:
        "Add data to thre treeview. "
        self.table.selectionModel().clear()
        self.table.model().layoutAboutToBeChanged.emit()
        self.table.model().setNewData(X)
        self.table.model().setTooltipdata(tooltipData)
        self.table.model().layoutChanged.emit()
        self.table.model().completeDataChanged()
        if dataID is not None:
            self.setDataID(dataID)

    def setColumnState(self,columnNames,newState) -> None:
        ""
        self.table.model().setColumnStateByData(columnNames,newState)

    def setGrouping(self,grouping : dict,groupingName : str) -> None:
        ""        
        if isinstance(grouping,dict):
            self.groupingName = groupingName
            groupColors = self.mC.grouping.getGroupColors()
            self.model.resetGrouping()
            for groupName, columnNames in grouping.items():
                #for idx in columnNames.index:
                modelDataIndex = self.model.getIndexFromNames(columnNames)
                
                self.model.setColumnStateByDataIndex(modelDataIndex,True)
                self.model.setGroupingColorByDataIndex(modelDataIndex,groupColors[groupName])
                self.model.setGroupNameByDataIndex(modelDataIndex,groupName)
                
            self.table.model().completeDataChanged()

    def setCurrentGrouping(self):
        ""
        if self.mC.grouping.groupingExists():
            self.setGrouping(self.mC.grouping.getCurrentGrouping(), self.mC.grouping.getCurrentGroupingName())
        else:
            self.model.resetGrouping()
            self.table.model().completeDataChanged()

class DataTreeModel(QAbstractTableModel):
    

    def __init__(self, labels = pd.Series(dtype="object"), parent=None):
        super(DataTreeModel, self).__init__(parent)
        self.initData(labels)
        

    def initData(self,labels):
        self._labels = labels
        self._inputLabels = labels.copy()
        self.tooltipData = OrderedDict()
        self.columnInGraph = pd.Series(np.zeros(shape=labels.index.size), index=labels.index)
        self.resetGrouping()
        self.lastSearchType = None

    def rowCount(self, parent=QModelIndex()):
        
        return self._labels.size

    def columnCount(self, parent=QModelIndex()):
        
        return 5

    def setDefaultSize(self,size=50):
        ""
        self.defaultSize = size

    def getDataIndex(self,row):
        ""
        if self.validDataIndex(row):
            return self._labels.index[row]
        
    def getColumnStateByDataIndex(self,dataIndex):
        ""
        return self.columnInGraph.loc[dataIndex] == 1

    def getColumnStateByTableIndex(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex is not None:
            return self.getColumnStateByDataIndex(dataIndex)

    def setColumnState(self,tableIndex, newState = None):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex is not None:
            if newState is None:
                newState = not self.columnInGraph.loc[dataIndex]
            self.columnInGraph.loc[dataIndex] = newState
            return newState
    
    def setColumnStateByData(self,columnNames,newState):
        ""
        idx = self._labels[self._labels.isin(columnNames)].index
        if not idx.empty:
            self.columnInGraph[idx] = newState

    def getGroupingStateByTableIndex(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex is not None:
            return self.getGroupingStateByDataIndex(dataIndex)

    def getGroupingStateByDataIndex(self,dataIndex):
        ""
        return self.columnInGrouping.loc[dataIndex] == 1
    
    def getIndexFromNames(self,columnNames):
        ""
        return self._labels.index[self._labels.isin(columnNames)]

    def setColumnStateByDataIndex(self,columnNameIndex,newState):
        ""
        idx = self._labels.index.intersection(columnNameIndex)
        if not idx.empty:
             self.columnInGrouping[idx] = newState

    def setGroupingState(self,tableIndex, newState = None):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex is not None:
            if newState is None:
                newState = not self.columnInGrouping.loc[dataIndex]
            self.columnInGrouping.loc[dataIndex] = newState
            return newState

    def setGroupingColorByDataIndex(self,columnNameIndex,hexColor):
        ""
        idx = self._labels.index.intersection(columnNameIndex)
        if not idx.empty:
             self.colorsInGrouping[idx] = hexColor
    
    def setGroupNameByDataIndex(self,columnNameIndex,hexColor):
        ""
        idx = self._labels.index.intersection(columnNameIndex)
        if not idx.empty:
             self.nameGrouping[idx] = hexColor

    def getGroupNameByTableIndex(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex is not None:
            return self.nameGrouping.loc[dataIndex]

    def getGroupColorByTableIndex(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex is not None:
            return self.getGroupColorByDataIndex(dataIndex)

    def getGroupColorByDataIndex(self,dataIndex):
        ""
        return self.colorsInGrouping.loc[dataIndex]
    
    def resetGrouping(self):
        ""
        self.columnInGrouping = pd.Series(np.zeros(shape=self._labels.index.size), index=self._labels.index)
        self.colorsInGrouping = pd.Series(np.zeros(shape=self._labels.index.size), index=self._labels.index)
        self.nameGrouping = pd.Series(np.zeros(shape=self._labels.index.size), index=self._labels.index)

    def updateData(self,value,index):
        ""
        dataIndex = self.getDataIndex(index.row())
        if dataIndex is not None:
            self._labels[dataIndex] = str(value)
            self._inputLabels = self._labels.copy()

    def validDataIndex(self,row):
        ""
        return row <= self._labels.index.size - 1

    def deleteEntriesByIndexList(self,indexList):
        ""
        dataIndices = [self.getDataIndex(tableIndex.row()) for tableIndex in indexList]
        self._labels = self._labels.drop(dataIndices)
        self._inputLabels = self._labels.copy()
        self.completeDataChanged()

    def deleteEntry(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex in self._inputLabels.index:
            self._labels = self._labels.drop(dataIndex)
            self._inputLabels = self._labels
            self.completeDataChanged()

    def getLabels(self):
        ""
        return self._labels

    def getSelectedData(self,indexList):
        ""
        dataIndices = [self.getDataIndex(tableIndex.row()) for tableIndex in indexList]
        return self._labels.loc[dataIndices]

    def setData(self,index,value,role):
        ""
        row =index.row()
        indexBottomRight = self.index(row,self.columnCount())
        if role == Qt.ItemDataRole.UserRole:
            self.dataChanged.emit(index,indexBottomRight)
            return True
        if role == Qt.ItemDataRole.CheckStateRole:
            self.setCheckState(index)
            self.dataChanged.emit(index,indexBottomRight)
            return True

        elif role == Qt.ItemDataRole.EditRole:
            if index.column() != 0:
                return False
            newValue = str(value)
            oldValue = str(self._labels.iloc[index.row()])
            columnNameMapper = {oldValue:newValue}
            if oldValue != newValue:
                self.parent().renameColumn(columnNameMapper)
                self.updateData(value,index)
                self.dataChanged.emit(index,index)
            return True

    def data(self, index, role=Qt.ItemDataRole.DisplayRole): 
        ""

        if not index.isValid(): 
            return QVariant()

        columnIndex = index.column()

        if role == Qt.ItemDataRole.DisplayRole and columnIndex == 0: 
            rowIndex = index.row() 
            if rowIndex >= 0 and rowIndex < self._labels.index.size:
                return str(self._labels.iloc[index.row()])
        elif role == Qt.ItemDataRole.FontRole:
            font = getStandardFont()
            return font
        elif role == Qt.ItemDataRole.ForegroundRole:
            return QColor(getStdTextColor())
        elif role == Qt.ItemDataRole.BackgroundRole:
            return QColor(getStdTextColor())
        elif role == Qt.ItemDataRole.ToolTipRole:
            if columnIndex == 3:
                groupName = self.getGroupNameByTableIndex(index)
                if groupName:
                    groupingName = self.parent().mC.grouping.getCurrentGroupingName()
                    return "Grouping: {}\nGroupName: {}".format(groupingName,groupName)
                else:
                    return "Group Indicator"
            elif columnIndex == 2:
                return "Filter Data"
            elif columnIndex == 4:
                return "Delete Column"
            elif columnIndex == 1:
                if self.getColumnStateByTableIndex(index):
                    return "Remove Column from Graph"
                else:
                    return "Add column to Graph"
            elif columnIndex == 0:
                dataIndex = self.getDataIndex(index.row())
                if dataIndex is not None and dataIndex in self._labels.index:
                    tooltipText = self._labels.loc[dataIndex]
                    if tooltipText in self.tooltipData:
                        return self.tooltipData[tooltipText]
                    else:
                        return ""
            else:
                return ""

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsDragEnabled  | Qt.ItemFlag.ItemIsEditable
        else:
            return Qt.ItemFlag.ItemIsEnabled

    def setNewData(self,labels):
        ""
        self.initData(labels)
        

    def setTooltipdata(self,tooltipData):
        ""
        if isinstance(tooltipData,dict):
            self.tooltipData = tooltipData.copy()

    def search(self,searchString):
        ""
        if self._inputLabels.size == 0:
            return
        if len(searchString) > 0:
      
            boolMask = self._labels.str.contains(searchString,case=False,regex=False)
            self._labels = self._labels.loc[boolMask]
        else:
            self._labels = self._inputLabels
        self.completeDataChanged()

    def sort(self, e = None, how = "ascending"):
        ""
        if self._inputLabels.size == 0:
            return
        if self.lastSearchType is None or self.lastSearchType != how:

            self._labels.sort_values(
                                    inplace = True,
                                    ascending = how == "ascending")
            self.lastSearchType = how

        else:
            self._labels.sort_index(
                                    inplace =  True,
                                    ascending=True)
            self.lastSearchType = None
        self.completeDataChanged()
    
    def completeDataChanged(self):
        ""
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1))

    def rowRangeChange(self,row1, row2):
        ""
        self.dataChanged.emit(self.index(row1,0),self.index(row2,self.columnCount()-1))

    def rowDataChanged(self, row):
        ""
        self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount()-1))

    def resetView(self):
        ""
        self._labels = pd.Series(dtype="object")
        self._inputLabels = self._labels.copy()
        self.completeDataChanged()


class DataTreeViewTable(QTableView):

    def __init__(self, parent=None, rowHeight = 22, mainController = None, sendToThread = None, tableID = None):
        super(DataTreeViewTable, self).__init__(parent)
       
        self.setMouseTracking(True)
        self.setShowGrid(False)
        self.verticalHeader().setDefaultSectionSize(rowHeight)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)
        self.rightClick = False
        self.rightClickMove = False
        self.focusRow = None
        self.focusColumn = None
        self.mC = mainController
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        
        self.rowHeight = rowHeight
        self.sendToThread = sendToThread
        self.tableID = tableID
        self.addMenu()
        self.setStyleSheet("""QTableView {background-color: white;border:None};""")
        
    def checkMenuItem(self,item):
        ""
        if isinstance(item["dataType"],str):
            return  item["dataType"] in ["All",self.tableID]
        elif isinstance(item["dataType"], list):
            return any(item in ["All",self.tableID] for item in item["dataType"])
        
    def addMenu(self):
        "Define Menu"
        try:
            if self.tableID in dataTypeSubMenu:
                mainMenu = createMenu()
                menuCollection = dict()
                menuCollection["main"] = mainMenu
                for menuName,subMenus in dataTypeSubMenu[self.tableID]:
                    
                    if menuName == "main":
                        menus = createSubMenu(main = mainMenu, subMenus = subMenus)
                        self.menu = mainMenu
                    else:
                        menus = createSubMenu(main=menuCollection[menuName],subMenus=subMenus)
                   
                    for menuName, menuObj in menus.items():
                        menuCollection[menuName] = menuObj
                filteredMenuBarItems = [item for item in menuBarItems if self.checkMenuItem(item)]
                for menuAction in filteredMenuBarItems:
                    if menuAction["subM"] in menuCollection:
                        if menuAction["subM"] == "main":
                           
                            action = self.menu.addAction(menuAction["name"])
                        else:
                            if menuAction["name"] == "Help":
                                menuCollection[menuAction["subM"]].addSeparator()
                                
                            action = menuCollection[menuAction["subM"]].addAction(menuAction["name"])
                            if "fnKwargs" in menuAction:
                                action.triggered.connect(lambda _,funcKey = menuAction["funcKey"], kwargs = menuAction["fnKwargs"]: self.prepareMenuAction(funcKey=funcKey,kwargs = kwargs))
                            else:
                                action.triggered.connect(lambda _,funcKey = menuAction["funcKey"]:self.prepareMenuAction(funcKey=funcKey))
                        if "checkable" in menuAction and menuAction["checkable"]:
                            action.setCheckable(True)
                            action.setChecked(self.mC.config.getParam(menuAction["fnKwargs"]["paramName"]))
        except Exception as e:
            print(e)
    
    def resizeColumns(self):
        #self.rowHeight
        columnWidths = [(0,400)] + [(n+1,10) for n in range(self.model().columnCount()-1)]
        for columnId,width in columnWidths:
            self.setColumnWidth(columnId,width)

    def mouseReleaseEvent(self,e):
        #reset move signal
        self.rightClickMove = False
        #find table index by event
        tableIndex = self.mouseEventToIndex(e)
        #tableColumn
        tableColumn = tableIndex.column()
        #get key modifiers
        keyMod = QApplication.keyboardModifiers()
        #is shift key pressed?
        shiftPressed = keyMod == Qt.KeyboardModifier.ShiftModifier
        #check if cmd(mac) or ctrl(windows) is clicked
        ctrlPressed = keyMod == Qt.KeyboardModifier.ControlModifier
        #cast menu if right click
        
        if self.rightClick and not self.rightClickMove and tableColumn == 0:
            if hasattr(self,"menu"):
                self.menu.exec(self.mapToGlobal(e.pos()))
        
        elif tableColumn == 0 and not self.rightClick:
            if not shiftPressed and not ctrlPressed:
                self.selectionModel().clear() 
                self.selectionModel().select(tableIndex,QItemSelectionModel.SelectionFlag.Select)
                #update data (e.g remove selction grey area)
                self.model().completeDataChanged()
            else:
                self.removeSelection(e)  

        elif tableColumn == 1:
            self.addRemoveItems(tableIndex)

        elif tableColumn == 2:
            self.applyFilter(False,tableIndex)
        elif tableColumn == 3:
        
            self.copyDataToClipboard(tableIndex)
             
        
        elif tableColumn == 4:

            self.dropColumn(tableIndex)
        
    

    def keyPressEvent(self,e):
        ""
        if e.key() in [Qt.Key.Key_Delete, Qt.Key.Key_Backspace]:

            #delete selected rows
            self.model().layoutAboutToBeChanged.emit()
            selectedRows = self.selectionModel().selectedRows()
           # self.model().deleteEntriesByIndexList(selectedRows)
            self.selectionModel().clear()
           # self.model().layoutChanged.emit()

            deletedColumns = self.model().getSelectedData(selectedRows)
            self.dropColumnsInDataFrame(deletedColumns)

        elif e.key() == Qt.Key.Key_Escape:
            #clear selection
            self.selectionModel().clear()

        elif e.key() == Qt.Key.Key_Tab:

            if self.focusColumn is not None and self.focusRow is not None:
                self.focusColumn += 1
                if self.focusColumn == self.model().columnCount():
                    self.focusColumn = 0
                self.model().rowDataChanged(self.focusRow)

        elif e.key() in [Qt.Key.Key_Enter, Qt.Key.Key_Return]:
            if self.focusRow is None or self.focusColumn is None:
                super().keyPressEvent(e)
                return 
            #get focused index
            tableIndex = self.model().index(self.focusRow,self.focusColumn)

            if self.focusColumn == 1:
                #handles hitting enter while focu is on add delegate (column 1)
                self.addRemoveItems(tableIndex)

            elif self.focusColumn == 2:

                self.mC.mainFrames["sliceMarks"].applyFilter(columnNames = self.getSelectedData([tableIndex]))

            elif self.focusColumn == 3:

                self.copyDataToClipboard(tableIndex)

            elif self.focusColumn == 4:
                #delete items
                self.dropColumn(tableIndex)
        else:
            if self.focusColumn is not None and self.focusRow is not None:
                currentFocusRow = int(self.focusRow)
                if e.key() == Qt.Key.Key_Down:
                        
                        if self.focusRow < self.model().rowCount() - 1:
                            self.focusRow += 1
                        self.model().rowRangeChange(currentFocusRow,self.focusRow)
 
                if e.key() == Qt.Key.Key_Up:
                        
                        if self.focusRow > 0 and self.focusRow < self.model().rowCount():
                            self.focusRow -= 1
                        self.model().rowRangeChange(self.focusRow,currentFocusRow)
    
    def rowWiseCalculations(self,*args,**kwargs):
        ""
        dlg = BasicOperationDialog(self.mC,dataID = self.mC.getDataID(), selectedColumns = self.getSelectedData())
        dlg.exec()

    def applyFilter(self,calledFromMenu = False, tableIndex = None, **kwargs):
        "Apply filtering (numeric or categorical)"
        if not calledFromMenu:
            if tableIndex is None:
                tableIndex = self.model().index(self.focusRow,0)
            if "columnNames" not in kwargs:
                kwargs["columnNames"] = self.getSelectedData([tableIndex])
        else:
            kwargs["columnNames"] = self.getSelectedData()

        self.mC.mainFrames["sliceMarks"].applyFilter(dragType = self.tableID, **kwargs)
        
    def applyNumericFilterForSelection(self,*args,**kwargs):
        ""

        columnNames = self.getSelectedData()
        if columnNames.size < 2:
            self.mC.sendToWarningDialog(infoText="Please select at least 2 column headers.")
            return 
        dlg = ICNumericFilterForSelection(self.mC,columnNames)
        dlg.exec()
        


    def leaveEvent(self, event):
   
        if self.rightClick:
            return
        if self.state() == QAbstractItemView.State.EditingState:
            return 
        if self.focusRow is not None:
            dataRow = int(self.focusRow)
            self.focusColumn = None
            self.focusRow = None
            self.model().rowDataChanged(dataRow)

    def mousePressEvent(self,e):
        ""
        tableIndex = self.mouseEventToIndex(e)
        if e.buttons() == Qt.MouseButton.RightButton:
            self.startIndex = tableIndex
            self.rightClick = True
        elif e.buttons() == Qt.MouseButton.LeftButton:
            if tableIndex.column() == 0:
                self.removeSelection(e)
                super(QTableView,self).mousePressEvent(e)
                self.saveDragStart()
            self.rightClick = False
        else:
            self.rightClick = False

    def addRemoveItems(self,tableIndex):
        ""
        if self.model().getColumnStateByTableIndex(tableIndex):
            self.removeItemFromRecieverbox(tableIndex)
        else:
            self.sendItemToRecieverbox(tableIndex)
        #update state (clicked) and refresh table
       
        self.model().setColumnState(tableIndex)
        self.model().rowDataChanged(tableIndex.row())

    def isGroupigActive(self):
        ""
        return np.any(self.model().columnInGrouping)

    def sendItemToRecieverbox(self, tableIndex):
        ""
        items = self.getSelectedData(indices=[tableIndex])
        funcProps = {"key":"receiverBox:addItems","kwargs":{"columnNames":items,"dataType":self.tableID}}
        funcProps["kwargs"]["dataID"] = self.mC.getDataID()
        self.mC.sendRequest(funcProps)

    def removeItemFromRecieverbox(self, tableIndex):
        ""
        items = self.getSelectedData(indices=[tableIndex])
        funcProps = {"key":"receiverBox:removeItems","kwargs":{"columnNames":items}}
        self.mC.sendRequest(funcProps)

    def saveDragStart(self):
        ""
        self.currentDragIndx = self.selectionModel().selectedRows()
        selectedData = self.getSelectedData(self.currentDragIndx)
         
        self.parent().parent().parent().parent().updateDragData(selectedData,self.tableID)
       
    def setColumnStateOfDraggedColumns(self):
        
        if hasattr(self,"currentDragIndx"):
            for indx in self.currentDragIndx:
                self.model().setColumnState(indx,newState=True)


    def mouseMoveEvent(self,event):
        
        #check if table is being edited, if yes - return
        if self.state() == QAbstractItemView.State.EditingState:
            return 
        if event.buttons() == Qt.MouseButton.LeftButton:
            
            super(QTableView,self).mouseMoveEvent(event)

        elif event.buttons() == Qt.MouseButton.RightButton:

            endIndex = self.mouseEventToIndex(event)
            for tableIndex in self.getIndicesForRow(endIndex.row()):
                self.selectionModel().select(tableIndex,QItemSelectionModel.SelectionFlag.Select)
            self.rightClickMove = True
    
        else:
            self.focusRow = self.rowAt(event.pos().y())
            index = self.model().index(self.focusRow,0)
            self.focusColumn = self.columnAt(event.pos().x())
            self.model().setData(index,self.focusRow,Qt.ItemDataRole.UserRole)

    def removeSelection(self,e):
        ""
        tableIndex = self.mouseEventToIndex(e)
        self.model().rowDataChanged(tableIndex.row())

    def isSelected(self,tableIndex):
        return tableIndex in self.selectionModel().selectedRows()

    def getIndicesForRow(self,row):
        ""
        numColumns = self.model().columnCount()
        indexList = [self.model().index(row,column) for column in range(numColumns)]
        return indexList

    def getSelectedData(self, indices = None):
        ""
        if indices is None:
            indices = self.selectionModel().selectedRows()
        return self.model().getSelectedData(indices)

    def filterFasta(self, event=None,*args,**kwargs):
        ""

        dlg = AskForFile(placeHolderEdit="Select fasta file.")
        if dlg.exec():
            columnNames = self.getSelectedData()
            if columnNames.index.size == 0:
                self.mC.sendToWarningDialog(infoText = "Select a column containing the identifiers you want to filter on.")
                return
            funcProps = {
                    "key":"data::filterFasta",
                    "kwargs":{
                        "columnNames":columnNames,
                        "fastaFile" : dlg.state
                        }
                    } 
            
            self.sendToThread(funcProps = funcProps, addDataID=True)

    def exportData(self,txtFileFormat,*args,**kwargs):
        ""
        if txtFileFormat == "clipboard":
            self.mC.mainFrames["data"].copyDataFrameToClipboard()
        else:
            self.mC.mainFrames["data"].exportData(txtFileFormat)
   

    def compareGroups(self, event=None, test = None, *args, **kwargs):
        ""
        try:
            if not self.mC.grouping.groupingExists():
                w = WarningMessage(infoText="No Grouping found. Please annotate Groups first.",iconDir = self.mC.mainPath)
                w.exec()
                return
            else: 
                dlg = ICCompareGroups(mainController = self.mC, test = test)
                dlg.exec()
        except Exception as e:
            print(e)

    def summarizeGroups(self,event=None, metric = "min"):
        ""
        if not self.mC.grouping.groupingExists():
            w = WarningMessage(infoText="No Grouping found. Please annotate Groups first.",iconDir = self.mC.mainPath)
            w.exec()
            return
                
        funcProps = {"key":"data::summarizeGroups",
                    "kwargs":{
                        "metric":metric,
                        "grouping":self.mC.grouping.getCurrentGrouping()}}
        self.sendToThread(funcProps,addDataID = True)

    def toggleParam(self,paramName):
        ""
        self.mC.config.toggleParam(paramName)#"perform.transformation.in.place"
        self.sender().setChecked(self.mC.config.getParam("perform.transformation.in.place"))


    def createGroups(self, event=None,**kwargs):
        ""

        if self.mC.data.hasData():
            groupDialog = ICGrouper(self.mC,parent=self)
            groupDialog.exec()
        

    def mouseEventToIndex(self,event):
        "Converts mouse event on table to tableIndex"
        row = self.rowAt(event.pos().y())
        column = self.columnAt(event.pos().x())
        return self.model().index(row,column)

    def renameColumn(self,columnNameMapper):
        ""
        funcProps = {"key":"data::renameColumns",
                    "kwargs":{"columnNameMapper":columnNameMapper,
                            "dataID":self.mC.getDataID()}}
        self.mC.sendRequestToThread(funcProps)
    
    def dropColumn(self, tableIndex):
        "Drops column by table Index"
        selectedRows = [tableIndex]
        #get selected data by index
        deletedColumns = self.model().getSelectedData(selectedRows)
        #self.model().deleteEntriesByIndexList(selectedRows)
        self.dropColumnsInDataFrame(deletedColumns)
        
    def dropColumnsInDataFrame(self,deletedColumns):
        ""
        funcProps = {"key":"data::dropColumns","kwargs":{"columnNames":deletedColumns,"dataID":self.mC.getDataID()}}
        self.mC.sendRequest(funcProps)

    def copyDataToClipboard(self,tableIndex):
      
        funcProps = {"key":"data::copyDataFrameSelection",
                    "kwargs":{"columnNames":self.model().getSelectedData([tableIndex])}}   

        self.sendToThread(funcProps=funcProps,addSelectionOfAllDataTypes=True,addDataID=True)


    def customSorting(self,*args,**kwargs):
        ""
        columnName = self.getSelectedData().values[0]
        dataID = self.mC.mainFrames["data"].getDataID()
        uniqueValues = self.mC.data.getUniqueValues(dataID,columnName)
        customSort = ResortableTable(inputLabels = uniqueValues)
        if customSort.exec():
            sortedValues = customSort.savedData.values
            funcProps = {"key":"data::sortDataByValues",
                         "kwargs":{"columnName":columnName,
                                    "dataID":dataID,
                                    "values":sortedValues}}
            #print(funcProps)
            self.sendToThread(funcProps=funcProps)
    
    def featureSelection(self,*args,**kwargs):
        ""
        ""
        #check if recursive feature elimination should be performed
        if 'RFEVC' in kwargs and kwargs['RFEVC']:
            runRFEVC = True
        else:
            runRFEVC = False

        if self.mC.grouping.groupingExists():
            try:
                columnNames = self.mC.grouping.getColumnNames()
                grouping = self.mC.grouping.getCurrentGrouping() 
                groupFactors = self.mC.grouping.getFactorizedColumns() 
                model = self.sender().text()
                createSubset = self.mC.config.getParam("feature.create.subset")
                fnKwargs = {"columnNames":columnNames,"grouping":grouping,"model":model,"groupFactors":groupFactors,"RFECV":runRFEVC,"createSubset":createSubset}
                self.prepareMenuAction("stats::featureSelection",fnKwargs,addColumnSelection=False)
            except Exception as e:
                print(e)

        else:

            self.mC.sendMessageRequest({"title":"Error..","message":"No Grouping found."})
       
    def fitModel(self,*args,**kwargs):
        if self.mC.grouping.groupingExists():
            try:
                #columnNames = self.mC.grouping.getColumnNames()
                #grouping = self.mC.grouping.getCurrentGrouping()
                if self.sender().text() == "Linear fit":
                    w = ICLinearFitModel(self.mC)
                else:
                    w = ICModelBase(self.mC)
                w.exec()
               # fnKwargs = {"columnNames":columnNames,"grouping":grouping}
                #self.prepareMenuAction("data::smartReplace",fnKwargs,addColumnSelection=False)
            except Exception as e:
                print(e)

        else:

            self.mC.sendMessageRequest({"title":"Error..","message":"No Grouping found."})

    def smartReplace(self,*args,**kwargs):
        ""
        if self.mC.grouping.groupingExists():
            try:
                columnNames = self.mC.grouping.getColumnNames()
                grouping = self.mC.grouping.getCurrentGrouping()
                fnKwargs = {"columnNames":columnNames,"grouping":grouping}
                self.prepareMenuAction("data::smartReplace",fnKwargs,addColumnSelection=False)
            except Exception as e:
                print(e)

        else:

            self.mC.sendMessageRequest({"title":"Error..","message":"No Grouping found."})



    def removeOutliersFromGroup(self,**kwargs):
        ""
        if self.mC.grouping.groupingExists():
            try:
               # columnNames = self.mC.grouping.getColumnNames()
                grouping = self.mC.grouping.getCurrentGrouping()
                fnKwargs = {"grouping":grouping}
                self.prepareMenuAction("data::replaceGroupOutlierWithNaN",fnKwargs,addColumnSelection=False)
            except Exception as e:
                print(e)

        else:

            self.mC.sendMessageRequest({"title":"Error..","message":"No Grouping found."})


    def sendSelectionToQuickSelect(self,mode="unique"):
        "Sends filter props to quick select widget"
        columnName = self.getSelectedData().values[0]
        quickSelect = self.mC.mainFrames["data"].qS
        sep = self.mC.config.getParam("quick.select.separator")
        
        filterProps = {'mode': mode, 'sep': sep, 'columnName': columnName}
        quickSelect.updateQuickSelectData(columnName,filterProps)

    def getUserInput(self,**kwargs):
        #print(kwargs)
        if "requiredColumns" in kwargs:
         
            askUserForColumns = kwargs["requiredColumns"]
            dataID = self.mC.mainFrames["data"].getDataID()
            dragColumns = self.mC.mainFrames["data"].getDragColumns()
            categoricalColumns = self.mC.data.getCategoricalColumns(dataID).values.tolist()
            
            sel = SelectionDialog(selectionNames = askUserForColumns, 
                    selectionOptions = dict([(selectionName, categoricalColumns) for selectionName in askUserForColumns]),
                    
                    selectionDefaultIndex = dict([(askUserForColumns[n],dragColumn) for \
                        n,dragColumn in enumerate(dragColumns) if n < len(askUserForColumns)]))

            if sel.exec():
                fnKwargs = sel.savedSelection
                if "otherKwargs" in kwargs:
                    fnKwargs = {**fnKwargs,**kwargs["otherKwargs"]}
                self.prepareMenuAction(funcKey=kwargs["funcKey"],kwargs=fnKwargs, addColumnSelection=False if not "addColumns" in kwargs else kwargs["addColumns"])

        elif "requireMultipleColumns" in kwargs:
            dataID = self.mC.mainFrames["data"].getDataID()
            categoricalColumns = self.mC.data.getCategoricalColumns(dataID).values.tolist()
            if len(categoricalColumns) == 0:
                self.mC.sendToWarningDialog(infoText="This method requires a categorical column.")
                return
            selectedColumns = self.mC.askForItemSelection(items=categoricalColumns,title = "Please select" if "title" not in kwargs else kwargs["title"])
            if selectedColumns is not None:
                fnKwargs = {kwargs["requireMultipleColumns"]:selectedColumns}
                if "otherKwargs" in kwargs:
                    fnKwargs = {**fnKwargs,**kwargs["otherKwargs"]}
                self.prepareMenuAction(funcKey=kwargs["funcKey"],kwargs=fnKwargs, addColumnSelection=False if not "addColumns" in kwargs else kwargs["addColumns"])

        elif "requiredStr" in  kwargs:
            
            labelText = kwargs["info"]
            splitString, ok = QInputDialog().getText(self,"Provide String",labelText)
            if ok: 
                self.prepareMenuAction(funcKey = kwargs["funcKey"],kwargs = {kwargs["requiredStr"]:splitString})
        
        elif "requiredFloat" in kwargs:
            
            labelText, minValue, maxValue = kwargs["info"], kwargs["min"], kwargs["max"]
            if maxValue == "nColumns":
                maxValue = self.getSelectedData().size
            if "default" in kwargs:
                if isinstance(kwargs["default"],str) and self.mC.config.paramExists(kwargs["default"]): 
                    defaultValue = self.mC.config.getParam(kwargs["default"])
                elif isinstance(kwargs["default"],float) or isinstance(kwargs["default"],int):
                    defaultValue = kwargs["default"]
                else:
                    defaultValue = float(kwargs["default"])
            else:
                defaultValue = (maxValue-minValue)/2

            
            number, ok = QInputDialog().getDouble(self,"Provide float",labelText, defaultValue ,minValue, maxValue, 2 )
            if ok:
                fnKwargs = {kwargs["requiredFloat"]:number}
                if "otherKwargs" in kwargs:
                    fnKwargs = {**fnKwargs,**kwargs["otherKwargs"]}
                self.prepareMenuAction(funcKey = kwargs["funcKey"],kwargs = fnKwargs)
        
        elif "requiredInt" in kwargs:
            
            labelText, minValue, maxValue = kwargs["info"], kwargs["min"], kwargs["max"]
            if maxValue == "nColumns":
                maxValue = self.getSelectedData().size
            elif maxValue == "nDataRows":
                dataID = self.mC.mainFrames["data"].getDataID()
                maxValue = self.mC.data.getRowNumber(dataID)
            
            if "default" in kwargs:
                if isinstance(kwargs["default"],str) and self.mC.config.paramExists(kwargs["default"]): 
                    defaultValue = self.mC.config.getParam(kwargs["default"])
                elif isinstance(kwargs["default"],float) or isinstance(kwargs["default"],int):
                    defaultValue = kwargs["default"]
                else:
                    defaultValue = int(kwargs["default"])
            else:
                defaultValue = int((maxValue-minValue)/2)
            
            number, ok = QInputDialog().getInt(self,"Provide integer",labelText, defaultValue ,minValue, maxValue)
        
            if ok:
                fnKwargs = {kwargs["requiredInt"]:number}
                if "otherKwargs" in kwargs:
                    fnKwargs = {**fnKwargs,**kwargs["otherKwargs"]}
                self.prepareMenuAction(funcKey = kwargs["funcKey"],kwargs = fnKwargs) 
           
           #
        elif "selectFromGroupings" in kwargs:
            if self.mC.grouping.groupingExists():
                try:
                # columnNames = self.mC.grouping.getColumnNames()
                    
                    funcKey = kwargs["funcKey"]
                    groupingKwargs = {"kwargs":{}}
                    groupingKwargs = self.mC.askForGroupingSelection(groupingKwargs,False,
                                    title="Select groupings.",
                                    kwargName = "groupingNames")
                    fnKwargs = groupingKwargs["kwargs"]
                    if "otherKwargs" in kwargs:
                        fnKwargs = {**fnKwargs["kwargs"],**kwargs["otherKwargs"]}
                    funcKey = kwargs["funcKey"]
                    self.prepareMenuAction(funcKey,fnKwargs,addColumnSelection=False,addDataID=True)
                except Exception as e:
                    print(e)

            else:
                self.mC.sendToWarningDialog(infoText="No grouping founds. Please add a grouping first.")
                

        elif "requiredGrouping" in kwargs:
            if self.mC.grouping.groupingExists():
                groupingName = kwargs["requiredGrouping"][0]
                grouping = self.mC.grouping.getCurrentGroupingName()
                fnKwargs = {groupingName:grouping}
                if "otherKwargs" in kwargs:
                    fnKwargs = {**fnKwargs,**kwargs["otherKwargs"]}
                funcKey = kwargs["funcKey"]
                #(fnKwargs)
                self.prepareMenuAction(funcKey,fnKwargs,addColumnSelection=False,addDataID=True)
        
            else:
                self.mC.sendToWarningDialog(infoText="No grouping founds. Please add a grouping first.")
    

    def runExponentialFit(self,fitType,*args,**kwargs):
        ""
        if self.mC.grouping.groupingExists():
            groupingNames = self.mC.grouping.getNames()
            if len(groupingNames) < 3:
                groupingNames = groupingNames + ["None"] * (3-len(groupingNames))
            groupings = ["timeGrouping","replicateGrouping","comparisonGrouping"]
            options = dict([(k,["None"] + groupingNames) for k in groupings])
            defaults = dict([(k,groupingNames[n] if len(groupingNames) > n+1 else groupingNames[0]) for n,k in enumerate(groupings)])
            selDiag = SelectionDialog(groupings,
                                options,
                                defaults,
                                title="Select Grouping for Exponential Fit.")
                        
            if selDiag.exec():

                fnKwargs = selDiag.savedSelection
                fnKwargs["dataID"] = self.mC.getDataID()
                fnKwargs["fitType"] = fitType
                funcProps = {
                    "key" : "stats::fitExponentialCurve",
                    "kwargs" : fnKwargs
                    }
                self.mC.sendRequestToThread(funcProps)
        else:
            self.mC.sendToWarningDialog(infoText = "No grouping founds. Please add a grouping first. (context menu - Groupings - Annotate Groups)")

    def run1DEnrichment(self,**kwargs):
        "Run a 1D Enrichment "
        
        selectedColumns = self.getSelectedData()
        categoricalColumns = self.mC.data.getCategoricalColumns(self.mC.getDataID())
        dlg = ICDSelectItems(data = pd.DataFrame(categoricalColumns), title = "Categorical Columns used in 1D Enrichment.")
        if dlg.exec():
            
            selectedCategoricalColumns = dlg.getSelection().values.flatten()
            labelColumns = [columnName for columnName in categoricalColumns if columnName not in selectedCategoricalColumns]
            selectedLabelColumns = []
            if len(labelColumns) > 0:
                selectedLabelColumnsByUser = self.mC.askForItemSelection(
                        items =  labelColumns, 
                        title="Select columns that should be used to annotate categorical groups.")
                if selectedLabelColumnsByUser is not None:
                    selectedLabelColumns = selectedLabelColumnsByUser.values.flatten().tolist()

            funcProps = {"key":"stats::oneDEnrichment"}
            funcKwargs = {"columnNames":selectedColumns,
                      "labelColumns" : selectedLabelColumns,
                      "alternative":self.mC.config.getParam("1D.enrichment.alternative"),
                      "splitString":self.mC.config.getParam("1D.enrichment.split.string"),
                      "categoricalColumns":selectedCategoricalColumns,
                      "dataID":self.mC.getDataID()}
            funcProps["kwargs"] = funcKwargs
            self.mC.sendRequestToThread(funcProps)

    def runNWayANOVA(self,**kwargs):
        ""
        if self.mC.grouping.groupingExists():
            groupingNames = self.mC.grouping.getNames()
            #currentGrouping = self.mC.grouping.getCurrentGroupingName() 
            N, ok = QInputDialog.getInt(self, 'N-Way Anova', 'Enter number of factors:',min=1,max=self.mC.grouping.getNumberOfGroups())
            if ok:
                groupings = ["Grouping {}".format(n) for n in range(N)]
                options = dict([(k,groupingNames) for k in groupings])
                defaults = dict([(k,groupingNames[n]) for n,k in enumerate(groupings)])
                selDiag = SelectionDialog(
                                groupings,
                                options,
                                defaults,
                                title="Select Grouping for N-Way ANOVA.")
                if selDiag.exec():
                    fnKwargs = {}
                    fnKwargs["groupings"] = np.array([x for x in selDiag.savedSelection.values()])
                    if "otherKwargs" in kwargs:
                        fnKwargs = {**fnKwargs,**kwargs["otherKwargs"]}
                    funcProps = {"key":"stats::runNWayANOVA","kwargs":fnKwargs}
                    funcProps["kwargs"]["dataID"] = self.mC.getDataID()
                    
                    self.mC.sendRequestToThread(funcProps)
        else:
            self.mC.sendToWarningDialog(infoText = "No grouping founds. Please add a grouping first. (context menu - Groupings - Annotate Groups)")

    def runRMOneTwoWayANOVA(self,**kwargs):

        if self.mC.grouping.groupingExists():
            groupingNames = self.mC.grouping.getNames()
            groupings = ["withinGrouping1","withinGrouping2","subjectGrouping"]
            options = dict([(k,["None"] + groupingNames) for k in groupings])
            defaults = dict([(k,groupingNames[n]) for n,k in enumerate(groupings)])
            selDiag = SelectionDialog(groupings,
                            options,
                            defaults,
                            title="Select Grouping for Repeated measure one/two-way ANOVA.")
                    
            if selDiag.exec():
                fnKwargs = selDiag.savedSelection
                if fnKwargs["withinGrouping1"] == fnKwargs["withinGrouping2"]:
                    self.mC.sendToWarningDialog(infoText = "withinGroupings cannot be the same. Set withinGrouping2 to None to perform 1W ANOVA.")
                    return
                if fnKwargs["subjectGrouping"] == "None":
                    self.mC.sendToWarningDialog(infoText = "subjectGroupings is required!")
                    return

                if "otherKwargs" in kwargs:
                    fnKwargs = {**fnKwargs,**kwargs["otherKwargs"]}
                funcProps = {"key":"stats::runRMTwoWayANOVA","kwargs":fnKwargs}
                funcProps["kwargs"]["dataID"] = self.mC.getDataID()
                self.mC.sendRequest(funcProps)

        else:
            self.mC.sendToWarningDialog(infoText = "No grouping founds. Please add a grouping first. (context menu - Groupings - Annotate Groups)")

    def runMixedANOVA(self,**kwargs):
        ""
        if self.mC.grouping.groupingExists():
            groupingNames = self.mC.grouping.getNames()
            currentGrouping = self.mC.grouping.getCurrentGroupingName() 
            selDiag = SelectionDialog(["groupingWithin","groupingBetween","groupingSubject"],
                            {"groupingWithin":groupingNames,"groupingBetween":groupingNames,"groupingSubject":groupingNames},
                            {"groupingWithin":currentGrouping,"groupingBetween":currentGrouping,"groupingSubject":currentGrouping},
                            title="Select Grouping for Mixed two-way ANOVA.")
            if selDiag.exec():
                fnKwargs = selDiag.savedSelection
                if "otherKwargs" in kwargs:
                    fnKwargs = {**fnKwargs,**kwargs["otherKwargs"]}
                funcProps = {"key":"stats::runMixedTwoWayANOVA","kwargs":fnKwargs}
                funcProps["kwargs"]["dataID"] = self.mC.getDataID()
                self.mC.sendRequest(funcProps)
        else:
            self.mC.sendToWarningDialog(infoText = "No grouping found. Please add a grouping first. (context menu - Groupings - Annotate Groups)")


    def fisherCategoricalEnrichmentTest(self,*args,**kwargs):
        ""
        dataID = self.mC.getDataID()
        categoricalColumns = self.mC.data.getCategoricalColumns(dataID)
        
        selectedColumn = self.getSelectedData()
        
        selDiag = SelectionDialog(
                ["categoricalColumn","alternative","splitString"],
                {"categoricalColumn":categoricalColumns,
                "alternative":["two-sided","greater","less"],
                "splitString":[";",">","<","_",",","."]},
                {"categoricalColumn":selectedColumn.values[0],"alternative":"two-sided","splitString":";"},
                title="Categorical Enrichment Settings.")
  
        if selDiag.exec():
            #select test settings (side, splitSTring, etc)
            categoricalColumn = selDiag.savedSelection["categoricalColumn"]
            dlg = ICDSelectItems(
                title="Choose categorical column to perform the enrichment on.",
                data = pd.DataFrame(categoricalColumns[categoricalColumns != categoricalColumn]))#filter test column out
            
            if dlg.exec():

                testColumns = dlg.getSelection()
                labelColumns = [columnName for columnName in categoricalColumns if columnName not in testColumns]
                selectedLabelColumns = []

                if len(labelColumns) > 0:
                    selectedLabelColumnsByUser = self.mC.askForItemSelection(
                            items =  labelColumns, 
                            title="Select columns that should be used to annotate categorical groups (such as Gene names).")
                    if selectedLabelColumnsByUser is not None:
                        selectedLabelColumns = selectedLabelColumnsByUser.values.flatten().tolist()
        
                fkey = "stats::runFisherEnrichment"
                columnNames = [categoricalColumn] + selectedLabelColumns + testColumns.values.flatten().tolist() 
                kwargs = {
                            "categoricalColumn":categoricalColumn,
                            "alternative":selDiag.savedSelection["alternative"],
                            "testColumns":testColumns,
                            "labelColumns":selectedLabelColumns,
                            "splitString":selDiag.savedSelection["splitString"],
                            "data" : self.mC.data.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"].copy()
                        }
                funcProps = {"key":fkey,"kwargs":kwargs}
                self.mC.sendRequestToThread(funcProps)
            

    def normalizeToSpecificGroup(self,*args,**kwargs):
        ""
        
        if self.mC.grouping.groupingExists():
            groupingNames = self.mC.grouping.getNames()
            currentGrouping = self.mC.grouping.getCurrentGroupingName()
            groupItems = self.mC.grouping.getGrouping(currentGrouping)
            selectableGroups = pd.Series(groupItems.keys())
            selectedColumns = self.mC.askForItemSelection(items=selectableGroups,title = "Please provide group to normalize to.")
            if selectedColumns is None: return
            fkey = "normalize::toSpecificGroup"
            defaultKwargs = {
                        "dataID": self.mC.getDataID(),
                        "groupingName":currentGrouping,
                        "toGroups": selectedColumns.values.flatten(),
                        "withinGroupingName": None
                    }
          
            if len(groupingNames) > 1:
                w = AskQuestionMessage(
                    parent=self.mC,
                    infoText = "Another grouping was found. Would you like to normalize within another grouping?", 
                    title="Within Grouping",
                    iconDir = self.mC.mainPath,
                    yesCallback = None)
                if w.exec():
                    
                    withinGroupings = pd.Series([x for x in groupingNames if x != currentGrouping])
                    withinGrouping = self.mC.askForItemSelection(items=withinGroupings,title = "Please select one within grouping.", singleSelection=True).values.flatten()[0]
                    if withinGrouping is None:return
                    defaultKwargs["withinGroupingName"] = withinGrouping
                    
            funcProps = {"key":fkey,"kwargs":defaultKwargs}
            self.mC.sendRequestToThread(funcProps)

        else:
            self.mC.sendMessageRequest({"title":"Error..","message":"No Grouping found. Please add a grouping first."})
            
    def getCustomGroupByInput(self,*args,**kwargs):
        ""

        selectableMetrices = OrderedDict([
                    ("sum","sum"),
                    ("median","median"),
                    ("std","std"),
                    ("variance","var"),
                    ("ptp",np.ptp),
                    ("unique values","nunique"),
                    ("count","count"),
                    ("size","size"),
                    ("min","min"),
                    ("max","max"),
                    ("sem","sem")])
        
        selectedMetrices = self.mC.askForItemSelection(items=pd.Series(list(selectableMetrices.keys())),title = "Select aggreagate metrices")
        if selectedMetrices is not None:
            dataID = self.mC.mainFrames["data"].getDataID()
            categoricalColumns = self.mC.data.getCategoricalColumns(dataID).values.tolist()
            selectedColumns = self.mC.askForItemSelection(items=categoricalColumns,title = "Select groupby columns")
            if selectedColumns is not None:
                
                fnKwargs = {"metric":[selectableMetrices[k] for k in selectedMetrices.values.flatten()]}
                fnKwargs["groupbyColumn"]=selectedColumns
                self.prepareMenuAction(kwargs["funcKey"],fnKwargs)

    def openBatchCorrectionDialog(self):
        ""

    def openWebsite(self,link):
        "Opens a website"
        if isinstance(link,str):
            webbrowser.open(link)

    def matchModPeptideSequenceToSites(self,*args,**kwargs):
        ""
        dataID = self.mC.getDataID()
        categoricalColumns = self.mC.data.getCategoricalColumns(dataID).values.tolist()
        selectedColumn = [colName for colName in self.getSelectedData() if colName in categoricalColumns]

        selDiag = SelectionDialog(
                ["proteinGroupColumn","modifiedPeptideColumn"],
                {"proteinGroupColumn":categoricalColumns,"modifiedPeptideColumn":categoricalColumns},
                {"proteinGroupColumn":"","modifiedPeptideColumn":""},
                title="Select the column that contains the modified sequence and protein group (for example uniprot)\nMust match the fasta reg ex (Settings).")
        if selDiag.exec():
            fnKwargs = selDiag.savedSelection
            dlg = AskForFile(placeHolderEdit="Select fasta file to match the modified sequences.")
            if dlg.exec():
                fnKwargs["fastaFilePath"] = dlg.state 
                fnKwargs["dataID"] = dataID
                funcProps = {"key":"proteomics::matchModSequenceToSites","kwargs":fnKwargs}
                self.mC.sendRequestToThread(funcProps)
               # print(fnKwargs)




    def runCombat(self,**kwargs):
        ""
        if self.mC.grouping.groupingExists():
            groupingNames = self.mC.grouping.getNames()
            currentGrouping = self.mC.grouping.getCurrentGroupingName() 
            
            selDiag = SelectionDialog(["groupingName"],{"groupingName":groupingNames},{"groupingName":currentGrouping},title="Select Grouping for Batch correction.")
            
            if selDiag.exec() :
                # askProceed = AskQuestionMessage(infoText="Combat is not thread safe resulting in a freeze of the graphical user interface. Proceed?")
                # askProceed.exec()
                # if askProceed.state:
                fnKwargs = selDiag.savedSelection
                if "otherKwargs" in kwargs:
                    fnKwargs = {**fnKwargs,**kwargs["otherKwargs"]}
                fnKwargs["grouping"] = self.mC.grouping.getGrouping(fnKwargs["groupingName"])
                
                funcProps = {"key":kwargs["funcKey"],"kwargs":fnKwargs}
                funcProps["kwargs"]["dataID"] = self.mC.getDataID()
                self.mC.sendRequestToThread(funcProps)
                #self.prepareMenuAction(funcKey=kwargs["funcKey"],kwargs=fnKwargs, addColumnSelection=False)
        else:
            self.mC.sendMessageRequest({"title":"Error..","message":"No Grouping found."})

    def prepareMenuAction(self,funcKey,kwargs = {},addDataID=True,addColumnSelection=True):

        ""
        try:
            if hasattr(self,funcKey):
                getattr(self,funcKey)(**kwargs)
            else:
                
                funcProps = {"key":funcKey,"kwargs":kwargs}
                if addColumnSelection:
                    funcProps["kwargs"]["columnNames"] = self.getSelectedData()
                    #funcProps["kwargs"]["transformGraph"] = True
            
                self.sendToThread(funcProps=funcProps,addDataID=addDataID)
        except Exception as e:
            print(e)
        
