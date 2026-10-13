import numpy as np
import pandas as pd
import pytest

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.data import DataCollection
from backend.plotting.plotterCalculations import distKDEData
from backend.utils.misc import scaleBetween
from backend.utils.stringOperations import mergeListToString


@pytest.fixture
def dc():
    d = DataCollection.__new__(DataCollection)
    return d


def test_evaluate_column_name_no_collision(dc):
    assert dc.evaluateColumnName("foo", columnList=["bar"]) == "foo"


def test_evaluate_column_name_appends_suffix(dc):
    assert dc.evaluateColumnName("foo", columnList=["foo"]) == "foo_00"


def test_evaluate_column_name_increments_numeric_suffix(dc):
    # regression from 0.12.1: genuine numbered columns were reset to _00
    assert dc.evaluateColumnName("gene_2023", columnList=["gene_2023"]) == "gene_2024"


def test_evaluate_column_name_multiple_collisions(dc):
    existing = ["x", "x_00", "x_01"]
    assert dc.evaluateColumnName("x", columnList=existing) == "x_02"


def test_kernel_density_from_df_index_matches(dc):
    # regression from 0.12.1: index of the original data was returned while
    # the histogram was computed on dropna() data -> length mismatch crash
    data = pd.DataFrame({"a": [1.0, 2.0, np.nan, 3.0, 4.0, np.nan]})
    kdeData, kdeIndex = dc.getKernelDensityFromDf(data)
    assert kdeData.size == kdeIndex.size == 4


def test_kernel_density_from_df_no_nan(dc):
    data = pd.DataFrame({"a": np.random.default_rng(0).normal(size=50)})
    kdeData, kdeIndex = dc.getKernelDensityFromDf(data)
    assert kdeData.size == 50
    assert np.all(np.isfinite(kdeData))
    assert (kdeData >= 0).all()


def test_kernel_density_from_df_all_nan(dc):
    data = pd.DataFrame({"a": [np.nan, np.nan]})
    kdeData, kdeIndex = dc.getKernelDensityFromDf(data)
    assert kdeData.size == 0


def test_dist_kde_data_uniform_shape():
    rng = np.random.default_rng(0)
    out = distKDEData(rng.uniform(size=100))
    assert out.shape == (100,)
    assert np.all(np.isfinite(out))


def test_scale_between():
    out = scaleBetween(np.array([1.0, 2.0, 3.0]), feature_range=(0, 10))
    assert out.min() == 0 and out.max() == 10


def test_merge_list_to_string():
    assert mergeListToString(["a", "b"], joinString=";") == "a;b"
