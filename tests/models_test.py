import anonypyx
from anonypyx import models

import pandas as pd
import pytest

@pytest.fixture
def k_anonymity_df():
    data = [
        [1, "A", 5],
        [1, "A", 6],
        [1, "A", 7]
    ]
    df = pd.DataFrame(data=data, columns=["col1", "col2", "col3"])
    df["col2"] = df["col2"].astype("category")

    return df

@pytest.fixture
def distinct_l_diversity_df():
    data = [
        [1, "A", 5],
        [1, "A", 5],
        [1, "A", 6],
        [1, "A", 6],
        [1, "A", 7]
    ]
    df = pd.DataFrame(data=data, columns=["col1", "col2", "col3"])
    df["col2"] = df["col2"].astype("category")

    return df

@pytest.fixture
def t_closeness_df():
    parent_data = [
        ["1", "A", "1"],
        ["1", "B", "1"],
        ["1", "A", "2"],
        ["1", "B", "3"],
        ["1", "A", "3"],
        ["1", "B", "4"],
    ]
    parent_df = pd.DataFrame(data=parent_data, columns=["col1", "col2", "col3"])
    parent_df["col2"] = parent_df["col2"].astype("category")
    parent_df["col3"] = parent_df["col3"].astype("category")
    
    return parent_df

def test_max_distance_metric():
    dist1 = {"A": 0.2, "B": 0.4, "C": 0.6}
    dist2 = {"A": 0.05, "B": 0.3, "C": 0.65}

    assert pytest.approx(0.15, 0.001) == models.max_distance_metric(dist1, dist2)

def test_k_anonymity(k_anonymity_df):
    sensitive_column = "col3"
    assert models.kAnonymity(3).is_enforcable(k_anonymity_df)

def test_k_anonymity_detects_violation(k_anonymity_df):
    sensitive_column = "col3"
    assert not (models.kAnonymity(4).is_enforcable(k_anonymity_df))

def test_distinct_l_diversity(distinct_l_diversity_df):
    sensitive_column = "col3"
    assert (models.DistinctLDiversity(3, sensitive_column).is_enforcable(distinct_l_diversity_df))

def test_distinct_l_diversity_detects_violation(distinct_l_diversity_df):
    sensitive_column = "col3"
    assert not (models.DistinctLDiversity(4, sensitive_column).is_enforcable(distinct_l_diversity_df))

def test_maxdistance_t_closeness(t_closeness_df):
    df = t_closeness_df.loc[[0, 2]] 
    sensitive_column = "col3"
    assert (models.tCloseness(2.01/6.0, t_closeness_df, sensitive_column, models.max_distance_metric).is_enforcable(df))

def test_maxdistance_t_closeness_detects_violation(t_closeness_df):
    df = t_closeness_df.loc[[0, 2]] 
    sensitive_column = "col3"
    assert not (models.tCloseness(1.99/6.0, t_closeness_df, sensitive_column, models.max_distance_metric).is_enforcable(df))

def test_emd_t_closeness_categorical(t_closeness_df):
    df = t_closeness_df.loc[[0, 2]] 
    sensitive_column = "col3"
    assert (models.tCloseness(0.5, t_closeness_df, sensitive_column, models.earth_movers_distance_categorical).is_enforcable(df))

def test_emd_t_closeness_detects_violation_categorical(t_closeness_df):
    df = t_closeness_df.loc[[0, 2]] 
    sensitive_column = "col3"
    assert not (models.tCloseness(0.49, t_closeness_df, sensitive_column, models.earth_movers_distance_categorical).is_enforcable(df))

def test_k_anonymity_no_sensitive_attribute(k_anonymity_df):
    sensitive_column = None
    assert models.kAnonymity(3).is_enforcable(k_anonymity_df)

def test_l_diverse_no_sensitive_attribute(distinct_l_diversity_df):
    sensitive_column = None
    assert not (models.DistinctLDiversity(3, sensitive_column).is_enforcable(distinct_l_diversity_df))

def test_t_closeness_no_sensitive_attribute(t_closeness_df):
    df = t_closeness_df.loc[[0, 2]] 
    sensitive_column = None
    assert not (models.tCloseness(0.5, t_closeness_df, sensitive_column, models.earth_movers_distance_categorical).is_enforcable(df))
