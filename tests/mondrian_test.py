import anonypyx
from anonypyx import models
from anonypyx import mondrian
import pandas as pd

data = [
    [6, "1", "test1", "x", 20],
    [6, "1", "test1", "x", 30],
    [8, "2", "test2", "x", 50],
    [8, "2", "test3", "w", 45],
    [8, "1", "test2", "y", 35],
    [4, "2", "test3", "y", 20],
    [4, "1", "test3", "y", 20],
    [2, "1", "test3", "z", 22],
    [2, "2", "test3", "y", 32],
]

columns = ["col1", "col2", "col3", "col4", "col5"]
categorical = set(("col2", "col3", "col4"))

def test_k_anonymity():
    df = pd.DataFrame(data=data, columns=columns)

    for name in categorical:
        df[name] = df[name].astype("category")

    feature_columns = ["col1", "col2", "col3"]
    m = mondrian.Mondrian(df, feature_columns)
    partitions = m.partition([models.kAnonymity(2)])
    print(f"partitions: {partitions}")


def test_distinct_l_diversity():
    df = pd.DataFrame(data=data, columns=columns)

    for name in categorical:
        df[name] = df[name].astype("category")

    feature_columns = ["col1", "col2", "col3"]
    sensitive_column = "col4"

    m = mondrian.Mondrian(df, feature_columns)
    partitions = m.partition([models.DistinctLDiversity(2, sensitive_column)])

    print(f"partitions: {partitions}")


def test_t_closeness():
    df = pd.DataFrame(data=data, columns=columns)

    for name in categorical:
        df[name] = df[name].astype("category")

    feature_columns = ["col1", "col2", "col3"]
    sensitive_column = "col4"

    m = mondrian.Mondrian(df, feature_columns)
    partitions = m.partition([models.tCloseness(0.2, df, sensitive_column, models.max_distance_metric)])

    print(f"partitions: {partitions}")


def test_get_spans():
    df = pd.DataFrame(data=data, columns=columns)

    for name in categorical:
        df[name] = df[name].astype("category")

    feature_columns = ["col1", "col2", "col3"]

    m = mondrian.Mondrian(df, feature_columns)
    spans = m.get_spans(df.index)

    assert {"col1": 6, "col2": 2, "col3": 3} == spans

def test_get_spans_with_scale():
    df = pd.DataFrame(data=data, columns=columns)
    scale = {"col1": 6, "col2": 4, "col3": 5}

    for name in categorical:
        df[name] = df[name].astype("category")

    feature_columns = ["col1", "col2", "col3"]

    m = mondrian.Mondrian(df, feature_columns)
    spans = m.get_spans(df.index, scale)

    assert {"col1": 6/6, "col2": 2/4, "col3": 3/5} == spans
