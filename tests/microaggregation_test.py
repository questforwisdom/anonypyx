import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pytest
from anonypyx.algorithms.microaggregation import (
    MDAVGeneric,
    FMDAV,
    RandomChoiceAggregation,
)


def test_MDAVGeneric_continuous_data_already_normalized_for_debugging():
    data = [
        [1.5, 1.5, 1.5],  # farthest from mean at (0,0,0)
        [0.5, 0.5, 0.5],  # cloest to this point
        [-1.5, -1.5, -0.5],  # farthest from (1.5, 1.5, 1.5)
        [0, -0.5, -1.5],  # closest to (-1.5, -1.5, -0.5)
        [0, 0, 0],  # remaining
        [0.5, 0, 0],  # remaining
    ]
    columns = ["num1", "num2", "num3"]
    df = pd.DataFrame(data=data, columns=columns)

    k = 2
    mdav_partitioner = MDAVGeneric(k, columns)
    fmdav_partitioner = FMDAV(k, columns)

    mdav_actual = mdav_partitioner.partition(df)
    fmdav_actual = fmdav_partitioner.partition(df)

    sorted_mdav_actual = sorted([sorted(subset) for subset in mdav_actual])
    sorted_fmdav_actual = sorted([sorted(subset) for subset in fmdav_actual])

    # Expected clusters based on distance from centroid
    # Centroid is near (0,0,0). Point [0] is farthest, pairs with [1] (closest to it).
    # Point [2] is next farthest, pairs with [3]. Points [4,5] remain.
    expected = [[0, 1], [2, 3], [4, 5]]

    assert (
        sorted_mdav_actual == expected
    ), "MDAVGeneric did not produce expected clusters"
    assert sorted_fmdav_actual == expected, "FMDAV did not produce expected clusters"


def test_continuous_data():
    data = [
        [0, 0, 0],
        [0, 1, -1],
        [100, 99.9, 101.1],
        [100, 100, 100],
        [0, 200, 0],
        [0, 200, 0],
        [-500, 3, 2],
        [-510, 3.141, 2],
    ]
    columns = ["num1", "num2", "num3"]
    df = pd.DataFrame(data=data, columns=columns)

    k = 2
    mdav_partitioner = MDAVGeneric(k, columns)
    fmdav_partitioner = FMDAV(k, columns)

    mdav_actual = mdav_partitioner.partition(df)
    fmdav_actual = fmdav_partitioner.partition(df)

    sorted_mdav_actual = sorted([sorted(subset) for subset in mdav_actual])
    sorted_fmdav_actual = sorted([sorted(subset) for subset in fmdav_actual])

    # Expected clusters based on distance after normalization
    # After normalization, points [0,1], [2,3], [4,5], [6,7] should pair due to proximity
    expected = [[0, 1], [2, 3], [4, 5], [6, 7]]

    assert (
        sorted_mdav_actual == expected
    ), "MDAVGeneric did not produce expected clusters"
    assert sorted_fmdav_actual == expected, "FMDAV did not produce expected clusters"


def test_categorical_data():
    data = [
        ["A", "1", "foo", "mode", "a"],
        ["A", "1", "foo", "mode", "b"],
        ["A", "1", "bar", "mode", "c"],
        ["A", "1", "bar", "mode", "d"],
        ["B", "2", "bar", "no mode", "e"],
        ["B", "2", "bar", "no mode", "f"],
    ]
    columns = ["col1", "col2", "col3", "col4", "col5"]
    categorical = set(columns)

    df = pd.DataFrame(data=data, columns=columns)
    for name in categorical:
        df[name] = df[name].astype("category")

    k = 2
    mdav_partitioner = MDAVGeneric(k, columns)
    fmdav_partitioner = FMDAV(k, columns)

    mdav_actual = mdav_partitioner.partition(df)
    fmdav_actual = fmdav_partitioner.partition(df)

    sorted_mdav_actual = sorted([sorted(subset) for subset in mdav_actual])
    sorted_fmdav_actual = sorted([sorted(subset) for subset in fmdav_actual])

    # Expected clusters based on Hamming distance after factorization
    # Points [0,1] and [2,3] share most categorical values, [4,5] differ
    expected = [[0, 1], [2, 3], [4, 5]]

    assert (
        sorted_mdav_actual == expected
    ), "MDAVGeneric did not produce expected clusters"
    assert sorted_fmdav_actual == expected, "FMDAV did not produce expected clusters"


def test_mixed_data():
    data = [
        [0, 0, 0, "A"],
        [0, 1, -1, "A"],
        [50, 49.9, 101.1, "B"],
        [50, 50, 100, "B"],
        [0, 200, 0, "C"],
        [0, 200, 0, "C"],
        [-500, 3, 2, "C"],
        [-510, 3.141, 2, "C"],
    ]
    columns = ["num1", "num2", "num3", "cat4"]
    categorical = set(["cat4"])

    df = pd.DataFrame(data=data, columns=columns)
    for name in categorical:
        df[name] = df[name].astype("category")

    k = 2
    mdav_partitioner = MDAVGeneric(k, columns)
    fmdav_partitioner = FMDAV(k, columns)

    mdav_actual = mdav_partitioner.partition(df)
    fmdav_actual = fmdav_partitioner.partition(df)

    sorted_mdav_actual = sorted([sorted(subset) for subset in mdav_actual])
    sorted_fmdav_actual = sorted([sorted(subset) for subset in fmdav_actual])

    # Expected clusters based on combined Euclidean (normalized) and Hamming distance
    # Points [0,1], [2,3], [4,5], [6,7] pair due to numerical proximity and same category
    expected = [[0, 1], [2, 3], [4, 5], [6, 7]]

    assert (
        sorted_mdav_actual == expected
    ), "MDAVGeneric did not produce expected clusters"
    assert sorted_fmdav_actual == expected, "FMDAV did not produce expected clusters"


def test_excluding_columns_works():
    data = [
        [0, 0, 0],
        [0, 1, -1],
        [100, 99.9, 101.1],
        [100, 100, 100],
        [0, 200, 0],
        [0, 200, 0],
    ]
    columns = ["num1", "num2", "num3"]
    df = pd.DataFrame(data=data, columns=columns)

    k = 2
    mdav_partitioner = MDAVGeneric(k, ["num1", "num2"])
    fmdav_partitioner = FMDAV(k, ["num1", "num2"])

    mdav_actual = mdav_partitioner.partition(df)
    fmdav_actual = fmdav_partitioner.partition(df)

    sorted_mdav_actual = sorted([sorted(subset) for subset in mdav_actual])
    sorted_fmdav_actual = sorted([sorted(subset) for subset in fmdav_actual])

    # Expected clusters based on distance in num1, num2 only
    expected = [[0, 1], [2, 3], [4, 5]]

    assert (
        sorted_mdav_actual == expected
    ), "MDAVGeneric did not produce expected clusters"
    assert sorted_fmdav_actual == expected, "FMDAV did not produce expected clusters"


def test_does_not_alter_data():
    data = [
        [0, 0, 0],
        [0, 1, -1],
        [100, 99.9, 101.1],
        [100, 100, 100],
        [0, 200, 0],
        [0, 200, 0],
    ]
    columns = ["num1", "num2", "num3"]
    df = pd.DataFrame(data=data, columns=columns)

    expected = df.copy()

    k = 2
    mdav_partitioner = MDAVGeneric(k, ["num1", "num2"])
    fmdav_partitioner = FMDAV(k, ["num1", "num2"])

    mdav_partitioner.partition(df)

    assert df.equals(expected), "Input DataFrame was altered by MDAVGeneric"

    fmdav_partitioner.partition(df)

    assert df.equals(expected), "Input DataFrame was altered by FMDAV"


def test_RandomChoice_continuous_data():
    data = [
        [0, 0, 0],
        [0, 1, -1],
        [100, 99.9, 101.1],
        [100, 100, 100],
        [0, 200, 0],
        [0, 200, 0],
        [-500, 3, 2],
        [-510, 3.141, 2],
    ]
    columns = ["num1", "num2", "num3"]
    df = pd.DataFrame(data=data, columns=columns)

    partitioner = RandomChoiceAggregation(2, columns)

    expected = [list(df.index[i:i+2]) for i in range(0, 8, 2)]
    actual = partitioner.partition(df)
    # sort to make order deterministic
    sorted_actual = sorted([sorted(subset) for subset in actual])

    assert expected == sorted_actual

def test_RandomChoice_excluding_columns_works():
    # dataset chosen such that clustering is predetermined despite random choice
    data = [
        [0, 0, 0],
        [0, 1, -1],
        [100, 99.9, 101.1],
        [100, 100, 100],
        [0, 200, 0],
        [0, 200, 0],
    ]
    columns = ["num1", "num2", "num3"]
    df = pd.DataFrame(data=data, columns=columns)

    partitioner = RandomChoiceAggregation(2, ["num1", "num2"])

    expected = [list(df.index[i:i+2]) for i in range(0, 6, 2)]
    actual = partitioner.partition(df)
    # sort to make order deterministic
    sorted_actual = sorted([sorted(subset) for subset in actual])

    assert expected == sorted_actual

def test_RandomChoice_does_not_alter_data():
    data = [
        [0, 0, 0],
        [0, 1, -1],
        [100, 99.9, 101.1],
        [100, 100, 100],
        [0, 200, 0],
        [0, 200, 0],
    ]
    columns = ["num1", "num2", "num3"]
    df = pd.DataFrame(data=data, columns=columns)

    expected = df.copy()

    partitioner = RandomChoiceAggregation(2, ["num1", "num2"])
    partitioner.partition(df)

    assert df.equals(expected)
