import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from anonypyx.microaggregation import MDAVGeneric, RandomChoiceAggregation

def test_MDAVGeneric_continuous_data_already_normalized_for_debugging():
    data = [
        [ 1.5,  1.5,  1.5],    # farthest from mean at (0,0,0)
        [ 0.5,  0.5,  0.5],    # cloest to this point
        [-1.5, -1.5, -0.5], # farthest from (1.5, 1.5, 1.5)
        [   0, -0.5, -1.5],    # closest to (-1.5, -1.5, -0.5)
        [   0,    0,    0],          # remaining
        [ 0.5,    0,    0],        # remaining
    ]
    columns = ["num1", "num2", "num3"]
    df = pd.DataFrame(data=data, columns=columns)

    partitioner = MDAVGeneric(df, columns)

    expected = [list(df.index[i:i+2]) for i in range(0, 6, 2)]
    actual = partitioner.partition(2)
    # sort to make order deterministic
    sorted_actual = sorted([sorted(subset) for subset in actual])

    assert expected == sorted_actual

def test_MDAVGeneric_continuous_data():
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

    partitioner = MDAVGeneric(df, columns)

    expected = [list(df.index[i:i+2]) for i in range(0, 8, 2)]
    actual = partitioner.partition(2)
    # sort to make order deterministic
    sorted_actual = sorted([sorted(subset) for subset in actual])

    assert expected == sorted_actual

def test_MDAVGeneric_categorical_data():
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

    partitioner = MDAVGeneric(df, columns)

    expected = [list(df.index[i:i+2]) for i in range(0, 6, 2)]
    actual = partitioner.partition(2)
    # sort to make order deterministic
    sorted_actual = sorted([sorted(subset) for subset in actual])

    assert expected == sorted_actual

def test_MDAVGeneric_mixed_data():
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

    partitioner = MDAVGeneric(df, columns)

    expected = [list(df.index[i:i+2]) for i in range(0, 8, 2)]
    actual = partitioner.partition(2)
    # sort to make order deterministic
    sorted_actual = sorted([sorted(subset) for subset in actual])

    assert expected == sorted_actual

def test_MDAVGeneric_excluding_columns_works():
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

    partitioner = MDAVGeneric(df, ["num1", "num2"])

    expected = [list(df.index[i:i+2]) for i in range(0, 6, 2)]
    actual = partitioner.partition(2)
    # sort to make order deterministic
    sorted_actual = sorted([sorted(subset) for subset in actual])

    assert expected == sorted_actual

def test_MDAVGeneric_does_not_alter_data():
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

    partitioner = MDAVGeneric(df, ["num1", "num2"])
    partitioner.partition(2)

    assert df.equals(expected)

def test_RandomcChoice_continuous_data():
    # dataset chosen such that clustering is predetermined despite random choice
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

    partitioner = RandomChoiceAggregation(df, columns)

    expected = [list(df.index[i:i+2]) for i in range(0, 8, 2)]
    actual = partitioner.partition(2)
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

    partitioner = RandomChoiceAggregation(df, ["num1", "num2"])

    expected = [list(df.index[i:i+2]) for i in range(0, 6, 2)]
    actual = partitioner.partition(2)
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

    partitioner = RandomChoiceAggregation(df, ["num1", "num2"])
    partitioner.partition(2)

    assert df.equals(expected)
