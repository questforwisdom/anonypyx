from anonypyx.algorithms.minvariance import MInvariance

import pandas as pd

def test_first_release():
    df = pd.DataFrame({
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'B'],
        'QI2': [10, 12, 20, 21],
        'S': [1, 2, 3, 4]
    })

    expected = [[0, 1], [2, 3]]

    algorithm = MInvariance(2)

    actual = algorithm.partition(df)

    assert expected == actual

def test_second_release_no_change():
    last_df = pd.DataFrame({
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'B'],
        'QI2': [10, 12, 20, 21],
        'S': [1, 2, 3, 4]
    })
    last_partition = [[0, 1], [2, 3]]
    df = pd.DataFrame({
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'B'],
        'QI2': [10, 12, 20, 21],
        'S': [1, 2, 3, 4]
    })
    expected = [[0, 1], [2, 3]]
    expected_counterfeits = {}

    algorithm = MInvariance(2, last_df, last_partition)

    actual, actual_counterfeits = algorithm.partition(df)

    assert expected == actual
    assert expected_counterfeits == actual_counterfeits

def test_second_release_two_insertions():
    last_df = pd.DataFrame({
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'B'],
        'QI2': [10, 12, 20, 21],
        'S': [1, 2, 3, 4]
    })
    last_partition = [[0, 1], [2, 3]]
    df = pd.DataFrame({
        'ID': [0, 1, 2, 3, 4, 5],
        'QI1': ['A', 'A', 'B', 'B', 'C', 'A'],
        'QI2': [10, 12, 20, 20, 10],
        'S': [1, 2, 3, 4, 5, 1]
    })
    expected = [[0, 1], [2, 3], [4, 5]]
    expected_counterfeits = {}

    algorithm = MInvariance(2, last_df, last_partition)

    actual, actual_counterfeits = algorithm.partition(df)

    assert expected == actual
    assert expected_counterfeits == actual_counterfeits

def test_second_release_single_deletion():
    last_df = pd.DataFrame({
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'B'],
        'QI2': [10, 12, 20, 21],
        'S': [1, 2, 3, 4]
    })
    last_partition = [[0, 1], [2, 3]]
    df = pd.DataFrame({
        'ID': [0, 2, 3],
        'QI1': ['A', 'B', 'B'],
        'QI2': [10, 20, 21],
        'S': [1, 3, 4]
    })
    expected = [[0], [1, 2]]
    expected_counterfeits = {0: {2: 1}}

    algorithm = MInvariance(2, last_df, last_partition)

    actual , actual_counterfeits= algorithm.partition(df)

    assert expected == actual
    assert expected_counterfeits == actual_counterfeits

def test_second_release_single_replacement_same_sensitive_value():
    last_df = pd.DataFrame({
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'B'],
        'QI2': [10, 12, 20, 21],
        'S': [1, 2, 3, 4]
    })
    last_partition = [[0, 1], [2, 3]]
    df = pd.DataFrame({
        'ID': [0, 1, 2, 4],
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [10, 12, 20, 20],
        'S': [1, 2, 3, 4]
    })
    expected = [[0, 1], [2, 3]]
    expected_counterfeits = {}

    algorithm = MInvariance(2, last_df, last_partition)

    actual , actual_counterfeits= algorithm.partition(df)

    assert expected == actual
    assert expected_counterfeits == actual_counterfeits

def test_second_release_single_replacement_different_sensitive_value():
    last_df = pd.DataFrame({
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'B'],
        'QI2': [10, 12, 20, 21],
        'S': [1, 2, 3, 4]
    })
    last_partition = [[0, 1], [2, 3]]
    df = pd.DataFrame({
        'ID': [0, 1, 2, 4],
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [10, 12, 20, 20],
        'S': [1, 2, 3, 5]
    })
    expected = [[0, 1], [2], [3]]

    algorithm = MInvariance(2, last_df, last_partition)

    actual, actual_counterfeits = algorithm.partition(df)

    assert expected == actual

    assert len(actual_counterfeits) == 2
    assert actual_counterfeits[1] == {4: 1}
    assert len(actual_counterfeits[2]) == 1
    for sensitive_value in actual_counterfeits[2]:
        assert sensitive_value != 5
        assert actual_counterfeits[2][sensitive_value] == 1
