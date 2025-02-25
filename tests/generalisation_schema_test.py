from anonypyx.generalisation.schema import *

from tests.util import *

import pandas as pd
import pytest

def test_build_column_groups():
    df = pd.DataFrame(data={
        'QI1': ['A', 'B', 'C'],
        'QI2': [1, 2, 3],
        'QI3': ['foo', 'bar', 'qux'],
        'S': [100, 200, 300]
    })
    categorical_expected = ['QI1', 'QI3']
    integer_expected = ['QI2']
    unaltered_expected = ['S']

    for col in categorical_expected:
        df[col] = df[col].astype("category")

    categorical_actual, integer_actual, unaltered_actual = build_column_groups(df, ['QI1', 'QI2', 'QI3'])

    assert categorical_actual == categorical_expected
    assert integer_actual == integer_expected
    assert unaltered_actual == unaltered_expected

def test_count_sensitive_values():
    df = pd.DataFrame(data={
        'QI1': ['A', 'B', 'C', 'D'],
        'QI2': [1, 2, 3, 4],
        'QI3': ['foo', 'bar', 'qux', 'bar'],
        'S': [100, 200, 300, 200]
    })
    expected = pd.DataFrame(data={
        'S': [100, 200],
        'count': [1, 2]
    })
    actual = count_sensitive_values_in_partition(df, [0, 1, 3], ['S'])

    assert_data_set_equal(actual, expected)

def test_count_sensitive_values_with_multiple_columns():
    df = pd.DataFrame(data={
        'QI1': ['A', 'B', 'C', 'D', 'E'],
        'QI2': [1, 2, 3, 4, 5],
        'QI3': ['foo', 'bar', 'qux', 'bar', 'foo'],
        'S1': [100, 200, 300, 200, 200],
        'S2': ['a', 'b', 'c', 'd', 'b']
    })
    expected = pd.DataFrame(data={
        'S1': [100, 200, 200],
        'S2': ['a', 'b', 'd'],
        'count': [1, 2, 1]
    })
    actual = count_sensitive_values_in_partition(df, [0, 1, 3, 4], ['S1', 'S2'])

    assert_data_set_equal(actual, expected)
