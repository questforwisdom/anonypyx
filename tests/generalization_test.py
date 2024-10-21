from anonypyx import generalization
import pandas as pd

import pytest
from collections import Counter

def assert_data_set_equal(left_df, right_df):
    # pandas is unable to ignore the row order when comparing dataframes for equality
    # (there is a parameter for ignoring the column order though)
    # this is a workaround

    # ensure that the columns are equal (ignoring order)
    left_columns = left_df.columns.to_list()
    right_columns = right_df.columns.to_list()

    assert set(left_columns) == set(right_columns)

    # ensure that rows are ordered similarily
    left_sorted = left_df.sort_values(left_columns)
    right_sorted = right_df.sort_values(left_columns)

    # reset the index so that pandas forgets the old order
    left_sorted = left_sorted.copy().reset_index(drop=True)
    right_sorted = right_sorted.copy().reset_index(drop=True)

    # do the actual comparison, ignoring column order
    pd.testing.assert_frame_equal(left_sorted, right_sorted, check_like=True)

def assert_multiset_equal(left_list, right_list):
    # equality check for lists where order of elements is ignored

    left_counter = Counter([tuple(element) for element in left_list])
    right_counter = Counter([tuple(element) for element in right_list])

    assert left_counter == right_counter

@pytest.fixture
def numerical_df_fixture():
    df = pd.DataFrame({
        "QI1": [101,102,103,110,110], 
        "QI2": [5,5,4,4,8], 
        "S":[10,20,10,21,10]
    }, index=[1,2,3,4,5])
    partition = [1,2,3]
    sensitive_column = "S"
    return df, partition, [sensitive_column]

@pytest.fixture
def mixed_df_fixture():
    df = pd.DataFrame({
        "QI1": [101,102,103,110,110], 
        "QI2": ["A","A","B","B","C"], 
        "S":[10,20,10,21,10]
    }, index=[1,2,3,4,5])
    partition = [1,2,3]
    sensitive_column = "S"
    return df, partition, [sensitive_column]

def test_to_interval():
    series = pd.Series([5,5,4,4,8])
    expected = [4,8]
    generaliser = generalization.Interval()
    result = generaliser.generalise(series)

    assert result == expected

def test_one_hot_with_one():
    series = pd.Series([0,1,0,0,1])
    expected = [1]
    generaliser = generalization.OneHot()
    result = generaliser.generalise(series)

    assert result == expected

def test_one_hot_all_zeros():
    series = pd.Series([0,0,0,0,0])
    expected = [0]
    generaliser = generalization.OneHot()
    result = generaliser.generalise(series)

    assert result == expected

def test_to_human_readable_set():
    series = pd.Series(['A', 'B', 'B', 'D', 'C'])
    expected = ["A,B,C,D"]
    generaliser = generalization.HumanReadableSet()
    result = generaliser.generalise(series)

    assert result == expected

def test_to_human_readable_interval():
    series = pd.Series([5,5,4,4,8])
    expected = ["4-8"]
    generaliser = generalization.HumanReadableInterval()
    result = generaliser.generalise(series)

    assert result == expected

def test_count_sensitive_values_in_partition(mixed_df_fixture):
    df, partition, unaltered_columns = mixed_df_fixture
    result = generalization.count_sensitive_values_in_partition(df, partition, unaltered_columns) 
    expected = pd.DataFrame([{'S': 10, 'count': 2}, {'S': 20, 'count': 1}])
    assert_data_set_equal(result, expected)

def test_generalize_partition(numerical_df_fixture):
    df, partition, unaltered_columns = numerical_df_fixture
    aggregations = {"QI1": generalization.HumanReadableInterval(), "QI2": generalization.HumanReadableInterval()}
    result = generalization.generalize_partition(df, partition, aggregations, unaltered_columns)
    expected = [
        ['101-103', '4-5', 10, 2],
        ['101-103', '4-5', 20, 1]
    ]
    assert_multiset_equal(result, expected)

def test_generalization_works_without_sensitive_attribute(numerical_df_fixture):
    df, partition, _ = numerical_df_fixture
    unaltered_columns = []
    aggregations = {"QI1": generalization.HumanReadableInterval(), "QI2": generalization.HumanReadableInterval(), "S": generalization.HumanReadableInterval()}
    result = generalization.generalize_partition(df, partition, aggregations, unaltered_columns)
    expected = [
        ['101-103', '4-5', '10-20', 3]
    ]
    assert_multiset_equal(result, expected)

def test_generalization_works_with_ignored_attributes(numerical_df_fixture):
    df, partition, _ = numerical_df_fixture
    unaltered_columns = ["QI2", "S"]
    aggregations = {"QI1": generalization.HumanReadableInterval()}
    result = generalization.generalize_partition(df, partition, aggregations, unaltered_columns)
    expected = [
        ['101-103', 5, 10, 1], 
        ['101-103', 5, 20, 1], 
        ['101-103', 4, 10, 1]
    ]

    assert_multiset_equal(result, expected)

def test_generalisation_can_replace_columns(numerical_df_fixture):
    df, partition, _ = numerical_df_fixture
    unaltered_columns = ["QI2", "S"]
    aggregations = {"QI1": generalization.Interval()}
    result = generalization.generalize_partition(df, partition, aggregations, unaltered_columns)
    expected = [
        [101, 103, 5, 10, 1], 
        [101, 103, 5, 20, 1], 
        [101, 103, 4, 10, 1]
    ]

    assert_multiset_equal(result, expected)

