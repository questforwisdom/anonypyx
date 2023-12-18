from anonypyx import generalization
import pandas as pd

def test_agg_categorical_column():
    series = pd.Series(['A','B'])
    result = generalization.agg_categorical_column(series) 
    assert (result == "A,B" or result == "B,A")


def test_agg_numerical_column():
    series = pd.Series([10,6,7,7,6,4,2,3])
    assert generalization.agg_numerical_column(series) == "2-10"

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

def test_count_sensitive_values_in_partition():
    df = pd.DataFrame({"QI1": [101,102,103,110,110], "QI2": ["A","A","B","B","C"], "S":[10,20,10,21,10]}, index=[1,2,3,4,5])
    partition = [1,2,3]
    sensitive_column = "S"
    result = generalization.count_sensitive_values_in_partition(df, partition, sensitive_column) 
    expected = pd.DataFrame([{'S': 10, 'count': 2}, {'S': 20, 'count': 1}])
    assert_data_set_equal(result, expected)

def test_generalize_partition():
    df = pd.DataFrame({"QI1": [101,102,103,110,110], "QI2": [5,5,4,4,8], "S":[10,20,10,21,10]}, index=[1,2,3,4,5])
    partition = [1,2,3]
    sensitive_column = "S"
    aggregations = {"QI1": generalization.agg_numerical_column, "QI2": generalization.agg_numerical_column}
    result = generalization.generalize_partition(df, partition, aggregations, sensitive_column)
    result = pd.DataFrame(result)
    expected = pd.DataFrame([{'QI1': '101-103', 'QI2': '4-5', 'S': 10, 'count': 2}, {'QI1': '101-103', 'QI2': '4-5', 'S': 20, 'count': 1}])
    assert_data_set_equal(result, expected)

def test_generalization_works_without_sensitive_attribute():
    df = pd.DataFrame({"QI1": [101,102,103,110,110], "QI2": [5,5,4,4,8], "S":[10,20,10,21,10]}, index=[1,2,3,4,5])
    partition = [1,2,3]
    unaltered_columns = []
    aggregations = {"QI1": generalization.agg_numerical_column, "QI2": generalization.agg_numerical_column, "S": generalization.agg_numerical_column}
    result = generalization.generalize_partition(df, partition, aggregations, unaltered_columns)
    result = pd.DataFrame(result)
    expected = pd.DataFrame([{'QI1': '101-103', 'QI2': '4-5', 'S': '10-20', 'count': 3}])
    assert_data_set_equal(result, expected)

def test_generalization_works_with_ignored_attributes():
    df = pd.DataFrame({"QI1": [101,102,103,110,110], "QI2": [5,5,4,4,8], "S":[10,20,10,21,10]}, index=[1,2,3,4,5])
    partition = [1,2,3]
    unaltered_columns = ["QI2", "S"]
    aggregations = {"QI1": generalization.agg_numerical_column}
    result = generalization.generalize_partition(df, partition, aggregations, unaltered_columns)
    result = pd.DataFrame(result)
    expected = pd.DataFrame([{'QI1': '101-103', 'QI2': 5, 'S': 10, 'count': 1}, {'QI1': '101-103', 'QI2': 5, 'S': 20, 'count': 1}, {'QI1': '101-103', 'QI2': 4, 'S': 10, 'count': 1}])

    assert_data_set_equal(result, expected)

