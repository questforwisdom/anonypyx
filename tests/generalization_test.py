from anonypyx import generalization
import pandas as pd

def test_agg_categorical_column():
    series = pd.Series(['A','B'])
    result = generalization.agg_categorical_column(series) 
    assert (result == "A,B" or result == "B,A")


def test_agg_numerical_column():
    series = pd.Series([10,6,7,7,6,4,2,3])
    assert generalization.agg_numerical_column(series) == "2-10"


def test_count_sensitive_values_in_partition():
    df = pd.DataFrame({"QI1": [101,102,103,110,110], "QI2": ["A","A","B","B","C"], "S":[10,20,10,21,10]}, index=[1,2,3,4,5])
    partition = [1,2,3]
    sensitive_column = "S"
    result = generalization.count_sensitive_values_in_partition(df, partition, sensitive_column) 
    assert result.to_dict() == {'S': {10: 2, 20: 1}}


def test_generalize_partition():
    df = pd.DataFrame({"QI1": [101,102,103,110,110], "QI2": [5,5,4,4,8], "S":[10,20,10,21,10]}, index=[1,2,3,4,5])
    partition = [1,2,3]
    sensitive_column = "S"
    aggregations = {"QI1": generalization.agg_numerical_column, "QI2": generalization.agg_numerical_column}
    result = generalization.generalize_partition(df, partition, aggregations, sensitive_column)
    result = pd.DataFrame(result)
    assert result.equals(pd.DataFrame([{'QI1': '101-103', 'QI2': '4-5', 'S': 10, 'count': 2}, {'QI1': '101-103', 'QI2': '4-5', 'S': 20, 'count': 1}]))
