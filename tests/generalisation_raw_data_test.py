
from anonypyx.generalisation.rawdata import *
from tests.util import *
import pandas as pd
from pandas import testing as tm

import pytest

@pytest.fixture
def mixed_df_fixture():
    df = pd.DataFrame({
        "QI1": [101,102,103,102,110,110], 
        "QI2": ["A","A","B","A","B","C"], 
        "S":[10,20,10,20,21,10]
    }, index=[1,2,3,4,5,6])
    df["QI2"] = df["QI2"].astype("category")
    partition = [1,2,3,4]
    return df, partition

@pytest.fixture
def mixed_schema():
    mixed_schema = RawData(['QI2'], ['QI1', 'S'], ['QI1', 'QI2'])
    return mixed_schema

@pytest.fixture
def mixed_schema_with_df():
    mixed_schema = RawData(['QI2'], ['QI1', 'S'], ['QI1', 'QI2'])
    df = pd.DataFrame({
        "QI1": [101,102,103,110,110], 
        "QI2": ["A","A","B","B","C"], 
        "S":[10,20,10,21,10]
    }, index=[1,2,3,4,5])
    df["QI2"] = df["QI2"].astype("category")
    return mixed_schema, df

def test_schema_creation_does_not_alter_original_df(mixed_df_fixture):
    df, _ = mixed_df_fixture
    expected = df.copy()
    schema = RawData.create_for_data(df, ['QI1', 'QI2'])

    assert_data_set_equal(df, expected)

def test_generalisation_does_not_alter_original_df(mixed_df_fixture):
    df, partition = mixed_df_fixture
    expected = df.copy()
    schema = RawData.create_for_data(df, ['QI1', 'QI2'])
    schema.generalise(df, [partition])

    assert_data_set_equal(df, expected)

def test_generalisation_only_counts_duplicates(mixed_df_fixture):
    df, partition = mixed_df_fixture
    old_df = df.drop(4) # drop duplicate
    old_df = old_df.drop(5) # not in partition
    old_df = old_df.drop(6) # not in partition
    old_df['count'] = [1,2,1]

    schema = RawData.create_for_data(df, ['QI1', 'QI2'])
    df = schema.generalise(df, [partition])

    assert_data_set_equal(df, old_df)

def test_generalisation_unaltered_attributes_keep_dtype(mixed_df_fixture):
    df, partition = mixed_df_fixture
    df['S'] = df['S'].astype('category')
    old_df = df.drop(4) # drop duplicate
    old_df = old_df.drop(5) # not in partition
    old_df = old_df.drop(6) # not in partition
    old_df['count'] = [1,2,1]

    schema = RawData.create_for_data(df, ['QI1', 'QI2'])
    df = schema.generalise(df, [partition])

    assert_data_set_equal(df, old_df)

def test_overlap(mixed_schema):
    columns = mixed_schema.quasi_identifier() + ['S']
    prior_knowledge = {'QI1': 4, 'QI2': 'B'}
    release = pd.DataFrame([
        [0, 'A', 200, 4], # 0
        [3, 'B', 300, 2], # 1
        [2, 'C', 100, 1], # 2
        [4, 'B', 500, 1], # 3
        [4, 'C', 700, 1], # 4
        [4, 'B', 400, 2], # 5
        [9, 'B', 800, 3]  # 6
    ], columns = columns + ['count'])

    expected = [3, 5]
    expected = release.iloc[[3,5]].copy()
    result = mixed_schema.match(release, prior_knowledge, on=['QI1', 'QI2'])
    
    assert_data_set_equal(result, expected)

def test_record_intersection(mixed_schema):
    record_a = pd.Series({'QI1': 1, 'QI2': 'A', 'S': 4})
    record_b = record_a.copy()
    expected = record_a.copy()
    actual = mixed_schema.intersect(record_a, record_b, ['QI1', 'QI2', 'S'], [], [])

    tm.assert_series_equal(expected, actual, check_like=True)

def test_record_intersection_when_empty(mixed_schema):
    record_a = pd.Series({'QI1': 1, 'QI2': 'A', 'S': 4})
    record_b = pd.Series({'QI1': 3, 'QI2': 'B', 'S': 4})

    assert mixed_schema.intersect(record_a, record_b, ['QI1', 'QI2', 'S'], [], []) is None

def test_record_intersection_copy_values(mixed_schema):
    record_a = pd.Series({'QI1': 1, 'QI2': 'A', 'S': 4})
    record_b = pd.Series({'QI1': 3, 'QI2': 'B', 'S': 4})
    expected = pd.Series({'QI1': 1, 'QI2': 'B', 'S': 4})
    actual = mixed_schema.intersect(record_a, record_b, ['S'], ['QI1'], ['QI2'])

    tm.assert_series_equal(expected, actual, check_like=True)

def test_get_values(mixed_schema):
    record = pd.Series({'QI1': 1, 'QI2': 'A', 'S': 4})

    assert mixed_schema.values_for(record, 'QI1') == {1}
    assert mixed_schema.values_for(record, 'QI2') == {'A'}
    assert mixed_schema.values_for(record, 'S') == {4}

def test_set_cardinality_is_one(mixed_schema):
    record = pd.Series({'QI1': 1, 'QI2': 'A', 'S': 4})
    assert 1 == mixed_schema.set_cardinality(record, ['QI1'])
    assert 1 == mixed_schema.set_cardinality(record, ['QI2'])
    assert 1 == mixed_schema.set_cardinality(record, ['S'])
    assert 1 == mixed_schema.set_cardinality(record, ['QI1', 'QI2', 'S'])

def test_select_numerical(mixed_schema_with_df):
    mixed_schema, df = mixed_schema_with_df
    query = {'QI1': (101, 102)}
    assert {1, 2} == set(mixed_schema.select(df, query))

def test_select_categorical(mixed_schema_with_df):
    mixed_schema, df = mixed_schema_with_df
    query = {'QI2': {'C'}}
    assert {5} == set(mixed_schema.select(df, query))

def test_select_unaltered_numerical(mixed_schema_with_df):
    mixed_schema, df = mixed_schema_with_df
    query = {'S': (20, 22)}
    assert {2, 4} == set(mixed_schema.select(df, query))

def test_select_unaltered_categorical(mixed_schema_with_df):
    mixed_schema, df = mixed_schema_with_df
    df['S'] = df['S'].astype('category')
    query = {'S': {10, 21}}
    assert {1, 3, 4, 5} == set(mixed_schema.select(df, query))

def test_select_mixed_df(mixed_schema_with_df):
    mixed_schema, df = mixed_schema_with_df
    query = {'QI1': (101, 102), 'QI2': {'A', 'C'}, 'S': (5, 15)}
    assert {1} == set(mixed_schema.select(df, query))

def test_select_empty_selction(mixed_schema_with_df):
    mixed_schema, df = mixed_schema_with_df
    query = {'QI1': (100, 105), 'QI2': {'C'}}
    assert set() == set(mixed_schema.select(df, query))

def test_query_overlap_categorical(mixed_schema):
    query = {'QI2': {'A', 'B'}}
    record = pd.Series({'QI1': -1, 'QI2': 'A', 'S': 4})
    assert 1 == mixed_schema.query_overlap(record, query)

def test_query_overlap_numerical(mixed_schema):
    query = {'QI1': (-1, 1)}
    record = pd.Series({'QI1': -1, 'QI2': 'A', 'S': 4})
    assert 1 == mixed_schema.query_overlap(record, query)

def test_query_overlap_unaltered_categorical(mixed_schema):
    query = {'S': {1,4,6}}
    record = pd.Series({'QI1': -1, 'QI2': 'A', 'S': 4})
    assert 1 == mixed_schema.query_overlap(record, query)

def test_query_overlap_unaltered_numerical(mixed_schema):
    query = {'S': (3, 4)}
    record = pd.Series({'QI1': -1, 'QI2': 'A', 'S': 4})
    assert 1 == mixed_schema.query_overlap(record, query)

def test_query_overlap_multiple_attributes(mixed_schema):
    query = {'QI1': (-1, 1), 'QI2': {'A', 'B'}, 'S': (3,4)}
    record = pd.Series({'QI1': -1, 'QI2': 'A', 'S': 4})
    assert 1 == mixed_schema.query_overlap(record, query)

def test_query_overlap_empty_overlap(mixed_schema):
    query = {'QI1': (-1, 1), 'QI2': {'C'}}
    record = pd.Series({'QI1': -1, 'QI2': 'A', 'S': 4})
    assert 0 == mixed_schema.query_overlap(record, query)
