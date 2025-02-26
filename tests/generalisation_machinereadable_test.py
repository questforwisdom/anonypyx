from anonypyx.generalisation.machinereadable import *
from tests.util import *
import pandas as pd
from pandas import testing as tm

import pytest

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
    df["QI2"] = df["QI2"].astype("category")
    partition = [1,2,3]
    sensitive_column = "S"
    return df, partition, [sensitive_column]

@pytest.fixture
def single_value_mixed_df_fixture():
    df = pd.DataFrame({
        "QI1": [101,101,101,110,110], 
        "QI2": ["A","A","A","B","C"], 
        "S":[10,20,10,21,10]
    }, index=[1,2,3,4,5])
    df["QI2"] = df["QI2"].astype("category")
    partition = [1,2,3]
    sensitive_column = "S"
    return df, partition, [sensitive_column]
@pytest.fixture
def mixed_schema():
    mixed_schema = MachineReadable({'QI2': ['QI2_A', 'QI2_B', 'QI2_C']}, {'QI1': ('QI1_min', 'QI1_max')}, ['S'])
    return mixed_schema

def test_generalise_integer_data_set(numerical_df_fixture):
    df, partition, _ = numerical_df_fixture
    strategy = MachineReadable.create_for_data(df, ['QI1', 'QI2'])
    result = strategy.generalise(df, [partition])

    expected = pd.DataFrame({
        'QI1_min': 101,
        'QI1_max': 103,
        'QI2_min': 4,
        'QI2_max': 5, 
        'S': [10, 20],
        'count': [2, 1],
        'group_id': 0
    })

    assert_data_set_equal(result, expected)

def test_generalise_single_value_machine_readable(single_value_mixed_df_fixture):
    df, partition, _ = single_value_mixed_df_fixture
    strategy = MachineReadable.create_for_data(df, ['QI1', 'QI2'])
    result = strategy.generalise(df, [partition])

    expected = pd.DataFrame({
        'QI1_min': 101,
        'QI1_max': 101,
        'QI2_A': True,
        'QI2_B': False,
        'QI2_C': False,
        'S': [10, 20],
        'count': [2, 1],
        'group_id': 0
    })

    assert_data_set_equal(result, expected)

def test_generalise_mixed_data_set(mixed_df_fixture):
    df, partition, _ = mixed_df_fixture
    strategy = MachineReadable.create_for_data(df, ['QI1', 'QI2'])
    result = strategy.generalise(df, [partition])

    expected = pd.DataFrame({
        'QI1_min': 101,
        'QI1_max': 103,
        'QI2_A': True,
        'QI2_B': True, 
        'QI2_C': False,
        'S': [10, 20],
        'count': [2, 1],
        'group_id': 0
    })

    assert_data_set_equal(result, expected)

def test_schema_creation_does_not_alter_original_df(mixed_df_fixture):
    df, _, _ = mixed_df_fixture
    expected = df.copy()
    strategy = MachineReadable.create_for_data(df, ['QI1', 'QI2'])

    assert_data_set_equal(df, expected)

def test_generalisation_does_not_alter_original_df(mixed_df_fixture):
    df, partition, _ = mixed_df_fixture
    expected = df.copy()
    strategy = MachineReadable.create_for_data(df, ['QI1', 'QI2'])
    strategy.generalise(df, [partition])

    assert_data_set_equal(df, expected)

def test_generalisation_works_without_sensitive_attribute(numerical_df_fixture):
    df, partition, _ = numerical_df_fixture
    strategy = MachineReadable.create_for_data(df, ['QI1', 'QI2', 'S'])
    result = strategy.generalise(df, [partition])

    expected = pd.DataFrame({
        'QI1_min': 101,
        'QI1_max': 103,
        'QI2_min': 4,
        'QI2_max': 5, 
        'S_min': 10,
        'S_max': 20,
        'count': [3],
        'group_id': 0
    })

    assert_data_set_equal(result, expected)

def test_generalise_record_to_schema(mixed_schema):
    record = pd.DataFrame(data={'QI1': 4, 'QI2': 'B', 'S': 2}, index=[0])
    expected = pd.DataFrame(data={'QI1_min': 4, 'QI1_max': 4, 'QI2_A': False, 'QI2_B': True, 'QI2_C': False, 'S': 2}, index=[0])
    result = mixed_schema.generalise(record, [[0]])
    result = result.drop(['count', 'group_id'], axis=1)

    assert_data_set_equal(result, expected)

def test_overlap(mixed_schema):
    columns = mixed_schema.quasi_identifier() + ['S']
    prior_knowledge = {'QI1_min': 4, 'QI1_max': 4, 'QI2_A': 0, 'QI2_B': True, 'QI2_C': 0}
    release = pd.DataFrame([
        [0, 3, True,  True,  False, 200, 1], # 0
        [0, 3, True,  True,  False, 300, 2], # 1
        [2, 4, False, False, True, 100, 3], # 2
        [3, 4, False, True,  True, 500, 2], # 3
        [3, 4, False, True,  True, 700, 1], # 4
        [5, 8, True,  True,  True, 400, 3], # 5
        [9, 9, False, True,  True, 800, 3]  # 6
    ], columns = columns + ['count'])

    expected = [3, 4]
    result = mixed_schema.match(release, prior_knowledge, on=['QI1', 'QI2'])

    assert list(result) == expected

def test_record_intersection_unaltered_column(mixed_schema):
    record_a = pd.Series({'QI1_min': 1, 'QI1_max': 2, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    record_b = pd.Series({'QI1_min': 3, 'QI1_max': 3, 'QI2_A': False, 'QI2_B': True, 'QI2_C': True, 'S': 4})
    expected = pd.Series({'S': 4})
    actual = mixed_schema.intersect(record_a, record_b, ['S'], [], [])

    tm.assert_series_equal(expected, actual)

def test_record_intersection_unaltered_column_empty_intersection(mixed_schema):
    record_a = pd.Series({'QI1_min': 1, 'QI1_max': 2, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    record_b = pd.Series({'QI1_min': 3, 'QI1_max': 3, 'QI2_A': False, 'QI2_B': True, 'QI2_C': True, 'S': 5})
    actual = mixed_schema.intersect(record_a, record_b, ['S'], [], [])

    assert actual is None

def test_record_intersection_interval(mixed_schema):
    record_a = pd.Series({'QI1_min': 1, 'QI1_max': 3, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    record_b = pd.Series({'QI1_min': 2, 'QI1_max': 4, 'QI2_A': False, 'QI2_B': True, 'QI2_C': True, 'S': 4})
    expected = pd.Series({'QI1_min': 2, 'QI1_max': 3})
    actual = mixed_schema.intersect(record_a, record_b, ['QI1'], [], [])

    tm.assert_series_equal(expected, actual)

def test_record_intersection_interval_empty_intersection(mixed_schema):
    record_a = pd.Series({'QI1_min': 1, 'QI1_max': 2, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    record_b = pd.Series({'QI1_min': 3, 'QI1_max': 4, 'QI2_A': False, 'QI2_B': True, 'QI2_C': True, 'S': 4})
    actual = mixed_schema.intersect(record_a, record_b, ['QI1'], [], [])

    assert actual is None

def test_record_intersection_one_hot_set(mixed_schema):
    record_a = pd.Series({'QI1_min': 1, 'QI1_max': 2, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    record_b = pd.Series({'QI1_min': 3, 'QI1_max': 3, 'QI2_A': False, 'QI2_B': True, 'QI2_C': True, 'S': 4})
    expected = pd.Series({'QI2_A': False, 'QI2_B': True, 'QI2_C': False})
    actual = mixed_schema.intersect(record_a, record_b, ['QI2'], [], [])

    tm.assert_series_equal(expected, actual)

def test_record_intersection_one_hot_set_empty_intersection(mixed_schema):
    record_a = pd.Series({'QI1_min': 1, 'QI1_max': 2, 'QI2_A': True, 'QI2_B': False, 'QI2_C': False, 'S': 4})
    record_b = pd.Series({'QI1_min': 3, 'QI1_max': 3, 'QI2_A': False, 'QI2_B': True, 'QI2_C': True, 'S': 4})
    actual = mixed_schema.intersect(record_a, record_b, ['QI2'], [], [])

    assert actual is None

def test_record_intersection_on_multiple_attributes(mixed_schema):
    record_a = pd.Series({'QI1_min': 1, 'QI1_max': 2, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    record_b = pd.Series({'QI1_min': 2, 'QI1_max': 3, 'QI2_A': False, 'QI2_B': True, 'QI2_C': True, 'S': 4})
    expected = pd.Series({'QI1_min': 2, 'QI1_max': 2, 'QI2_A': False, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    actual = mixed_schema.intersect(record_a, record_b, ['QI1', 'QI2', 'S'], [], [])

    tm.assert_series_equal(expected, actual, check_like=True)

def test_record_intersection_copy_values(mixed_schema):
    record_a = pd.Series({'QI1_min': 1, 'QI1_max': 2, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    record_b = pd.Series({'QI1_min': 2, 'QI1_max': 3, 'QI2_A': False, 'QI2_B': True, 'QI2_C': True, 'S': 4})
    expected = pd.Series({'QI1_min': 1, 'QI1_max': 2, 'QI2_A': False, 'QI2_B': True, 'QI2_C': True, 'S': 4})
    actual = mixed_schema.intersect(record_a, record_b, ['S'], ['QI1'], ['QI2'])

    tm.assert_series_equal(expected, actual, check_like=True)

def test_get_values_unaltered_value(mixed_schema):
    record = pd.Series({'QI1_min': 1, 'QI1_max': 3, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    assert mixed_schema.values_for(record, 'S') == {4}

def test_get_values_one_hot_set(mixed_schema):
    record = pd.Series({'QI1_min': 1, 'QI1_max': 3, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    assert mixed_schema.values_for(record, 'QI2') == {'A', 'B'}

def test_get_values_interval(mixed_schema):
    record = pd.Series({'QI1_min': 1, 'QI1_max': 3, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    assert mixed_schema.values_for(record, 'QI1') == {1, 2, 3}

def test_get_values_interval_single_value(mixed_schema):
    record = pd.Series({'QI1_min': 3, 'QI1_max': 3, 'QI2_A': True, 'QI2_B': True, 'QI2_C': False, 'S': 4})
    assert mixed_schema.values_for(record, 'QI1') == {3}
