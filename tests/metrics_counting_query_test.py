import pandas as pd

import pytest

from anonypyx.metrics.query_error import counting_query, counting_query_error
from anonypyx.generalisation import MachineReadable, RawData

@pytest.fixture
def raw_data_df():
    df = pd.DataFrame(data={
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4],
        'count': [1, 1, 1, 2]
    })
    df['QI1'] = df['QI1'].astype('category')
    schema = RawData(['QI1'], ['QI2', 'S'], ['QI1', 'QI2'])
    return df, schema

@pytest.fixture
def machine_readable_df():
    df = pd.DataFrame(data={
        'QI1_A': [True, True, False, False],
        'QI1_B': [False, False, True, True],
        'QI1_C': [False, False, True, True],
        'QI2_min': [1, 1, -300, -300],
        'QI2_max': [10, 10, -42, -42],
        'S': [1, 2, 3, 4],
        'count': [1, 1, 1, 2]
    })
    schema = MachineReadable({'QI1': ['QI1_A', 'QI1_B', 'QI1_C']}, {'QI2': ['QI2_min', 'QI2_max']}, ['S'])
    return df, schema

def test_counting_query_raw_data_categorical_attribute(raw_data_df):
    df, schema = raw_data_df
    query = {'QI1': {'A', 'B'}}
    assert pytest.approx(3.0) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_counting_query_raw_data_numerical_attribute(raw_data_df):
    df, schema = raw_data_df
    query = {'QI2': (-1000, 20)}
    assert pytest.approx(4.0) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_counting_query_raw_data_multiple_attributes(raw_data_df):
    df, schema = raw_data_df
    query = {'QI1': {'A', 'B'}, 'QI2': (1, 10), 'S': (1,3)}
    assert pytest.approx(2.0) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_counting_query_raw_data_no_match(raw_data_df):
    df, schema = raw_data_df
    query = {'QI1': {'A'}, 'QI2': (-300, 300), 'S': (3,4)}
    assert pytest.approx(0.0) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_counting_query_raw_data_duplicates(raw_data_df):
    df, schema = raw_data_df
    query = {'QI1': {'A', 'C'}, 'QI2': (-300, -10)}
    assert pytest.approx(2.0) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_counting_query_machine_readable_data_no_generalisation():
    df = pd.DataFrame(data={
        'QI1_A': [True, True, False, False],
        'QI1_B': [False, False, True, False],
        'QI1_C': [False, False, False, True],
        'QI2_min': [1, 10, 42, -300],
        'QI2_max': [1, 10, 42, -300],
        'S': [1, 2, 3, 4],
        'count': [1, 1, 1, 2]
    })
    schema = MachineReadable({'QI1': ['QI1_A', 'QI1_B', 'QI1_C']}, {'QI2': ['QI2_min', 'QI2_max']}, ['S'])
    query = {'QI1': {'A', 'B'}, 'QI2': (1, 10), 'S': (1,3)}
    assert pytest.approx(2.0) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_counting_query_machine_readable_data_complete_overlap(machine_readable_df):
    df, schema = machine_readable_df
    query = {'QI1': {'A', 'B'}, 'QI2': (1, 10), 'S': (2,3)}
    assert pytest.approx(1.0) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_counting_query_machine_readable_data_partial_overlap(machine_readable_df):
    df, schema = machine_readable_df
    query = {'QI1': {'A', 'B'}, 'QI2': (6, 10), 'S': (2,3)}
    assert pytest.approx(0.5) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_counting_query_machine_readable_duplicates(machine_readable_df):
    df, schema = machine_readable_df
    query = {'QI1': {'B', 'C'}, 'QI2': (-300, -172), 'S': (4,4)}
    # 1 equivalence class with 3 records matches the query over the quasi-identifiers
    # 2 of these 3 records match the query over the sensitive attribute S
    # the query overlaps with roughly 50% of the region covered by the generlised quasi-identifiers
    assert pytest.approx(2.0*(300-172+1)/(300-42+1)) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_counting_query_machine_readable_data_no_match(machine_readable_df):
    df, schema = machine_readable_df
    query = {'QI1': {'C'}, 'QI2': (1, 10), 'S': (2,3)}
    assert pytest.approx(0.0) == counting_query(query, df, schema, ['QI1', 'QI2'])

def test_query_error():
    raw_df = pd.DataFrame(data={
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    raw_df['QI1'] = raw_df['QI1'].astype('category')
    anon_df = pd.DataFrame(data={
        'QI1_A': [True, True, False, False],
        'QI1_B': [False, False, True, True],
        'QI1_C': [False, False, True, True],
        'QI2_min': [1, 1, -300, -300],
        'QI2_max': [10, 10, -42, -42],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    schema = MachineReadable({'QI1': ['QI1_A', 'QI1_B', 'QI1_C']}, {'QI2': ['QI2_min', 'QI2_max']}, ['S'])
    query = {'QI1': {'A', 'C'}, 'QI2': (6, 10), 'S': (2,3)}

    assert pytest.approx(0.5) == counting_query_error(query, raw_df, anon_df, schema, ['QI1', 'QI2'])

def test_query_error_categorical_sensitive_attribute():
    raw_df = pd.DataFrame(data={
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    raw_df['QI1'] = raw_df['QI1'].astype('category')
    raw_df['S'] = raw_df['S'].astype('category')
    anon_df = pd.DataFrame(data={
        'QI1_A': [True, True, False, False],
        'QI1_B': [False, False, True, True],
        'QI1_C': [False, False, True, True],
        'QI2_min': [1, 1, -300, -300],
        'QI2_max': [10, 10, -42, -42],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    anon_df['S'] = anon_df['S'].astype('category')
    schema = MachineReadable({'QI1': ['QI1_A', 'QI1_B', 'QI1_C']}, {'QI2': ['QI2_min', 'QI2_max']}, ['S'])
    query = {'QI1': {'A', 'C'}, 'QI2': (6, 10), 'S': {2,3}}

    assert pytest.approx(0.5) == counting_query_error(query, raw_df, anon_df, schema, ['QI1', 'QI2'])
