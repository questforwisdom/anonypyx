import pandas as pd

import pytest

from anonypyx.metrics.query_error import counting_query, counting_query_error, CountingQueryGenerator
from anonypyx.metrics.preprocessing import PreparedUtilityDataFrame
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
    return PreparedUtilityDataFrame(df, schema, ['QI1', 'QI2'])

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
    return PreparedUtilityDataFrame(df, schema, ['QI1', 'QI2'])

def test_counting_query_raw_data_categorical_attribute(raw_data_df):
    query = {'QI1': {'A', 'B'}}
    assert pytest.approx(3.0) == counting_query(query, raw_data_df)
def test_counting_query_raw_data_numerical_attribute(raw_data_df):
    query = {'QI2': (-1000, 20)}
    assert pytest.approx(4.0) == counting_query(query, raw_data_df)

def test_counting_query_raw_data_multiple_attributes(raw_data_df):
    query = {'QI1': {'A', 'B'}, 'QI2': (1, 10), 'S': (1,3)}
    assert pytest.approx(2.0) == counting_query(query, raw_data_df)

def test_counting_query_raw_data_no_match(raw_data_df):
    query = {'QI1': {'A'}, 'QI2': (-300, 300), 'S': (3,4)}
    assert pytest.approx(0.0) == counting_query(query, raw_data_df)

def test_counting_query_raw_data_duplicates(raw_data_df):
    query = {'QI1': {'A', 'C'}, 'QI2': (-300, -10)}
    assert pytest.approx(2.0) == counting_query(query, raw_data_df)

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
    prepared = PreparedUtilityDataFrame(df, schema, ['QI1', 'QI2'])
    query = {'QI1': {'A', 'B'}, 'QI2': (1, 10), 'S': (1,3)}
    assert pytest.approx(2.0) == counting_query(query, prepared)

def test_counting_query_machine_readable_data_complete_overlap(machine_readable_df):
    query = {'QI1': {'A', 'B'}, 'QI2': (1, 10), 'S': (2,3)}
    assert pytest.approx(1.0) == counting_query(query, machine_readable_df)

def test_counting_query_machine_readable_data_partial_overlap(machine_readable_df):
    query = {'QI1': {'A', 'B'}, 'QI2': (6, 10), 'S': (2,3)}
    assert pytest.approx(0.5) == counting_query(query, machine_readable_df)

def test_counting_query_machine_readable_duplicates(machine_readable_df):
    query = {'QI1': {'B', 'C'}, 'QI2': (-300, -172), 'S': (4,4)}
    # 1 equivalence class with 3 records matches the query over the quasi-identifiers
    # 2 of these 3 records match the query over the sensitive attribute S
    # the query overlaps with roughly 50% of the region covered by the generlised quasi-identifiers
    assert pytest.approx(2.0*(300-172+1)/(300-42+1)) == counting_query(query, machine_readable_df)

def test_counting_query_machine_readable_data_no_match(machine_readable_df):
    query = {'QI1': {'C'}, 'QI2': (1, 10), 'S': (2,3)}
    assert pytest.approx(0.0) == counting_query(query, machine_readable_df)

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
    anon_schema = MachineReadable({'QI1': ['QI1_A', 'QI1_B', 'QI1_C']}, {'QI2': ['QI2_min', 'QI2_max']}, ['S'])
    raw_schema = RawData(['QI1'], ['QI2', 'S'], ['QI1', 'QI2'])
    raw_prepared = PreparedUtilityDataFrame(raw_df, raw_schema, ['QI1', 'QI2'])
    anon_prepared = PreparedUtilityDataFrame(anon_df, anon_schema, ['QI1', 'QI2'])
    query = {'QI1': {'A', 'C'}, 'QI2': (6, 10), 'S': (2,3)}
    assert pytest.approx(0.5) == counting_query_error(query, raw_prepared, anon_prepared)

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
    anon_schema = MachineReadable({'QI1': ['QI1_A', 'QI1_B', 'QI1_C']}, {'QI2': ['QI2_min', 'QI2_max']}, ['S'])
    raw_schema = RawData(['QI1', 'S'], ['QI2'], ['QI1', 'QI2'])
    raw_prepared = PreparedUtilityDataFrame(raw_df, raw_schema, ['QI1', 'QI2'])
    anon_prepared = PreparedUtilityDataFrame(anon_df, anon_schema, ['QI1', 'QI2'])
    query = {'QI1': {'A', 'C'}, 'QI2': (6, 10), 'S': {2,3}}

    assert pytest.approx(0.5) == counting_query_error(query, raw_prepared, anon_prepared)

def test_query_generator_for_all_attributes():
    raw_df = pd.DataFrame(data={
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    raw_df['QI1'] = raw_df['QI1'].astype('category')
    raw_df['S'] = raw_df['S'].astype('category')
    raw_schema = RawData(['QI1', 'S'], ['QI2'], ['QI1', 'QI2'])
    raw_prepared = PreparedUtilityDataFrame(raw_df, raw_schema, ['QI1', 'QI2'])
    generator = CountingQueryGenerator(raw_prepared)

    query = generator.generate(3, 1, True)

    assert query == {'QI1': {'A', 'B', 'C'}, 'QI2': (-300, 42), 'S': {1, 2, 3, 4}}

def test_query_generator_for_all_quasi_identifiers():
    raw_df = pd.DataFrame(data={
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    raw_df['QI1'] = raw_df['QI1'].astype('category')
    raw_df['S'] = raw_df['S'].astype('category')

    raw_schema = RawData(['QI1', 'S'], ['QI2'], ['QI1', 'QI2'])
    raw_prepared = PreparedUtilityDataFrame(raw_df, raw_schema, ['QI1', 'QI2'])
    generator = CountingQueryGenerator(raw_prepared)

    query = generator.generate(2, 1, False)

    assert query == {'QI1': {'A', 'B', 'C'}, 'QI2': (-300, 42)}

def test_query_generator_volume_ratio():
    raw_df = pd.DataFrame(data={
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    raw_df['QI1'] = raw_df['QI1'].astype('category')
    raw_df['S'] = raw_df['S'].astype('category')

    raw_schema = RawData(['QI1', 'S'], ['QI2'], ['QI1', 'QI2'])
    raw_prepared = PreparedUtilityDataFrame(raw_df, raw_schema, ['QI1', 'QI2'])
    generator = CountingQueryGenerator(raw_prepared)

    query = generator.generate(3, 0.5, True)

    # the categorical domains are small, this leads to large rounding errors
    # the cubic root of 0.5 is roughly 0.794
    # this corresponds to choosing 2 values for 'QI1', 3 for 'S' and 272 for 'QI1'
    expected_volume = 2 * 3 * 272
    actual_volume = len(query['QI1']) * len(query['S']) * (query['QI2'][1] - query['QI2'][0] + 1)

    assert pytest.approx(expected_volume) == actual_volume
