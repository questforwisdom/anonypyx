import pandas as pd

from pandas.testing import assert_frame_equal
import pytest

from anonypyx.metrics import preprocessing as pp
from anonypyx.generalisation import MachineReadable

from tests.util import *

def test_preprocess_original_data_for_privacy():
    df = pd.DataFrame(data={
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4]
    })
    expected = pd.DataFrame(data={
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4]
    }, index=pd.Index(data=[0,1,2,3], name='ID'))

    actual = pp.preprocess_original_data_for_privacy(df)

    assert_frame_equal(expected, actual)

def test_preprocess_original_data_for_utility():
    df = pd.DataFrame(data={
        'ID': [0, 1, 2, 3, 4],
        'QI1': ['A', 'A', 'A', 'B', 'C'],
        'QI2': [1, 10, 1, 42, -300],
        'S': [1, 2, 1, 3, 4]
    })
    df['QI1'] = df['QI1'].astype('category')
    expected = pd.DataFrame(data={
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    expected['QI1'] = expected['QI1'].astype('category')

    prepared = pp.PreparedUtilityDataFrame.from_raw_data(df, ['QI1', 'QI2'])
    actual = prepared.df().drop(columns='group_id') # drop this column because order is nondeterministic

    assert_frame_equal(expected, actual, check_like=True)

def test_preprocess_original_data_for_utility_categorical_sensitive_attribute():
    df = pd.DataFrame(data={
        'ID': [0, 1, 2, 3, 4],
        'QI1': ['A', 'A', 'A', 'B', 'C'],
        'QI2': [1, 10, 1, 42, -300],
        'S': [1, 2, 1, 3, 4]
    })
    df['QI1'] = df['QI1'].astype('category')
    df['S'] = df['S'].astype('category')
    expected = pd.DataFrame(data={
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    expected['QI1'] = expected['QI1'].astype('category')
    expected['S'] = expected['S'].astype('category')

    prepared = pp.PreparedUtilityDataFrame.from_raw_data(df, ['QI1', 'QI2'])
    actual = prepared.df().drop(columns='group_id') # drop this column because order is nondeterministic

    assert_frame_equal(expected, actual, check_like=True)

def test_preprocess_prediction():
    df = pd.DataFrame(data={
        'ID':  [0, 1, 2, 3],
        '1': [2, 1, 0, 1],
        '2': [1, 1, 0, 2],
        '3': [0, 3, 2, 3],
        '4': [0, 0, 0, 4]
    })
    expected = pd.DataFrame(data={
        '1': [2/3, 1/5,   0, 1/10],
        '2': [1/3, 1/5,   0, 2/10],
        '3': [  0, 3/5,   1, 3/10],
        '4': [  0,   0,   0, 4/10]
    }, index=pd.Index(data=[0,1,2,3],name='ID'))

    actual = pp.preprocess_prediction(df)

    assert_frame_equal(expected, actual)

def test_utility_group_sizes():
    df = pd.DataFrame(data={
        'QI1_A': [True, True, True, False, False],
        'QI1_B': [False, False, False, True, True],
        'QI1_C': [False, False, False, True, True],
        'QI2_min': [1, 1, -42, -300, -300],
        'QI2_max': [10, 10, -42, 42, 42],
        'S': [1, 2, 1, 3, 4],
        'count': [1, 1, 3, 2, 2]
    })
    schema = MachineReadable({'QI1': ['QI1_A', 'QI1_B', 'QI1_C']}, {'QI2': ('QI2_min', 'QI2_max')}, ['S'])
    prepared = pp.PreparedUtilityDataFrame(df, schema, ['QI1', 'QI2'])

    assert {2, 3, 4} == {prepared.group_size(0), prepared.group_size(1), prepared.group_size(2)}
