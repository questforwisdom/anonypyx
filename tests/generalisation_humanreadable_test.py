from anonypyx.generalisation.humanreadable import *
from tests.util import *
import pandas as pd
from pandas import testing as tm

import pytest

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

def test_generalise_single_value_human_readable(single_value_mixed_df_fixture):
    df, partition, _ = single_value_mixed_df_fixture
    strategy = HumanReadable.create_for_data(df, ['QI1', 'QI2'])
    result = strategy.generalise(df, [partition])

    expected = pd.DataFrame({
        'QI1': '101',
        'QI2': 'A',
        'S': [10, 20],
        'count': [2, 1],
        'group_id': 0
    })

    assert_data_set_equal(result, expected)

def test_generalise_human_readable_output(mixed_df_fixture):
    df, partition, _ = mixed_df_fixture
    strategy = HumanReadable.create_for_data(df, ['QI1', 'QI2'])
    result = strategy.generalise(df, [partition])

    expected = pd.DataFrame({
        'QI1': '101-103',
        'QI2': 'A,B',
        'S': [10, 20],
        'count': [2, 1],
        'group_id': 0
    })

    assert_data_set_equal(result, expected)
