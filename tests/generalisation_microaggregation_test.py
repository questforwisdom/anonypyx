from anonypyx.generalisation import Microaggregation
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

def test_generalise_by_microaggregation(numerical_df_fixture):
    df, partition, _ = numerical_df_fixture
    strategy = Microaggregation.create_for_data(df, ['QI1', 'QI2', 'S'])
    result = strategy.generalise(df, [partition])

    expected = pd.DataFrame({
        'QI1': [df.loc[partition]['QI1'].mean()],
        'QI2': [df.loc[partition]['QI2'].mean()],
        'S': [df.loc[partition]['S'].mean()],
        'count': [3]
    })
    assert_data_set_equal(result, expected)


