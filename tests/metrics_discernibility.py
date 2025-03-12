import pandas as pd

import pytest

from anonypyx.metrics.discernibility import discernibility_penalty
from anonypyx.metrics.preprocessing import PreparedUtilityDataFrame
from anonypyx.generalisation import MachineReadable

def test_discernibility_penalty():
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
    prepared = PreparedUtilityDataFrame(df, schema, ['QI1', 'QI2'])

    assert 13 == discernibility_penalty(prepared)

def test_discernibility_penalty_no_generalisation():
    df = pd.DataFrame(data={
        'QI1_A': [True, True, False, False, False],
        'QI1_B': [False, False, False, True, True],
        'QI1_C': [False, False, True, False, False],
        'QI2_min': [1, 10, -300, -42, 0],
        'QI2_max': [1, 10, -300, -42, 0],
        'S': [1, 2, 3, 4, 4],
        'count': [1, 1, 1, 1, 1]
    })
    schema = MachineReadable({'QI1': ['QI1_A', 'QI1_B', 'QI1_C']}, {'QI2': ['QI2_min', 'QI2_max']}, ['S'])
    prepared = PreparedUtilityDataFrame(df, schema, ['QI1', 'QI2'])

    assert 5 == discernibility_penalty(prepared)
