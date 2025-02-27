import pandas as pd

from pandas.testing import assert_frame_equal
import pytest

from anonypyx.metrics import preprocessing as pp
def test_preprocess_original_data():
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

    actual = pp.preprocess_original_data(df)

    assert_frame_equal(expected, actual)

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
