import pandas as pd

import pytest

from anonypyx.metrics import pvpc

def test_pvpc_confidence_levels():
    ground_truth = pd.DataFrame(data={
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4]
    })
    prediction = pd.DataFrame(data={
        'ID':  [  0,   1, 2,   3],
        '1': [1/2, 1/3, 0, 1/4],
        '2': [1/2, 1/3, 0, 1/4],
        '3': [  0, 1/3, 1, 1/4],
        '4': [  0,   0, 0, 1/4]
    })

    assert pytest.approx(0.25) == pvpc.percentage_of_vulnerable_population(ground_truth, prediction, 1.0, 'S')
    assert pytest.approx(0.5) == pvpc.percentage_of_vulnerable_population(ground_truth, prediction, 1.0/2.0, 'S')
    assert pytest.approx(0.75) == pvpc.percentage_of_vulnerable_population(ground_truth, prediction, 1.0/3.0, 'S')
    assert pytest.approx(1.0) == pvpc.percentage_of_vulnerable_population(ground_truth, prediction, 1.0/4.0, 'S')

def test_pvpc_multiple_occurrences_of_some_confidence():
    ground_truth = pd.DataFrame(data={
        'ID': [0, 1, 2, 3],
        'QI1': ['A', 'A', 'B', 'C'],
        'QI2': [1, 10, 42, -300],
        'S': [1, 2, 3, 4]
    })
    prediction = pd.DataFrame(data={
        'ID':  [  0,   1,   2,   3],
        '1': [1/2,   0, 1/4,   0],
        '2': [1/2, 1/2, 1/4, 1/2],
        '3': [  0, 1/2, 1/4,   0],
        '4': [  0,   0, 1/4, 1/2]
    })

    assert pytest.approx(0) == pvpc.percentage_of_vulnerable_population(ground_truth, prediction, 1.0, 'S')
    assert pytest.approx(0.75) == pvpc.percentage_of_vulnerable_population(ground_truth, prediction, 1.0/2.0, 'S')
    assert pytest.approx(0.75) == pvpc.percentage_of_vulnerable_population(ground_truth, prediction, 1.0/3.0, 'S')
    assert pytest.approx(1.0) == pvpc.percentage_of_vulnerable_population(ground_truth, prediction, 1.0/4.0, 'S')


