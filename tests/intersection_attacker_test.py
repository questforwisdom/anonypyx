import pytest
import pandas as pd

from anonypyx import generalisation
from anonypyx.attackers.intersection_attacker import IntersectionAttacker

@pytest.fixture
def mixed_schema():
    prior_knowledge = pd.DataFrame(data={
        'ID': [0, 1],
        'QI1_min': [1, 3],
        'QI1_max': [1, 3],
        'QI2_A': [True, False],
        'QI2_B': [False, True]
    })
    schema = generalisation.MachineReadable({'QI2': ['QI2_A', 'QI2_B']}, {'QI1': ['QI1_min', 'QI1_max']}, ['S'])
    quasi_identifiers = ['QI1', 'QI2']
    attacker = IntersectionAttacker(prior_knowledge, quasi_identifiers, 'S', schema)

    return attacker, quasi_identifiers

def test_linking_attack(mixed_schema):
    attacker, quasi_identifiers = mixed_schema
    release = pd.DataFrame(data={
        'QI1_min': [1, 1, 3, 3],
        'QI1_max': [2, 2, 3, 3],
        'QI2_A': [True, True, True, True],
        'QI2_B': [False, False, True, True],
        'S': [1, 2, 3, 4],
        'count': [1, 2, 2, 1]
    })
    attacker.observe(release, quasi_identifiers +['S'], [0, 1])

    assert attacker.predict(0, 'S') == {1: 1, 2: 1}
    assert attacker.predict(1, 'S') == {3: 1, 4: 1}

def test_intersection_attack(mixed_schema):
    attacker, quasi_identifiers = mixed_schema

    release_1 = pd.DataFrame(data={
        'QI1_min': [1, 1, 3, 3],
        'QI1_max': [2, 2, 3, 3],
        'QI2_A': [True, True, True, True],
        'QI2_B': [False, False, True, True],
        'S': [1, 2, 3, 4],
        'count': [1, 2, 2, 1]
    })
    release_2 = pd.DataFrame(data={
        'QI1_min': [1, 1, 2, 2],
        'QI1_max': [1, 1, 3, 3],
        'QI2_A': [True, True, False, False],
        'QI2_B': [True, True, True, True],
        'S': [1, 5, 3, 4],
        'count': [2, 1, 2, 1]
    })
    attacker.observe(release_1, quasi_identifiers +['S'], [0, 1])
    attacker.observe(release_2, quasi_identifiers +['S'], [0, 1])

    assert attacker.predict(0, 'S') == {1: 1}
    assert attacker.predict(1, 'S') == {3: 1, 4: 1}

def test_absent_records(mixed_schema):
    attacker, quasi_identifiers = mixed_schema
    release_1 = pd.DataFrame(data={
        'QI1_min': [1, 1, 3, 3],
        'QI1_max': [2, 2, 3, 3],
        'QI2_A': [True, True, True, True],
        'QI2_B': [False, False, True, True],
        'S': [1, 2, 3, 4],
        'count': [1, 2, 2, 1]
    })
    release_2 = pd.DataFrame(data={
        'QI1_min': [1, 1, 2, 2],
        'QI1_max': [1, 1, 3, 3],
        'QI2_A': [True, True, False, False],
        'QI2_B': [True, True, True, True],
        'S': [1, 5, 3, 4],
        'count': [2, 1, 2, 1]
    })
    attacker.observe(release_1, quasi_identifiers +['S'], [1])
    attacker.observe(release_2, quasi_identifiers +['S'], [0, 1])

    assert attacker.predict(0, 'S') == {1: 1, 5: 1}
    assert attacker.predict(1, 'S') == {3: 1, 4: 1}
