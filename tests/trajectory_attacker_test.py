import pytest
import pandas as pd

from anonypyx import generalisation
from anonypyx.attackers.trajectory_attacker import TrajectoryAttacker

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
    attacker = TrajectoryAttacker(prior_knowledge, quasi_identifiers, schema)

    return attacker, quasi_identifiers

def test_attacker_works_when_id_is_not_the_first_column():
    prior_knowledge = pd.DataFrame(data={
        'QI1_min': [1, 3],
        'QI1_max': [1, 3],
        'QI2_A': [True, False],
        'ID': [0, 1],
        'QI2_B': [False, True]
    })
    schema = generalisation.MachineReadable({'QI2': ['QI2_A', 'QI2_B']}, {'QI1': ['QI1_min', 'QI1_max']}, ['S'])
    quasi_identifiers = ['QI1', 'QI2']
    attacker = TrajectoryAttacker(prior_knowledge, quasi_identifiers, schema)
    release = pd.DataFrame(data={
        'QI1_min': [1, 1, 3, 3],
        'QI1_max': [2, 2, 3, 3],
        'QI2_A': [True, True, True, True],
        'QI2_B': [False, False, True, True],
        'S': [1, 2, 3, 4],
        'count': [1, 2, 2, 1]
    })
    attacker.observe(release, quasi_identifiers + ['S'], [0, 1])

    assert attacker.predict(0, 'S') == {1: 1,2: 2}
    assert attacker.predict(1, 'S') == {3: 2,4: 1}

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
    attacker.observe(release, quasi_identifiers + ['S'], [0, 1])

    assert attacker.predict(0, 'S') == {1: 1,2: 2}
    assert attacker.predict(1, 'S') == {3: 2,4: 1}

def test_equivalent_permutations(mixed_schema):
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
        'QI1_min': [1, 1, 3, 3],
        'QI1_max': [2, 2, 3, 3],
        'QI2_A': [True, True, True, True],
        'QI2_B': [False, False, True, True],
        'S': [1, 2, 3, 4],
        'count': [1, 1, 3, 2]
    })
    attacker.observe(release_1, quasi_identifiers + ['S'], [0, 1])
    attacker.observe(release_2, quasi_identifiers + ['S'], [0, 1])

    # there is only one option if ID=0 has S=1
    # there are two options if ID=0 has S=2:
    # it can be either the first or second record in release_1
    # and must be the single record in release_2
    assert attacker.predict(0, 'S') == {1: 1, 2: 2}

    # the case S=3 for ID=1 corresponds to three records from
    # release_1 and two records from release_2, thus there are
    # 6 options in total
    assert attacker.predict(1, 'S') == {3: 6, 4: 2}

def test_absent_records(mixed_schema):
    attacker, quasi_identifiers = mixed_schema
    release = pd.DataFrame(data={
        'QI1_min': [1, 1, 3, 3],
        'QI1_max': [2, 2, 3, 3],
        'QI2_A': [True, True, True, True],
        'QI2_B': [False, False, True, True],
        'S': [1, 2, 3, 4],
        'count': [1, 2, 2, 1]
    })
    attacker.observe(release, quasi_identifiers + ['S'], [0])

    assert attacker.predict(0, 'S') == {1: 1,2: 2}
    assert attacker.predict(1, 'S') is None

def test_prior_knowledge_one_sensitive_value():
    prior_knowledge = pd.DataFrame(data={
        # S=2 ruled out for ID=0
        'ID': [0, 0, 1],
        'QI1_min': [1, 1, 3],
        'QI1_max': [1, 1, 3],
        'QI2_A': [True, True, False],
        'QI2_B': [False, False, True],
        'S_min': [1, 3, 1],
        'S_max': [1, 4, 4]
    })
    schema = generalisation.MachineReadable({'QI2': ['QI2_A', 'QI2_B']}, {'QI1': ['QI1_min', 'QI1_max'], 'S': ['S_min', 'S_max']}, [])
    quasi_identifiers = ['QI1', 'QI2']
    attacker = TrajectoryAttacker(prior_knowledge, quasi_identifiers + ['S'], schema)
    release = pd.DataFrame(data={
        'QI1_min': [1, 1, 3, 3],
        'QI1_max': [2, 2, 3, 3],
        'QI2_A': [True, True, True, True],
        'QI2_B': [False, False, True, True],
        'S_min': [1, 2, 3, 4],
        'S_max': [1, 2, 3, 4],
        'count': [1, 2, 2, 1]
    })
    attacker.observe(release, quasi_identifiers + ['S'], [0, 1])

    assert attacker.predict(0, 'S') == {1: 1}
    assert attacker.predict(1, 'S') == {3: 2, 4: 1}

def test_insertion_attack():
    prior_knowledge = pd.DataFrame(data={
        # S=2 ruled out for ID=0
        'ID': [0, 1, 2, 3, 4],
        'QI1_min': [1, 2, 3, 4, 5],
        'QI1_max': [1, 2, 3, 4, 5],
        'QI2_A': [True, True, False, True, False],
        'QI2_B': [False, False, True, False, True]
    })
    schema = generalisation.MachineReadable({'QI2': ['QI2_A', 'QI2_B']}, {'QI1': ['QI1_min', 'QI1_max']}, ['S'])
    quasi_identifiers = ['QI1', 'QI2']
    attacker = TrajectoryAttacker(prior_knowledge, quasi_identifiers, schema)
    release_1 = pd.DataFrame(data={
        'QI1_min': [1, 3, 3],
        'QI1_max': [2, 4, 4],
        'QI2_A': [True, True, True],
        'QI2_B': [False, True, True],
        'S': [1, 2, 3],
        'count': [2, 1, 1]
    })
    release_2 = pd.DataFrame(data={
        'QI1_min': [1, 3, 3, 3],
        'QI1_max': [2, 5, 5, 5],
        'QI2_A': [True, True, True, True],
        'QI2_B': [False, True, True, True],
        'S': [1, 2, 3, 4],
        'count': [2, 1, 1, 1]
    })
    attacker.observe(release_1, quasi_identifiers + ['S'], [0, 1, 2, 3])
    attacker.observe(release_2, quasi_identifiers + ['S'], [0, 1, 2, 3, 4])

    attacker.prune_multiset_exact_cover()

    assert attacker.predict(0, 'S') == {1: 4}
    assert attacker.predict(1, 'S') == {1: 4}
    assert attacker.predict(2, 'S') == {2: 1, 3: 1}
    assert attacker.predict(3, 'S') == {2: 1, 3: 1}
    assert attacker.predict(4, 'S') == {4: 1}

