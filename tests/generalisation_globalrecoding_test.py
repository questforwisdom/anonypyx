import pytest
import pandas as pd
from anonypyx.generalisation.globalrecoding import GlobalRecoding, Taxonomy
from tests.util import *

@pytest.fixture
def mixed_schema():
    age_taxonomy = Taxonomy('any') \
        .add_generalised(Taxonomy('child') \
            .add_raw_values(range(0, 13)) \
        ).add_generalised(Taxonomy('teenager') \
            .add_raw_values(range(13, 20)) \
        ).add_generalised(Taxonomy('adult') \
            .add_generalised(Taxonomy('young adult') \
                .add_raw_values(range(20, 40)) \
            ).add_generalised(Taxonomy('middle aged') \
                .add_raw_values(range(40, 70)) \
            ).add_generalised(Taxonomy('old age') \
                .add_raw_values(range(70, 150)) \
            ) \
        )
    sex_taxonomy = Taxonomy('any') \
        .add_generalised(Taxonomy('binary') \
            .add_raw_values(['female', 'male']) \
        ).add_raw_values(['nonbinary'])

    return GlobalRecoding({'age': age_taxonomy, 'sex': sex_taxonomy}, ['S'])


@pytest.fixture
def mixed_schema_with_json(mixed_schema):
    json_dict = {
        'taxonomy': {
            'age': {
                'generalised': ['any', 'child', 'teenager', 'adult', 'young adult', 'middle aged', 'old age'],
                'definitions': {
                    'any': ['child', 'teenager', 'adult'],
                    'adult': ['young adult', 'middle aged', 'old age'],
                    'child': list(range(0, 13)),
                    'teenager': list(range(13, 20)),
                    'young adult': list(range(20, 40)),
                    'middle aged': list(range(40, 70)),
                    'old age': list(range(70, 150))
                }
            },
            'sex': {
                'generalised': ['any', 'binary'],
                'definitions': {
                    'any': ['binary', 'nonbinary'],
                    'binary': ['female', 'male']
                }
            }
        },
        'unaltered': ['S']
    }
    return mixed_schema, json_dict

def test_to_json_dict(mixed_schema_with_json):
    mixed_schema, json_dict = mixed_schema_with_json

    assert mixed_schema.to_json_dict() == json_dict

def test_from_json_dict(mixed_schema_with_json):
    mixed_schema, json_dict = mixed_schema_with_json

    recreated = GlobalRecoding.from_json_dict(json_dict)

    df = pd.DataFrame({
        'age': [13, 15, 20, 75],
        'sex': ['male', 'female', 'female', 'female'],
        'S': [1, 2, 1, 1],
    })
    partitions = [[0,1], [2,3]]
    record_a = {'age': 'adult', 'sex': 'female', 'S': 20}
    record_b = {'age': 'middle aged', 'sex': 'binary'}

    recoded_1 = mixed_schema.generalise(df, partitions)
    recoded_2 = recreated.generalise(df, partitions)

    intersection_1 = mixed_schema.intersect(record_a, record_b, ['age', 'sex'], ['S'], [])
    intersection_2 = recreated.intersect(record_a, record_b, ['age', 'sex'], ['S'], [])

    assert_data_set_equal(recoded_1, recoded_2)
    assert mixed_schema.quasi_identifier() == recreated.quasi_identifier()
    assert intersection_1 == intersection_2

def test_generalisation(mixed_schema):
    df = pd.DataFrame({
        'age': [13, 15, 20, 75],
        'sex': ['male', 'female', 'female', 'female'],
        'S': [1, 2, 1, 1],
    })

    partitions = [[0,1], [2,3]]

    result = mixed_schema.generalise(df, partitions)

    expected = pd.DataFrame({
        'age': ['teenager', 'teenager', 'adult'],
        'sex': ['binary', 'binary', 'binary'],
        'S': [1, 2, 1],
        'count': [1, 1, 2]
    })

    assert_data_set_equal(result, expected)

def test_matching_same_generalisation(mixed_schema):
    df = pd.DataFrame({
        'age': ['teenager', 'teenager', 'adult'],
        'sex': ['binary', 'binary', 'binary'],
        'S': [1, 2, 1],
        'count': [1, 1, 2]
    })
    record = {'age': 'teenager', 'sex': 'binary'}
    result = mixed_schema.match(df, record, ['age', 'sex'])
    expected = df.iloc[[0, 1]].copy()

    assert_data_set_equal(result, expected)

def test_matching_different_generalisation(mixed_schema):
    df = pd.DataFrame({
        'age': ['teenager', 'teenager', 'adult'],
        'sex': ['binary', 'binary', 'binary'],
        'S': [1, 2, 1],
        'count': [1, 1, 2]
    })
    record = {'age': 'middle aged', 'sex': 'any'}
    result = mixed_schema.match(df, record, ['age', 'sex'])
    expected = df.iloc[[2]].copy()

    assert_data_set_equal(result, expected)

def test_intersection_with_overlap(mixed_schema):
    record_a = {'age': 'adult', 'sex': 'female', 'S': 20}
    record_b = {'age': 'middle aged', 'sex': 'binary'}

    result = mixed_schema.intersect(record_a, record_b, ['age', 'sex'], ['S'], [])

    expected = {'age': 'middle aged', 'sex': 'female', 'S': 20}

    assert result == expected

def test_intersection_no_overlap(mixed_schema):
    record_a = {'age': 'child', 'sex': 'female', 'S': 20}
    record_b = {'age': 42, 'sex': 'binary'}

    result = mixed_schema.intersect(record_a, record_b, ['age', 'sex'], ['S'], [])

    assert result is None

def test_values_for(mixed_schema):
    record = {'age': 'child', 'sex': 'female', 'S': 20}
    result = mixed_schema.values_for(record, 'age')

    assert result == set(range(0, 13))

def test_values_for_raw_value(mixed_schema):
    record = {'age': 'child', 'sex': 'female', 'S': 20}
    result = mixed_schema.values_for(record, 'sex')

    assert result == {'female'}

def test_quasi_identifier(mixed_schema):
    assert mixed_schema.quasi_identifier() == ['age', 'sex']

def test_set_cardinality(mixed_schema):
    record = {'age': 'teenager', 'sex': 'binary', 'S': 20}

    assert mixed_schema.set_cardinality(record, ['age', 'sex', 'S']) == 7 * 2 * 1
    assert mixed_schema.set_cardinality(record, ['age', 'sex']) == 7 * 2
    assert mixed_schema.set_cardinality(record, ['sex']) == 2

def test_select_with_matches(mixed_schema):
    df = pd.DataFrame({
        'age': ['teenager', 'teenager', 'adult'],
        'sex': ['binary', 'binary', 'binary'],
        'S': [1, 2, 1],
        'count': [1, 1, 2]
    })
    query = {'age': (15, 25), 'S': (1, 1)}

    result = mixed_schema.select(df, query)

    assert sorted(list(result)) == [0, 2]

def test_select_without_matches(mixed_schema):
    df = pd.DataFrame({
        'age': ['teenager', 'teenager', 'adult'],
        'sex': ['binary', 'binary', 'binary'],
        'S': [1, 2, 1],
        'count': [1, 1, 2]
    })
    query = {'age': (0, 5), 'sex': {'female', 'male'}}

    result = mixed_schema.select(df, query)

    assert sorted(list(result)) == []

def test_query_overlap(mixed_schema):
    record = {'age': 'child', 'sex': 'binary', 'S': 20}
    query = {'age': (10, 15), 'sex': {'female', 'male'}}

    result = mixed_schema.query_overlap(record, query)

    assert result == 3 * 2

def test_query_overlap_no_overlap(mixed_schema):
    record = {'age': 'child', 'sex': 'nonbinary', 'S': 20}
    query = {'age': (10, 15), 'sex': {'female', 'male'}}

    result = mixed_schema.query_overlap(record, query)

    assert result == 0
