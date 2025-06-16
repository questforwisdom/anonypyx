import pytest
import pandas as pd
from anonypyx.generalisation.humanreadable import HumanReadable
from tests.util import *

@pytest.fixture
def mixed_df_fixture():
    df = pd.DataFrame(
        {
            "QI1": [101, 102, 103, 110, 110],
            "QI2": ["A", "A", "B", "B", "C"],
            "S": [10, 20, 10, 21, 10],
        },
        index=[1, 2, 3, 4, 5],
    )
    df["QI2"] = df["QI2"].astype("category")
    partition = [1, 2, 3]
    sensitive_column = "S"
    return df, partition, [sensitive_column]

@pytest.fixture
def single_value_mixed_df_fixture():
    df = pd.DataFrame(
        {
            "QI1": [101, 101, 101, 110, 110],
            "QI2": ["A", "A", "A", "B", "C"],
            "S": [10, 20, 10, 21, 10],
        },
        index=[1, 2, 3, 4, 5],
    )
    df["QI2"] = df["QI2"].astype("category")
    partition = [1, 2, 3]
    sensitive_column = "S"
    return df, partition, [sensitive_column]

@pytest.fixture
def human_readable_schema(mixed_df_fixture):
    return HumanReadable(["QI2"], ["QI1"], ["S"])

@pytest.fixture
def generalised_mixed_df():
    df = pd.DataFrame({
        "QI1": ["101-103", "101-103", "110", "110"],
        "QI2": ["A,B", "A,B", "B,C", "B,C"],
        "S": [10, 20, 21, 10],
        "count": [2, 1, 1, 1]
    })
    schema = HumanReadable(["QI2"], ["QI1"], ["S"])
    return df, schema

def test_generalise_single_value_human_readable(single_value_mixed_df_fixture):
    df, partition, _ = single_value_mixed_df_fixture
    strategy = HumanReadable.create_for_data(df, ["QI1", "QI2"])
    result = strategy.generalise(df, [partition])

    expected = pd.DataFrame(
        {
            "QI1": "101",
            "QI2": "A",
            "S": [10, 20],
            "count": [2, 1],
        }
    )

    assert_data_set_equal(result, expected)

def test_intersect(human_readable_schema):
    record_a = {"QI1": "101-103", "QI2": "A,B", "S": 3}
    record_b = {"QI1": "102-110", "QI2": "B,C", "S": 3}
    result = human_readable_schema.intersect(record_a, record_b, ["QI1", "QI2"], [], [])
    assert result == {"QI1": "102-103", "QI2": "B"}

def test_values_for(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 3}
    assert human_readable_schema.values_for(record, "QI1") == {101, 102, 103}
    assert human_readable_schema.values_for(record, "QI2") == {"A", "B"}
    assert human_readable_schema.values_for(record, "S") == {3}

def test_set_cardinality(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 3}
    assert human_readable_schema.set_cardinality(record, ["QI1", "QI2"]) == 3 * 2

def test_query_overlap(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 3}
    query = {"QI1": (102, 105), "QI2": {"B"}}
    assert human_readable_schema.query_overlap(record, query) == 2 * 1

def test_query_overlap_non_overlapping(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 3}
    query = {"QI1": (200, 250), "QI2": {"C"}}
    assert human_readable_schema.query_overlap(record, query) == 0

def test_query_overlap_multiple_categorical_values(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 3}
    query = {"QI1": (101, 103), "QI2": {"A", "B"}}
    assert human_readable_schema.query_overlap(record, query) == 3 * 2

def test_query_overlap_with_unaltered_column_match(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 10}
    query = {"QI1": (101, 103), "QI2": {"A"}, "S": {10}}
    assert human_readable_schema.query_overlap(record, query) == 3

def test_query_overlap_with_unaltered_column_no_match(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 10}
    query = {"QI1": (101, 103), "QI2": {"A"}, "S": {99}}
    assert human_readable_schema.query_overlap(record, query) == 0

def test_query_overlap_single_value_numerical(human_readable_schema):
    record = {"QI1": "101", "QI2": "A,B", "S": 3}
    query = {"QI1": (101, 101), "QI2": {"A"}}
    assert human_readable_schema.query_overlap(record, query) == 1

def test_query_overlap_single_value_categorical(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A", "S": 3}
    query = {"QI1": (101, 103), "QI2": {"A"}}
    assert human_readable_schema.query_overlap(record, query) == 3

def test_query_overlap_single_value_in_interval(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 3}
    query = {"QI1": (101, 101), "QI2": {"A"}}
    assert human_readable_schema.query_overlap(record, query) == 1

def test_cardinality_with_unaltered_attributes(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 10}
    assert human_readable_schema.set_cardinality(record, ["QI1", "QI2", "S"]) == 6

def test_intersect_with_unaltered_attributes(human_readable_schema):
    record_a = {"QI1": "101-103", "QI2": "A,B", "S": 10}
    record_b = {"QI1": "102-105", "QI2": "B,C", "S": 10}
    result = human_readable_schema.intersect(
        record_a, record_b, ["QI1", "QI2", "S"], [], []
    )
    assert result == {"QI1": "102-103", "QI2": "B", "S": 10}

def test_empty_overlap_numerical(human_readable_schema):
    record_a = {"QI1": "101-103", "QI2": "A,B", "S": 10}
    record_b = {"QI1": "104-106", "QI2": "A,B", "S": 10}
    result = human_readable_schema.intersect(
        record_a, record_b, ["QI1", "QI2", "S"], [], []
    )
    assert result is None

def test_empty_overlap_categorical(human_readable_schema):
    record_a = {"QI1": "101-103", "QI2": "A", "S": 10}
    record_b = {"QI1": "101-103", "QI2": "B,C", "S": 10}
    result = human_readable_schema.intersect(
        record_a, record_b, ["QI1", "QI2", "S"], [], []
    )
    assert result is None

def test_empty_overlap_unaltered(human_readable_schema):
    record_a = {"QI1": "101-103", "QI2": "A,B", "S": 10}
    record_b = {"QI1": "101-103", "QI2": "A,B", "S": 20}
    result = human_readable_schema.intersect(
        record_a, record_b, ["QI1", "QI2", "S"], [], []
    )
    assert result is None

def test_overlap_with_correct_unaltered_column_handling(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "A,B", "S": 10}
    query = {"QI1": (101, 103), "QI2": {"A"}, "S": {10}}
    assert human_readable_schema.query_overlap(record, query) == 3

def test_values_for_single_value_categorical(human_readable_schema):
    record = {"QI1": "101-103", "QI2": "B", "S": 3}
    assert human_readable_schema.values_for(record, "QI2") == {"B"}

def test_values_for_single_value_numerical(human_readable_schema):
    record = {"QI1": "105", "QI2": "A", "S": 3}
    assert human_readable_schema.values_for(record, "QI1") == {105}

def test_set_cardinality_single_value(human_readable_schema):
    record = {"QI1": "110", "QI2": "C", "S": 3}
    assert human_readable_schema.set_cardinality(record, ["QI1", "QI2"]) == 1

def test_intersect_categorical_mismatch(human_readable_schema):
    record_a = {"QI1": "101-103", "QI2": "A", "S": 3}
    record_b = {"QI1": "101-103", "QI2": "C", "S": 3}
    result = human_readable_schema.intersect(record_a, record_b, ["QI1", "QI2"], [], [])
    assert result is None

def test_select_method(generalised_mixed_df):
    df, schema = generalised_mixed_df
    query = {"QI1": (101, 103), "QI2": {"A", "C", "D"}}
    result_index = schema.select(df, query)
    assert list(result_index) == [0, 1]

def test_select_non_qi_interval(generalised_mixed_df):
    df, schema = generalised_mixed_df
    query = {"S": (10, 15)}
    assert list(schema.select(df, query)) == [0, 3]

def test_select_categorical_qi(generalised_mixed_df):
    df, schema = generalised_mixed_df
    query = {"QI2": {"C", "D"}}
    assert list(schema.select(df, query)) == [2, 3]

def test_query_overlap_multiple_generalised_blocks(human_readable_schema):
    query = {"QI1": (102, 103), "QI2": {"A"}}
    record = {'QI1': "101-103", 'QI2': "A,B", 'S': 20, 'count': 1}
    overlap = human_readable_schema.query_overlap(record, query)

    assert overlap == 2 * 1

def test_query_overlap_with_negative_numbers(human_readable_schema):
    record = {"QI1": "-5--3", "QI2": "A,B", "S": 3}
    query = {"QI1": (-6, -4), "QI2": {"A"}}
    assert human_readable_schema.query_overlap(record, query) == 2

def test_match_no_columns(generalised_mixed_df):
    df, schema = generalised_mixed_df
    record = {"QI1": "101-103", "QI2": "A", "S": 10}
    result = schema.match(df, record, [])
    result = list(result)

    assert result == [0, 1, 2, 3]

def test_match_matching_record(generalised_mixed_df):
    df, schema = generalised_mixed_df
    record = {"QI1": "101-103", "QI2": "A,C", "S": 10}
    result = schema.match(df, record, ["QI1", "QI2", "S"])
    result = list(result)

    assert result == [0]

def test_match_non_matching_record(generalised_mixed_df):
    df, schema = generalised_mixed_df
    record = {"QI1": "200-300", "QI2": "Z", "S": 99}
    result = schema.match(df, record, ["QI1", "QI2", "S"])
    result = list(result)

    assert result == []

def test_match_single_numerical_quasi_identifier(generalised_mixed_df):
    df, schema = generalised_mixed_df
    record = {"QI1": "101-105", "QI2": "A,B", "S": 10}
    result = schema.match(df, record, ["QI1"])
    result = list(result)

    assert result == [0, 1]

def test_match_single_categorical_quasi_identifier(generalised_mixed_df):
    df, schema = generalised_mixed_df
    record = {"QI1": "101-110", "QI2": "A", "S": 10}
    result = schema.match(df, record, ["QI2"])
    result = list(result)

    assert result == [0, 1]
