import anonypyx
import pandas as pd

import pytest

@pytest.fixture
def prepared_df():
    data = [
        [6, "1", "test1", "x", 20],
        [6, "1", "test1", "x", 30],
        [8, "2", "test2", "x", 50],
        [8, "2", "test3", "w", 45],
        [8, "1", "test2", "y", 35],
        [4, "2", "test3", "y", 20],
        [4, "1", "test3", "y", 20],
        [2, "1", "test3", "z", 22],
        [2, "2", "test3", "y", 32],
    ]
    df = pd.DataFrame(data=data, columns=["col1", "col2", "col3", "col4", "col5"])
    for name in ("col2", "col3", "col4"):
        df[name] = df[name].astype("category")

    return df

def test_k_anonymity(prepared_df):
    a = anonypyx.Anonymizer(prepared_df, k=2, feature_columns=["col1", "col2", "col3"], sensitive_column="col4")
    rows = a.anonymize()

    dfn = pd.DataFrame(rows)
    print(dfn)


def test_l_diversity(prepared_df):
    a = anonypyx.Anonymizer(prepared_df, k=2, l=2, diversity_definition="distinct", feature_columns=["col1", "col2", "col3"], sensitive_column="col4")
    rows = a.anonymize()

    dfn = pd.DataFrame(rows)
    print(dfn)


def test_t_closeness(prepared_df):
    a = anonypyx.Anonymizer(prepared_df, k=2, t=0.2, closeness_metric="max distance", feature_columns=["col1", "col2", "col3"], sensitive_column="col4")
    rows = a.anonymize()

    dfn = pd.DataFrame(rows)
    print(dfn)

