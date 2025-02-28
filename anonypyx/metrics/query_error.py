import pandas as pd

import anonypyx.generalisation

def counting_query_error(query, original_df, anon_df, anon_schema, quasi_identifier):
    """
    Computes the error of a counting query.

    Parameters
    ----------
    query : dict 
        A dictionary describing the query's predicates. Keys must be column names from the 
        original data. Numercial columns must be mapped to a tuple of two values which define
        the lower and upper bound of the query (both inclusive). Categorical attributes must
        be mapped to a set of values. The query is the conjunction of all predicates (logical
        AND).
    original_df : pd.DataFrame
        The original data frame before anonymisation. It must comply with the output format
        of `anonypyx.metrics.preprocess_original_data_for_utility()`.
    anon_df : pd.DataFrame
        The generalised version of original_df.
    anon_df : GeneralisedSchema
        The schema of anon_df.
    quasi_identifier : list of str
        The names of the columns from original_df which are used as a quasi-identifier.
        
    """
    categorical = [c for c in original_df.columns if original_df[c].dtype == 'category']
    numerical = [c for c in original_df.columns if original_df[c].dtype != 'category']
    raw_schema = anonypyx.generalisation.RawData(categorical, numerical, quasi_identifier)

    true_count = counting_query(query, original_df, raw_schema, quasi_identifier)
    anon_count = counting_query(query, anon_df, anon_schema, quasi_identifier)

    return true_count - anon_count

def counting_query(query, df, schema, quasi_identifier):
    full_matches = schema.select(df, query)

    matches_per_ec = df.loc[full_matches].groupby(by=schema.quasi_identifier(), observed=True)

    result = 0.0

    qi_query = {k:v for k,v in query.items() if k in quasi_identifier}

    for qi, group_df in matches_per_ec:
        count = group_df['count'].sum()
        counterfeits = 0
        qi_record = pd.Series(qi, index=schema.quasi_identifier())
        ec_size = df.iloc[schema.match(df, qi_record, on=quasi_identifier)]['count'].sum()
        ec_region = schema.set_cardinality(qi_record, quasi_identifier)
        overlapping_region = schema.query_overlap(qi_record, qi_query)

        c1 = overlapping_region / ec_region
        c2 = count / ec_size

        result += (ec_size - counterfeits) * c1 * c2

    return result
