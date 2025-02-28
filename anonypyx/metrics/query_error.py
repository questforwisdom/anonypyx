import pandas as pd

import anonypyx.generalisation

def counting_query(query, df, schema, quasi_identifier):
    full_matches = schema.select(df, query)
    matches_per_ec = df.loc[full_matches].value_counts(schema.quasi_identifier())

    result = 0.0

    qi_query = {k:v for k,v in query.items() if k in quasi_identifier}

    for qi, count in matches_per_ec.items():
        counterfeits = 0
        qi_record = pd.Series(qi, index=schema.quasi_identifier())
        ec_size = len(schema.match(df, qi_record, on=quasi_identifier))
        ec_region = schema.set_cardinality(qi_record, quasi_identifier)
        overlapping_region = schema.query_overlap(qi_record, qi_query)

        c1 = overlapping_region / ec_region
        c2 = count / ec_size

        result += (ec_size - counterfeits) * c1 * c2

    return result

def counting_query_error(query, raw_df, anon_df, anon_schema, quasi_identifier):
    categorical = [c for c in raw_df.columns if raw_df[c].dtype == 'category']
    numerical = [c for c in raw_df.columns if raw_df[c].dtype != 'category']
    raw_schema = anonypyx.generalisation.RawData(categorical, numerical, quasi_identifier)

    true_count = counting_query(query, raw_df, raw_schema, quasi_identifier)
    anon_count = counting_query(query, anon_df, anon_schema, quasi_identifier)

    return true_count - anon_count
