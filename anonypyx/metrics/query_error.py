"""
Defines the error of counting queries (also called aggregate queries) between
generalised data and the raw data as a utility metric. This metric was proposed
in the following paper:

Xiao, X., & Tao, Y. (2007). M-invariance: Towards privacy preserving re-publication of dynamic datasets. Proceedings of the 2007 ACM SIGMOD International Conference on Management of Data, 689–700. https://doi.org/10.1145/1247480.1247556
"""
import random

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

        # computing the region sizes over the entire quasi-identifier led to overflows on larger
        # data sets such as Adult
        c1 = 1.0
        for col in qi_query:
            ec_region = schema.set_cardinality(qi_record, [col])
            overlapping_region = schema.query_overlap(qi_record, {col: qi_query[col]})
            c1 *= overlapping_region / ec_region

        c2 = count / ec_size

        result += (ec_size - counterfeits) * c1 * c2

    return result

class CountingQueryGenerator:
    """
    Generates random counting queries for a given data frame.
    """
    def __init__(self, df, quasi_identifiers):
        """
        Constructor. 

        Parameters
        ----------
        query : dict 
        df : pd.DataFrame
            The original data frame before anonymisation. It must comply with the output format
            of `anonypyx.metrics.preprocess_original_data_for_utility()`.
        quasi_identifier : list of str
            The names of the columns which are used as a quasi-identifier.
        """
        self._quasi_identifiers = quasi_identifiers[:]
        self._unaltered = [col for col in df.columns if col not in self._quasi_identifiers and col != 'count']

        self._integer_domains = {}
        self._categorical_domains = {}

        for col in df.columns:
            if df[col].dtype.name == 'category':
                self._categorical_domains[col] = list(df[col].unique())
            else:
                self._integer_domains[col] = (df[col].min(), df[col].max())

    def generate(self, num_predicates, expected_selectivity, use_sensitive=True):
        """
        Generates a random counting query for the data frame for which this instance was
        created.

        Parameters
        ----------
        num_predicates : int
            Number of predicates contained in the generated query. A predicate is a range
            query over a single column.
        exptected_selectivity : float
            A number between 0 and 1 which indicates the ratio between the volume of the data
            space defined by the query and the volume covered by the data frame for which this 
            instance was created. For instance, if the data frame contains only two attributes
            age and sex with domains [0, 99] and {male, female, other} (volume is 100 * 3 = 300)
            and this parameter is set to 0.3333, a query with a volume of 100 such as sex=male AND
            0<=age<=99 might be returned.
        use_sensitive : Boolean
            If this is set to False, only quasi_identifiers are queries. If this is set to True,
            exactly one randomly chosen unaltered/sensitive column from the data frame is included
            in the query. (Default is True)

        Returns
        -------
        A dictionary describing the query's predicates. Keys must be column names from the
        original data. Numercial columns must be mapped to a tuple of two values which define
        the lower and upper bound of the query (both inclusive). Categorical attributes must
        be mapped to a set of values. The query is the conjunction of all predicates (logical
        AND).

        """
        if num_predicates == 0:
            return 0

        query = {}
        weight = expected_selectivity ** (1.0/num_predicates)

        if use_sensitive:
            column = random.choice(self._unaltered)
            query[column] = self._build_predicate(column, weight)
            num_predicates -= 1

        selected = random.sample(self._quasi_identifiers, num_predicates)

        for i in range(num_predicates):
            query[selected[i]] = self._build_predicate(selected[i], weight)

        return query

    def _build_predicate(self, column, weight):
        if column in self._categorical_domains:
            domain = self._categorical_domains[column]
            num_values = round(weight * len(domain))
            return set(random.sample(domain, num_values))
        else:
            domain = self._integer_domains[column]
            domain_size = domain[1] - domain[0]
            num_values = round(domain_size * weight)
            start = random.randint(domain[0], domain[1] - num_values)
            end = start + num_values
            return (start, end)
