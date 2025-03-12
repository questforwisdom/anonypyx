"""
Defines the error of counting queries (also called aggregate queries) between
generalised data and the raw data as a utility metric. This metric was proposed
in the following paper:

Xiao, X., & Tao, Y. (2007). M-invariance: Towards privacy preserving re-publication of dynamic datasets. Proceedings of the 2007 ACM SIGMOD International Conference on Management of Data, 689–700. https://doi.org/10.1145/1247480.1247556
"""
import random

import pandas as pd

import anonypyx.generalisation

def counting_query_error(query, original_df, anon_df):
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
    original_df : anonypyx.metrics.PreparedUtilityDataFrame
        The original data frame before anonymisation. 
    anon_df : anonypyx.metrics.PreparedUtilityDataFrame
        The generalised version of original_df.

    Returns
    -------
    The relative absolute error of the query run on anon_df compared to original_df.

    Raises
    ------
    ValueError if the query does not match any data point from the original data frame
    (computing the relative error would entail a divide by zero in this case).
    """
    true_count = counting_query(query, original_df)
    anon_count = counting_query(query, anon_df)

    if true_count == 0.0:
        raise ValueError('Query does not match any original data points.')

    return abs(true_count - anon_count) / true_count

def counting_query(query, prepared_df):
    """
    Processes a counting query on a generalised data frame.

    Parameters
    ----------
    query : dict 
        A dictionary describing the query's predicates. Keys must be column names from the 
        original data. Numercial columns must be mapped to a tuple of two values which define
        the lower and upper bound of the query (both inclusive). Categorical attributes must
        be mapped to a set of values. The query is the conjunction of all predicates (logical
        AND).
    prepared_df : anonypyx.metrics.PreparedUtilityDataFrame
        The data frame to query.
    
    Returns
    -------
    The approximate number of data points matching the query as a floating-point number.
    """
    qi_query = {k:v for k,v in query.items() if k in prepared_df.original_quasi_identifier()}

    matches = prepared_df.df().loc[prepared_df.schema().select(prepared_df.df(), query)].groupby('group_id', sort=False)

    full_matches = matches['count'].sum()

    qis = matches.nth(0)

    if len(qis) == 0:
        return 0.0

    def count_in_ec(row):
        ec_size  = prepared_df.group_size(row['group_id'])
        counterfeits = 0
        c1 = 1.0
        for col in qi_query:
            ec_region = prepared_df.schema().set_cardinality(row, [col])
            overlapping_region = prepared_df.schema().query_overlap(row, {col: qi_query[col]})
            c1 *= overlapping_region / ec_region

        c2 = full_matches.loc[row['group_id']] / ec_size
        return (ec_size - counterfeits) * c1 * c2

    return qis.apply(count_in_ec, axis=1).sum()

class CountingQueryGenerator:
    """
    Generates random counting queries for a given data frame.
    """
    def __init__(self, prepared_df):
        """
        Constructor. 

        Parameters
        ----------
        query : dict 
        prepared_df : anonypyx.metrics.PreparedUtilityDataFrame
            The original data frame before anonymisation.
        """
        self._quasi_identifiers = prepared_df.original_quasi_identifier()[:]
        self._unaltered = [col for col in prepared_df.df().columns if col not in self._quasi_identifiers and col not in ['group_id', 'count']]

        self._integer_domains = {}
        self._categorical_domains = {}

        for col in prepared_df.df().columns:
            if prepared_df.df()[col].dtype.name == 'category':
                self._categorical_domains[col] = list(prepared_df.df()[col].unique())
            else:
                self._integer_domains[col] = (prepared_df.df()[col].min(), prepared_df.df()[col].max())

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
