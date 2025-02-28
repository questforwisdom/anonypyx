import anonypyx.generalisation

def preprocess_prediction(prediction_df):
    """
    Prepares the adversary's prediction for most privacy metrics.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        The adversary's prediction. It must contain the unique identifier of the targets in a 
        column ID and then one column for every distinct value of the sensitive attribute (names
        must match the values) which contains the adversary's confidence in the
        corresponding value for the given individual. These confidence values do not have to be
        normalised (i.e. rows such as (ID=0, S_1=0, S_2=1, S_3=2, S_4=5) may be used here). 
        There must be exactly one record/row per targeted individual.

    Returns
    -------
        A data frame where the ID column is used as an index and the confidence levels are
        normalised in the range [0, 1].
    """
    prediction_df = prediction_df.set_index('ID')
    prediction_df = prediction_df.div(prediction_df.sum(axis=1), axis=0)
    return prediction_df

def preprocess_original_data_for_privacy(original_df):
    """
    Prepares the original input data frame for most privacy metrics.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original data frame before anonymisation. If multiple data sets are released, this
        must contain all records, i.e. provide the union of all original data sets instead. It 
        must contain a column ID assigning a unique identifier to each targeted individual.
        IDs must start at 0 and increase by 1 for every target.
        There must be exactly one record/row per targeted individual.

    Returns
    -------
        A data frame where the ID column is used as an index.
    """
    return original_df.set_index('ID')

def preprocess_original_data_for_utility(original_df, quasi_identifier):
    """
    Prepares the original input data frame for most utility metrics.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original data frame before anonymisation. If multiple data sets are released, this
        must contain all records, i.e. provide the union of all original data sets instead. It 
        must contain a column ID assigning a unique identifier to each targeted individual.
        IDs must start at 0 and increase by 1 for every target.
        There must be exactly one record/row per targeted individual.
    quasi_identifier : list of str
        The names of the columns which serve as a quasi-identifier.

    Returns
    -------
    A data frame where the ID column is removed and a count column is added.
    """
    original_df = original_df.drop('ID', axis=1)
    categorical = [c for c in original_df.columns if original_df[c].dtype == 'category']
    numerical = [c for c in original_df.columns if original_df[c].dtype != 'category']
    raw_schema = anonypyx.generalisation.RawData(categorical, numerical, quasi_identifier)
    return raw_schema.generalise(original_df, [[i] for i in original_df.index])
