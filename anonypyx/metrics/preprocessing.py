def preprocess_prediction(prediction_df):
    """
    Prepares the adversary's prediction for most privacy metrics.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        The adversary's prediction. It must contain the unique identifier of the targets in a 
        column ID and then one column for every distinct value of the sensitive attribute (naming
        scheme is <sensitive_column>_<value>) which contains the adversary's confidence in the
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

def preprocess_original_data(original_df):
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

