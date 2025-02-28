def percentage_of_vulnerable_population(original_df, prediction_df, confidence_level, sensitive_column):
    """
    Computes the Percentage of Vulnerable Population (PVP_C) [1]. The PVP_C is the ratio of 
    input records for which the adversary predicts the sensitive value correctly with
    a confidence level of at least C.

    [1]: Ganta, S. R., Kasiviswanathan, S. P., & Smith, A. (2008). Composition attacks and 
    auxiliary information in data privacy. Proceedings of the 14th ACM SIGKDD International
    Conference on Knowledge Discovery and Data Mining, 265–273.


    Parameters
    ----------
    original_df : pd.DataFrame
        The original data frame before anonymisation. It must comply with the output format
        of `anonypyx.metrics.preprocess_original_data_for_privacy()`.
    prediction_df : pd.DataFrame
        The adversary's prediction. It must comply with the output format of 
        `anonypyx.metrics.preprocess_prediction()`.
    confidence_level : float
        The confidence threshold C. Must be between 0 and 1 (inclusive).
    sensitive_column : str
        The name of the sensitive column in the original_df, i.e. the one the adversary attempts
        to predict.

    Returns
    -------
    The PVP_C, a float between 0 (high privacy) and 1 (low privacy).
    """
    vulnerable_population = 0
    total_population = len(original_df)

    for target_id in range(total_population):
        ground_truth = original_df.at[target_id, sensitive_column]
        confidence = prediction_df.at[target_id, str(ground_truth)]

        if confidence >= confidence_level:
            vulnerable_population += 1

    return vulnerable_population / total_population
