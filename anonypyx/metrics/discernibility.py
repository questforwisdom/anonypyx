"""
Implements the utility metric "discernibility penalty" which is sum of squared equivalence class sizes.
In other words, it penalises every data point by the number of data points from which it is indistinguishable
when projected to the quasi-identifiers and sums these penalites over the entire data set.

Bayardo, R. J., & Agrawal, R. (2005). Data privacy through optimal k-anonymization. 21st International Conference on Data Engineering (ICDE’05), 217–228. https://doi.org/10.1109/ICDE.2005.42
"""

def discernibility_penalty(prepared_df):
    """
    Computes the discernibility penalty for the given data frame.
    The best value is the number of data points (attained if all records 
    are distinguishable when projected to the quasi-identifiers).
    The larger the penalty, the less utility.
    """
    result = 0
    for i in range(prepared_df.num_groups()):
        result += prepared_df.group_size(i) ** 2
    return result
