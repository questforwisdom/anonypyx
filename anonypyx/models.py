class kAnonymity:
    def __init__(self, k): 
        self.__k = k

    def is_enforcable(self, df):
        return len(df.index) >= self.__k

class DistinctLDiversity:
    def __init__(self, l, sensitive_column):
        self.__l = l
        self.__sensitive_column = sensitive_column

    def is_enforcable(self, df):
        if self.__sensitive_column is None:
            return False
        return self.__l <= len(df[self.__sensitive_column].unique())

def earth_movers_distance_categorical(distribution1, distribution2):
    diff_sum = 0.0

    for value, f1 in distribution1.items():
        f2 = distribution2[value]
        diff_sum += abs(f1 - f2)

    return diff_sum * 0.5

def max_distance_metric(distribution1, distribution2):
    max_diff = 0

    for value, f1 in distribution1.items():
        f2 = distribution2[value]
        max_diff = max(abs(f1 - f2), max_diff)

    return max_diff

class tCloseness:
    def __init__(self, t, df, sensitive_column, distance_metric):
        self.__t = t
        self.__sensitive_column = sensitive_column
        if self.__sensitive_column is not None:
            self.__global_frequencies = get_frequency(df, sensitive_column)
        self.__metric = distance_metric

    def is_enforcable(self, df):
        if self.__sensitive_column is None:
            return False
        total_count = float(len(df.index))

        if total_count == 0:
            return False

        # TODO: optimization possible: do not recalculate frequencies
        local_frequencies = get_frequency(df, self.__sensitive_column)

        return self.__metric(local_frequencies, self.__global_frequencies) <= self.__t

def get_frequency(df, sensitive_column):
    global_freqs = {}
    total_count = float(len(df))
    group_counts = df.groupby(sensitive_column, observed=False)[sensitive_column].agg("count")

    for value, count in group_counts.to_dict().items():
        p = count / total_count
        global_freqs[value] = p
    return global_freqs
