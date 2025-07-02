'''
Implements the Mondrian [1] algorithm for multidimensional partitioning.

[1]: LeFevre, K., DeWitt, D. J., & Ramakrishnan, R. (2006). Mondrian multidimensional K-anonymity. 22nd International Conference on Data Engineering (ICDE’06), 25–25. https://doi.org/10.1109/ICDE.2006.101
'''

class Mondrian:
    def __init__(self, privacy_models, feature_columns):
        self._privacy_models = privacy_models
        self._feature_columns = feature_columns

    def _get_spans(self, df, partition, scale=None):
        spans = {}
        for column in self._feature_columns:
            if df[column].dtype.name == "category":
                span = len(df[column][partition].unique())
            else:
                span = (
                    df[column][partition].max() - df[column][partition].min()
                )
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans

    def _split(self, df, column, partition):
        dfp = df[column][partition]
        if dfp.dtype.name == "category":
            values = dfp.unique()
            lv = set(values[: len(values) // 2])
            rv = set(values[len(values) // 2 :])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]
            return (dfl, dfr)

    def partition(self, df):
        scale = self._get_spans(df, df.index)

        finished_partitions = []
        partitions = [df.index]

        while partitions:
            partition = partitions.pop(0)
            normalized_spans = self._get_spans(df, partition, scale)
            for column, span in sorted(normalized_spans.items(), key=lambda x: -x[1]):
                left_part, right_part = self._split(df, column, partition)

                if self.__all_models_enforceable(df, left_part, right_part):
                    partitions.extend((left_part, right_part))
                    break
            else:
                finished_partitions.append(partition)
        return finished_partitions

    def __all_models_enforceable(self, df, left_part, right_part):
        return all(model.is_enforcable(df.loc[left_part]) and model.is_enforcable(df.loc[right_part]) for model in self._privacy_models)

