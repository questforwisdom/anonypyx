class Mondrian:
    def __init__(self, df, feature_columns):
        self.df = df
        self.feature_columns = feature_columns

    def get_spans(self, partition, scale=None):
        spans = {}
        for column in self.feature_columns:
            if self.df[column].dtype.name == "category":
                span = len(self.df[column][partition].unique())
            else:
                span = (
                    self.df[column][partition].max() - self.df[column][partition].min()
                )
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans

    def split(self, column, partition):
        dfp = self.df[column][partition]
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

    def partition(self, privacy_models):
        scale = self.get_spans(self.df.index)

        finished_partitions = []
        partitions = [self.df.index]

        while partitions:
            partition = partitions.pop(0)
            normalized_spans = self.get_spans(partition, scale)
            for column, span in sorted(normalized_spans.items(), key=lambda x: -x[1]):
                left_part, right_part = self.split(column, partition)

                if self.__all_models_enforceable(left_part, right_part, privacy_models):
                    partitions.extend((left_part, right_part))
                    break
            else:
                finished_partitions.append(partition)
        return finished_partitions

    def __all_models_enforceable(self, left_part, right_part, privacy_models):
        return all(model.is_enforcable(self.df.loc[left_part]) and model.is_enforcable(self.df.loc[right_part]) for model in privacy_models)

