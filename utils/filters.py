class UtilityFilter:

    @staticmethod
    def filter_meta(df_meta, topics=None, orientations=None):
        # filter by the settings set on the config.yaml
        if topics:
            df_meta = df_meta[df_meta["Topic"].isin(topics)]
        if orientations:
            df_meta = df_meta[df_meta["Party_orientation"].isin(orientations)]
        return df_meta

    @staticmethod
    def filter_years(years_folders, years=None):
        # filtering years folders, setting is set on the config.yaml
        if years:
            years_folders = list(set(years_folders) - set(years))
        return years_folders


class DataFilter:
    def __init__(self, df):
        self._df = df.copy()

    def apply(self):
        return self._df

    def replace_hyphen_with_undefined(self, column):
        """
        Change the "-" naming convention from the parlamint data set to "Undefined
        """
        self._df[column] = self._df[column].replace("-", "Undefined")
        return self

    def select_columns(self, cols_to_keep):
        self._df = self._df[cols_to_keep]
        return self

    def filter_nonempty_rows(self):
        """
        Remove Nonempty words. Empty words in the Dataset can be punctuation.
        """
        self._df = self._df[self._df["Words"] != 0]
        return self

    def drop_duplicate_texts(self):
        self._df = self._df.drop_duplicates("text")
        return self

    def filter_by_threshold(self, column, min_val=None, max_val=None):
        if min_val is not None:
            self._df = self._df[self._df[column].gt(min_val)]
        if max_val is not None:
            self._df = self._df[self._df[column].lt(max_val)]
        return self
