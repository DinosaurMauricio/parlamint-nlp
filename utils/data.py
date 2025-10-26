import os
import numpy as np
import pandas as pd
from utils.file_loader import ParlaMintFileLoader
from utils.filters import DataFilter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


def load_data(config):
    if os.path.exists(config.paths.preprocessed_data):
        return pd.read_parquet(config.paths.preprocessed_data)
    loader = ParlaMintFileLoader(config)
    samples, _ = loader.load_samples()
    return pd.DataFrame(samples)


class DataPipeline:
    def __init__(self, config):
        self.config = config

    def prepare_dataset(self, loaded_data, seed=42):

        column = self.config.dataset.target_column

        if column not in loaded_data.columns:
            raise ValueError("Column set on config not found")

        data = self._filter_data(
            loaded_data,
            self.config.dataset.word_count.min,
            self.config.dataset.word_count.max,
        )
        print(f"After filtering: {len(data)} samples")

        if self.config.dataset.sampling.enabled:
            print("Sampling...")
            data = self._undersample_classes(
                data, self.config.dataset.sampling.max_samples, column
            )
            print(f"After sampling: {len(data)} samples")

        class_weights = self._compute_class_weights(data, column)

        train_df, test_df, val_df = self._split_data(data, column, seed)

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "class_weights": class_weights,
        }

    def _filter_data(self, df, word_count_min, word_count_max):
        return (
            DataFilter(df)
            .select_columns(
                ["sent_id", "ID_meta", "text", "Party_orientation", "Words"]
            )
            .replace_hyphen_with_undefined("Party_orientation")
            .drop_duplicate_texts()
            .filter_nonempty_rows()
            .filter_by_threshold("Words", word_count_min, word_count_max)
            .apply()
        )

    def _undersample_classes(self, df, max_samples, class_column, seed=42):
        dfs = []
        for class_label in df[class_column].unique():
            temp_df = df[df[class_column] == class_label]
            size = len(temp_df)

            if size <= max_samples:
                # append all because samples size is lower than threshold
                dfs.append(temp_df)
            else:
                sampled = temp_df.sample(n=max_samples, random_state=seed)
                dfs.append(sampled)
        return pd.concat(dfs, ignore_index=True)

    def _compute_class_weights(self, df, column):
        return compute_class_weight(
            "balanced", classes=np.unique(df[column]), y=df[column]
        )

    def _split_data(self, df, column, seed=42):
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df[column], random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df[column],
            random_state=seed,
        )
        return train_df, val_df, test_df
