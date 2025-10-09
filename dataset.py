import os
import torch

from tqdm import tqdm

from constants import CONLLU_EXT, META_EXT, ANA_META_EXT
from utils.file_loader import ParlaMintLoader
from utils.filters import filter_meta, filter_years
from utils.merge_helper import merge_data_frames, process_df_samples


class ParlimentDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _load_samples(self):
        samples = []

        years_folders = ParlaMintLoader.load_years_folders(self.config.paths.conllu)
        years_folders = filter_years(years_folders, self.config.years)

        loaded_files_dict, _ = ParlaMintLoader.load_file_names_by_years(
            self.config.paths.conllu, years_folders, CONLLU_EXT
        )

        for files_dict in loaded_files_dict:
            files, year, number_of_files = files_dict.values()
            progres_bar_folder = tqdm(
                files, desc=f"Processing Files [Year] {year}: ", total=number_of_files
            )
            for file in progres_bar_folder:
                temp_path = os.path.join(self.config.paths.conllu, year, file)
                df_meta = ParlaMintLoader.load_tsv_file(
                    temp_path + META_EXT,
                    ["Text_ID", "ID", "Party_orientation", "Topic"],
                )
                df_meta = filter_meta(
                    df_meta, self.config.topics, self.config.orientations
                )

                if df_meta.empty:
                    continue

                df_ana_meta = ParlaMintLoader.load_tsv_file(
                    temp_path + ANA_META_EXT, ["ID", "Parent_ID"]
                )
                df_segment_data = ParlaMintLoader.load_segments_sentiment_data(
                    temp_path + CONLLU_EXT
                )

                df = merge_data_frames(df_meta, df_ana_meta, df_segment_data)

                # attach year
                df["year"] = year

                sample = process_df_samples(df)
                samples.append(sample)

        return samples
