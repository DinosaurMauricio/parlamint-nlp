import os
import torch

from tqdm import tqdm

from utils.file_loader import ParlaMintFileLoader
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

        loaded_files_dict = ParlaMintFileLoader.get_yearly_files(self.config)

        for files_dict in loaded_files_dict:
            files, year, number_of_files = files_dict.values()
            progres_bar_folder = tqdm(
                files, desc=f"Processing Files [Year] {year}: ", total=number_of_files
            )
            for file in progres_bar_folder:

                temp_path = os.path.join(self.config.paths.conllu, year, file)
                df_meta, df_ana_meta, df_segment_data = (
                    ParlaMintFileLoader.load_parlamint_records(temp_path, self.config)
                )

                if df_meta.empty:
                    continue

                df = merge_data_frames(df_meta, df_ana_meta, df_segment_data)

                # attach year
                df["year"] = year

                sample = process_df_samples(df)
                samples.append(sample)

        return samples
