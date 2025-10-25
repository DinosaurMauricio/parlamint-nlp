import os
import pandas as pd

from tqdm import tqdm
from conllu import parse_incr
from dataclasses import dataclass

from utils.constants import TXT_EXT, ANA_META_EXT, META_EXT, CONLLU_EXT
from utils.filters import UtilityFilter
from utils.merge_helper import merge_data_frames


@dataclass
class FileLoaderStats:
    tottal_utterances: int
    files_processed: int


class ParlaMintFileLoader:
    def __init__(self, config):
        self.base_path = config.paths.conllu
        self.years = config.years
        self.topics = config.topics
        self.orientations = config.orientations

    def load_samples(self):
        """
        Returns:
            tuple:
                - pd.DataFrame: processed data
                - tuple: (total_files_loaded, total_num_utt)
        """
        total_num_utt = 0
        dfs = []

        loaded_files_dict = self._get_yearly_files(CONLLU_EXT)
        total_files_loaded = self._count_files(loaded_files_dict)

        for files_dict in loaded_files_dict:
            files, year, number_of_files = files_dict.values()
            progres_bar_folder = tqdm(
                files, desc=f"Processing Files [{year}] : ", total=number_of_files
            )

            for file in progres_bar_folder:

                df = self._load_file_data(file, year)

                if df.empty:
                    continue

                total_num_utt += df.shape[0]

                dfs.append(df)

        final_df = pd.concat(dfs, ignore_index=True)

        return final_df, FileLoaderStats(total_num_utt, total_files_loaded)

    def _get_yearly_files(
        self,
        extension,
    ):
        """
        Collects all file names grouped by year.
        Returns:
            list[dict]:
                A list of dictionaries, each containing:
                    {
                        "files": list of file base names (without extension),
                        "year": year folder name,
                        "number_of_files": total files for that year
                    }
        """
        years_folders = self._get_year_folders_list()
        years_folders = UtilityFilter.filter_years(years_folders, self.years)

        loaded_files_dict, _ = self._load_file_names_by_years(years_folders, extension)
        return loaded_files_dict

    def _count_files(self, loaded_files_dict):
        return sum([files_dict["number_of_files"] for files_dict in loaded_files_dict])

    def _get_year_folders_list(self):
        years_folders = os.listdir(self.base_path)
        # could just delete the file... but its fine like this
        if "00README.txt" in years_folders:
            years_folders.remove("00README.txt")
        return years_folders

    def _load_file_names_by_years(self, years, extension):
        file_names_by_year = []
        total_files = 0

        for year in years:
            path_with_year = os.path.join(self.base_path, year)
            files, number_of_files = self._load_file_names(path_with_year, extension)

            total_files += number_of_files

            file_names_by_year.append(
                {"files": files, "year": year, "number_of_files": number_of_files}
            )
        return file_names_by_year, total_files

    def _load_file_names(self, full_path, extension):
        dirs = os.listdir(full_path)
        files = (file[: -len(extension)] for file in dirs if file.endswith(extension))
        # divide by 3 because each date is compsed by 3 files (.conllu, . meta and .ana-meta)
        number_of_files = len(dirs) / 3
        return files, number_of_files

    def _load_file_data(self, file, year):
        temp_path = os.path.join(self.base_path, year, file)

        df_meta, df_ana_meta, df_segment_data = self._load_parlamint_records(temp_path)

        if df_meta.empty:
            return pd.DataFrame()

        df = merge_data_frames(df_meta, df_ana_meta, df_segment_data)

        # attach year
        df["year"] = year
        return df

    def _load_parlamint_records(self, path):
        df_meta = ParlaMintFileLoader._load_tsv_file(
            path + META_EXT,
            [
                "Text_ID",
                "ID",
                "Date",
                "Speaker_party",
                "Party_orientation",
                "Speaker_ID",
                "Speaker_name",
                "Speaker_gender",
                "Topic",
            ],
        )

        df_meta = UtilityFilter.filter_meta(df_meta, self.topics, self.orientations)

        if df_meta.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_ana_meta = ParlaMintFileLoader._load_tsv_file(
            path + ANA_META_EXT, ["ID", "Parent_ID", "Words"]
        )
        df_segment_data = self._load_segments_sentiment_data(path + CONLLU_EXT)

        return df_meta, df_ana_meta, df_segment_data

    @staticmethod
    def _load_tsv_file(file_path, columns):
        return pd.read_csv(
            file_path,
            sep="\t",
            usecols=columns,
        )

    @staticmethod
    def _load_segments_sentiment_data(file_path):
        """
        Load sentiment data from CoNLL-U formatted file.
        """
        data = []

        with open(file_path, encoding="utf-8") as f:
            files = parse_incr(f)
            for file in files:
                meta = file.metadata
                data.append(
                    {
                        "sent_id": meta.get("sent_id", ""),
                        "text": meta.get("text", ""),
                        "senti_3": meta.get("senti_3", ""),
                        "senti_6": meta.get("senti_6", ""),
                        "senti_n": float(meta.get("senti_n", "0")),
                    }
                )
        return pd.DataFrame(data)

    # NOTE: text files include non-verbal cues
    # the segment text from conllu does not, won't delete this till I get a better idea whats best lol
    # There are 13,335 different cues on the Italian corpus
    # e.g. of these non-verbal cues:
    # [[Applause.]], [[He stands up.]] [[The microphone automatically switches off]])
    # [[Applause from the Mixed Group and Senator {name}]].
    @staticmethod
    def load_texts(path):
        df_uttrances = pd.read_csv(
            path + TXT_EXT, sep="\t", names=["ID", "Text"], header=None
        )

        return df_uttrances
