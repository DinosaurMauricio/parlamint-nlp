import os
import pandas as pd

from conllu import parse_incr

from utils.constants import TXT_EXT, ANA_META_EXT, META_EXT, CONLLU_EXT
from utils.filters import filter_meta, filter_years


class ParlaMintFileLoader:
    @staticmethod
    def get_yearly_files(path, extension, years_to_filter=None):
        """
        Collects all file names grouped by year.
        Args:
            str:
                path to files
            str:
                extension of file
            years_to_filter:
                the years we want to filter, default set to None
        Returns:
            list[dict]:
                A list of dictionaries, each containing:
                    {
                        "files": list of file base names (without extension),
                        "year": year folder name,
                        "number_of_files": total files for that year
                    }
        """
        years_folders = ParlaMintFileLoader.get_year_folders_list(path)
        years_folders = filter_years(years_folders, years_to_filter)

        loaded_files_dict, _ = ParlaMintFileLoader.load_file_names_by_years(
            path, years_folders, extension
        )
        return loaded_files_dict

    @staticmethod
    def load_file_names_by_years(path, years, extension):
        file_names_by_year = []
        total_files = 0

        for year in years:
            path_with_year = os.path.join(path, year)
            files, number_of_files = ParlaMintFileLoader.load_file_names(
                path_with_year, extension
            )

            total_files += number_of_files

            file_names_by_year.append(
                {"files": files, "year": year, "number_of_files": number_of_files}
            )
        return file_names_by_year, total_files

    @staticmethod
    def load_file_names(path, extension):
        dirs = os.listdir(path)
        files = (file[: -len(extension)] for file in dirs if file.endswith(extension))
        # divide by 3 because each date is compsed by 3 files (.conllu, . meta and .ana-meta)
        number_of_files = len(dirs) / 3
        return files, number_of_files

    @staticmethod
    def load_tsv_file(file_path, columns):
        return pd.read_csv(
            file_path,
            sep="\t",
            usecols=columns,
        )

    @staticmethod
    def load_segments_sentiment_data(file_path):
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

    @staticmethod
    def get_year_folders_list(path):
        years_folders = os.listdir(path)
        # could just delete the file... but its fine like this
        if "00README.txt" in years_folders:
            years_folders.remove("00README.txt")
        return years_folders

    @staticmethod
    def load_parlamint_records(path, config):

        df_meta = ParlaMintFileLoader.load_tsv_file(
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

        df_meta = filter_meta(df_meta, config.topics, config.orientations)

        if df_meta.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_ana_meta = ParlaMintFileLoader.load_tsv_file(
            path + ANA_META_EXT, ["ID", "Parent_ID", "Words"]
        )
        df_segment_data = ParlaMintFileLoader.load_segments_sentiment_data(
            path + CONLLU_EXT
        )

        return df_meta, df_ana_meta, df_segment_data

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
