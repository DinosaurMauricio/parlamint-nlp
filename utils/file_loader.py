import os
import pandas as pd

from conllu import parse_incr

from constants import TXT_EXT


class ParlaMintLoader:
    @staticmethod
    def load_file_names_by_years(path, years, extension):
        file_names_by_year = []
        total_files = 0
        for year in years:
            path_with_year = os.path.join(path, year)
            files, number_of_files = ParlaMintLoader.load_file_names(
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
    def load_years_folders(path):
        years_folders = os.listdir(path)
        # could just delete the file... but its fine like this
        if "00README.txt" in years_folders:
            years_folders.remove("00README.txt")
        return years_folders

    # NOTE: segments include non-verbal cues (e.g. [[Applause.]]) # raw text doesn't.
    # Won't delete this till I get a better idea whats best lol
    @staticmethod
    def load_texts(
        texts_path,
        row,
    ):
        text_file = os.path.join(texts_path, row.Text_ID + TXT_EXT)

        df_uttrances = pd.read_csv(
            text_file, sep="\t", names=["ID", "Text"], header=None
        )

        df_uttrances = df_uttrances[df_uttrances["ID"] == row.ID]

        return [row_utt.Text for _, row_utt in df_uttrances.iterrows()]
