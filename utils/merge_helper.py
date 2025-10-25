import pandas as pd


def merge_data_frames(df_meta, df_ana_meta, df_sentiment_data):
    # merge with ana_meta because we want to know the segments from the utterance
    merged = pd.merge(
        df_meta,
        df_ana_meta,
        left_on="ID",
        right_on="Parent_ID",
        suffixes=("_meta", "_ana_meta"),
    )

    # merge with the sentiment data (mostly for the text segments...)
    merged = pd.merge(
        merged, df_sentiment_data, left_on="ID_ana_meta", right_on="sent_id"
    )
    return merged
