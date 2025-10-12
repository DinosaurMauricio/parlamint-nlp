import pandas as pd


def process_df_samples(df):
    samples = []

    for parent_id, group in df.groupby("Parent_ID"):
        text_segments = group[
            ["sent_id", "text", "senti_3", "senti_6", "senti_n"]
        ].to_dict(
            orient="records"
        )  # orient so it takes the headers of the keys

        text = " ".join([seg["text"] for seg in text_segments])

        head = group.iloc[0]

        temp_sample = {
            "id": parent_id,
            "party_orientation": head.Party_orientation,
            "topic": head.Topic,
            "text": text,
            "text_segments": text_segments,
            "year": head.year,
        }

        # we take only this value for analyzing, the text included in the conllu
        # might include other information (i.e. such as actions).
        # Only used when .txt file is loaded
        if "Text" in df.columns:
            temp_sample["raw_text"] = head.Text

        samples.append(temp_sample)

    return samples


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
