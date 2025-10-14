def aggregate_words(df, filters):
    group_cols = ["year", "Topic"]
    if filters.get("Party_orientation"):
        group_cols.append("Party_orientation")

    df_grouped = df.groupby(group_cols)["Words"].sum().reset_index()

    for key in group_cols:
        df_grouped = df_grouped[df_grouped[key].isin(filters.get(key, []))]

    return df_grouped


def aggregate_topics_by_party(df, filters, orientation):
    topic_df = df[df["Topic"].isin(filters.get("Topic", []))]
    group_cols = ["Party_orientation", "Topic"]

    topic_counts = topic_df.groupby(group_cols).size().reset_index(name="Count")

    party_data = topic_counts[topic_counts["Party_orientation"] == orientation]

    return party_data


def aggregate_count_by_columns(df, filters, columns, filter_keys=[]):

    for key in filter_keys:
        df = df[df[key].isin(filters.get(key, []))]

    df = df.groupby(columns).size().reset_index(name="Count")

    return df
