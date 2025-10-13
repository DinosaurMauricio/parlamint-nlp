import plotly.express as px


def aggregate_words(df, filters):
    group_cols = ["year", "Topic"]
    if filters.get("Party_orientation"):
        group_cols.append("Party_orientation")

    df_grouped = df.groupby(group_cols)["Words"].sum().reset_index()

    for key in group_cols:
        df_grouped = df_grouped[df_grouped[key].isin(filters.get(key, []))]

    return df_grouped


def build_word_count_bar_chart(df_grouped, filters):
    party_active = bool(filters.get("Party_orientation"))

    args = {
        "x": "year",
        "y": "Words",
        "title": "Total Tokens by Year"
        + (" and Party Orientation" if party_active else ""),
        "labels": {"Words": "Word Count", "year": "Year"},
        "hover_data": ["Topic"],
    }

    if party_active:
        args.update({"color": "Party_orientation", "barmode": "group"})

    return px.bar(df_grouped, **args)


def aggregate_topics_by_party(df, filters, orientation):
    topic_df = df[df["Topic"].isin(filters.get("Topic", []))]
    group_cols = ["Party_orientation", "Topic"]

    topic_counts = topic_df.groupby(group_cols).size().reset_index(name="Count")

    party_data = topic_counts[topic_counts["Party_orientation"] == orientation]

    return party_data


def build_pie_chart(df, orientation):

    return px.pie(
        df,
        names="Topic",
        values="Count",
        title=f"Topic Mentions by {orientation}",
    )


def aggregate_gender_by_year(df):
    return df.groupby(["year", "Speaker_gender"]).size().reset_index(name="Count")


def build_gender_by_year_line_chart(df):
    return px.line(
        df,
        x="year",
        y="Count",
        color="Speaker_gender",
        markers=True,
        title="Gender Over Time",
    )


def aggregate_count_by_topic_and_orientation(df):
    return df.groupby(["Topic", "Party_orientation"]).size().reset_index(name="Count")


def build_count_by_topic_and_orientation_bar_chart(df):
    return px.bar(
        df,
        x="Topic",
        y="Party_orientation",
        color="Party_orientation",
        barmode="group",  # shows bars side-by-side
        title="Topic Mentions by Party Orientation",
        labels={"Count": "Number of Mentions", "Topic": "Topic"},
    )
