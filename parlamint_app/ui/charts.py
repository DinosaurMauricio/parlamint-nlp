import plotly.express as px


def aggregate_words(df, filters):
    group_cols = ["year", "Topic"]
    if filters.get("Party_orientation"):
        group_cols.append("Party_orientation")

    df_grouped = df.groupby(group_cols)["Words"].sum().reset_index()

    for key in group_cols:
        if key in filters and filters[key]:
            df_grouped = df_grouped[df_grouped[key].isin(filters[key])]

    return df_grouped


def build_bar_chart(df_grouped, filters):
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
