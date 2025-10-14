import plotly.express as px


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


def build_pie_chart(df, orientation):

    return px.pie(
        df,
        names="Topic",
        values="Count",
        title=f"Topic Mentions by {orientation}",
    )


def build_gender_by_year_line_chart(df):
    return px.line(
        df,
        x="year",
        y="Count",
        color="Speaker_gender",
        markers=True,
        title="Gender Over Time",
    )


def build_count_by_topic_and_orientation_bar_chart(df):
    return px.bar(
        df,
        x="Topic",
        y="Count",
        color="Party_orientation",
        barmode="group",  # shows bars side-by-side
        title="Topic Mentions by Party Orientation",
        labels={"Count": "Number of Mentions", "Topic": "Topic"},
    )


def build_topics_per_year_chart(df):
    return px.bar(
        df,
        x="year",
        y="Count",
        color="Party_orientation",
        barmode="group",
        title="Topics Mentioned per Year by Party",
        hover_data=["Topic"],
        labels={"Count": "Mentions", "year": "Year"},
    )
