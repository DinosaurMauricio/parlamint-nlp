import streamlit as st

import plotly.express as px


def create_view(filters, df):
    # Define columns to group by
    to_group_by = ["year", "Topic"]
    party_filter_active = bool(filters.get("Party_orientation"))
    if party_filter_active:
        to_group_by.append("Party_orientation")

    tokens_by_year = df.groupby(to_group_by)["Words"].sum().reset_index()

    for key in to_group_by:
        tokens_by_year = tokens_by_year[tokens_by_year[key].isin(filters[key])]

    bar_args = {
        "x": "year",
        "y": "Words",
        "title": "Total Tokens by Year"
        + (" and Party Orientation" if party_filter_active else ""),
        "labels": {"Words": "Word Count", "year": "Year"},
        "hover_data": ["Topic"],
    }

    if party_filter_active:
        bar_args["color"] = "Party_orientation"
        bar_args["barmode"] = "group"

    fig = px.bar(tokens_by_year, **bar_args)

    # Streamlit display
    st.plotly_chart(fig, use_container_width=True)
