import streamlit as st
import plotly.express as px

from filters import sidebar, selected_filters_from_sidebar
from utils.data import load_data, get_unique_df
from config import DATA_PATH

# Config
st.set_page_config(
    page_title="ParlaMint",
)

st.title("ParlaMint Dashboard")

# Data
if "df" not in st.session_state:
    with st.spinner("Loading data..."):
        df = load_data(DATA_PATH)
        st.session_state.df = df
else:
    df = st.session_state.df


if "unique_df" not in st.session_state:
    with st.spinner("Filtering some data..."):
        unique_df = get_unique_df(df)
        st.session_state.unique_df = unique_df
else:
    unique_df = st.session_state.unique_df


sidebar(df)

filters = selected_filters_from_sidebar(df)

# Define columns to group by
to_filter = ["year", "Topic"]
party_filter_active = bool(filters.get("Party_orientation"))
if party_filter_active:
    to_filter.append("Party_orientation")

# Group and sum words
tokens_by_year = unique_df.groupby(to_filter)["Words"].sum().reset_index()

# Apply filters
tokens_by_year = tokens_by_year[tokens_by_year["year"].isin(filters["year"])]
tokens_by_year = tokens_by_year[tokens_by_year["Topic"].isin(filters["Topic"])]
if party_filter_active:
    tokens_by_year = tokens_by_year[
        tokens_by_year["Party_orientation"].isin(filters["Party_orientation"])
    ]

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
