import streamlit as st
import plotly.express as px


from views.general import create_view
from filters import sidebar, selected_filters_from_sidebar, get_view_selector
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

if "view_selector" not in st.session_state:
    # TODO: When using multiple countries this has to change
    # as we could select a specific country now just for party orientation is fine
    st.session_state.view_selector = get_view_selector(df)

sidebar(df)

filters = selected_filters_from_sidebar(df)

if st.session_state.view_selector == "General":
    create_view(filters, unique_df)
else:
    topic_tab, gender_tab, orientation_tab = st.tabs(
        [
            "Topics",
            "Gender",
            "Party",
        ]
    )

    # Filter dataframe by selected topics (if any)
    topic_df = unique_df[unique_df["Topic"].isin(filters.get("Topic", []))]

    # Group by Party_orientation and Topic, and count occurrences
    topic_counts = (
        topic_df.groupby(["Party_orientation", "Topic"])
        .size()
        .reset_index(name="Count")
    )

    # Get unique party orientations
    orientations = topic_counts["Party_orientation"].unique()

    # Create one pie chart per orientation
    for orientation in orientations:
        party_data = topic_counts[topic_counts["Party_orientation"] == orientation]

        fig = px.pie(
            party_data,
            names="Topic",
            values="Count",
            title=f"Topic Mentions by {orientation}",
        )

        st.plotly_chart(fig, use_container_width=True)

    words_by_year_gender = (
        unique_df.groupby(["year", "Speaker_gender"]).size().reset_index(name="Count")
    )

    fig3 = px.line(
        words_by_year_gender,
        x="year",
        y="Count",
        color="Speaker_gender",
        markers=True,
        title="Gender Over Time",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Group by Topic and Party_orientation, count occurrences
    topic_party_counts = (
        unique_df.groupby(["Topic", "Party_orientation"])
        .size()
        .reset_index(name="Count")
    )

    fig_bar = px.bar(
        topic_party_counts,
        x="Topic",
        y="Count",
        color="Party_orientation",
        barmode="group",  # shows bars side-by-side
        title="Topic Mentions by Party Orientation",
        labels={"Count": "Number of Mentions", "Topic": "Topic"},
    )

    st.plotly_chart(fig_bar, use_container_width=True)
