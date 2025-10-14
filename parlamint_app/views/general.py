import streamlit as st

from config import FILTERS
from data.loader import get_view_options
from ui.charts import (
    build_word_count_bar_chart,
    build_gender_by_year_line_chart,
    build_count_by_topic_and_orientation_bar_chart,
    build_topics_per_year_chart,
)
from ui.aggregations import (
    aggregate_words,
    aggregate_count_by_columns,
)


def create_sidebar(df):
    with st.sidebar:
        st.header("Overview")
        st.session_state.view_selector = st.selectbox(
            "Choose Party Orientation:", get_view_options(df)
        )

        st.header("Filters")
        with st.container(border=True):
            st.markdown("**Select Filters**")

            for filter in FILTERS:
                header, key = filter
                with st.expander(header):
                    select_all = st.checkbox(
                        "Select All", value=True, key=f"select_all_{header.lower()}"
                    )
                    st.markdown(f"**Select {header}:**")
                    for y in sorted(df[key].unique()):
                        st.checkbox(str(y), value=select_all, key=f"{key.lower()}_{y}")


def create_view(df, filters):
    # TODO: This for sure could be abstracted in a more simple redable way
    # for now, simple like this, is fine
    tab = st.selectbox("Choose Insights view", ["Text Overview", "Gender", "Topic"])

    if tab == "Text Overview":
        # TODO: Use for statistics
        # word_count_tab, gender_tab, topic_tab = st.tabs(["Word", "Gender", "Topic"])
        # with word_count_tab:
        grouped = aggregate_words(df, filters)
        if grouped.empty:
            # TODO: Improve the error messages for each selected case
            st.warning("No data available for the selected filters.")
        else:
            fig = build_word_count_bar_chart(grouped, filters)
            st.plotly_chart(fig, use_container_width=True)

    elif tab == "Gender":
        grouped = aggregate_count_by_columns(
            df,
            filters,
            ["year", "Speaker_gender"],
            ["year"],
        )
        if grouped.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig = build_gender_by_year_line_chart(grouped)
            st.plotly_chart(fig, use_container_width=True)

    elif tab == "Topic":
        grouped = aggregate_count_by_columns(
            df,
            filters,
            ["Party_orientation", "Topic"],
            ["Party_orientation", "Topic"],
        )
        if grouped.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig = build_count_by_topic_and_orientation_bar_chart(grouped)
            st.plotly_chart(fig, use_container_width=True)

        grouped_by_year = aggregate_count_by_columns(
            df,
            filters,
            ["Party_orientation", "Topic", "year"],
            ["Party_orientation", "Topic", "year"],
        )

        if grouped.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig = build_topics_per_year_chart(grouped_by_year)
            st.plotly_chart(fig, use_container_width=True)
