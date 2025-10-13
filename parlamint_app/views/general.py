import streamlit as st

from config import FILTERS
from data.loader import get_view_options
from ui.charts import aggregate_words, build_bar_chart


def sidebar(df):
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
    grouped = aggregate_words(df, filters)
    fig = build_bar_chart(grouped, filters)
    st.plotly_chart(fig, use_container_width=True)
