import streamlit as st

from config import FILTERS


@st.cache_data
def get_view_selector(df):
    return ["General"] + list(df["Party_orientation"].unique())


def sidebar(df):
    with st.sidebar:
        st.header("Overview")
        st.session_state.view_selector = st.selectbox(
            "Choose Party Orientation:", get_view_selector(df)
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


def selected_filters_from_sidebar(df):

    filtered_settings = {
        key: [
            y
            for y in df[key].unique()
            if st.session_state.get(f"{key.lower()}_{y}", False)
        ]
        for _, key in FILTERS
    }

    return filtered_settings
