import streamlit as st
from data.loader import load_data, load_unique_df
from config import DATA_PATH


def load_app_state():
    if "df" not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data(DATA_PATH)

    if "unique_df" not in st.session_state:
        with st.spinner("Filtering unique records..."):
            st.session_state.unique_df = load_unique_df(st.session_state.df)

    if "view_selector" not in st.session_state:
        st.session_state.view_selector = "General"

    return st.session_state.df, st.session_state.unique_df
