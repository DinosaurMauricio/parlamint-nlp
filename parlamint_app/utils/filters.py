import streamlit as st
from config import FILTERS


def get_active_filters(df):

    filtered_settings = {
        key: [
            y
            for y in df[key].unique()
            if st.session_state.get(f"{key.lower()}_{y}", False)
        ]
        for _, key in FILTERS
    }

    return filtered_settings
