import pandas as pd
import streamlit as st


@st.cache_data
def load_data(filepath):
    df = pd.read_parquet(filepath)
    return df


@st.cache_data
def load_unique_df(df):
    """
    Keep only unique text to avoid repeated segments.
    """
    return df.drop_duplicates(subset="ID_meta", keep="first").reset_index(drop=True)


@st.cache_data
def get_view_options(df):
    """Return available views based on party orientation."""
    return ["General"] + sorted(df["Party_orientation"].dropna().unique())
