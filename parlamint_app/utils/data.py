import pandas as pd
import streamlit as st


@st.cache_data
def load_data(filepath):
    df = pd.read_parquet(filepath)
    return df


@st.cache_data
def get_unique_df(df):
    """
    Keep only unique text to avoid repeated segments.
    """
    return df.drop_duplicates(subset="ID_meta", keep="first").reset_index(drop=True)


# def get_grouped_data_by_filters(df, years=None, orientations=None, topics= None):
#    if years:
#        df = df[df["year"].isin(years)]
#    if orientation:
#        df = df[df["speaker"] == speaker]
#    return df
#
#
# tokens_by_year = unique_df.groupby("year")["Words"].sum().reset_index()
# tokens_by_year = tokens_by_year[tokens_by_year["year"].isin(filters["Years"])]
