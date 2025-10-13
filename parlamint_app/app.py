import streamlit as st
import plotly.express as px


from views.general import create_view
from utils.filters import get_active_filters
from data.loader import load_data, load_unique_df, get_view_options
from views.general import sidebar, create_view
from views.party_orientation import create_orientation_view
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
        unique_df = load_unique_df(df)
        st.session_state.unique_df = unique_df
else:
    unique_df = st.session_state.unique_df

if "view_selector" not in st.session_state:
    # TODO: When using multiple countries this has to change
    # as we could select a specific country now just for party orientation is fine
    st.session_state.view_selector = "General"

sidebar(df)

filters = get_active_filters(df)

if st.session_state.view_selector == "General":
    create_view(unique_df, filters)
else:
    create_orientation_view(df, filters, st.session_state.view_selector)
