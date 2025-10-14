import streamlit as st
import plotly.express as px


from views.general import create_view
from utils.filters import get_active_filters
from state import load_app_state
from views.general import create_sidebar, create_view
from views.party_orientation import create_orientation_view
from config import DATA_PATH

# Config
st.set_page_config(
    page_title="ParlaMint",
)

# Data
df, unique_df = load_app_state()

# Views
st.title("ParlaMint Dashboard")
create_sidebar(df)

filters = get_active_filters(df)

if st.session_state.view_selector == "General":
    create_view(unique_df, filters)
else:
    create_orientation_view(df, filters, st.session_state.view_selector)
