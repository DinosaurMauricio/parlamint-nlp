import streamlit as st

from ui.charts import aggregate_topics_by_party, build_pie_chart


def create_orientation_view(df, filters, orientation):
    grouped = aggregate_topics_by_party(df, filters, orientation)
    fig = build_pie_chart(grouped, orientation)
    st.plotly_chart(fig, use_container_width=True)
