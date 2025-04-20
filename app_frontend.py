# app_frontend.py

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Stock Recommender", layout="wide")
st.title("📈 Today's Top Stock Picks")

# 读取推荐结果
try:
    df = pd.read_csv("today_recommendations.csv")
    st.success("✅ Loaded recommendation file successfully.")
except FileNotFoundError:
    st.error("❌ File not found: today_recommendations.csv")
    st.stop()

# 可选筛选器
top_n = st.slider("Show top N stocks by score", min_value=1, max_value=20, value=10)
df_sorted = df.sort_values(by="score", ascending=False).head(top_n)

# 显示结果表格
st.dataframe(df_sorted, use_container_width=True)

# 显示分析图（可选）
st.line_chart(df_sorted["score"])