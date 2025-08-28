import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Phân tích Mô tả", page_icon="📈", layout="wide")

st.title("📈 Phân tích Mô tả")

# Kiểm tra xem dữ liệu đã được tải lên chưa
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Vui lòng quay lại trang chủ và tải lên dữ liệu trước.")
    st.stop()

df = st.session_state.df
var_types = st.session_state.var_types

# Tách biến định lượng và định tính
numeric_vars = [var for var, type in var_types.items() if type == 'Định lượng']
categorical_vars = [var for var, type in var_types.items() if type == 'Định tính']

# --- Thống kê tổng quan ---
st.header("1. Thống kê Tổng quan")
total_obs = len(df)
total_vars = len(df.columns)
missing_cells = df.isnull().sum().sum()
missing_percentage = (missing_cells / (total_obs * total_vars)) * 100 if (total_obs * total_vars) > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Số lượng Quan sát", f"{total_obs}")
col2.metric("Số lượng Biến số", f"{total_vars}")
col3.metric("Tỷ lệ Thiếu (%)", f"{missing_percentage:.2f}%")

st.markdown("---")

# --- Phân tích Biến Định lượng ---
if numeric_vars:
    st.header("2. Phân tích Biến Định lượng")
    
    # Bảng thống kê
    st.subheader("Bảng Thống kê")
    desc_numeric = df[numeric_vars].describe().transpose()
    st.dataframe(desc_numeric.style.format('{:.2f}'))

    # Biểu đồ
    st.subheader("Biểu đồ Phân phối")
    selected_numeric_var = st.selectbox("Chọn một biến định lượng để vẽ biểu đồ:", numeric_vars)
    
    if selected_numeric_var:
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(df, x=selected_numeric_var, title=f"Histogram của {selected_numeric_var}")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_box = px.box(df, y=selected_numeric_var, title=f"Boxplot của {selected_numeric_var}")
            st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# --- Phân tích Biến Định tính ---
if categorical_vars:
    st.header("3. Phân tích Biến Định tính")

    # Bảng thống kê
    st.subheader("Bảng Tần suất")
    selected_cat_var_table = st.selectbox("Chọn một biến định tính để xem bảng tần suất:", categorical_vars, key="cat_table")
    if selected_cat_var_table:
        freq_table = df[selected_cat_var_table].value_counts().reset_index()
        freq_table.columns = ['Giá trị', 'Tần suất']
        freq_table['Tỷ lệ (%)'] = (freq_table['Tần suất'] / total_obs * 100).round(2)
        st.dataframe(freq_table)

    # Biểu đồ
    st.subheader("Biểu đồ Tần suất")
    selected_cat_var_plot = st.selectbox("Chọn một biến định tính để vẽ biểu đồ:", categorical_vars, key="cat_plot")
    if selected_cat_var_plot:
        fig_bar = px.bar(
            freq_table, 
            x='Giá trị', 
            y='Tần suất', 
            title=f"Biểu đồ cột của {selected_cat_var_plot}",
            text='Tần suất'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
