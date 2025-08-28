import streamlit as st
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.mosaicplot import mosaic

st.set_page_config(page_title="Phân tích Tương quan", page_icon="🔗", layout="wide")

st.title("🔗 Phân tích Tương quan (Bivariate Analysis)")

# Kiểm tra dữ liệu
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Vui lòng quay lại trang chủ và tải lên dữ liệu trước.")
    st.stop()

df = st.session_state.df
var_types = st.session_state.var_types

# --- Giao diện lựa chọn biến ---
st.header("1. Lựa chọn Biến số")
col1, col2 = st.columns(2)
with col1:
    y_var = st.selectbox("Chọn biến phụ thuộc (Y):", df.columns)
with col2:
    x_var = st.selectbox("Chọn biến độc lập (X):", [col for col in df.columns if col != y_var])

analyze_btn = st.button("Thực hiện Phân tích", type="primary")

# --- Thực hiện phân tích khi nhấn nút ---
if analyze_btn:
    y_type = var_types.get(y_var)
    x_type = var_types.get(x_var)
    
    # Làm sạch dữ liệu cho phân tích
    analysis_df = df[[y_var, x_var]].dropna()
    
    st.header("2. Kết quả Phân tích")
    
    # Trường hợp 1: Định lượng vs. Định lượng
    if y_type == 'Định lượng' and x_type == 'Định lượng':
        st.subheader(f"Tương quan giữa {y_var} và {x_var}")
        corr = pg.corr(analysis_df[x_var], analysis_df[y_var])
        st.dataframe(corr.style.format('{:.4f}'))
        
        fig = px.scatter(analysis_df, x=x_var, y=y_var, trendline="ols",
                         title=f"Biểu đồ phân tán giữa {y_var} và {x_var}")
        st.plotly_chart(fig, use_container_width=True)

    # Trường hợp 2: Định lượng vs. Định tính
    elif y_type == 'Định lượng' and x_type == 'Định tính':
        st.subheader(f"So sánh {y_var} theo nhóm {x_var}")
        n_groups = analysis_df[x_var].nunique()
        
        if n_groups == 2:
            test_result = pg.ttest(data=analysis_df, dv=y_var, between=x_var)
            st.write("Kết quả T-test:")
        elif n_groups > 2:
            test_result = pg.anova(data=analysis_df, dv=y_var, between=x_var)
            st.write("Kết quả ANOVA:")
        else:
            st.warning("Biến độc lập chỉ có 1 nhóm, không thể so sánh.")
            st.stop()
            
        st.dataframe(test_result.style.format('{:.4f}'))
        
        fig = px.violin(analysis_df, x=x_var, y=y_var, box=True, points="all",
                        title=f"Biểu đồ Violin so sánh {y_var} theo {x_var}")
        st.plotly_chart(fig, use_container_width=True)

    # Trường hợp 3: Định tính vs. Định tính
    elif y_type == 'Định tính' and x_type == 'Định tính':
        st.subheader(f"Mối liên quan giữa {y_var} và {x_var}")
        
        # Bảng chéo (Crosstab)
        contingency_table = pd.crosstab(analysis_df[y_var], analysis_df[x_var])
        st.write("Bảng chéo (Observed Frequencies):")
        st.dataframe(contingency_table)
        
        # Kiểm định Chi-square
        expected, observed, stats = pg.chi2_independence(data=analysis_df, x=x_var, y=y_var)
        st.write("Kết quả kiểm định Chi-square:")
        st.dataframe(stats.style.format('{:.4f}'))
        
        # Biểu đồ Mosaic
        st.write("Biểu đồ Mosaic:")
        fig, ax = mosaic(contingency_table.stack(), title=f'Mosaic plot of {y_var} vs {x_var}', gap=0.02)
        st.pyplot(fig)

    else:
        st.error("Sự kết hợp loại biến này chưa được hỗ trợ (ví dụ: Y định tính vs X định lượng). Vui lòng chọn lại.")
