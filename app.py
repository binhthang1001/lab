import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Công cụ Phân tích Hồi quy",
    page_icon="📊",
    layout="wide"
)

# --- Hàm tính toán & Phân tích ---

def get_descriptive_stats(df):
    """Tạo thống kê mô tả cho dữ liệu."""
    return df.describe()

def run_linear_regression(df, dependent_var, independent_vars):
    """Thực hiện hồi quy tuyến tính và trả về kết quả."""
    X = df[independent_vars]
    y = df[dependent_var]
    X = sm.add_constant(X)  # Thêm hằng số (intercept) vào mô hình
    model = sm.OLS(y, X).fit()
    return model

def run_logistic_regression(df, dependent_var, independent_vars):
    """Thực hiện hồi quy logistic và trả về kết quả."""
    # Kiểm tra biến phụ thuộc có phải là nhị phân (0/1) không
    if not all(df[dependent_var].isin([0, 1])):
        st.error(f"Lỗi: Biến phụ thuộc '{dependent_var}' cho Hồi quy Logistic phải là biến nhị phân (chỉ chứa giá trị 0 và 1).")
        return None
        
    X = df[independent_vars]
    y = df[dependent_var]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit()
    return model

def get_logistic_summary(model):
    """Tạo bảng tóm tắt cho hồi quy logistic với Odds Ratios."""
    params = model.params
    conf = model.conf_int()
    conf['Odds Ratio'] = params
    conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
    conf = np.exp(conf)
    
    p_values = model.pvalues
    summary_df = pd.DataFrame({
        'Hệ số (log-odds)': params,
        'P-value': p_values
    })
    summary_df = summary_df.join(conf)
    return summary_df

# --- Giao diện ứng dụng ---

st.title("📊 Công cụ Phân tích Hồi quy Tuyến tính & Logistic")
st.write("Tải lên file CSV của bạn, chọn các biến và loại mô hình để xem kết quả phân tích chi tiết.")

# --- Sidebar cho việc tải file và lựa chọn ---
with st.sidebar:
    st.header("1. Tải dữ liệu lên")
    uploaded_file = st.file_uploader("Chọn một file CSV", type="csv")

    st.markdown("---")
    
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Tải file thành công!")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
    else:
        st.info("Vui lòng tải lên một file CSV để bắt đầu.")

if df is not None:
    # Lấy danh sách các cột số và tất cả các cột
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    with st.sidebar:
        st.header("2. Lựa chọn Mô hình & Biến số")
        
        model_type = st.radio(
            "Chọn loại mô hình:",
            ('Hồi quy Tuyến tính (Linear Regression)', 'Hồi quy Logistic (Logistic Regression)')
        )

        # Lựa chọn biến dựa trên loại mô hình
        if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
            # Biến phụ thuộc phải là biến số
            dependent_var = st.selectbox(
                "Chọn biến phụ thuộc (Y):",
                options=numeric_cols,
                help="Biến bạn muốn dự đoán. Phải là một biến số."
            )
        else: # Logistic Regression
            # Biến phụ thuộc có thể là bất kỳ biến nào (sẽ được kiểm tra sau)
            dependent_var = st.selectbox(
                "Chọn biến phụ thuộc (Y):",
                options=all_cols,
                help="Biến bạn muốn dự đoán. Phải là biến nhị phân (0 hoặc 1)."
            )

        # Biến độc lập có thể là bất kỳ biến nào
        available_indep_vars = [col for col in all_cols if col != dependent_var]
        independent_vars = st.multiselect(
            "Chọn các biến độc lập (X):",
            options=available_indep_vars,
            default=available_indep_vars[0] if available_indep_vars else None,
            help="Các biến được sử dụng để dự đoán biến phụ thuộc."
        )

        st.markdown("---")
        
        # Nút thực hiện phân tích
        analyze_button = st.button("Thực hiện Phân tích", type="primary")

    # --- Khu vực hiển thị kết quả ---
    if analyze_button:
        if not dependent_var or not independent_vars:
            st.warning("Vui lòng chọn cả biến phụ thuộc và ít nhất một biến độc lập.")
        else:
            st.header("Kết quả Phân tích")
            
            # Xử lý dữ liệu: loại bỏ các dòng có giá trị thiếu trong các cột đã chọn
            selected_cols = [dependent_var] + independent_vars
            df_analysis = df[selected_cols].dropna()
            
            if len(df_analysis) < len(df):
                st.info(f"Đã loại bỏ {len(df) - len(df_analysis)} dòng có giá trị thiếu (missing values).")

            # Tạo các tab để hiển thị kết quả
            tab1, tab2, tab3, tab4 = st.tabs(["Xem trước Dữ liệu", "Thống kê Mô tả", "Kết quả Mô hình", "Trực quan hóa"])

            with tab1:
                st.subheader("5 dòng dữ liệu đầu tiên")
                st.dataframe(df_analysis.head())

            with tab2:
                st.subheader("Thống kê mô tả cho các biến đã chọn")
                st.dataframe(get_descriptive_stats(df_analysis))

            with tab3:
                st.subheader("Kết quả Hồi quy")
                try:
                    if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
                        model = run_linear_regression(df_analysis, dependent_var, independent_vars)
                        st.text(model.summary())
                        st.write(f"**R-squared:** {model.rsquared:.4f}")
                        st.write(f"**Adj. R-squared:** {model.rsquared_adj:.4f}")
                    else: # Logistic Regression
                        model = run_logistic_regression(df_analysis, dependent_var, independent_vars)
                        if model:
                            st.write("Bảng tóm tắt với Tỷ số chênh (Odds Ratios):")
                            summary_df = get_logistic_summary(model)
                            st.dataframe(summary_df)
                            st.write(f"**Pseudo R-squ.:** {model.prsquared:.4f}")

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")

            with tab4:
                st.subheader("Biểu đồ Minh họa")
                try:
                    if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
                        # Tạo giá trị dự đoán
                        df_analysis['predicted'] = model.predict(sm.add_constant(df_analysis[independent_vars]))
                        
                        fig = px.scatter(
                            df_analysis, 
                            x=dependent_var, 
                            y='predicted',
                            title=f'Biểu đồ phân tán: Giá trị Thực tế vs. Dự đoán',
                            labels={dependent_var: 'Giá trị Thực tế', 'predicted': 'Giá trị Dự đoán'},
                            trendline='ols', # Thêm đường hồi quy
                            trendline_color_override='red'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else: # Logistic Regression
                        if model:
                            # Tính toán đường cong ROC
                            y_true = df_analysis[dependent_var]
                            y_prob = model.predict(sm.add_constant(df_analysis[independent_vars]))
                            
                            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                            roc_auc = auc(fpr, tpr)

                            fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
                            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                            
                            fig.update_layout(
                                title_text='Đường cong ROC (Receiver Operating Characteristic)',
                                xaxis_title='Tỷ lệ Dương tính Giả (False Positive Rate)',
                                yaxis_title='Tỷ lệ Dương tính Thật (True Positive Rate)',
                                width=700, height=600
                            )
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi khi tạo biểu đồ: {e}")

