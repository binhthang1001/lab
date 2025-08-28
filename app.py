import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Công cụ Phân tích Hồi quy",
    page_icon="📊",
    layout="wide"
)

# --- Hàm tiện ích ---

@st.cache_data
def to_excel(df):
    """Chuyển đổi DataFrame thành file Excel trong bộ nhớ."""
    output = io.BytesIO()
    # Nếu df là một Styler object, lấy dữ liệu gốc
    if isinstance(df, pd.io.formats.style.Styler):
        df = df.data
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=True, sheet_name='KetQuaHoiQuy')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def preprocess_data(df, dependent_var, independent_vars):
    """Tự động xử lý biến định tính bằng cách tạo biến giả (dummy variables)."""
    X_df = df[independent_vars]
    y = df[dependent_var]

    # Tách các biến định tính và định lượng
    categorical_vars = X_df.select_dtypes(include=['object', 'category']).columns
    numeric_vars = X_df.select_dtypes(include=np.number).columns

    # Tạo biến giả cho các biến định tính
    if not categorical_vars.empty:
        dummies = pd.get_dummies(X_df[categorical_vars], drop_first=True, dtype=int)
        X_processed = pd.concat([X_df[numeric_vars], dummies], axis=1)
        st.info(f"Đã tự động chuyển đổi các biến định tính sau thành biến giả: {', '.join(categorical_vars)}")
    else:
        X_processed = X_df[numeric_vars]

    X_processed = sm.add_constant(X_processed) # Thêm hằng số
    return X_processed, y


# --- Hàm tính toán & Phân tích ---

def get_descriptive_stats(df):
    """Tạo thống kê mô tả cho dữ liệu."""
    return df.describe(include='all')

def run_linear_regression(X, y):
    """Thực hiện hồi quy tuyến tính và trả về kết quả."""
    model = sm.OLS(y, X).fit()
    return model

def run_logistic_regression(X, y):
    """Thực hiện hồi quy logistic và trả về kết quả."""
    if not all(y.isin([0, 1])):
        st.error(f"Lỗi: Biến phụ thuộc cho Hồi quy Logistic phải là biến nhị phân (chỉ chứa giá trị 0 và 1).")
        return None
    model = sm.Logit(y, X).fit()
    return model

def get_linear_summary_df(model):
    """Tạo DataFrame tóm tắt kết quả cho Hồi quy Tuyến tính."""
    summary_df = pd.DataFrame({
        'Hệ số': model.params,
        'Sai số chuẩn': model.bse,
        't-value': model.tvalues,
        'p-value': model.pvalues
    })
    conf_int = model.conf_int()
    summary_df['CI 2.5%'] = conf_int[0]
    summary_df['CI 97.5%'] = conf_int[1]
    return summary_df

def get_logistic_summary_df(model):
    """Tạo DataFrame tóm tắt kết quả cho Hồi quy Logistic với Odds Ratios."""
    params = model.params
    conf = model.conf_int()
    conf['Odds Ratio'] = params
    conf.columns = ['CI 2.5%', 'CI 97.5%', 'Odds Ratio']
    conf = np.exp(conf)
    
    p_values = model.pvalues
    summary_df = pd.DataFrame({
        'p-value': p_values
    })
    summary_df = conf.join(summary_df)
    summary_df = summary_df[['Odds Ratio', 'CI 2.5%', 'CI 97.5%', 'p-value']]
    return summary_df

# --- Giao diện ứng dụng ---

st.title("📊 Công cụ Phân tích Hồi quy Tuyến tính & Logistic")
st.write("Tải lên file CSV của bạn, chọn các biến và loại mô hình để xem kết quả phân tích chi tiết.")

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
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    with st.sidebar:
        st.header("2. Lựa chọn Mô hình & Biến số")
        model_type = st.radio(
            "Chọn loại mô hình:",
            ('Hồi quy Tuyến tính (Linear Regression)', 'Hồi quy Logistic (Logistic Regression)')
        )

        if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
            dependent_var = st.selectbox("Chọn biến phụ thuộc (Y):", options=numeric_cols)
        else:
            dependent_var = st.selectbox("Chọn biến phụ thuộc (Y):", options=all_cols, help="Phải là biến nhị phân (0/1).")

        available_indep_vars = [col for col in all_cols if col != dependent_var]
        independent_vars = st.multiselect(
            "Chọn các biến độc lập (X):",
            options=available_indep_vars,
            default=available_indep_vars[0] if available_indep_vars else None
        )
        st.markdown("---")
        analyze_button = st.button("Thực hiện Phân tích", type="primary")

    if analyze_button:
        if not dependent_var or not independent_vars:
            st.warning("Vui lòng chọn cả biến phụ thuộc và ít nhất một biến độc lập.")
        else:
            st.header("Kết quả Phân tích")
            
            selected_cols = [dependent_var] + independent_vars
            df_analysis = df[selected_cols].dropna()
            
            if len(df_analysis) < len(df):
                st.info(f"Đã loại bỏ {len(df) - len(df_analysis)} dòng có giá trị thiếu (missing values).")

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
                    # Tiền xử lý dữ liệu
                    X_processed, y_processed = preprocess_data(df_analysis, dependent_var, independent_vars)

                    if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
                        model = run_linear_regression(X_processed, y_processed)
                        results_df = get_linear_summary_df(model)
                        
                        st.subheader("Bảng Hệ số Hồi quy")
                        st.dataframe(results_df.style.format('{:.4f}'))
                        
                        st.download_button(
                            label="📥 Tải xuống dưới dạng Excel",
                            data=to_excel(results_df),
                            file_name='linear_regression_results.xlsx'
                        )
                        
                        st.subheader("Độ phù hợp của Mô hình")
                        col1, col2 = st.columns(2)
                        col1.metric("R-squared", f"{model.rsquared:.4f}")
                        col2.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
                        
                    else: # Logistic Regression
                        model = run_logistic_regression(X_processed, y_processed)
                        if model:
                            results_df = get_logistic_summary_df(model)
                            st.subheader("Bảng Tỷ số chênh (Odds Ratios)")
                            st.dataframe(results_df.style.format('{:.4f}'))
                            
                            st.download_button(
                                label="📥 Tải xuống dưới dạng Excel",
                                data=to_excel(results_df),
                                file_name='logistic_regression_results.xlsx'
                            )

                            st.subheader("Độ phù hợp của Mô hình")
                            st.metric("Pseudo R-squ.", f"{model.prsquared:.4f}")

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")

            with tab4:
                st.subheader("Biểu đồ Minh họa")
                try:
                    X_processed, y_processed = preprocess_data(df_analysis, dependent_var, independent_vars)
                    
                    if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
                        model = run_linear_regression(X_processed, y_processed)
                        df_analysis['predicted'] = model.predict(X_processed)
                        
                        fig = px.scatter(
                            df_analysis, x=dependent_var, y='predicted',
                            title='Biểu đồ phân tán: Giá trị Thực tế vs. Dự đoán',
                            labels={dependent_var: 'Giá trị Thực tế', 'predicted': 'Giá trị Dự đoán'},
                            trendline='ols', trendline_color_override='red'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else: # Logistic Regression
                        model = run_logistic_regression(X_processed, y_processed)
                        if model:
                            y_prob = model.predict(X_processed)
                            fpr, tpr, _ = roc_curve(y_processed, y_prob)
                            roc_auc = auc(fpr, tpr)

                            fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
                            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                            fig.update_layout(
                                title_text='Đường cong ROC (Receiver Operating Characteristic)',
                                xaxis_title='Tỷ lệ Dương tính Giả', yaxis_title='Tỷ lệ Dương tính Thật'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi khi tạo biểu đồ: {e}")
