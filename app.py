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
    if isinstance(df, pd.io.formats.style.Styler):
        df = df.data
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=True, sheet_name='KetQuaHoiQuy')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def preprocess_data(df, dependent_var, independent_vars, var_types):
    """Xử lý biến dựa trên định nghĩa của người dùng."""
    X_df = df[independent_vars]
    y = df[dependent_var]

    categorical_vars = [var for var, type in var_types.items() if type == 'Định tính' and var in independent_vars]
    numeric_vars = [var for var, type in var_types.items() if type == 'Định lượng' and var in independent_vars]

    X_processed = pd.DataFrame(index=X_df.index)

    if numeric_vars:
        X_processed[numeric_vars] = X_df[numeric_vars]

    if categorical_vars:
        dummies = pd.get_dummies(X_df[categorical_vars], drop_first=True, dtype=int)
        X_processed = pd.concat([X_processed, dummies], axis=1)
        st.info(f"Đã tự động chuyển đổi các biến định tính sau thành biến giả: {', '.join(categorical_vars)}")
    
    X_processed = sm.add_constant(X_processed, has_constant='add')
    return X_processed, y

@st.cache_data
def calculate_auc_ci(y_true, y_scores, n_bootstraps=1000, alpha=0.95):
    """Tính toán khoảng tin cậy cho AUC bằng bootstrapping."""
    bootstrapped_aucs = []
    rng = np.random.RandomState(42)
    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true_arr), len(y_true_arr))
        if len(np.unique(y_true_arr[indices])) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(y_true_arr[indices], y_scores_arr[indices])
        bootstrapped_aucs.append(auc(fpr, tpr))

    sorted_aucs = np.array(bootstrapped_aucs)
    sorted_aucs.sort()
    
    lower_percentile = (1.0 - alpha) / 2.0 * 100
    upper_percentile = (alpha + (1.0 - alpha) / 2.0) * 100
    lower_bound = np.percentile(sorted_aucs, lower_percentile)
    upper_bound = np.percentile(sorted_aucs, upper_percentile)
    
    return lower_bound, upper_bound

# --- Hàm tính toán & Phân tích ---

def get_descriptive_stats(df):
    return df.describe(include='all')

def run_linear_regression(X, y):
    model = sm.OLS(y, X).fit()
    return model

def run_logistic_regression(X, y):
    if not all(y.isin([0, 1])):
        st.error(f"Lỗi: Biến phụ thuộc cho Hồi quy Logistic phải là biến nhị phân (0/1).")
        return None
    model = sm.Logit(y, X).fit()
    return model

def get_linear_summary_df(model):
    summary_df = pd.DataFrame({'Hệ số': model.params, 'Sai số chuẩn': model.bse, 't-value': model.tvalues, 'p-value': model.pvalues})
    conf_int = model.conf_int()
    summary_df['CI 2.5%'] = conf_int[0]
    summary_df['CI 97.5%'] = conf_int[1]
    return summary_df

def get_logistic_summary_df(model):
    params = model.params
    conf = model.conf_int()
    conf['Odds Ratio'] = params
    conf.columns = ['CI 2.5%', 'CI 97.5%', 'Odds Ratio']
    conf = np.exp(conf)
    p_values = model.pvalues
    summary_df = pd.DataFrame({'p-value': p_values})
    summary_df = conf.join(summary_df)
    summary_df = summary_df[['Odds Ratio', 'CI 2.5%', 'CI 97.5%', 'p-value']]
    return summary_df

# --- Giao diện ứng dụng ---

st.title("📊 Công cụ Phân tích Hồi quy")
st.write("Tải lên file CSV, định nghĩa biến, chọn mô hình và xem kết quả phân tích chi tiết.")

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
    all_cols = df.columns.tolist()

    with st.sidebar:
        st.header("2. Định nghĩa loại biến số")
        var_types = {}
        with st.expander("Nhấp để định nghĩa loại cho từng biến"):
            for col in all_cols:
                # Tự động đoán loại biến
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                is_binary_numeric = is_numeric and df[col].nunique() == 2 and df[col].min() == 0 and df[col].max() == 1
                
                default_type = 'Định lượng' if is_numeric and not is_binary_numeric else 'Định tính'
                
                var_types[col] = st.radio(f"Loại của biến '{col}':", ('Định lượng', 'Định tính'), index=0 if default_type == 'Định lượng' else 1, key=f"type_{col}", horizontal=True)
        
        st.markdown("---")
        st.header("3. Lựa chọn Mô hình & Biến số")
        model_type = st.radio("Chọn loại mô hình:", ('Hồi quy Tuyến tính (Linear Regression)', 'Hồi quy Logistic (Logistic Regression)'))

        # Lọc các biến phù hợp cho Y
        numeric_vars_for_y = [var for var, type in var_types.items() if type == 'Định lượng']
        
        if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
            dependent_var = st.selectbox("Chọn biến phụ thuộc (Y):", options=numeric_vars_for_y)
        else:
            dependent_var = st.selectbox("Chọn biến phụ thuộc (Y):", options=all_cols, help="Phải là biến nhị phân (0/1).")

        available_indep_vars = [col for col in all_cols if col != dependent_var]
        independent_vars = st.multiselect("Chọn các biến độc lập (X):", options=available_indep_vars, default=available_indep_vars[0] if available_indep_vars else None)
        
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
                st.info(f"Đã loại bỏ {len(df) - len(df_analysis)} dòng có giá trị thiếu.")

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
                    X_processed, y_processed = preprocess_data(df_analysis, dependent_var, independent_vars, var_types)

                    if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
                        model = run_linear_regression(X_processed, y_processed)
                        results_df = get_linear_summary_df(model)
                        st.subheader("Bảng Hệ số Hồi quy")
                        st.dataframe(results_df.style.format('{:.4f}'))
                        st.download_button(label="📥 Tải xuống Excel", data=to_excel(results_df), file_name='linear_regression.xlsx')
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
                            st.download_button(label="📥 Tải xuống Excel", data=to_excel(results_df), file_name='logistic_regression.xlsx')
                            st.subheader("Độ phù hợp của Mô hình")
                            st.metric("Pseudo R-squ.", f"{model.prsquared:.4f}")

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")

            with tab4:
                st.subheader("Biểu đồ Minh họa")
                try:
                    X_processed, y_processed = preprocess_data(df_analysis, dependent_var, independent_vars, var_types)
                    
                    if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
                        model = run_linear_regression(X_processed, y_processed)
                        df_analysis['predicted'] = model.predict(X_processed)
                        fig = px.scatter(df_analysis, x=dependent_var, y='predicted', title='Biểu đồ phân tán: Giá trị Thực tế vs. Dự đoán', labels={dependent_var: 'Giá trị Thực tế', 'predicted': 'Giá trị Dự đoán'}, trendline='ols', trendline_color_override='red')
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else: # Logistic Regression
                        model = run_logistic_regression(X_processed, y_processed)
                        if model:
                            y_prob = model.predict(X_processed)
                            fpr, tpr, _ = roc_curve(y_processed, y_prob)
                            roc_auc = auc(fpr, tpr)
                            
                            # Tính 95% CI cho AUC
                            auc_lower, auc_upper = calculate_auc_ci(y_processed, y_prob)
                            
                            st.metric(
                                label="Area Under Curve (AUC)", 
                                value=f"{roc_auc:.4f}",
                                help=f"Khoảng tin cậy 95% cho AUC: ({auc_lower:.4f} - {auc_upper:.4f})"
                            )

                            fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve'))
                            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, 
