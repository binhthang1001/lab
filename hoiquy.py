import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io

# --- Hàm tiện ích (đã được định nghĩa trong các immersive trước) ---
@st.cache_data
def to_excel(df):
    output = io.BytesIO()
    if isinstance(df, pd.io.formats.style.Styler):
        df = df.data
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=True, sheet_name='KetQuaHoiQuy')
    writer.close()
    return output.getvalue()

@st.cache_data
def calculate_auc_ci(y_true, y_scores, n_bootstraps=1000, alpha=0.95):
    bootstrapped_aucs = []
    rng = np.random.RandomState(42)
    y_true_arr, y_scores_arr = np.array(y_true), np.array(y_scores)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true_arr), len(y_true_arr))
        if len(np.unique(y_true_arr[indices])) < 2: continue
        fpr, tpr, _ = roc_curve(y_true_arr[indices], y_scores_arr[indices])
        bootstrapped_aucs.append(auc(fpr, tpr))
    sorted_aucs = np.array(bootstrapped_aucs)
    sorted_aucs.sort()
    lower_bound = np.percentile(sorted_aucs, (1.0 - alpha) / 2.0 * 100)
    upper_bound = np.percentile(sorted_aucs, (alpha + (1.0 - alpha) / 2.0) * 100)
    return lower_bound, upper_bound

# --- Bắt đầu giao diện ---
st.set_page_config(page_title="Phân tích Hồi quy", page_icon="📊", layout="wide")
st.title("📊 Phân tích Hồi quy (Multivariate Analysis)")

if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Vui lòng quay lại trang chủ và tải lên dữ liệu trước.")
    st.stop()

df = st.session_state.df
var_types = st.session_state.var_types
all_cols = df.columns.tolist()

# --- Giao diện lựa chọn ---
st.header("1. Lựa chọn Mô hình & Biến số")

model_type = st.radio("Chọn loại mô hình:", ('Hồi quy Tuyến tính (Linear Regression)', 'Hồi quy Logistic (Logistic Regression)'))

numeric_vars_for_y = [var for var, type in var_types.items() if type == 'Định lượng']
if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
    dependent_var = st.selectbox("Chọn biến phụ thuộc (Y):", options=numeric_vars_for_y)
else:
    dependent_var = st.selectbox("Chọn biến phụ thuộc (Y):", options=all_cols, help="Phải là biến nhị phân (0/1).")

available_indep_vars = [col for col in all_cols if col != dependent_var]
independent_vars = st.multiselect("Chọn các biến độc lập (X):", options=available_indep_vars, default=available_indep_vars[0] if available_indep_vars else None)

analyze_button = st.button("Thực hiện Phân tích Hồi quy", type="primary")

# --- Thực hiện phân tích ---
if analyze_button:
    if not dependent_var or not independent_vars:
        st.warning("Vui lòng chọn cả biến phụ thuộc và ít nhất một biến độc lập.")
    else:
        st.header("2. Kết quả Phân tích Hồi quy")
        
        selected_cols = [dependent_var] + independent_vars
        df_analysis = df[selected_cols].dropna()

        # Tiền xử lý
        X_df = df_analysis[independent_vars]
        y = df_analysis[dependent_var]
        categorical_vars = [var for var, type in var_types.items() if type == 'Định tính' and var in independent_vars]
        numeric_vars = [var for var, type in var_types.items() if type == 'Định lượng' and var in independent_vars]
        
        X_processed = pd.DataFrame(index=X_df.index)
        if numeric_vars: X_processed[numeric_vars] = X_df[numeric_vars]
        if categorical_vars:
            dummies = pd.get_dummies(X_df[categorical_vars], drop_first=True, dtype=int)
            X_processed = pd.concat([X_processed, dummies], axis=1)
            st.info(f"Đã chuyển đổi các biến định tính: {', '.join(categorical_vars)}")
        X_processed = sm.add_constant(X_processed, has_constant='add')

        # Chạy mô hình
        try:
            if model_type == 'Hồi quy Tuyến tính (Linear Regression)':
                model = sm.OLS(y, X_processed).fit()
                results_df = pd.DataFrame({'Hệ số': model.params, 'Sai số chuẩn': model.bse, 'p-value': model.pvalues})
                conf_int = model.conf_int()
                results_df['CI 2.5%'] = conf_int[0]
                results_df['CI 97.5%'] = conf_int[1]
                
                st.subheader("Bảng Kết quả Hồi quy Tuyến tính")
                st.dataframe(results_df.style.format('{:.4f}'))
                st.download_button("📥 Tải xuống Excel", to_excel(results_df), 'linear_regression.xlsx')
                
                st.subheader("Độ phù hợp của Mô hình")
                col1, col2 = st.columns(2)
                col1.metric("R-squared", f"{model.rsquared:.4f}")
                col2.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")

                st.subheader("Kiểm tra Giả định Mô hình")
                residuals = model.resid
                fitted = model.fittedvalues
                fig_resid = px.scatter(x=fitted, y=residuals, labels={'x': 'Giá trị phù hợp (Fitted)', 'y': 'Phần dư (Residuals)'}, title="Biểu đồ Phần dư vs. Giá trị phù hợp")
                fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_resid, use_container_width=True)

            else: # Logistic Regression
                if not all(y.isin([0, 1])):
                    st.error("Biến phụ thuộc cho Hồi quy Logistic phải là nhị phân (0/1).")
                else:
                    model = sm.Logit(y, X_processed).fit()
                    params = model.params
                    conf = model.conf_int()
                    conf['Odds Ratio'] = params
                    conf.columns = ['CI 2.5%', 'CI 97.5%', 'Odds Ratio']
                    conf = np.exp(conf)
                    results_df = conf.join(pd.DataFrame({'p-value': model.pvalues}))
                    results_df = results_df[['Odds Ratio', 'CI 2.5%', 'CI 97.5%', 'p-value']]
                    
                    st.subheader("Bảng Kết quả Hồi quy Logistic (Odds Ratios)")
                    st.dataframe(results_df.style.format('{:.4f}'))
                    st.download_button("📥 Tải xuống Excel", to_excel(results_df), 'logistic_regression.xlsx')
                    
                    st.subheader("Độ phù hợp của Mô hình")
                    st.metric("Pseudo R-squ.", f"{model.prsquared:.4f}")
                    
                    st.subheader("Đánh giá Mô hình")
                    y_prob = model.predict(X_processed)
                    fpr, tpr, _ = roc_curve(y, y_prob)
                    roc_auc = auc(fpr, tpr)
                    auc_lower, auc_upper = calculate_auc_ci(y, y_prob)
                    
                    st.metric("Area Under Curve (AUC)", f"{roc_auc:.4f}", help=f"95% CI: ({auc_lower:.4f} - {auc_upper:.4f})")
                    
                    fig_roc = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines'))
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    fig_roc.update_layout(title_text=f'Đường cong ROC (AUC = {roc_auc:.4f}, 95% CI: {auc_lower:.4f}-{auc_upper:.4f})',
                                          xaxis_title='Tỷ lệ Dương tính Giả', yaxis_title='Tỷ lệ Dương tính Thật')
                    st.plotly_chart(fig_roc, use_container_width=True)

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")
