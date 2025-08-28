import streamlit as st
import pandas as pd
import numpy as np

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Trang chủ | Công cụ Phân tích Dữ liệu",
    page_icon="🏠",
    layout="wide"
)

# --- Khởi tạo Session State ---
# Dùng để lưu trữ và chia sẻ dữ liệu giữa các trang
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_types' not in st.session_state:
    st.session_state.var_types = None

# --- Giao diện Trang chủ ---
st.title("🏠 Công cụ Phân tích Dữ liệu Trực tuyến")
st.write("""
Chào mừng bạn đến với công cụ phân tích dữ liệu chuyên nghiệp. Ứng dụng này cho phép bạn thực hiện các phân tích từ mô tả cơ bản đến các mô hình hồi quy phức tạp.
""")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("Bắt đầu")
    st.write("Vui lòng thực hiện theo các bước dưới đây:")

    # Bước 1: Tải dữ liệu
    st.subheader("1. Tải dữ liệu lên")
    uploaded_file = st.file_uploader("Chọn một file CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("Tải file thành công!")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
            st.session_state.df = None
    
    # Nếu đã có dữ liệu, hiển thị các bước tiếp theo
    if st.session_state.df is not None:
        df = st.session_state.df
        all_cols = df.columns.tolist()

        # Bước 2: Định nghĩa loại biến
        st.subheader("2. Định nghĩa loại biến số")
        var_types = {}
        with st.expander("Nhấp để định nghĩa loại cho từng biến"):
            for col in all_cols:
                # Tự động đoán loại biến
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                is_binary_numeric = is_numeric and df[col].nunique() == 2 and df[col].min() == 0 and df[col].max() == 1
                
                default_type = 'Định lượng' if is_numeric and not is_binary_numeric else 'Định tính'
                
                var_types[col] = st.radio(
                    f"Loại của biến '{col}':", 
                    ('Định lượng', 'Định tính'), 
                    index=0 if default_type == 'Định lượng' else 1, 
                    key=f"type_{col}", 
                    horizontal=True
                )
        st.session_state.var_types = var_types
        
        st.success("Dữ liệu và loại biến đã sẵn sàng!")
        st.info("Bây giờ bạn có thể điều hướng đến các trang phân tích ở bên trái.")

# --- Nội dung chính của trang chủ ---
if st.session_state.df is not None:
    st.header("Xem trước Dữ liệu")
    st.write(f"Dữ liệu của bạn có **{st.session_state.df.shape[0]}** quan sát và **{st.session_state.df.shape[1]}** biến số.")
    st.dataframe(st.session_state.df.head())
else:
    st.info("Vui lòng tải lên một file CSV trong thanh công cụ bên trái để bắt đầu phân tích.")

st.markdown("---")
st.write("Phát triển bởi Gemini | Hỗ trợ bởi Streamlit")

