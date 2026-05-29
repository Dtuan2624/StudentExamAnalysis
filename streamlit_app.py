import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import os

# Thêm đường dẫn để import các module trong project
sys.path.append(str(Path(__file__).parent))

try:
    from src.model import preprocess_data, predict_new_data, get_feature_importance
    from src.visualize import (
        plot_score_distribution, plot_study_vs_score, plot_correlation_heatmap,
        plot_score_by_gender, plot_score_by_school_type, plot_avg_score_by_parental_involvement,
        plot_attendance_vs_score, plot_motivation_level_distribution, plot_pair_plot,
        plot_actual_vs_predicted, plot_feature_importance, plot_residuals,
        plot_prediction_errors, plot_model_metrics, plot_cv_scores
    )
except ImportError:
    st.warning("⚠️ Chưa tìm thấy module src. Đang chạy ở chế độ demo.")

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Student Exam Performance",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
    <style>
    /* Styling for Sidebar and Radio Buttons */
    .css-1d391kg {
        background-color: #F8FAFC;
    }
    
    /* Elegant gradients and soft shadows */
    .main-header {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2563EB !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B !important;
    }
    
    /* Elegant premium buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.1), 0 2px 4px -1px rgba(37, 99, 235, 0.06);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2), 0 4px 6px -4px rgba(37, 99, 235, 0.1);
        border: none;
    }
    
    /* Clean container panels */
    .premium-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    
    /* Info alert premium touch */
    .stAlert {
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown("<h1 class='main-header'>📊 Student Exam Performance</h1>", unsafe_allow_html=True)
st.markdown("##### Phân Tích Các Yếu Tố Ảnh Hưởng & Dự Đoán Điểm Số Bằng Machine Learning")
st.markdown("---")

# ====================== SIDEBAR ======================
st.sidebar.markdown("### 📚 MENU ĐIỀU HƯỚNG")
page = st.sidebar.radio("Chọn tính năng:",
                        ["🏠 Tổng Quan Đề Tài", "📊 Phân Tích Dữ Liệu (EDA)", "🔮 Dự Đoán Điểm Thi", "📈 Đánh Giá Hiệu Năng Mô Hình"])

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model_files():
    model_path = Path("models/student_performance_model.pkl")
    encoders_path = Path("models/label_encoders.pkl")
    
    model = None
    label_encoders = {}
    
    if model_path.exists():
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.sidebar.error(f"❌ Không load được model: {e}")
            
    if encoders_path.exists():
        try:
            label_encoders = joblib.load(encoders_path)
        except Exception as e:
            st.sidebar.error(f"❌ Không load được encoders: {e}")
            
    return model, label_encoders

model, label_encoders = load_model_files()

if model is not None:
    st.sidebar.success("✅ Hệ thống AI đang hoạt động!")
else:
    st.sidebar.warning("⚠️ Model chưa huấn luyện hoặc chưa lưu. Vui lòng chạy main.py trước.")

# ====================== PAGES ======================

# PAGE 1: OVERVIEW
if page == "🏠 Tổng Quan Đề Tài":
    st.markdown("### 🏠 Tổng Quan Về Đề Tài Nghiên Cứu")
    
    # KPI metrics panel
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Thuật toán", "Random Forest Regressor")
    with col2:
        st.metric("Hệ số R² Score", "84.12%")
    with col3:
        st.metric("Sai số MAE", "0.96 Điểm")
    with col4:
        st.metric("Số lượng mẫu (Dòng)", "6,607 dòng")
        
    st.markdown("""
    #### 🎯 Mục tiêu của Đề tài:
    Đề tài nghiên cứu xây dựng giải pháp phân tích đa chiều các yếu tố gia đình, nhà trường và bản thân học sinh ảnh hưởng trực tiếp đến kết quả thi cuối kỳ. Từ đó, xây dựng mô hình học máy để dự báo điểm số thi chính xác của học sinh, giúp nhà trường và gia đình phát hiện sớm các nguy cơ học tập sa sút để hỗ trợ kịp thời.
    
    #### 🏗️ Quy trình triển khai (Pipeline):
    1. **Thu thập dữ liệu:** Sử dụng tập dữ liệu **Student Performance Factors** gồm 20 cột chứa dữ liệu thô về các mặt hành vi, thể chất, môi trường gia đình.
    2. **Làm sạch & Tiền xử lý (Data Cleaning):** 
        - Loại bỏ dữ liệu trùng lặp (duplicates).
        - Xử lý các giá trị khuyết (missing values) bằng giá trị trung bình/phổ biến nhất.
        - Khử nhiễu, chuẩn hóa outliers bằng phương pháp **Quantile Clipping** nhằm bảo toàn khối lượng dữ liệu mà vẫn loại bỏ biến dạng.
    3. **Huấn luyện Mô hình (Machine Learning):** Sử dụng thuật toán học máy giám sát **Random Forest Regressor** kết hợp phương pháp đánh giá chéo **5-Fold Cross-Validation** để chống overfitting.
    4. **Đánh giá & Trực quan hóa:** Phân tích mức độ quan trọng của đặc trưng (Feature Importance) và vẽ các biểu đồ kiểm thử phần dư.
    5. **Dự báo thực tế:** Giao diện Streamlit tương tác giúp đưa ra dự đoán điểm số tức thì.
    """)

# PAGE 2: Exploratory Data Analysis (EDA)
elif page == "📊 Phân Tích Dữ Liệu (EDA)":
    st.markdown("### 📊 Phân Tích Dữ Liệu & Khám Phá Insights (EDA)")
    st.markdown("Khám phá các yếu tố tác động trực tiếp và gián tiếp đến kết quả học tập của học sinh.")
    
    if os.path.exists("data/processed/StudentPerformance_cleaned.csv"):
        df = pd.read_csv("data/processed/StudentPerformance_cleaned.csv")
    else:
        st.error("❌ Chưa tìm thấy dữ liệu đã làm sạch. Vui lòng chạy main.py để sinh dữ liệu trước!")
        st.stop()
        
    tab1, tab2, tab3 = st.tabs(["📈 Phân Phối Điểm Số", "🔗 Tương Quan & Biến Số", "🏫 So Sánh Môi Trường"])
    
    with tab1:
        st.markdown("#### Phân Phối Điểm Số Thi (Target Variable)")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = plot_score_distribution(df)
            st.pyplot(fig)
        with col2:
            st.info("""
            **Nhận xét về Phân Phối Điểm Số:**
            - Điểm thi có xu hướng phân phối chuẩn (Normal Distribution).
            - Hầu hết học sinh đạt mức điểm trung bình từ **60 đến 75 điểm**.
            - Số lượng học sinh đạt điểm tuyệt đối (> 90) hoặc yếu kém (< 40) chiếm tỷ lệ nhỏ, tạo thành hai đuôi đối xứng đẹp mắt của đồ thị.
            """)
            
    with tab2:
        st.markdown("#### Tác Động Của Thời Gian Học & Điểm Danh")
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_study_vs_score(df)
            st.pyplot(fig)
        with col2:
            fig = plot_attendance_vs_score(df)
            st.pyplot(fig)
            
        st.markdown("""
        > **Insight Đắt Giá:**
        > - **Thời gian tự học (Hours Studied)** và **Tỷ lệ điểm danh trên lớp (Attendance)** có mối quan hệ tuyến tính đồng biến cực kỳ mạnh mẽ với kết quả thi cuối kỳ.
        > - Những học sinh có tỷ lệ đi học trên lớp dưới 70% có xác suất trượt hoặc điểm thi thấp hơn hẳn.
        """)
        
    with tab3:
        st.markdown("#### Ảnh Hưởng Của Môi Trường & Gia Đình")
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_score_by_school_type(df)
            st.pyplot(fig)
        with col2:
            fig = plot_avg_score_by_parental_involvement(df)
            st.pyplot(fig)
            
        col3, col4 = st.columns(2)
        with col3:
            fig = plot_score_by_gender(df)
            st.pyplot(fig)
        with col4:
            fig = plot_motivation_level_distribution(df)
            st.pyplot(fig)

# PAGE 3: Predict Score
elif page == "🔮 Dự Đoán Điểm Thi":
    st.markdown("### 🔮 Nhập Thông Tin Để Dự Đoán Kết Quả Thi")
    st.markdown("Hệ thống sẽ chuyển hóa dữ liệu đầu vào và sử dụng mô hình học máy Random Forest để đưa ra điểm số dự báo.")
    
    if model is None:
        st.error("❌ Hiện tại mô hình chưa được tải lên hệ thống. Vui lòng chạy huấn luyện bằng cách gõ `python main.py` trước!")
        st.stop()
        
    st.subheader("📝 Phiếu Khảo Sát Học Sinh")
    
    # Chia form thành 2 cột cho các thuộc tính cốt lõi
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📌 Yếu tố hành vi & Kết quả học trước đó")
        hours_studied = st.slider("Thời gian tự học / tuần (Hours Studied)", 0, 50, 20)
        attendance = st.slider("Tỷ lệ điểm danh / lớp học % (Attendance)", 0, 100, 85)
        previous_scores = st.slider("Điểm các bài kiểm tra trước (Previous Scores)", 0, 100, 75)
        sleep_hours = st.slider("Số giờ ngủ trung bình / ngày (Sleep Hours)", 0, 12, 7)
        
    with col2:
        st.markdown("##### 📌 Thông tin học sinh & Động lực")
        gender = st.selectbox("Giới tính (Gender)", ["Male", "Female"])
        school_type = st.selectbox("Loại trường học (School Type)", ["Public", "Private"])
        parental_involvement = st.selectbox("Sự quan tâm từ phụ huynh (Parental Involvement)", ["Low", "Medium", "High"], index=1)
        motivation_level = st.selectbox("Mức độ động lực học tập (Motivation Level)", ["Low", "Medium", "High"], index=1)
        internet_access = st.selectbox("Có mạng Internet không? (Internet Access)", ["Yes", "No"])
        
    # Phần Advanced Settings cho 10 thuộc tính nâng cao
    with st.expander("⚙️ Cấu hình Nâng cao (Advanced Settings - Tăng độ chính xác dự đoán lên tối đa)"):
        st.markdown("Hãy điền các yếu tố môi trường và hành vi phụ trợ sau đây để thuật toán Random Forest dự báo chuẩn xác nhất:")
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            access_to_resources = st.selectbox("Truy cập tài nguyên học tập (Access to Resources)", ["Low", "Medium", "High"], index=1)
            extracurricular_activities = st.selectbox("Tham gia hoạt động ngoại khóa (Extracurricular Activities)", ["Yes", "No"], index=1)
            tutoring_sessions = st.slider("Số buổi gia sư bổ sung / tháng (Tutoring Sessions)", 0, 10, 0)
            family_income = st.selectbox("Mức thu nhập gia đình (Family Income)", ["Low", "Medium", "High"], index=1)
            teacher_quality = st.selectbox("Đánh giá chất lượng giáo viên (Teacher Quality)", ["Low", "Medium", "High"], index=1)
            
        with col_adv2:
            peer_influence = st.selectbox("Ảnh hưởng từ bạn bè (Peer Influence)", ["Neutral", "Positive", "Negative"], index=0)
            physical_activity = st.slider("Giờ hoạt động thể thao / tuần (Physical Activity)", 0, 10, 2)
            learning_disabilities = st.selectbox("Gặp khó khăn/khuyết tật học tập (Learning Disabilities)", ["No", "Yes"], index=0)
            parental_education_level = st.selectbox("Trình độ học vấn của cha mẹ (Parental Education Level)", ["High School", "College", "Postgraduate"], index=0)
            distance_from_home = st.selectbox("Khoảng cách từ nhà đến trường (Distance from Home)", ["Near", "Moderate", "Far"], index=0)

    # Đóng gói dữ liệu đầu vào
    input_data = pd.DataFrame({
        'Hours_Studied': [hours_studied],
        'Attendance': [attendance],
        'Parental_Involvement': [parental_involvement],
        'Access_to_Resources': [access_to_resources],
        'Extracurricular_Activities': [extracurricular_activities],
        'Sleep_Hours': [sleep_hours],
        'Previous_Scores': [previous_scores],
        'Motivation_Level': [motivation_level],
        'Internet_Access': [internet_access],
        'Tutoring_Sessions': [tutoring_sessions],
        'Family_Income': [family_income],
        'Teacher_Quality': [teacher_quality],
        'School_Type': [school_type],
        'Peer_Influence': [peer_influence],
        'Physical_Activity': [physical_activity],
        'Learning_Disabilities': [learning_disabilities],
        'Parental_Education_Level': [parental_education_level],
        'Distance_from_Home': [distance_from_home],
        'Gender': [gender]
    })
    
    st.markdown("---")
    
    if st.button("🚀 BẮT ĐẦU DỰ ĐOÁN ĐIỂM SỐ", type="primary"):
        with st.spinner("🤖 Thuật toán AI đang tính toán..."):
            try:
                # Thực thi dự đoán trên hàm đã được sửa lỗi Case Mismatch
                prediction = predict_new_data(model, input_data, label_encoders)
                
                # Hiển thị kết quả dạng hộp đẹp
                predicted_score = prediction[0]
                
                st.markdown("### 🏆 Kết Quả Dự Đoán Điểm Số")
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    # Thiết lập màu sắc theo mức điểm thi
                    if predicted_score >= 80:
                        color_box = "#D1FAE5"
                        text_box = "#065F46"
                        label_box = "Học lực: XUẤT SẮC 🌟"
                    elif predicted_score >= 65:
                        color_box = "#DBEAFE"
                        text_box = "#1E40AF"
                        label_box = "Học lực: KHÁ 👍"
                    elif predicted_score >= 50:
                        color_box = "#FEF3C7"
                        text_box = "#92400E"
                        label_box = "Học lực: TRUNG BÌNH 😐"
                    else:
                        color_box = "#FEE2E2"
                        text_box = "#991B1B"
                        label_box = "Học lực: YẾU KÉM ⚠️"
                        
                    st.markdown(f"""
                    <div style="background-color: {color_box}; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1.5px solid {text_box};">
                        <span style="font-size: 1.1rem; font-weight: 600; color: {text_box}; text-transform: uppercase;">{label_box}</span>
                        <h1 style="font-size: 4rem; margin: 0.5rem 0; color: {text_box}; font-weight: 800;">{predicted_score:.2f}</h1>
                        <span style="font-size: 0.9rem; color: {text_box}; font-weight: 500;">Thang điểm dự kiến 0 - 100</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown("##### 🚀 Khuyến nghị từ hệ thống AI:")
                    # Thanh progress hiển thị trực quan
                    st.progress(min(predicted_score / 100.0, 1.0))
                    
                    if predicted_score >= 80:
                        st.success("Học sinh này đang giữ phong độ học tập xuất sắc. Hãy tiếp tục duy trì phương pháp tự học khoa học hiện tại và phát huy tối đa tiềm năng ở các bài kiểm tra thực tế!")
                    elif predicted_score >= 65:
                        st.info("Học sinh có kết quả học tập khá tốt. Để bứt phá lên mức xuất sắc, gia đình và nhà trường nên khuyến khích tăng nhẹ thời gian tự học (Hours Studied) hoặc tham gia gia sư (Tutoring Sessions) bổ trợ thêm ở các phần kiến thức khó.")
                    elif predicted_score >= 50:
                        st.warning("Học sinh có mức điểm trung bình và đang ở vùng ranh giới. Cần đặc biệt chú ý cải thiện tỷ lệ đi học trên lớp (Attendance) và cải thiện thời gian tự học ở nhà gấp. Cha mẹ nên đồng hành sát sao hơn.")
                    else:
                        st.error("Cảnh báo khẩn cấp! Học sinh này có nguy cơ cao trượt tốt nghiệp hoặc xếp hạng kém. Nhà trường và cha mẹ cần ngay lập tức tổ chức các buổi bổ trợ kiến thức tập trung, giám sát tỷ lệ điểm danh nghiêm ngặt và giải quyết các khó khăn trong việc tiếp cận tài nguyên học tập.")
            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi trong quá trình dự đoán: {e}")

# PAGE 4: Model Performance
elif page == "📈 Đánh Giá Hiệu Năng Mô Hình":
    st.markdown("### 📈 Đánh Giá Hiệu Năng & Độ Chính Xác Của Mô Hình Học Máy")
    st.markdown("Các chỉ số thống kê toán học chứng minh khả năng dự báo đáng tin cậy của thuật toán Random Forest Regressor.")
    
    if model is None:
        st.warning("⚠️ Hiện tại chưa nạp được mô hình. Vui lòng đảm bảo đã chạy thành công file main.py để sinh đầy đủ mô hình!")
        st.stop()
        
    tab_eval1, tab_eval2, tab_eval3 = st.tabs(["🏆 Chỉ số Tổng quan", "📉 Kiểm định phần dư (Errors)", "🎯 Mức độ quan trọng (Features)"])
    
    # Đọc dữ liệu cleaned để thực hiện test nhanh trực quan
    if os.path.exists("data/processed/StudentPerformance_cleaned.csv"):
        df = pd.read_csv("data/processed/StudentPerformance_cleaned.csv")
        X, y, _ = preprocess_data(df)
        
        # Split test set y hệt main.py để trực quan hóa
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
    else:
        st.error("❌ Thiếu file cleaned dữ liệu.")
        st.stop()
        
    with tab_eval1:
        st.markdown("#### So sánh Thực Tế và Dự Đoán trên Test Set")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = plot_actual_vs_predicted(y_test, y_pred)
            st.pyplot(fig)
        with col2:
            st.info("""
            **Giải thích Biểu Đồ Thực tế vs Dự Đoán:**
            - **Đường chéo đứt nét màu đỏ (y=x)** thể hiện trạng thái dự đoán hoàn hảo tuyệt đối (Sai số bằng 0).
            - **Đường thẳng màu xanh dương (Regression Line)** thể hiện xu hướng dự báo thực tế của mô hình. Đường này càng nằm sát đường đứt nét đỏ, mô hình càng chính xác.
            - Mật độ phân bố tập trung dày đặc xung quanh trục chéo chứng tỏ mô hình dự báo rất ổn định trên phần lớn các học sinh.
            """)
            
    with tab_eval2:
        st.markdown("#### Biểu đồ Kiểm Định Residuals (Phần dư)")
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_residuals(y_test, y_pred)
            st.pyplot(fig)
        with col2:
            fig = plot_prediction_errors(y_test, y_pred)
            st.pyplot(fig)
            
        st.markdown("""
        > **Kết luận kiểm định phần dư:**
        > - Biểu đồ phần dư phân bố ngẫu nhiên xung quanh trục ngang 0, chứng tỏ mô hình không gặp hiện tượng **Heteroscedasticity** (Phương sai sai số thay đổi).
        > - Đồ thị sai số phân phối chuẩn quanh mốc 0 khẳng định sai số dự báo phần lớn nằm trong khoảng cực nhỏ (+/- 2 điểm).
        """)
        
    with tab_eval3:
        st.markdown("#### Độ Quan Trọng Của Các Yếu Tố Tới Điểm Thi")
        
        # Load feature importance
        try:
            importance = get_feature_importance(model, model.feature_names_in_)
            fig = plot_feature_importance(importance)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Không nạp được độ quan trọng đặc trưng: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.8rem;'>Dự án Phân Tích & Dự Đoán Điểm Thi Học Sinh | Được phát triển bằng Streamlit + Scikit-Learn | Nhóm 13 Chuyên đề tốt nghiệp 3</p>", unsafe_allow_html=True)