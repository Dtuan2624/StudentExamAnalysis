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
        plot_score_by_gender, plot_score_by_school_type, plot_actual_vs_predicted,
        plot_feature_importance
    )
except ImportError:
    st.warning("⚠️ Chưa tìm thấy module src. Đang chạy ở chế độ demo.")

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Student Exam Performance",
    page_icon="📚",
    layout="wide"
)

st.title("📊 Student Exam Performance Analysis & Prediction")
st.markdown("### Dự đoán điểm thi học sinh sử dụng Machine Learning")

# ====================== SIDEBAR ======================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Chọn trang:",
                        ["🏠 Overview", "📊 Exploratory Data Analysis", "🔮 Predict Score", "📈 Model Performance"])


# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model_path = Path("models/student_performance_model.pkl")
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            st.success("✅ Model loaded successfully!")
            return model
        except:
            st.error("❌ Không load được model")
            return None
    else:
        st.warning("⚠️ Model chưa được train. Vui lòng chạy training trước.")
        return None


model = load_model()

# ====================== MAIN PAGES ======================
if page == "🏠 Overview":
    st.header("Tổng quan về Project")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mô hình", "Random Forest Regressor")
    with col2:
        st.metric("Thuật toán", "Supervised Learning")
    with col3:
        st.metric("Mục tiêu", "Dự đoán Exam Score")

    st.info("""
    **Project Features:**
    - Data Cleaning & Preprocessing
    - Exploratory Data Analysis (EDA)
    - Random Forest Model
    - Feature Importance Analysis
    - Interactive Prediction
    """)

elif page == "📊 Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    # Tải dữ liệu
    if os.path.exists("data/processed/StudentPerformance_cleaned.csv"):
        df = pd.read_csv("data/processed/StudentPerformance_cleaned.csv")
    else:
        st.error("Chưa có dữ liệu cleaned. Vui lòng chạy clean_data.py trước.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Distribution", "Relationships", "Comparisons"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            plot_score_distribution(df)
            st.image("figure/eda/score_distribution.png")
    with tab2:
        plot_study_vs_score(df)
        st.image("figure/eda/study_vs_score.png")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            plot_score_by_gender(df)
            st.image("figure/eda/score_by_gender.png")
        with col2:
            plot_score_by_school_type(df)
            st.image("figure/eda/score_by_school.png")

elif page == "🔮 Predict Score":
    st.header("🔮 Dự đoán điểm thi")

    if model is None:
        st.error("Vui lòng train model trước khi sử dụng tính năng dự đoán!")
        st.stop()

    st.subheader("Nhập thông tin học sinh")

    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.slider("Hours Studied", 0, 50, 20)
        attendance = st.slider("Attendance (%)", 0, 100, 85)
        previous_scores = st.slider("Previous Scores", 0, 100, 75)
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        school_type = st.selectbox("School Type", ["Public", "Private"])
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])

    # Tạo DataFrame
    input_data = pd.DataFrame({
        'Hours_Studied': [hours_studied],
        'Attendance': [attendance],
        'Previous_Scores': [previous_scores],
        'Sleep_Hours': [sleep_hours],
        'Gender': [gender],
        'School_Type': [school_type],
        'Parental_Involvement': [parental_involvement],
        'Motivation_Level': [motivation_level],
        'Internet_Access': [internet_access],
        # Thêm các cột còn lại nếu cần (có thể để default)
    })

    if st.button("🚀 Predict Score", type="primary"):
        try:
            # Load label encoders (nếu có)
            encoders_path = Path("models/label_encoders.pkl")
            label_encoders = joblib.load(encoders_path) if encoders_path.exists() else {}

            prediction = predict_new_data(model, input_data, label_encoders)

            st.success(f"### Điểm dự đoán: **{prediction[0]:.2f}**")

            # Gauge chart đơn giản
            st.progress(min(prediction[0] / 100, 1.0))

        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

elif page == "📈 Model Performance":
    st.header("Model Performance")

    if model:
        st.success("Model đã được huấn luyện")

        # Feature Importance
        if 'feature_names' in locals() or True:
            st.subheader("Feature Importance")
            # Giả sử hoặc load feature names
            try:
                importance = get_feature_importance(model, model.feature_names_in_)
                plot_feature_importance(importance)
                st.image("figure/evaluation/feature_importance.png")
            except:
                st.info("Feature importance sẽ hiển thị sau khi train model.")
    else:
        st.warning("Model chưa được train")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Student Performance Prediction\nBuilt with Streamlit + Scikit-learn")