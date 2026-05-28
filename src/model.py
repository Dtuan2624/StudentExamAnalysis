import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from pathlib import Path

# Define model output directory
MODEL_DIR = Path(__file__).parent.parent / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_data(df):
    """
    Tiền xử lý dữ liệu: mã hóa categorical, chuẩn bị features và target
    """
    # Sao chép dataframe để không ảnh hưởng gốc
    df_processed = df.copy()

    # Mã hóa các cột categorical
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    # Tách features và target
    X = df_processed.drop('Exam_Score', axis=1)
    y = df_processed['Exam_Score']

    return X, y, label_encoders

def train_model(X, y, label_encoders=None):
    """
    Huấn luyện mô hình Random Forest và lưu model + label_encoders
    """
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Khởi tạo và huấn luyện mô hình
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán trên test set
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    # Lưu mô hình
    model_path = MODEL_DIR / 'student_performance_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")

    # Lưu label encoders (quan trọng cho predict)
    if label_encoders is not None:
        encoders_path = MODEL_DIR / 'label_encoders.pkl'
        joblib.dump(label_encoders, encoders_path)
        print(f"✅ Label encoders saved to: {encoders_path}")

    # Lưu feature names để đồng bộ cột khi predict
    feature_names_path = MODEL_DIR / 'feature_names.pkl'
    joblib.dump(list(X.columns), feature_names_path)
    print(f"✅ Feature names saved to: {feature_names_path}")

    return model, metrics, cv_scores, X_test, y_test, y_pred

def predict_new_data(model, new_data, label_encoders):
    """
    Dự đoán trên dữ liệu mới.
    Tự động sắp xếp cột đúng thứ tự model đã học và xử lý missing features.
    """
    # Sao chép để không ảnh hưởng gốc
    new_data_processed = new_data.copy()

    # Mã hóa categorical
    for col, le in label_encoders.items():
        if col in new_data_processed.columns:
            new_data_processed[col] = new_data_processed[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Đồng bộ cột theo thứ tự model đã học
    if hasattr(model, 'feature_names_in_'):
        expected_cols = list(model.feature_names_in_)
        # Thêm cột thiếu với giá trị 0
        for col in expected_cols:
            if col not in new_data_processed.columns:
                new_data_processed[col] = 0
        # Sắp xếp đúng thứ tự và chỉ giữ cột cần thiết
        new_data_processed = new_data_processed[expected_cols]

    # Dự đoán
    predictions = model.predict(new_data_processed)

    return predictions

def get_feature_importance(model, feature_names):
    """
    Lấy độ quan trọng của các đặc trưng
    """
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, importances))
    return feature_importance_dict