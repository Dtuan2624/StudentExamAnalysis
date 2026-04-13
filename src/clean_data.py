import pandas as pd
import os

def clean_dataset(input_path, output_path):
    """
    input_path: đường dẫn dataset gốc (CSV)
    output_path: đường dẫn lưu dataset đã clean
    """

    # 1. Load dataset
    df = pd.read_csv(input_path)
    print("Original dataset shape:", df.shape)

    # 3. Xử lý missing values
    for col in df.columns:
        if df[col].dtype == 'object':  # text hoặc categorical
            # fill bằng giá trị mode (giá trị xuất hiện nhiều nhất)
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = df[col].str.strip().str.lower()  # chuẩn hóa text
        elif pd.api.types.is_numeric_dtype(df[col]):  # chỉ numeric thật
            # fill bằng mean, bỏ qua các giá trị không convert được
            df[col] = pd.to_numeric(df[col], errors='coerce')  # convert lỗi -> NaN
            df[col] = df[col].fillna(df[col].mean())
        else:
            # các kiểu dữ liệu khác (datetime...) có thể thêm xử lý riêng
            pass
    # 4. Chuyển kiểu dữ liệu nếu cần (ví dụ tuổi là int)
    for col in df.select_dtypes(include='float64').columns:
        df[col] = df[col].astype(float)

    # 5. Xử lý outliers (theo IQR) cho các cột số
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    print("After cleaning:", df.shape)
    # 2. Xóa duplicates
    df = df.drop_duplicates()
    print("After dropping duplicates:", df.shape)

    # 6. Lưu dataset sạch
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Clean dataset saved to:", output_path)

# Example usage
if __name__ == "__main__":
    input_csv = "D:/IHatePython/StudentExamAnalysis/data/raw/StudentPerformanceFactors.csv"
    output_csv = "D:/IHatePython/StudentExamAnalysis/data/processed/StudentPerformance_cleaned.csv"
    clean_dataset(input_csv, output_csv)