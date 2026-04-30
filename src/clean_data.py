import pandas as pd
import os

def clean_dataset(input_path, output_path):

    # 1. Load
    df = pd.read_csv(input_path)
    print(df.head())
    print("Original:", df.shape)
    # 2. Drop duplicate trước
    df = df.drop_duplicates()
    print("After drop duplicates:", df.shape)

    # 3. Handle missing
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.lower()
            df[col] = df[col].fillna(df[col].mode()[0])
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Xử lý outlier (KHÔNG drop toàn bộ)
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # 👉 clip thay vì drop
        df[col] = df[col].clip(lower, upper)

    # 5. Fill lại missing numeric (sau khi xử lý outlier)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    print("After cleaning:", df.shape)

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Saved to:", output_path)