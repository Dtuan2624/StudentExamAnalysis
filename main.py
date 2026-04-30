import src
import pandas as pd
from src import clean_data
from src import visualize
from src import model


# Example usage
if __name__ == "__main__":
    input_csv = "./data/raw/StudentPerformanceFactors.csv"
    output_csv = "./data/processed/StudentPerformance_cleaned.csv"
    clean_data.clean_dataset(input_csv, output_csv)

    df = pd.read_csv(output_csv)

    # Phân tích dữ liệu
    print("=== PHÂN TÍCH DỮ LIỆU ===")
    visualize.plot_score_distribution(df)
    visualize.plot_study_vs_score(df)
    visualize.plot_correlation_heatmap(df)
    visualize.plot_score_by_gender(df)
    visualize.plot_score_by_school_type(df)
    visualize.plot_avg_score_by_parental_involvement(df)
    visualize.plot_attendance_vs_score(df)
    visualize.plot_motivation_level_distribution(df)
    visualize.plot_pair_plot(df)

    print("saved to figure dir")
    # Mô hình hóa và dự đoán
    print("\n=== MÔ HÌNH HÓA VÀ DỰ ĐOÁN ===")
    X, y, label_encoders = model.preprocess_data(df)
    trained_model, metrics, cv_scores, X_test, y_test, y_pred = model.train_model(X, y)

    print("Các chỉ số hiệu suất mô hình:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Trực quan hóa kết quả mô hình
    visualize.plot_actual_vs_predicted(y_test, y_pred)
    visualize.plot_residuals(y_test, y_pred)
    visualize.plot_prediction_errors(y_test, y_pred)
    visualize.plot_model_metrics(metrics)
    visualize.plot_cv_scores(cv_scores)

    # Độ quan trọng của đặc trưng
    feature_importance = model.get_feature_importance(trained_model, X.columns)
    visualize.plot_feature_importance(feature_importance,
                                      title="Feature Importance - Random Forest")

    print("\nĐã hoàn thành phân tích và dự đoán!")