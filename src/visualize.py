import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# style đẹp hơn
sns.set(style="whitegrid")

# Define figure output directory
FIGURE_DIR = Path(__file__).parent.parent / 'figure'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR = FIGURE_DIR / 'eda'
EVAL_DIR = FIGURE_DIR / 'evaluation'
EDA_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ================== 1. PHÂN PHỐI ĐIỂM ==================
def plot_score_distribution(df):
    plt.figure(figsize=(8,6))
    
    sns.histplot(df['Exam_Score'], kde=True, bins=20)
    
    plt.title("Phân phối điểm số của học sinh", fontsize=14)
    plt.xlabel("Điểm số (Exam Score)")
    plt.ylabel("Số lượng học sinh")
    
    plt.savefig(EDA_DIR / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 2. THỜI GIAN HỌC VS ĐIỂM ==================
def plot_study_vs_score(df):
    plt.figure(figsize=(8,6))
    
    sns.scatterplot(x=df['Hours_Studied'], y=df['Exam_Score'])
    
    plt.title("Mối quan hệ giữa thời gian học và điểm số", fontsize=14)
    plt.xlabel("Thời gian học (Hours Studied)")
    plt.ylabel("Điểm số (Exam Score)")
    
    plt.savefig(EDA_DIR / 'study_vs_score.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 3. HEATMAP ==================
def plot_correlation_heatmap(df):
    plt.figure(figsize=(12,8))

    # chỉ lấy cột số
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    corr = numeric_df.corr()

    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Ma trận tương quan giữa các thuộc tính")

    plt.savefig(EDA_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 4. BOX PLOT: ĐIỂM SỐ THEO GIỚI TÍNH ==================
def plot_score_by_gender(df):
    plt.figure(figsize=(8,6))
    
    sns.boxplot(x='Gender', y='Exam_Score', data=df)
    
    plt.title("Phân phối điểm số theo giới tính", fontsize=14)
    plt.xlabel("Giới tính")
    plt.ylabel("Điểm số (Exam Score)")
    
    plt.savefig(EDA_DIR / 'score_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 5. BOX PLOT: ĐIỂM SỐ THEO LOẠI TRƯỜNG ==================
def plot_score_by_school_type(df):
    plt.figure(figsize=(8,6))
    
    sns.boxplot(x='School_Type', y='Exam_Score', data=df)
    
    plt.title("Phân phối điểm số theo loại trường", fontsize=14)
    plt.xlabel("Loại trường")
    plt.ylabel("Điểm số (Exam Score)")
    
    plt.savefig(EDA_DIR / 'score_by_school.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 6. BAR PLOT: ĐIỂM TRUNG BÌNH THEO SỰ THAM GIA CỦA PHỤ HUYNH ==================
def plot_avg_score_by_parental_involvement(df):
    plt.figure(figsize=(8,6))
    
    avg_scores = df.groupby('Parental_Involvement')['Exam_Score'].mean().reset_index()
    
    sns.barplot(x='Parental_Involvement', y='Exam_Score', data=avg_scores)
    
    plt.title("Điểm trung bình theo sự tham gia của phụ huynh", fontsize=14)
    plt.xlabel("Sự tham gia của phụ huynh")
    plt.ylabel("Điểm trung bình (Exam Score)")
    
    plt.savefig(EDA_DIR / 'avg_score_by_parental_involment.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 7. SCATTER PLOT VỚI REGRESSION: TỶ LỆ THAM GIA VS ĐIỂM ==================
def plot_attendance_vs_score(df):
    plt.figure(figsize=(8,6))
    
    sns.regplot(x='Attendance', y='Exam_Score', data=df, scatter_kws={'alpha':0.5})
    
    plt.title("Mối quan hệ giữa tỷ lệ tham gia và điểm số", fontsize=14)
    plt.xlabel("Tỷ lệ tham gia (%)")
    plt.ylabel("Điểm số (Exam Score)")
    
    plt.savefig(EDA_DIR / 'attendance_vs_score.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 8. COUNT PLOT: MỨC ĐỘ ĐỘNG VIÊN ==================
def plot_motivation_level_distribution(df):
    plt.figure(figsize=(8,6))
    
    sns.countplot(x='Motivation_Level', data=df, order=['Low', 'Medium', 'High'])
    
    plt.title("Phân phối mức độ động viên", fontsize=14)
    plt.xlabel("Mức độ động viên")
    plt.ylabel("Số lượng học sinh")
    
    plt.savefig(EDA_DIR / 'motivation_level_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 10. PAIR PLOT ==================
def plot_pair_plot(df):
    plt.figure(figsize=(12,10))
    
    # Chọn một số cột quan trọng để vẽ pair plot
    selected_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Exam_Score']
    g = sns.pairplot(df[selected_cols], diag_kind='kde', plot_kws={'alpha': 0.6})
    g.fig.suptitle("Pair Plot của Các Biến Quan Trọng", y=1.02, fontsize=14)

    g.fig.savefig(EDA_DIR / 'pair_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 11. ACTUAL VS PREDICTED ==================
def plot_actual_vs_predicted(y_actual, y_predicted):
    plt.figure(figsize=(8,6))
    
    plt.scatter(y_actual, y_predicted, alpha=0.6, edgecolors='k')
    
    # Vẽ đường hoàn hảo
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Hoàn hảo')
    
    plt.title("Giá trị Thực tế vs Dự đoán", fontsize=14)
    plt.xlabel("Giá trị Thực tế")
    plt.ylabel("Giá trị Dự đoán")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(EVAL_DIR / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 12. RESIDUALS PLOT ==================
def plot_residuals(y_actual, y_predicted):
    residuals = y_actual - y_predicted
    plt.figure(figsize=(8,6))
    
    plt.scatter(y_predicted, residuals, alpha=0.6, edgecolors='k')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    
    plt.title("Biểu đồ Phần dư", fontsize=14)
    plt.xlabel("Giá trị Dự đoán")
    plt.ylabel("Phần dư (Residuals)")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(EVAL_DIR / 'residuals.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 13. PREDICTION ERRORS DISTRIBUTION ==================
def plot_prediction_errors(y_actual, y_predicted):
    errors = y_actual - y_predicted
    plt.figure(figsize=(8,6))
    
    sns.histplot(errors, kde=True, bins=20)
    
    plt.title("Phân phối Lỗi Dự đoán", fontsize=14)
    plt.xlabel("Lỗi Dự đoán")
    plt.ylabel("Tần số")
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Lỗi = 0')
    plt.legend()
    
    plt.savefig(EVAL_DIR / 'prediction_eror_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 14. FEATURE IMPORTANCE ==================
def plot_feature_importance(feature_importance, title="Feature Importance"):
    """
    Vẽ biểu đồ độ quan trọng của các đặc trưng.
    Hỗ trợ cả dictionary và tuple (feature_names, importances)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Nếu truyền vào là dictionary
    if isinstance(feature_importance, dict):
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
    else:
        # Nếu truyền vào là 2 biến riêng
        feature_names, importances = feature_importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

    # Sắp xếp theo độ quan trọng giảm dần
    importance_df = importance_df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df,
        palette='viridis'
    )

    plt.title(title, fontsize=16)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()

    plt.savefig(EVAL_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
# ================== 15. MODEL PERFORMANCE METRICS ==================
def plot_model_metrics(metrics_dict):
    """
    Vẽ biểu đồ các chỉ số hiệu suất mô hình
    metrics_dict: dict với các chỉ số như {'MAE': 0.5, 'MSE': 0.3, 'R2': 0.85, ...}
    """
    plt.figure(figsize=(10, 6))
    
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    
    bars = sns.barplot(x=metrics_names, y=metrics_values, hue=metrics_names, palette='husl', legend=False)
    
    # Hiển thị giá trị trên mỗi cột
    for bar in bars.patches:
        height = bar.get_height()
        bars.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    plt.title("Các Chỉ số Hiệu suất Mô hình", fontsize=14)
    plt.ylabel("Giá trị")
    plt.ylim(0, 1.5)
    
    plt.savefig(EVAL_DIR / 'model_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 16. CROSS-VALIDATION SCORES ==================
def plot_cv_scores(cv_scores):
    """
    Vẽ biểu đồ điểm Cross-Validation
    cv_scores: array các điểm từ cross-validation
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(cv_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', lw=2, 
                label=f'Trung bình = {cv_scores.mean():.3f}')
    plt.fill_between(range(len(cv_scores)), cv_scores.min(), cv_scores.max(), alpha=0.2)
    
    plt.title("Điểm Cross-Validation", fontsize=14)
    plt.xlabel("Fold")
    plt.ylabel("Điểm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(EVAL_DIR / 'cv_scores.png', dpi=300, bbox_inches='tight')
    plt.close()