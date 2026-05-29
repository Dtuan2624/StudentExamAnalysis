import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Cấu hình thẩm mỹ cao cấp cho Matplotlib và Seaborn
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#E5E7EB',
    'axes.linewidth': 1.2,
    'grid.color': '#F3F4F6',
    'grid.alpha': 0.8,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FAFAFA',
    'text.color': '#1F2937',
    'axes.labelcolor': '#4B5563',
    'xtick.color': '#4B5563',
    'ytick.color': '#4B5563'
})

# Khai báo thư mục lưu biểu đồ
FIGURE_DIR = Path(__file__).parent.parent / 'figure'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR = FIGURE_DIR / 'eda'
EVAL_DIR = FIGURE_DIR / 'evaluation'
EDA_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Bảng màu thương hiệu cao cấp
PALETTE_PRIMARY = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']

# ================== 1. PHÂN PHỐI ĐIỂM ==================
def plot_score_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sns.histplot(df['Exam_Score'], kde=True, bins=20, color='#3B82F6', ax=ax, edgecolor='white', alpha=0.8)
    
    ax.set_title("Phân Phối Điểm Số Của Học Sinh", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Điểm số (Exam Score)", fontsize=11, fontweight='medium')
    ax.set_ylabel("Số lượng học sinh", fontsize=11, fontweight='medium')
    
    plt.savefig(EDA_DIR / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 2. THỜI GIAN HỌC VS ĐIỂM ==================
def plot_study_vs_score(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sns.scatterplot(x='Hours_Studied', y='Exam_Score', data=df, color='#3B82F6', alpha=0.6, edgecolor='white', s=50, ax=ax)
    sns.regplot(x='Hours_Studied', y='Exam_Score', data=df, scatter=False, color='#EF4444', ax=ax, line_kws={'linewidth': 2})
    
    ax.set_title("Mối Quan Hệ Giữa Thời Gian Học Và Điểm Số", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Thời gian học (Hours Studied)", fontsize=11, fontweight='medium')
    ax.set_ylabel("Điểm số (Exam Score)", fontsize=11, fontweight='medium')
    
    plt.savefig(EDA_DIR / 'study_vs_score.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 3. HEATMAP ==================
def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 9))
    
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", 
                linewidths=0.5, square=True, cbar_kws={"shrink": .8}, ax=ax)
    
    ax.set_title("Ma Trận Tương Quan Giữa Các Thuộc Tính Số", fontsize=16, fontweight='bold', pad=20, color='#111827')
    
    plt.savefig(EDA_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 4. BOX PLOT: ĐIỂM SỐ THEO GIỚI TÍNH ==================
def plot_score_by_gender(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.boxplot(x='Gender', y='Exam_Score', data=df, palette=['#3B82F6', '#EC4899'], ax=ax, width=0.5, linewidth=1.5)
    
    ax.set_title("Phân Phối Điểm Số Theo Giới Tính", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Giới tính", fontsize=11, fontweight='medium')
    ax.set_ylabel("Điểm số (Exam Score)", fontsize=11, fontweight='medium')
    
    plt.savefig(EDA_DIR / 'score_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 5. BOX PLOT: ĐIỂM SỐ THEO LOẠI TRƯỜNG ==================
def plot_score_by_school_type(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.boxplot(x='School_Type', y='Exam_Score', data=df, palette=['#10B981', '#F59E0B'], ax=ax, width=0.5, linewidth=1.5)
    
    ax.set_title("Phân Phối Điểm Số Theo Loại Trường", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Loại trường", fontsize=11, fontweight='medium')
    ax.set_ylabel("Điểm số (Exam Score)", fontsize=11, fontweight='medium')
    
    plt.savefig(EDA_DIR / 'score_by_school.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 6. BAR PLOT: ĐIỂM TRUNG BÌNH THEO SỰ THAM GIA CỦA PHỤ HUYNH ==================
def plot_avg_score_by_parental_involvement(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Chuẩn hóa để tránh lỗi key viết thường
    df_temp = df.copy()
    if 'parental_involvement' in df_temp.columns:
        df_temp['Parental_Involvement'] = df_temp['parental_involvement']
    
    # Đảm bảo thứ tự hiển thị hợp lý
    order_list = ['low', 'medium', 'high']
    if df_temp['Parental_Involvement'].iloc[0].istitle():
        order_list = ['Low', 'Medium', 'High']
        
    avg_scores = df_temp.groupby('Parental_Involvement')['Exam_Score'].mean().reindex(order_list).reset_index()
    
    sns.barplot(x='Parental_Involvement', y='Exam_Score', data=avg_scores, palette='Blues', ax=ax, edgecolor='#374151', linewidth=1)
    
    # Thêm số liệu cụ thể trên cột
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() - 5),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
                    
    ax.set_title("Điểm Trung Bình Theo Sự Tham Gia Của Phụ Huynh", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Sự tham gia của phụ huynh", fontsize=11, fontweight='medium')
    ax.set_ylabel("Điểm trung bình (Exam Score)", fontsize=11, fontweight='medium')
    ax.set_ylim(0, 100)
    
    plt.savefig(EDA_DIR / 'avg_score_by_parental_involment.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 7. SCATTER PLOT VỚI REGRESSION: TỶ LỆ THAM GIA VS ĐIỂM ==================
def plot_attendance_vs_score(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sns.regplot(x='Attendance', y='Exam_Score', data=df, scatter_kws={'alpha':0.4, 'color': '#10B981'}, 
                line_kws={'color': '#EF4444', 'linewidth': 2.5}, ax=ax)
    
    ax.set_title("Mối Quan Hệ Giữa Tỷ Lệ Tham Gia Học Lớp Và Điểm Số", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Tỷ lệ tham gia (%)", fontsize=11, fontweight='medium')
    ax.set_ylabel("Điểm số (Exam Score)", fontsize=11, fontweight='medium')
    
    plt.savefig(EDA_DIR / 'attendance_vs_score.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 8. COUNT PLOT: MỨC ĐỘ ĐỘNG VIÊN ==================
def plot_motivation_level_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Chuẩn hóa để tránh lỗi key viết thường
    df_temp = df.copy()
    if 'motivation_level' in df_temp.columns:
        df_temp['Motivation_Level'] = df_temp['motivation_level']
        
    order_list = ['low', 'medium', 'high']
    if df_temp['Motivation_Level'].iloc[0].istitle():
        order_list = ['Low', 'Medium', 'High']
        
    sns.countplot(x='Motivation_Level', data=df_temp, order=order_list, palette='viridis', ax=ax, edgecolor='#374151', linewidth=1)
    
    # Hiển thị số lượng trên cột
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() + 50),
                    ha='center', va='bottom', color='#374151', fontweight='bold')
                    
    ax.set_title("Phân Phối Mức Độ Động Lực Học Tập", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Mức độ động lực", fontsize=11, fontweight='medium')
    ax.set_ylabel("Số lượng học sinh", fontsize=11, fontweight='medium')
    
    plt.savefig(EDA_DIR / 'motivation_level_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 10. PAIR PLOT ==================
def plot_pair_plot(df):
    # Chọn một số cột quan trọng để vẽ pair plot
    selected_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Exam_Score']
    
    # pairplot trả về PairGrid chứ không phải figure đơn
    g = sns.pairplot(df[selected_cols], diag_kind='kde', plot_kws={'alpha': 0.5, 'color': '#3B82F6'})
    g.fig.suptitle("Biểu Đồ Cặp Tương Quan (Pair Plot) Các Biến Quan Trọng", y=1.02, fontsize=16, fontweight='bold', color='#111827')
    
    g.savefig(EDA_DIR / 'pair_plot.png', dpi=300, bbox_inches='tight')
    fig = g.fig
    plt.close(fig)
    return fig

# ================== 11. ACTUAL VS PREDICTED ==================
def plot_actual_vs_predicted(y_actual, y_predicted, model_name="Random Forest"):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    # Tính toán các metrics đánh giá
    r2 = r2_score(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mape = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100 if np.all(y_actual != 0) else np.nan
    
    residuals = y_actual - y_predicted
    bias = np.mean(residuals)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot biểu thị giá trị thực tế vs dự đoán
    scatter = ax.scatter(y_actual, y_predicted, alpha=0.6, c=residuals, cmap='coolwarm', edgecolors='k', linewidths=0.5, s=55)
    
    # Đường chéo hoàn hảo (y = x)
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Đường hoàn hảo (y = x)')
    
    # Đường hồi quy thực tế dự đoán
    sns.regplot(x=y_actual, y=y_predicted, scatter=False, color='#2563EB', ax=ax, line_kws={'linestyle': '-'}, label='Regression Line')
    
    fig.colorbar(scatter, label='Sai số (Phần dư: Actual - Predicted)')
    
    # Hộp chứa thông số hiệu năng
    metrics_text = (
        f"Mô hình: {model_name}\n"
        f"R² Score     = {r2:.4f}\n"
        f"MAE          = {mae:.2f}\n"
        f"RMSE         = {rmse:.2f}\n"
        f"MAPE         = {mape:.2f}%\n"
        f"Độ lệch (Bias) = {bias:.2f}\n"
        f"Tổng số mẫu  = {len(y_actual)}"
    )
    
    ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes, fontsize=11, fontweight='medium',
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="#F9FAFB", alpha=0.95, edgecolor='#D1D5DB'))
             
    ax.set_title("So Sánh Điểm Thi Thực Tế vs Điểm Số Dự Đoán", fontsize=16, fontweight='bold', pad=20, color='#111827')
    ax.set_xlabel("Điểm Thực Tế (Actual Exam Score)", fontsize=12, fontweight='medium')
    ax.set_ylabel("Điểm Dự Đoán (Predicted Exam Score)", fontsize=12, fontweight='medium')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#D1D5DB')
    
    plt.savefig(EVAL_DIR / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 12. RESIDUALS PLOT ==================
def plot_residuals(y_actual, y_predicted):
    residuals = y_actual - y_predicted
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.scatter(y_predicted, residuals, alpha=0.6, color='#8B5CF6', edgecolors='white', s=50)
    ax.axhline(y=0, color='#EF4444', linestyle='--', lw=2.5)
    
    ax.set_title("Biểu Đồ Phần Dư (Residuals Plot)", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Giá trị Dự đoán (Predicted Values)", fontsize=11, fontweight='medium')
    ax.set_ylabel("Phần dư (Residuals: Actual - Predicted)", fontsize=11, fontweight='medium')
    
    plt.savefig(EVAL_DIR / 'residuals.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 13. PREDICTION ERRORS DISTRIBUTION ==================
def plot_prediction_errors(y_actual, y_predicted):
    errors = y_actual - y_predicted
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sns.histplot(errors, kde=True, bins=20, color='#EC4899', edgecolor='white', alpha=0.8, ax=ax)
    ax.axvline(x=0, color='#EF4444', linestyle='--', lw=2, label='Sai số = 0')
    ax.legend(frameon=True, facecolor='white')
    
    ax.set_title("Biểu Đồ Phân Phối Sai Số Dự Đoán", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Giá trị Sai Số", fontsize=11, fontweight='medium')
    ax.set_ylabel("Tần suất xuất hiện", fontsize=11, fontweight='medium')
    
    plt.savefig(EVAL_DIR / 'prediction_eror_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 14. FEATURE IMPORTANCE ==================
def plot_feature_importance(feature_importance, title="Feature Importance - Random Forest"):
    if isinstance(feature_importance, dict):
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
    else:
        feature_names, importances = feature_importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sử dụng bảng màu gradient để thể hiện mức độ quan trọng
    sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df,
        palette='viridis',
        ax=ax,
        edgecolor='#374151',
        linewidth=0.5
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#111827')
    ax.set_xlabel('Điểm số Độ quan trọng (Importance Score)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Các Đặc Trưng (Features)', fontsize=12, fontweight='medium')
    
    plt.tight_layout()
    plt.savefig(EVAL_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 15. MODEL PERFORMANCE METRICS ==================
def plot_model_metrics(metrics_dict):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    
    bars = sns.barplot(x=metrics_names, y=metrics_values, palette='husl', ax=ax, edgecolor='#374151', linewidth=1)
    
    # Hiển thị chính xác giá trị số trên từng cột
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # offset 3 points vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
                    
    ax.set_title("Các Chỉ Số Hiệu Suất Mô Hình", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_ylabel("Giá trị", fontsize=11, fontweight='medium')
    
    # Giới hạn trục Y linh hoạt
    max_val = max(metrics_values)
    ax.set_ylim(0, max_val * 1.15)
    
    plt.savefig(EVAL_DIR / 'model_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

# ================== 16. CROSS-VALIDATION SCORES ==================
def plot_cv_scores(cv_scores):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.plot(cv_scores, marker='o', linestyle='-', linewidth=2.5, markersize=8, color='#3B82F6', label='R² Score từng Fold')
    ax.axhline(y=cv_scores.mean(), color='#EF4444', linestyle='--', lw=2.5, 
                label=f'R² Trung bình = {cv_scores.mean():.4f}')
                
    ax.fill_between(range(len(cv_scores)), cv_scores.min(), cv_scores.max(), alpha=0.15, color='#3B82F6')
    
    ax.set_title("Kết Quả Đánh Giá Chéo (Cross-Validation Scores)", fontsize=15, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel("Lần Lặp (Fold / Iteration)", fontsize=11, fontweight='medium')
    ax.set_ylabel("Hệ số Xác định R²", fontsize=11, fontweight='medium')
    ax.set_xticks(range(len(cv_scores)))
    ax.set_xticklabels([f"Fold {i+1}" for i in range(len(cv_scores))])
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#D1D5DB')
    
    plt.savefig(EVAL_DIR / 'cv_scores.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig