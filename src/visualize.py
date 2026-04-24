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

# ================== 1. PHÂN PHỐI ĐIỂM ==================
def plot_score_distribution(df):
    plt.figure(figsize=(8,6))
    
    sns.histplot(df['Exam_Score'], kde=True, bins=20)
    
    plt.title("Phân phối điểm số của học sinh", fontsize=14)
    plt.xlabel("Điểm số (Exam Score)")
    plt.ylabel("Số lượng học sinh")

    plt.show()
    plt.savefig(FIGURE_DIR / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 2. THỜI GIAN HỌC VS ĐIỂM ==================
def plot_study_vs_score(df):
    plt.figure(figsize=(8,6))
    
    sns.scatterplot(x=df['Hours_Studied'], y=df['Exam_Score'])
    
    plt.title("Mối quan hệ giữa thời gian học và điểm số", fontsize=14)
    plt.xlabel("Thời gian học (Hours Studied)")
    plt.ylabel("Điểm số (Exam Score)")

    plt.show()
    plt.savefig(FIGURE_DIR / 'study_vs_score.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 3. HEATMAP ==================
def plot_correlation_heatmap(df):
    plt.figure(figsize=(12,8))

    # chỉ lấy cột số
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    corr = numeric_df.corr()

    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Ma trận tương quan giữa các thuộc tính")

    plt.show()
    plt.savefig(FIGURE_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 4. BOX PLOT: ĐIỂM SỐ THEO GIỚI TÍNH ==================
def plot_score_by_gender(df):
    plt.figure(figsize=(8,6))
    
    sns.boxplot(x='Gender', y='Exam_Score', data=df)
    
    plt.title("Phân phối điểm số theo giới tính", fontsize=14)
    plt.xlabel("Giới tính")
    plt.ylabel("Điểm số (Exam Score)")

    plt.show()
    plt.savefig(FIGURE_DIR / 'score_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 5. BOX PLOT: ĐIỂM SỐ THEO LOẠI TRƯỜNG ==================
def plot_score_by_school_type(df):
    plt.figure(figsize=(8,6))
    
    sns.boxplot(x='School_Type', y='Exam_Score', data=df)
    
    plt.title("Phân phối điểm số theo loại trường", fontsize=14)
    plt.xlabel("Loại trường")
    plt.ylabel("Điểm số (Exam Score)")

    plt.show()
    plt.savefig(FIGURE_DIR / 'score_by_school_type.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 6. BAR PLOT: ĐIỂM TRUNG BÌNH THEO SỰ THAM GIA CỦA PHỤ HUYNH ==================
def plot_avg_score_by_parental_involvement(df):
    plt.figure(figsize=(8,6))
    
    avg_scores = df.groupby('Parental_Involvement')['Exam_Score'].mean().reset_index()
    
    sns.barplot(x='Parental_Involvement', y='Exam_Score', data=avg_scores)
    
    plt.title("Điểm trung bình theo sự tham gia của phụ huynh", fontsize=14)
    plt.xlabel("Sự tham gia của phụ huynh")
    plt.ylabel("Điểm trung bình (Exam Score)")

    plt.show()
    plt.savefig(FIGURE_DIR / 'avg_score_by_parental_involvement.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 7. SCATTER PLOT VỚI REGRESSION: TỶ LỆ THAM GIA VS ĐIỂM ==================
def plot_attendance_vs_score(df):
    plt.figure(figsize=(8,6))
    
    sns.regplot(x='Attendance', y='Exam_Score', data=df, scatter_kws={'alpha':0.5})
    
    plt.title("Mối quan hệ giữa tỷ lệ tham gia và điểm số", fontsize=14)
    plt.xlabel("Tỷ lệ tham gia (%)")
    plt.ylabel("Điểm số (Exam Score)")

    plt.show()
    plt.savefig(FIGURE_DIR / 'attendance_vs_score.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 8. COUNT PLOT: MỨC ĐỘ ĐỘNG VIÊN ==================
def plot_motivation_level_distribution(df):
    plt.figure(figsize=(8,6))
    
    sns.countplot(x='Motivation_Level', data=df, order=['low', 'medium', 'high'])
    
    plt.title("Phân phối mức độ động viên", fontsize=14)
    plt.xlabel("Mức độ động viên")
    plt.ylabel("Số lượng học sinh")

    plt.show()
    plt.savefig(FIGURE_DIR / 'motivation_level_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 9. VIOLIN PLOT: THỜI GIAN HỌC THEO KHuyết TẬT HỌC TẬP ==================
def plot_hours_by_learning_disabilities(df):
    plt.figure(figsize=(8,6))
    
    sns.violinplot(x='Learning_Disabilities', y='Hours_Studied', data=df)
    
    plt.title("Phân phối thời gian học theo Learning Disability", fontsize=14)
    plt.xlabel("Learning Disability")
    plt.ylabel("Thời gian học (giờ)")

    plt.show()
    plt.savefig(FIGURE_DIR / 'hours_by_learning_disabilities.png', dpi=300, bbox_inches='tight')
    plt.close()