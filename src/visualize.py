import matplotlib.pyplot as plt
import seaborn as sns

# style đẹp hơn
sns.set(style="whitegrid")

# ================== 1. PHÂN PHỐI ĐIỂM ==================
def plot_score_distribution(df):
    plt.figure(figsize=(8,6))
    
    sns.histplot(df['Exam_Score'], kde=True, bins=20)
    
    plt.title("Phân phối điểm số của học sinh", fontsize=14)
    plt.xlabel("Điểm số (Exam Score)")
    plt.ylabel("Số lượng học sinh")
    
    plt.show()


# ================== 2. THỜI GIAN HỌC VS ĐIỂM ==================
def plot_study_vs_score(df):
    plt.figure(figsize=(8,6))
    
    sns.scatterplot(x=df['Hours_Studied'], y=df['Exam_Score'])
    
    plt.title("Mối quan hệ giữa thời gian học và điểm số", fontsize=14)
    plt.xlabel("Thời gian học (Hours Studied)")
    plt.ylabel("Điểm số (Exam Score)")
    
    plt.show()


# ================== 3. HEATMAP ==================
def plot_correlation_heatmap(df):
    plt.figure(figsize=(12,8))

    # chỉ lấy cột số
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    corr = numeric_df.corr()

    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Ma trận tương quan giữa các thuộc tính")

    plt.show()