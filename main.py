import src
import pandas as pd
from src import clean_data
from src import visualize


# Example usage
if __name__ == "__main__":
    input_csv = "./data/raw/StudentPerformanceFactors.csv"
    output_csv = "./data/processed/StudentPerformance_cleaned.csv"
    clean_data.clean_dataset(input_csv, output_csv)

df = pd.read_csv(output_csv)
visualize.plot_score_distribution(df)
visualize.plot_study_vs_score(df)
visualize.plot_correlation_heatmap(df)
visualize.plot_score_by_gender(df)
visualize.plot_score_by_school_type(df)
visualize.plot_avg_score_by_parental_involvement(df)
visualize.plot_attendance_vs_score(df)
visualize.plot_motivation_level_distribution(df)
visualize.plot_hours_by_learning_disabilities(df)
visualize.plot_pair_plot(df)