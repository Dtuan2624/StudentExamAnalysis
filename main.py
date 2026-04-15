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