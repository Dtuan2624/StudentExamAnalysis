import src
import pandas as pd
from src import clean_data
from src import visualize


# Example usage
if __name__ == "__main__":
    input_csv = "./data/raw/StudentPerformanceFactors.csv"
    output_csv = "./data/processed/StudentPerformance_cleaned.csv"
    clean_data.clean_dataset(input_csv, output_csv)