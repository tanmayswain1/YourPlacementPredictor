import pandas as pd
import numpy as np

def preprocess_data(input_file="raw_student_data.csv"):
    print(f"Loading raw data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please provide the raw dataset.")
        return

    print("Starting preprocessing pipeline...")

    # 1. Handle Missing Values
    print("Handling missing values...")
    # Fill missing numeric values with the median to avoid outliers skewing data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill missing categorical values with the mode (most frequent)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Encode Categorical Variables
    print("Encoding categorical features...")
    # Map text responses to numeric vectors for Gradient Boosting
    binary_mapping = {"Yes": 1, "No": 0, "High": 2, "Medium": 1, "Low": 0}
    df = df.replace(binary_mapping)
    
    # Standardize column names if needed
    # df.rename(columns={"old_name": "new_name"}, inplace=True)

    # 3. Feature Selection
    # Keep only the features that actually contribute to the model
    target_features = [
        "CGPA", "Major Projects", "Workshops/Certificatios", "Mini Projects",
        "Skills", "Communication Skill Rating", "Internship", "Hackathon",
        "12th Percentage", "10th Percentage", "backlogs", 
        "PlacementStatus", "Salary_LPA"
    ]
    
    existing_features = [col for col in target_features if col in df.columns]
    df = df[existing_features]

    # 4. Generate Stage 1 Data (Classification: Placed vs Not Placed)
    print("Generating placement_data.csv for Stage 1 Model...")
    placement_df = df.drop(columns=["Salary_LPA"], errors="ignore")
    placement_df.to_csv("placement_data.csv", index=False)

    # 5. Generate Stage 2 Data (Regression: Salary for Placed Students only)
    if "PlacementStatus" in df.columns and "Salary_LPA" in df.columns:
        print("Generating salary_data.csv for Stage 2 Model...")
        # We only train salary regression on students who actually got placed
        salary_df = df[df["PlacementStatus"] == 1]
        salary_df.to_csv("salary_data.csv", index=False)

    print("Preprocessing complete! Data is ready for train_placement.py and train_salary.py.")

if __name__ == "__main__":
    preprocess_data()

