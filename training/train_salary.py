import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

# Features used for the second stage (Regression)
# Includes all base features PLUS the PlacementStatus
FEATURES = [
    "CGPA", "Major Projects", "Workshops/Certificatios", "Mini Projects",
    "Skills", "Communication Skill Rating", "Internship", "Hackathon",
    "12th Percentage", "10th Percentage", "backlogs", "PlacementStatus"
]

def train_salary_model(data_path="salary_data.csv", output_path="../model1.pkl"):
    print("Loading dataset...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset {data_path} not found. Please provide the dataset.")
        return

    X = df[FEATURES]
    y = df["Salary_LPA"] # Target: Salary in Lakhs Per Annum

    # Filter out people who were not placed (Salary = 0 or NaN)
    # df = df[df["PlacementStatus"] == 1]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Regressor
    print("Training Gradient Boosting Regressor for Salary...")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model R^2 Score on Test Data: {score:.3f}")

    # Save the model
    joblib.dump(model, output_path)
    print(f"Salary Regressor successfully saved to {output_path}")

if __name__ == "__main__":
    train_salary_model()

