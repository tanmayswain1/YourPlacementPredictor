import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

# Features used for the first stage (Classification)
FEATURES = [
    "CGPA", "Major Projects", "Workshops/Certificatios", "Mini Projects",
    "Skills", "Communication Skill Rating", "Internship", "Hackathon",
    "12th Percentage", "10th Percentage", "backlogs"
]

def train_model(data_path="placement_data.csv", output_path="../model.pkl"):
    print("Loading dataset...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset {data_path} not found. Please provide the dataset.")
        return

    X = df[FEATURES]
    y = df["PlacementStatus"] # 1 for Placed, 0 for Not Placed

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting Classifier
    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    # Save the model
    joblib.dump(model, output_path)
    print(f"Model successfully saved to {output_path}")

if __name__ == "__main__":
    train_model()

