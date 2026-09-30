# Model Training Scripts

This folder contains the scripts used to train the machine learning models for the Placement Prediction System.

Since the original datasets (`placement_data.csv` and `salary_data.csv`) contain student records, they are **not included** in this public GitHub repository for privacy reasons. 

However, these scripts are provided so collaborators can understand the exact architecture, algorithms, and features used to generate `model.pkl` and `model1.pkl`.

## Scripts

1. **`train_placement.py`**: Trains the Stage 1 Classifier (`model.pkl`) using `GradientBoostingClassifier` to predict if a student is likely to be placed (1) or not (0).
2. **`train_salary.py`**: Trains the Stage 2 Regressor (`model1.pkl`) using `GradientBoostingRegressor` to predict the expected salary package (in LPA) for placed students.

## How to use
If you have access to the original CSV datasets:
1. Place them in this directory.
2. Run `python train_placement.py` or `python train_salary.py`.
3. The scripts will automatically train the models and update the `.pkl` files in the main directory.

