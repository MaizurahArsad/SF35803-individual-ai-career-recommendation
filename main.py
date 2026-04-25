from config import RAW_DATA_PATH, OUTPUT_FIGURES_PATH, RANDOM_STATE, TEST_SIZE, N_ESTIMATORS, NUMERIC_COLUMNS, TEXT_COLUMNS, TARGET_COLUMN
from data.data_loader import load_career_data
from preprocessing.preprocessing import overview_data, preprocess_data
from models.model_training import train_model
from evaluation.evaluation import evaluate_model
from utils.visualization import plot_age_distribution, plot_education_count
import os

# Ensure output folder exists
os.makedirs(OUTPUT_FIGURES_PATH, exist_ok=True)

# Step 1: Load dataset
df = load_career_data(RAW_DATA_PATH)

# Step 2: Overview
overview_data(df)

# Step 3: EDA
plot_age_distribution(df, save_path=OUTPUT_FIGURES_PATH)
plot_education_count(df, save_path=OUTPUT_FIGURES_PATH)

# Step 4: Preprocess
X, vectorizer = preprocess_data(df, numeric_cols=NUMERIC_COLUMNS, text_cols=TEXT_COLUMNS)
y = df[TARGET_COLUMN]

# Step 5: Train & Evaluate
clf, X_train, X_test, y_train, y_test = train_model(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS)
evaluate_model(clf, X_test, y_test, class_names=y.unique())