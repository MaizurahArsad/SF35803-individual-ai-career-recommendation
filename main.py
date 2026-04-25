from data.data_loader import load_career_data
from preprocessing.preprocessing import overview_data, preprocess_data
from models.model_training import train_model
from evaluation.evaluation import evaluate_model
from utils.visualization import plot_age_distribution, plot_education_count

# Load
df = load_career_data()

# Overview
overview_data(df)

# EDA
plot_age_distribution(df)
plot_education_count(df)

# Preprocess
X, vectorizer = preprocess_data(df)
y = df['Recommended_Career']

# Train & Evaluate
clf, X_train, X_test, y_train, y_test = train_model(X, y)
evaluate_model(clf, X_test, y_test, class_names=y.unique())
