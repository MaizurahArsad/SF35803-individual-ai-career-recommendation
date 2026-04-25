# config.py

# File paths
RAW_DATA_PATH = "data/raw/ai.career_recommendation.csv"
OUTPUT_FIGURES_PATH = "outputs/figures/"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# Feature settings
TEXT_COLUMNS = ["Skills", "Interests"]
NUMERIC_COLUMNS = ["Age", "Education"]
TARGET_COLUMN = "Recommended_Career"
