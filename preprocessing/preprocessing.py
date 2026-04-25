from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

def overview_data(df):
    """Print dataset shape, info, and descriptive stats."""
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    df.info()
    print("\nDataset Description:")
    print(df.describe())

def preprocess_data(df, numeric_cols, text_cols):
    """
    Preprocess dataset and return feature matrix X.
    numeric_cols: list of numeric column names
    text_cols: list of text column names (Skills + Interests)
    """
    features = df[numeric_cols + text_cols].copy()
    
    # Encode Education
    features['Education_enc'] = LabelEncoder().fit_transform(features['Education'])
    
    # Combine text features
    features['Text_Features'] = features['Skills'].fillna('') + " " + features['Interests'].fillna('')
    
    vectorizer = CountVectorizer(tokenizer=lambda x: [item.strip() for item in x.split(";") if item.strip()], lowercase=True)
    text_features = vectorizer.fit_transform(features['Text_Features'])
    
    # Numeric features
    numeric_features = features[numeric_cols + ['Education_enc']].values
    
    # Combine numeric + text features
    X = hstack([numeric_features, text_features])
    
    return X, vectorizer