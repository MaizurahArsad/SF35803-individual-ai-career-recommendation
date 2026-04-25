from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

def overview_data(df):
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    df.info()
    print("\nDataset Description:")
    print(df.describe())

def preprocess_data(df, numeric_cols, text_cols):
    # Copy only numeric + text columns
    features = df[numeric_cols + text_cols].copy()

    # Encode categorical numeric-like columns
    if 'Education' in numeric_cols:
        features['Education_enc'] = LabelEncoder().fit_transform(features['Education'])
        numeric_cols = [c for c in numeric_cols if c != 'Education']
        numeric_cols.append('Education_enc')

    # Combine text columns
    features['Text_Features'] = features[text_cols[0]].fillna('')
    for col in text_cols[1:]:
        features['Text_Features'] += " " + features[col].fillna('')

    # Vectorize text features
    vectorizer = CountVectorizer(
        tokenizer=lambda x: [item.strip() for item in x.split(";") if item.strip()],
        lowercase=True
    )
    text_features = vectorizer.fit_transform(features['Text_Features'])

    # Numeric features
    numeric_features = features[numeric_cols].values

    # Combine numeric + text
    X = hstack([numeric_features, text_features])

    return X, vectorizer