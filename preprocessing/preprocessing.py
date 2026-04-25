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
    
    Parameters:
        df (pd.DataFrame): input dataset
        numeric_cols (list): list of numeric columns, e.g., ['Age', 'Education']
        text_cols (list): list of text columns, e.g., ['Skills', 'Interests']
    
    Returns:
        X (scipy.sparse.csr_matrix): combined feature matrix
        vectorizer (CountVectorizer): fitted CountVectorizer for text features
    """
    # Copy only the numeric and text columns (exclude CandidateID, Name)
    features = df[numeric_cols + text_cols].copy()
    
    # Encode categorical numeric-like features (like Education)
    if 'Education' in numeric_cols:
        features['Education_enc'] = LabelEncoder().fit_transform(features['Education'])
        numeric_cols = [col for col in numeric_cols if col != 'Education']  # remove original
        numeric_cols.append('Education_enc')
    
    # Combine text features into a single column
    features['Text_Features'] = features[text_cols[0]].fillna('')
    for col in text_cols[1:]:
        features['Text_Features'] += " " + features[col].fillna('')
    
    # Vectorize text features
    vectorizer = CountVectorizer(
        tokenizer=lambda x: [item.strip() for item in x.split(";") if item.strip()],
        lowercase=True
    )
    text_features = vectorizer.fit_transform(features['Text_Features'])
    
    # Numeric features as numpy array
    numeric_features = features[numeric_cols].values  # only numeric, no object dtypes
    
    # Combine numeric + text features
    X = hstack([numeric_features, text_features])
    
    return X, vectorizer