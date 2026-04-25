# preprocessing/preprocessing.py
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

def overview_data(df):
    print("Dataset Shape:", df.shape)
    df.info()
    print(df.describe())

def preprocess_data(df):
    features = df[['Age','Education','Skills','Interests']].copy()
    features['Education_enc'] = LabelEncoder().fit_transform(features['Education'])
    features['Text_Features'] = features['Skills'].fillna('') + " " + features['Interests'].fillna('')
    vectorizer = CountVectorizer(tokenizer=lambda x: [item.strip() for item in x.split(";") if item.strip()], lowercase=True)
    text_features = vectorizer.fit_transform(features['Text_Features'])
    numeric_features = features[['Age','Education_enc']].values
    X = hstack([numeric_features, text_features])
    return X, vectorizer
