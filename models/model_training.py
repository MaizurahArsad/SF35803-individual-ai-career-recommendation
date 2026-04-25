from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y, test_size=0.2, random_state=42, n_estimators=100):
    """Train RandomForest model and return model + dataset splits."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test
