import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_data():
    # ds = load_dataset("dair-ai/emotion", "split")
    df = pd.read_csv("./data/snappfood-comments.csv")

    # Preprocessing
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["comment"])
    feature_names = vectorizer.get_feature_names_out()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, df["label_id"], test_size=0.2, random_state=42
    )

    def preprocess_data(text):
        return vectorizer.transform([text])

    return X_train, X_test, y_train, y_test, feature_names, preprocess_data


def get_classification_metrics(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    matrix = confusion_matrix(y_test, pred)

    return accuracy, report, matrix
