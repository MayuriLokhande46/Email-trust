import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

VECT_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')


def build_vectorizer(texts, max_features=5000):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    return vect, X


def save_vectorizer(vect, path=VECT_PATH):
    joblib.dump(vect, path)


def load_vectorizer(path=VECT_PATH):
    return joblib.load(path)
