import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import json

from preprocess import clean_text
from features import build_vectorizer, save_vectorizer

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'spam.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Dataset not found at {DATA_PATH}. Run data/create_dataset.py first.")

print('Loading data from', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('Raw rows:', len(df))
df = df.dropna(subset=['text', 'label'])
print('After dropna rows:', len(df))

df['text_clean'] = df['text'].astype(str).map(clean_text)
empty_after_clean = (df['text_clean'].astype(str).str.strip() == '').sum()
print('Empty after clean():', empty_after_clean)

# Save cleaned CSV for inspection
cleaned_path = os.path.join(os.path.dirname(DATA_PATH), 'spam_clean.csv')
df.to_csv(cleaned_path, index=False)
print('Saved cleaned CSV to', cleaned_path)

# Drop rows where cleaned text is empty (they can't be vectorized)
empty_mask = df['text_clean'].astype(str).str.strip() == ''
empty_count = int(empty_mask.sum())
if empty_count > 0:
    print(f'Dropping {empty_count} rows with empty cleaned text')
    df = df.loc[~empty_mask].copy()

X_text = df['text_clean'].values
# binary label: spam=1, ham=0
y = (df['label'].str.lower() == 'spam').astype(int).values

print('Building TF-IDF vectorizer...')
vect, X = build_vectorizer(X_text, max_features=5000)
save_vectorizer(vect)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'nb': MultinomialNB(),
    'lr': LogisticRegression(max_iter=1000),
    'gbc': GradientBoostingClassifier(),
    'rf': RandomForestClassifier()
}

best_name = None
best_score = -1
best_model = None
for name, model in models.items():
    print('Training', name)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_name = name
        best_model = model

print('Best model:', best_name, 'score:', best_score)
joblib.dump(best_model, os.path.join(MODEL_DIR, 'spam_model.joblib'))
print('Saved model to', os.path.join(MODEL_DIR, 'spam_model.joblib'))

# Detailed report
pred = best_model.predict(X_test)
print(classification_report(y_test, pred))

# --- Save spam words if model is Logistic Regression ---
if best_name == 'lr':
    print('Saving spam words for Logistic Regression model...')
    try:
        feature_names = vect.get_feature_names_out()
        coefs = best_model.coef_[0]
        word_coefs = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)
        
        # Get top 200 words that are strong indicators of spam
        top_spam_words = [word for word, coef in word_coefs if coef > 0][:200]

        spam_words_path = os.path.join(MODEL_DIR, 'spam_words.json')
        with open(spam_words_path, 'w') as f:
            json.dump(top_spam_words, f)
        print(f'Saved top 200 spam words to {spam_words_path}')
    except Exception as e:
        print(f"Could not save spam words: {e}")

