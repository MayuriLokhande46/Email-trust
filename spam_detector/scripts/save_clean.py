import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.preprocess import clean_text

DATA_PATH = os.path.join(ROOT, 'data', 'spam.csv')
OUT_PATH = os.path.join(ROOT, 'data', 'spam_clean.csv')

def main():
    if not os.path.exists(DATA_PATH):
        print('Dataset not found at', DATA_PATH)
        return
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    df = df.dropna(subset=['text', 'label']).copy()
    df['text_clean'] = df['text'].astype(str).map(clean_text)
    df.to_csv(OUT_PATH, index=False)
    print('Saved cleaned CSV to', OUT_PATH)

if __name__ == '__main__':
    main()
