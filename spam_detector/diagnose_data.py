#!/usr/bin/env python3
"""Small diagnostics to check how many rows are dropped or become empty after preprocessing."""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from src.preprocess import clean_text

DATA_PATH = os.path.join(ROOT, "data", "spam.csv")

def main():
    if not os.path.exists(DATA_PATH):
        print("Dataset not found at:", DATA_PATH)
        return
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    print("Total rows (raw):", len(df))
    if 'text' in df.columns and 'label' in df.columns:
        print("Missing 'text':", df['text'].isnull().sum())
        print("Missing 'label':", df['label'].isnull().sum())
        print("Label counts:\n", df['label'].value_counts())
    else:
        print("Columns:", list(df.columns))

    df2 = df.dropna(subset=['text', 'label']).copy()
    dropped = len(df) - len(df2)
    print("After dropna rows:", len(df2), "(dropped:", dropped, ")")

    # apply cleaning and check for empty cleaned text
    df2['text_clean'] = df2['text'].astype(str).map(clean_text)
    empty_after_clean = (df2['text_clean'].astype(str).str.strip() == '').sum()
    print("Empty after clean():", empty_after_clean)
    if empty_after_clean:
        empties = df2[df2['text_clean'].astype(str).str.strip() == '']
        print("Sample empty rows (index,label,text):")
        for idx, row in empties.head(5).iterrows():
            txt = row.get('text', '')
            print(idx, row.get('label'), (txt[:120] + '...') if len(txt) > 120 else txt)

if __name__ == '__main__':
    main()
