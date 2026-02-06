import os
import pandas as pd

os.makedirs(os.path.dirname(__file__), exist_ok=True)

# Download a public SMS spam dataset and save as CSV with columns label,text
url = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)

out_path = os.path.join(os.path.dirname(__file__), "spam.csv")
try:
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path} with {len(df)} rows")
except Exception as e:
    print("Failed to download dataset:", e)
    # fallback: create a tiny sample file so the rest of the project can run
    sample = pd.DataFrame(
        {
            "label": ["ham", "spam"],
            "text": [
                "Hey, are you free tonight?",
                "WIN $1000 now! Click http://spam.example",
            ],
        }
    )
    sample.to_csv(out_path, index=False)
    print(f"Created sample dataset at {out_path}")

os.system(r".\spam_detector\.venv\Scripts\activate")
