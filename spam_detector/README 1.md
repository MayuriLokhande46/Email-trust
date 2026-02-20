@"
# Email Spam Detection using ML

Steps:
1. Activate venv: `.\.venv\Scripts\Activate.ps1`
2. Install: `python -m pip install -r requirements.txt`
3. Download data: `python .\data\create_dataset.py`
4. Train: `python .\src\train.py`
5. Predict: `python .\src\predict.py "sample text"`
"@ | Set-Content -Path .\README.md -Encoding UTF8