# Deep Learning Models for Spam Detection

This guide explains how to use the advanced LSTM and BERT models for improved spam detection accuracy.

## Overview

The project now includes three types of models:

1. **Traditional ML Models** (existing) - Fast, lightweight
2. **LSTM Models** - Deep learning RNN-based
3. **BERT Models** - State-of-the-art transformer-based
4. **Ensemble Models** - Combines multiple models for robustness

## Installation

### LSTM Model Requirements

```bash
pip install tensorflow>=2.10.0
pip install keras>=2.10.0
```

### BERT Model Requirements

```bash
pip install transformers>=4.25.0
pip install torch>=1.13.0
```

### Complete Installation

```bash
pip install -r requirements_deep_learning.txt
```

Create `requirements_deep_learning.txt`:
```
tensorflow>=2.10.0
torch>=1.13.0
transformers>=4.25.0
numpy>=1.21.0
scipy>=1.7.0
```

## LSTM Model

### Overview

LSTM (Long Short-Term Memory) uses Recurrent Neural Networks to understand sequential patterns in text.

**Advantages:**
- Better than traditional ML for sequence understanding
- Faster to train than BERT
- Good balance between speed and accuracy
- Works well with longer sequences

**Disadvantages:**
- Slower inference than traditional ML
- Requires more training data
- Less accurate than BERT for complex patterns

### Usage

#### 1. Training an LSTM Model

```python
from spam_detector.src.deep_learning import LSTMSpamDetector
import pandas as pd
from sklearn.model_selection import train_test_split
from spam_detector.src.preprocess import clean_text

# Load data
df = pd.read_csv('spam_detector/data/spam.csv')

# Preprocess
df['text_clean'] = df['text'].astype(str).map(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'].values,
    (df['label'].str.lower() == 'spam').astype(int).values,
    test_size=0.2,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

# Create and train model
lstm = LSTMSpamDetector(
    max_features=5000,
    max_length=100,
    embedding_dim=128,
    lstm_units=64
)

# Prepare texts
X_train_seq = lstm.prepare_texts(X_train, fit=True)
X_val_seq = lstm.prepare_texts(X_val, fit=False)
X_test_seq = lstm.prepare_texts(X_test, fit=False)

# Train
history = lstm.train(
    X_train_seq, y_train,
    X_val_seq, y_val,
    epochs=10,
    batch_size=32
)

# Save model
lstm.save_model('models/lstm_spam_detector.h5')

# Evaluate
predictions, probabilities = lstm.predict(X_test_seq)
accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")
```

#### 2. Predicting with LSTM

```python
from spam_detector.src.deep_learning import LSTMSpamDetector

# Load model
lstm = LSTMSpamDetector()
lstm.load_model('models/lstm_spam_detector.h5')

# Predict
texts = [
    "Congratulations! You won $1000!",
    "Meeting scheduled for tomorrow",
    "URGENT: Verify your account now!"
]

predictions, probabilities = lstm.predict(texts)

for text, pred, prob in zip(texts, predictions, probabilities):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"Text: {text[:50]}...")
    print(f"Label: {label}, Probability: {prob:.4f}\n")
```

## BERT Model

### Overview

BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer model.

**Advantages:**
- Highest accuracy for complex spam patterns
- Understands bidirectional context
- Pre-trained on massive text corpus
- Excellent for nuanced spam detection

**Disadvantages:**
- Slower inference (100ms+ per prediction)
- Requires more memory
- GPU recommended for production use
- More computationally expensive

### Usage

#### 1. Using Pre-trained BERT

```python
from spam_detector.src.deep_learning import BERTSpamDetector

# Initialize BERT detector
bert = BERTSpamDetector(model_name='bert-base-uncased')
bert.initialize_pretrained()

# Predict
texts = [
    "URGENT: Update your password immediately!",
    "Let's schedule a meeting for next week",
    "Click HERE!!! FREE MONEY!!!"
]

predictions, probabilities = bert.predict(texts)

for text, pred, prob in zip(texts, predictions, probabilities):
    label = "SPAM" if pred == 1 else "HAM"
    confidence = prob if pred == 1 else (1 - prob)
    print(f"Text: {text[:50]}...")
    print(f"Label: {label} (Confidence: {confidence:.4f})\n")
```

#### 2. Using Different BERT Models

```python
# DistilBERT - Faster, smaller version
bert_distil = BERTSpamDetector('distilbert-base-uncased')
bert_distil.initialize_pretrained()

# RoBERTa - Improved variant
bert_roberta = BERTSpamDetector('roberta-base')
bert_roberta.initialize_pretrained()

# Domain-specific - For email/SMS
bert_domain = BERTSpamDetector('bert-base-uncased')
bert_domain.initialize_pretrained()
```

#### 3. Fine-tuning BERT (Advanced)

For production use with domain-specific data:

```python
# Use HuggingFace Trainer for fine-tuning
from transformers import Trainer, TrainingArguments
import torch

# Prepare datasets
# ... (see HuggingFace documentation)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./models/bert_spam_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=2e-5,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
trainer.train()

# Save
trainer.save_model('models/bert_spam_finetuned')
```

## Ensemble Model

### Overview

Combines multiple models for robust, reliable predictions.

**Advantages:**
- More robust predictions
- Reduces bias from single model
- Can balance speed vs. accuracy
- Easier error recovery

**Example:**

```python
from spam_detector.src.deep_learning import EnsembleSpamDetector, LSTMSpamDetector, BERTSpamDetector

# Create ensemble
ensemble = EnsembleSpamDetector()

# Add models with weights
lstm = LSTMSpamDetector()
lstm.load_model('models/lstm_spam_detector.h5')

bert = BERTSpamDetector('distilbert-base-uncased')
bert.initialize_pretrained()

ensemble.add_model('lstm', lstm, weight=0.4)
ensemble.add_model('bert', bert, weight=0.6)

# Predict
texts = ["Test email content here"]
predictions, probabilities = ensemble.predict(texts)

# Get individual contributions
contributions = ensemble.get_model_contributions(texts[0])
print(f"LSTM: {contributions['lstm']:.4f}")
print(f"BERT: {contributions['bert']:.4f}")
print(f"Ensemble: {probabilities[0]:.4f}")
```

## Integration with API

### Using LSTM in FastAPI

```python
# In api.py
from deep_learning import LSTMSpamDetector

# Load model at startup
lstm_model = None

@app.on_event("startup")
async def load_models():
    global lstm_model
    lstm_model = LSTMSpamDetector()
    lstm_model.load_model('models/lstm_spam_detector.h5')

@app.post("/predict_lstm")
async def predict_lstm(input_data: EmailInput, current_user: str = Depends(get_current_user)):
    """Predict using LSTM model"""
    try:
        X = lstm_model.prepare_texts([input_data.text], fit=False)
        predictions, probs = lstm_model.predict([input_data.text])
        
        return {
            "label": "spam" if predictions[0] == 1 else "ham",
            "probability": float(probs[0]),
            "model": "lstm"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Using BERT in FastAPI

```python
@app.post("/predict_bert")
async def predict_bert(input_data: EmailInput, current_user: str = Depends(get_current_user)):
    """Predict using BERT model"""
    try:
        predictions, probs = bert_model.predict([input_data.text])
        
        return {
            "label": "spam" if predictions[0] == 1 else "ham",
            "probability": float(probs[0]),
            "model": "bert"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Performance Comparison

| Metric | Traditional ML | LSTM | BERT | Ensemble |
|--------|----------------|------|------|----------|
| Accuracy | 85-90% | 90-95% | 94-99% | 93-98% |
| Inference Time | 1ms | 50-100ms | 100-500ms | 30-150ms |
| Training Time | Minutes | Hours | Hours-Days | Hours |
| Memory | 10MB | 100MB | 500MB+ | 500MB+ |
| Production Ready | Yes | Yes | Yes (GPU) | Yes |

## Best Practices

### 1. Model Selection

**Use Traditional ML when:**
- Speed is critical (real-time, high volume)
- Resource-constrained (edge devices)
- Interpretability is important
- Limited training data

**Use LSTM when:**
- Better accuracy needed than traditional ML
- Moderate computational resources available
- Need to understand sequential patterns
- Good balance of speed and accuracy

**Use BERT when:**
- Maximum accuracy required
- Complex spam patterns to detect
- GPU resources available
- Can tolerate higher latency

### 2. Deployment Strategy

```python
# Start with fast traditional model
traditional_model = load_traditional_model()

# Use deep learning for uncertain predictions
if 0.4 < traditional_model.predict(text) < 0.6:
    # Run BERT for uncertain cases
    bert_prediction = bert_model.predict(text)
    final_prediction = bert_prediction
else:
    final_prediction = traditional_model.predict(text)
```

### 3. Monitoring

```python
# Track model performance
def log_prediction(text, model, prediction, actual):
    logger.info(f"Model: {model}, Prediction: {prediction}, Actual: {actual}")
    
    # Update metrics
    if prediction == actual:
        model_accuracy.labels(model=model).inc()
    else:
        model_errors.labels(model=model).inc()
```

### 4. Batch Processing

```python
# For high-volume predictions
def batch_predict(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        predictions, probs = model.predict(batch)
        results.extend(zip(predictions, probs))
    return results
```

## GPU Acceleration

### Setting up GPU Support

```bash
# For CUDA 11.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Using GPU in Code

```python
# TensorFlow
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# PyTorch
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

## Troubleshooting

### Out of Memory (OOM) Error

```python
# Reduce batch size
lstm.train(X_train, y_train, batch_size=8)

# Or use smaller model
lstm = LSTMSpamDetector(lstm_units=32)
```

### Slow Predictions

```python
# Use DistilBERT instead of full BERT
bert = BERTSpamDetector('distilbert-base-uncased')

# Enable batch processing
predictions = model.predict(texts)  # Pass list, not individual texts
```

### Model Not Converging

```python
# Reduce learning rate
# Increase training data
# Add regularization (Dropout)

lstm = LSTMSpamDetector()
model = lstm.build_model()
# Modify compile with lower learning rate
```

## References

- [LSTM Introduction](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [HuggingFace Documentation](https://huggingface.co/docs)
- [TensorFlow Keras](https://www.tensorflow.org/guide/keras)
- [PyTorch Documentation](https://pytorch.org/docs)

---

**Last Updated:** February 6, 2024
