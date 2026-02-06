"""
Deep Learning models for spam detection using LSTM and BERT.

This module provides advanced neural network models:
1. LSTM (Long Short-Term Memory) - RNN-based approach
2. BERT (Bidirectional Encoder Representations from Transformers) - Transformer-based approach

Note: Requires tensorflow/pytorch and huggingface transformers library.
"""

import logging
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning('TensorFlow not available. LSTM models will not work.')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from torch import nn
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning('Transformers not available. BERT models will not work.')


class LSTMSpamDetector:
    """LSTM-based spam detector using Keras/TensorFlow."""

    def __init__(self, max_features: int = 5000, max_length: int = 100, 
                 embedding_dim: int = 128, lstm_units: int = 64):
        """
        Initialize LSTM model.

        Args:
            max_features: Vocabulary size
            max_length: Maximum sequence length
            embedding_dim: Word embedding dimension
            lstm_units: Number of LSTM units
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError('TensorFlow is required for LSTM models. Install: pip install tensorflow')

        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = None
        self.model = None

    def build_model(self) -> 'tf.keras.Model':
        """
        Build LSTM model architecture.

        Architecture:
        - Embedding Layer (word embeddings)
        - Bidirectional LSTM
        - Dropout
        - Dense layer with ReLU
        - Dropout
        - Output layer with Sigmoid

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(self.lstm_units // 2)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        logger.info('LSTM model built successfully')
        return model

    def prepare_texts(self, texts: list, fit: bool = False) -> np.ndarray:
        """
        Tokenize and pad text sequences.

        Args:
            texts: List of text samples
            fit: Whether to fit tokenizer (set True for training data)

        Returns:
            Padded sequence array
        """
        if fit:
            self.tokenizer = Tokenizer(num_words=self.max_features)
            self.tokenizer.fit_on_texts(texts)
            logger.info(f'Tokenizer fitted on {len(texts)} texts')

        if self.tokenizer is None:
            raise ValueError('Tokenizer not initialized. Call with fit=True first.')

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 10, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        if self.model is None:
            self.model = self.build_model()

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        logger.info(f'Starting LSTM training for {epochs} epochs')
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        return history.history

    def predict(self, texts: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict spam probability for texts.

        Args:
            texts: List of text samples

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError('Model not trained or loaded')

        X = self.prepare_texts(texts, fit=False)
        probabilities = self.model.predict(X)
        predictions = (probabilities > 0.5).astype(int).flatten()

        return predictions, probabilities.flatten()

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise ValueError('No model to save')

        self.model.save(filepath)
        
        # Save tokenizer config
        tokenizer_path = filepath.replace('.h5', '_tokenizer.json')
        tokenizer_json = json.dumps(self.tokenizer.get_config())
        with open(tokenizer_path, 'w') as f:
            f.write(tokenizer_json)

        logger.info(f'Model saved to {filepath}')

    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        self.model = load_model(filepath)
        
        # Load tokenizer
        tokenizer_path = filepath.replace('.h5', '_tokenizer.json')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                tokenizer_config = json.load(f)
            self.tokenizer = Tokenizer.from_config(tokenizer_config)

        logger.info(f'Model loaded from {filepath}')


class BERTSpamDetector:
    """BERT-based spam detector using HuggingFace Transformers."""

    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize BERT detector.

        Args:
            model_name: HuggingFace model identifier
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                'Transformers and PyTorch required. '
                'Install: pip install transformers torch'
            )

        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipe = None

    def initialize_pretrained(self):
        """Load pretrained BERT model."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Create classification pipeline
            self.pipe = pipeline(
                'text-classification',
                model=self.model_name,
                device=0 if self._has_cuda() else -1  # Use GPU if available
            )

            logger.info(f'BERT model {self.model_name} loaded successfully')
        except Exception as e:
            logger.error(f'Failed to load BERT model: {str(e)}')
            raise

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def predict(self, texts: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict spam probability using BERT.

        Args:
            texts: List of text samples

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.pipe is None:
            raise ValueError('Model not initialized. Call initialize_pretrained() first.')

        predictions_list = []
        probabilities_list = []

        for text in texts:
            # Truncate if too long
            if len(text) > 512:
                text = text[:512]

            result = self.pipe(text)[0]
            
            # Parse result
            label = result['label'].lower()
            score = result['score']

            # Normalize: BERT might return "POSITIVE"/"NEGATIVE" or "1"/"0"
            if 'spam' in label or label == '1':
                predictions_list.append(1)
                probabilities_list.append(score)
            else:
                predictions_list.append(0)
                probabilities_list.append(1 - score)

        return np.array(predictions_list), np.array(probabilities_list)

    def fine_tune(self, X_train: list, y_train: np.ndarray,
                  X_val: Optional[list] = None, y_val: Optional[np.ndarray] = None,
                  epochs: int = 3, batch_size: int = 8):
        """
        Fine-tune BERT model on spam detection data.

        Note: This requires more setup and is computationally expensive.
        
        Args:
            X_train: Training texts
            y_train: Training labels (0=ham, 1=spam)
            X_val: Validation texts
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
        """
        logger.warning('BERT fine-tuning requires advanced setup. Consider using HuggingFace Trainer.')
        raise NotImplementedError(
            'Fine-tuning not yet implemented. Use HuggingFace Trainer for production fine-tuning.'
        )


class EnsembleSpamDetector:
    """Ensemble detector combining multiple models."""

    def __init__(self):
        """Initialize ensemble detector."""
        self.models = {}
        self.weights = {}

    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.

        Args:
            name: Model identifier
            model: Model object with predict method
            weight: Weight in ensemble (default: equal weight)
        """
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f'Added {name} to ensemble with weight {weight}')

    def predict(self, texts: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using ensemble voting.

        Args:
            texts: List of text samples

        Returns:
            Tuple of (predictions, ensemble_probabilities)
        """
        if not self.models:
            raise ValueError('No models in ensemble')

        ensemble_scores = np.zeros(len(texts))
        weight_sum = 0

        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            
            try:
                _, probs = model.predict(texts)
                ensemble_scores += probs * weight
                weight_sum += weight
            except Exception as e:
                logger.warning(f'Error in {name} prediction: {str(e)}')

        # Normalize by weight sum
        if weight_sum > 0:
            ensemble_scores /= weight_sum

        predictions = (ensemble_scores > 0.5).astype(int)
        return predictions, ensemble_scores

    def get_model_contributions(self, text: str) -> Dict[str, float]:
        """
        Get individual model contributions for a single text.

        Args:
            text: Text sample

        Returns:
            Dict of model predictions
        """
        contributions = {}

        for name, model in self.models.items():
            try:
                _, probs = model.predict([text])
                contributions[name] = float(probs[0])
            except Exception as e:
                logger.warning(f'Error getting contribution from {name}: {str(e)}')
                contributions[name] = None

        return contributions


if __name__ == '__main__':
    # Example usage
    print("Deep Learning Models for Spam Detection")
    print("=" * 50)

    # LSTM example
    if TENSORFLOW_AVAILABLE:
        print("\nLSTM Model:")
        lstm = LSTMSpamDetector()
        print(f"- Max features: {lstm.max_features}")
        print(f"- Max length: {lstm.max_length}")
        print(f"- Embedding dimension: {lstm.embedding_dim}")

    # BERT example
    if TRANSFORMERS_AVAILABLE:
        print("\nBERT Model:")
        bert = BERTSpamDetector()
        print(f"- Model: {bert.model_name}")
        print("- Use case: State-of-the-art NLP understanding")

    # Ensemble example
    print("\nEnsemble Detector:")
    print("- Combines multiple models for robust predictions")
    print("- Weighted voting for final decision")

    print("\n" + "=" * 50)
    print("Install required packages for deep learning:")
    print("pip install tensorflow")
    print("pip install transformers torch")
