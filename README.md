# LSTM Text Sequence Prediction

## Overview
This project demonstrates how to build and train a Long Short-Term Memory (LSTM) model for sequence prediction and text generation. The model is trained on English text data and can predict or generate sequences of words based on input sentences. It incorporates essential components such as data preprocessing, one-hot encoding, and regularization to ensure robust performance.

---

## Features
- **Data Preprocessing**:
  - Converts raw text into normalized ASCII format.
  - Splits the dataset into training, validation, and test sets.
  - Implements one-hot encoding for representing words as tensors.
- **LSTM Model**:
  - Two-layer LSTM with dropout regularization.
  - Predicts word sequences with End-of-Sequence (EOS) handling.
- **Training and Optimization**:
  - Implements CrossEntropyLoss for sequence modeling.
  - Uses the Adam optimizer with weight decay for regularization.
  - Integrates a learning rate scheduler and early stopping for efficient training.
- **Visualization**:
  - Tracks and plots training, validation, and test losses over epochs.
  - Visualizes training progress using Matplotlib.

---

## Dataset
The project uses a bilingual English-to-Spanish dataset from the [Tatoeba Project](https://tatoeba.org/), specifically the `spa.txt` file:
- **Source**: https://www.manythings.org/anki/spa-eng.zip
- **Preprocessing**:
  - Normalizes text by converting to lowercase, removing special characters, and trimming whitespace.
  - Splits sentences into individual words for input-output pair generation.

---

## Key Components
### 1. **Data Handling**
- Parses the dataset into normalized English sentence pairs.
- Creates one-hot encoded tensors for input and target sequences.
- Splits the dataset into training (1000 samples), validation (1000 samples), and test (1000 samples).

### 2. **Model Architecture**
- **Input Size**: Vocabulary size + 1 (for EOS token).
- **Hidden Size**: 128 features.
- **Output Size**: Vocabulary size + 1.
- Two-layer LSTM with dropout regularization (30%).
- Fully connected layer for sequence prediction.

### 3. **Training**
- **Loss Function**: CrossEntropyLoss for multi-class classification.
- **Optimizer**: Adam with a learning rate of 0.001 and weight decay of 0.01 for regularization.
- **Learning Rate Scheduler**: Exponential decay with gamma = 0.95.
- Implements early stopping to halt training if validation loss does not improve for 3 consecutive epochs.

### 4. **Prediction**
- Generates word sequences by predicting the most probable next word iteratively until the EOS token is reached or a maximum length is achieved.

---

## How to Use
### Prerequisites
- Python 3.x
- Libraries: `torch`, `numpy`, `matplotlib`, `scikit-learn`

### Steps to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
    ```bash
    pip install torch matplotlib scikit-learn numpy

    ```
3. Run the script to preprocess data, train the model, and generate predictions:
```bash
python lstm_text_prediction.py
```
## Future Enhancements
- Extend to multilingual datasets for broader sequence modeling.
- Implement attention mechanisms for improved long-range dependencies.
- Add BLEU score evaluation for better performance analysis.
