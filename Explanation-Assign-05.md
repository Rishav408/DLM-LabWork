# Assignment 05: RNN and LSTM (Sequence Modeling)

## 🎯 Objective
To implement **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** networks for sequential data tasks: Sentiment Analysis and Time Series Prediction.

## 🔑 Key Concepts
- **Recurrent Neural Network (RNN):** A type of neural network where the output from the previous step is fed as input to the current step, creating a "memory" effect.
- **Vanishing Gradient Problem:** Standard RNNs struggle to learn long-range dependencies because gradients can shrink exponentially during backpropagation.
- **LSTM (Long Short-Term Memory):** A specialized RNN architecture with "gates" (input, forget, output) that help maintain information over long sequences.
- **Word Embedding:** Converting text into dense numerical vectors that capture semantic relationships.

## 💻 Code Walkthrough

### 1. RNN for Text (IMDB)
We use an `Embedding` layer followed by a `SimpleRNN` layer.
```python
model = models.Sequential([
    layers.Embedding(input_dim=10000, output_dim=32),
    layers.SimpleRNN(32),
    layers.Dense(1, activation='sigmoid')
])
```

### 2. LSTM for Time Series
Predicting the next value in a sequence (e.g., a sine wave).
```python
model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(n_steps, n_features)),
    layers.LSTM(50),
    layers.Dense(1)
])
```

## 🎓 VIVA Preparation (FAQs)

**Q1: How does an RNN differ from a standard Feed-Forward network?**
*Answer:* RNNs have internal loops (feedback connections). This allows them to process sequences of data by maintaining a hidden state that carries information from previous time steps.

**Q2: What is the "Vanishing Gradient" problem?**
*Answer:* In deep networks and long sequences, gradients can become very small during backpropagation, making it impossible for the weights to update effectively. This is particularly problematic for standard RNNs.

**Q3: How does an LSTM solve the vanishing gradient problem?**
*Answer:* LSTMs use a "cell state" and "gates" (specifically the forget gate) to allow information to flow through the network with minimal change, enabling the learning of long-term dependencies.

**Q4: What is the purpose of `pad_sequences`?**
*Answer:* Deep learning models usually require fixed-size inputs. Since sentences have different lengths, we use padding to make all sequences the same length by adding zeros.
