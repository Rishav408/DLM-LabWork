# Assignment 03: Neural Network from Scratch (XOR Logic)

## 🎯 Objective
To implement a **Multi-Layer Perceptron (MLP)** from scratch using only `NumPy` to solve the XOR problem. This helps in understanding the internal mechanics of **Forward Propagation** and **Backpropagation**.

## 🔑 Key Concepts
- **Weights & Biases:** Parameters of the network that are adjusted during training.
- **Sigmoid Activation:** $\sigma(x) = \frac{1}{1+e^{-x}}$. It maps inputs to a range between 0 and 1.
- **Forward Propagation:** Calculating the output by passing inputs through layers.
- **Backpropagation:** Calculating gradients of the error with respect to weights using the Chain Rule.
- **Gradient Descent:** Updating weights using the calculated gradients: $W = W - \alpha \cdot \Delta W$.

## 💻 Code Walkthrough

### 1. Initialization
Weights are initialized with random values.
```python
w1 = np.random.uniform(size=(2, 2))
w2 = np.random.uniform(size=(2, 1))
```

### 2. Forward Pass
Calculates $Z = W \cdot X + b$ and $A = \sigma(Z)$.
```python
layer1_output = sigmoid(np.dot(X, w1))
final_output = sigmoid(np.dot(layer1_output, w2))
```

### 3. Backpropagation (The Math)
We calculate the error at the output and propagate it back to update the weights.
```python
error = y - final_output
d_weights2 = np.dot(layer1_output.T, error * sigmoid_derivative(final_output))
```

## 🎓 VIVA Preparation (FAQs)

**Q1: Why can't a single Perceptron solve the XOR problem?**
*Answer:* XOR is not linearly separable. A single perceptron can only learn linear boundaries. To solve XOR, we need at least one hidden layer to create non-linear boundaries.

**Q2: What is the purpose of Backpropagation?**
*Answer:* Backpropagation is used to calculate the gradient of the loss function with respect to each weight in the network, which is then used by an optimization algorithm to update the weights.

**Q3: What is the Learning Rate ($\alpha$)?**
*Answer:* The learning rate is a hyperparameter that controls how much we change the weights at each step. A large value might overshoot the minimum, while a small value might make training extremely slow.

**Q4: Why do we use the Sigmoid derivative in backpropagation?**
*Answer:* In the chain rule of calculus, the gradient depends on the slope of the activation function. The derivative of sigmoid is $f(x) \cdot (1-f(x))$, which tells us how much the output changes for a small change in input.
