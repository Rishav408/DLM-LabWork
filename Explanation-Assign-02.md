# Assignment 02: TensorFlow Basics

## 🎯 Objective
To understand the core components of **TensorFlow**, including tensor creation, manipulation, and the basics of building a computational graph using the Keras API.

## 🔑 Key Concepts
- **Tensors:** Multi-dimensional arrays with a uniform type (dtype). They are the fundamental building blocks of TensorFlow.
- **Broadcasting:** The ability to perform operations on tensors of different shapes under certain conditions.
- **Variables vs. Constants:** `tf.Variable` allows for mutable state (like weights in a model), whereas `tf.constant` creates immutable tensors.
- **Layer API:** Using `tf.keras.layers` to build models.

## 💻 Code Walkthrough

### 1. Creating Tensors
```python
import tensorflow as tf
a = tf.constant([[1, 2], [3, 4]])
b = tf.Variable([2.0, 3.0], name='weights')
```

### 2. Math Operations
TensorFlow supports standard matrix operations which are essential for deep learning.
```python
c = tf.matmul(a, a) # Matrix multiplication
```

### 3. Simple Model Structure
Understanding how layers stack to form a model.
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
```

## 🎓 VIVA Preparation (FAQs)

**Q1: What is a Tensor in TensorFlow?**
*Answer:* A Tensor is a multi-dimensional array, similar to a NumPy `ndarray`, but it can be used on GPUs/TPUs and is integrated into TensorFlow’s automatic differentiation system.

**Q2: Difference between `tf.Variable` and `tf.constant`?**
*Answer:* `tf.constant` is immutable; its value cannot change after creation. `tf.Variable` is mutable and is used to store parameters that need to be updated during training (like weights).

**Q3: What is the role of an Activation Function?**
*Answer:* Activation functions (like ReLU, Sigmoid) introduce non-linearity into the model, allowing it to learn complex patterns. Without them, a neural network would just behave like a linear regression model regardless of depth.

**Q4: What is Eager Execution?**
*Answer:* Eager execution is an imperative programming environment that evaluates operations immediately, without building a graph. This makes debugging and prototyping much easier in TensorFlow 2.x.
