# Assignment 04: Convolutional Neural Networks (CNN)

## 🎯 Objective
To build and train a **CNN** using TensorFlow/Keras to classify images from the **CIFAR-10** dataset (10 categories like airplane, dog, frog, etc.).

## 🔑 Key Concepts
- **Convolution Layer:** Extracts spatial features (edges, textures) using filters/kernels.
- **Pooling (MaxPooling):** Reduces the spatial dimensions (width, height) of the feature maps, reducing computation and preventing overfitting.
- **Flattening:** Converting the 2D feature maps into a 1D vector to feed into a Dense (fully connected) layer.
- **Softmax:** The final layer activation for multi-class classification, providing probabilities for each class.

## 💻 Code Walkthrough

### 1. Model Architecture
A typical CNN stack alternates between Conv and Pool layers.
```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
```

### 2. Training
We use `categorical_crossentropy` for multi-class labels.
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
```

## 🎓 VIVA Preparation (FAQs)

**Q1: Why are CNNs better than MLPs for image data?**
*Answer:* CNNs preserve spatial structure and use parameter sharing (kernels), making them much more efficient at detecting local patterns (like edges) regardless of their position in the image.

**Q2: What is the role of Padding in Convolution?**
*Answer:* Padding ('same' or 'valid') ensures that the spatial dimensions of the input are preserved or controlled after convolution. 'Same' padding adds zeros around the edges so the output size is the same as the input.

**Q3: What does a Pooling layer do?**
*Answer:* Pooling downsamples the input. It reduces the number of parameters and computation in the network, and it also provides a form of translation invariance.

**Q4: What is CIFAR-10?**
*Answer:* It is a standard computer vision dataset consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
