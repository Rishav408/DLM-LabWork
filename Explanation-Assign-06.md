# Assignment 06: Transfer Learning (ResNet50)

## 🎯 Objective
To apply **Transfer Learning** by leveraging a pre-trained **ResNet50** model (trained on ImageNet) to classify CIFAR-10 images with high accuracy and minimal training time.

## 🔑 Key Concepts
- **Transfer Learning:** Taking a model developed for one task and reusing it as the starting point for a model on a second task.
- **Pre-trained Model:** A model that has been previously trained on a massive dataset (like ImageNet).
- **Fine-tuning:** Unfreezing some layers of the pre-trained model and training them with a very low learning rate to adapt to the new specific dataset.
- **Feature Extraction:** Using the pre-trained model as a fixed "feature extractor" by freezing all its weights and only training the new output layers.

## 💻 Code Walkthrough

### 1. Loading the Base Model
We load ResNet50 without the top (classification) layer.
```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False  # Freeze the weights
```

### 2. Adding Custom Layers
We add a global average pooling and a dense layer for our 10 classes.
```python
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

## 🎓 VIVA Preparation (FAQs)

**Q1: What are the advantages of Transfer Learning?**
*Answer:* It saves significant time and computational resources. Since the model has already learned features like edges and shapes from a massive dataset, it converges much faster and often achieves better accuracy on smaller datasets.

**Q2: When should we use "Feature Extraction" vs. "Fine-tuning"?**
*Answer:* Use feature extraction if the new dataset is small and similar to the original dataset. Use fine-tuning if you have a larger dataset or if the new data is very different from what the model was originally trained on.

**Q3: Why do we remove the "Top Layer" of the pre-trained model?**
*Answer:* The top layer is the specific classification head (e.g., 1000 classes for ImageNet). We replace it with a new head that matches the number of classes in our specific task (e.g., 10 for CIFAR-10).

**Q4: What is "ImageNet"?**
*Answer:* ImageNet is a massive database of over 14 million labeled images in more than 20,000 categories, commonly used for training state-of-the-art computer vision models.
