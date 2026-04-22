# Assignment 07: Generative Adversarial Networks (GANs)

## 🎯 Objective
To understand and implement a **Generative Adversarial Network (GAN)** to generate synthetic handwritten digits (similar to the MNIST dataset).

## 🔑 Key Concepts
- **Generator:** A network that takes random noise as input and tries to create realistic data (images) to "fool" the discriminator.
- **Discriminator:** A network that acts as a classifier, trying to distinguish between "Real" data (from the dataset) and "Fake" data (from the generator).
- **Adversarial Training:** The two networks are trained simultaneously in a game-theoretic manner until the generator produces data indistinguishable from real data.
- **Latent Space:** The space of random input vectors (noise) from which the generator creates images.

## 💻 Code Walkthrough

### 1. The Generator
Uses `Conv2DTranspose` (upsampling) to turn noise into a 28x28 image.
```python
model.add(layers.Dense(7*7*256, input_shape=(100,)))
model.add(layers.Reshape((7, 7, 256)))
model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
```

### 2. The Discriminator
A standard CNN classifier.
```python
model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
model.add(layers.LeakyReLU())
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
```

## 🎓 VIVA Preparation (FAQs)

**Q1: Explain the "minimax" nature of GAN training.**
*Answer:* The generator tries to minimize the probability that the discriminator correctly identifies its output as fake, while the discriminator tries to maximize its ability to distinguish real from fake.

**Q2: What is Mode Collapse?**
*Answer:* Mode collapse is a failure mode in GAN training where the generator produces a very limited variety of outputs (e.g., generating only the digit '1' for all different noise inputs).

**Q3: Why do we use LeakyReLU instead of ReLU in the Discriminator?**
*Answer:* LeakyReLU allows a small gradient to flow through even when the neuron is inactive ($x < 0$), which helps avoid "dead neurons" during the adversarial competition.

**Q4: What is the purpose of the Latent Dimension?**
*Answer:* It represents a compressed, abstract representation of the data. By sampling different points in this latent space, we can generate diverse outputs.
