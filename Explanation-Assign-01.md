# Assignment 01: Data Exploration and Visualization

## 🎯 Objective
The primary goal of this assignment is to perform **Exploratory Data Analysis (EDA)** on a real-world dataset (MrBeast YouTube statistics) to uncover patterns, trends, and correlations using Python's data science stack.

## 🔑 Key Concepts
- **EDA (Exploratory Data Analysis):** The process of analyzing datasets to summarize their main characteristics, often using visual methods.
- **Data Cleaning:** Handling missing values, removing duplicates, and correcting data types.
- **Statistical Summaries:** Using measures like mean, median, standard deviation, and quartiles to understand data distribution.
- **Correlation:** Measuring the strength of the relationship between numerical variables (e.g., Views vs. Likes).

## 💻 Code Walkthrough

### 1. Data Loading
We use `pandas` to load the dataset and `kagglehub` to fetch it directly if needed.
```python
import pandas as pd
df = pd.read_csv('mrbeast_stats.csv')
df.head()
```

### 2. Data Cleaning
We check for null values and handle them to ensure analysis accuracy.
```python
df.isnull().sum()
df.dropna(inplace=True)
```

### 3. Visualization
Using `matplotlib` and `seaborn` to create insightful plots.
- **Histogram:** To see the distribution of video durations.
- **Scatter Plot:** To visualize the relationship between views and likes.
- **Heatmap:** To show correlation between different numerical features.

## 🎓 VIVA Preparation (FAQs)

**Q1: Why is EDA the first step in any Machine Learning project?**
*Answer:* EDA helps in understanding the data's structure, identifying outliers, detecting missing values, and choosing the right features for modeling.

**Q2: What is the difference between a Histogram and a Bar Plot?**
*Answer:* A histogram is used for continuous numerical data (frequency distribution), while a bar plot is used for categorical data comparison.

**Q3: How do you handle missing values in a dataset?**
*Answer:* You can either remove the rows (dropping), fill them with a constant/mean/median (imputation), or predict the missing values using another model.

**Q4: What does a correlation value of 0.9 signify?**
*Answer:* It indicates a very strong positive linear relationship between the two variables.
