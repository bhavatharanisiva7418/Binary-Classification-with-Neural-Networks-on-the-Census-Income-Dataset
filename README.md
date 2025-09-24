# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset

Build a binary classification model using a neural network to predict whether an individual earns more than $50K annually based on demographic and work-related features.

### Model Steps:
### 1. Data Loading
Load the income.csv dataset (30,000 entries) using Pandas.
Inspect the dataset to understand the features and target variable.
#### 2.Data Preprocessing
Separate columns into:

cat_cols: Categorical features (sex, education, etc.)
cont_cols: Continuous features (age, hours-per-week)
y_col: Label column (label)
Convert all categorical columns to category dtype for encoding.
### 3.Embedding Setup
Determine the number of unique categories in each categorical column.
Define embedding sizes using the rule: (category_size, min(50, (category_size + 1)//2))
### 4.Data Conversion
Convert categorical columns to category codes and stack into a NumPy array.

Convert continuous columns to a NumPy array.

Convert both arrays into PyTorch tensors:

cats: Categorical tensor (int64)
conts: Continuous tensor (float32)
y: Label tensor (flattened)
### 5.Train/Test Split
Split the data into:

Training set (25,000 records)
Testing set (5,000 records)
### 6.Model Architecture
Create a custom PyTorch model class TabularModel that:

Handles embedding layers for categorical data.
Applies batch normalization and dropout.
Uses one or more fully connected layers for prediction.
Final output layer has 2 units (binary classification).

### 7.Training Setup
Define:

criterion: CrossEntropyLoss
optimizer: Adam optimizer with lr=0.001
Set random seed for reproducibility.

### 8.Model Training
Train for 300 epochs.
Track and store loss values per epoch.
### 9.Loss Visualization
Plot CrossEntropy Loss vs Epochs to observe convergence.
### 10.Model Evaluation
Evaluate on the test set:

Calculate Cross Entropy Loss
Compute accuracy of predictions
