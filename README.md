# Machine Learning Analysis of Diabetes Progression

## Project Overview
This project is **Lab 2** for a machine learning course, focusing on predicting diabetes progression one year after baseline using the Scikit-Learn Diabetes dataset. The objective is to develop a screening tool to assist physicians in identifying patients at risk of diabetes progression. The analysis involves building and comparing multiple machine learning models, evaluating their performance, and providing insights into their strengths and limitations.

The project is implemented in a Jupyter Notebook (`lab2.ipynb`) and follows a structured approach with three main parts:
1. **Part 1**: Data acquisition, problem framing, exploratory data analysis (EDA), data cleaning, and dataset splitting.
2. **Part 2**: Univariate polynomial regression models (degrees 0–5) using the BMI feature.
3. **Part 3**: Multivariate models, including polynomial regression (degrees 2 and 3), decision trees (max_depth 3 and 5), and k-Nearest Neighbors (k=5 and k=10).

Models are evaluated using R-squared, Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) metrics, with a train-validation-test split (75%-10%-15%). The notebook emphasizes readability, clarity, and comprehensive documentation.

## Dataset
The project uses the **Scikit-Learn Diabetes dataset**, a toy dataset available in Scikit-Learn. Key details:
- **Source**: [Scikit-Learn Documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- **Description**: The dataset includes 442 samples, 10 numerical features, and a continuous target variable representing disease progression one year after baseline.
- **Features**:
  - `age`: Age of the patient.
  - `sex`: Sex of the patient.
  - `bmi`: Body Mass Index.
  - `bp`: Average blood pressure.
  - `s1`–`s6`: Six blood serum measurements (e.g., total cholesterol, HDL, LDL).
- **Target**: Quantitative measure of diabetes progression (range: 25–346).
- **Preprocessing**: Features are standardized (mean=0, variance=1) using `scaled=True` in `load_diabetes`.

The dataset is loaded in the notebook using:
```python
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(as_frame=True, scaled=True)
X = diabetes.data
y = diabetes.target
```

## Objectives
- **Primary Goal**: Build a predictive model to estimate diabetes progression as a screening tool for physicians.
- **Models**:
  - **Univariate (Part 2)**: Polynomial regression (degrees 0–5) using BMI.
  - **Multivariate (Part 3)**: Polynomial regression (degrees 2 and 3), decision trees (max_depth 3 and 5), kNN (k=5 and k=10).
- **Evaluation Metrics**:
  - R-squared: Measures the proportion of variance explained.
  - MAE: Average absolute difference between predictions and actual values.
  - MAPE: Average absolute percentage error.
- **Dataset Split**:
  - Training: 75% (332 samples)
  - Validation: 10% (44 samples)
  - Test: 15% (66 samples)

## Methodology
The project follows a standard machine learning workflow, implemented in `lab2.ipynb`:

### Part 1: Data Acquisition and Exploration
1. **Data Loading**: Load the dataset as a DataFrame with standardized features.
2. **Problem Framing**: Define the regression task to predict disease progression.
3. **EDA**:
   - Compute descriptive statistics (`X.describe()`).
   - Visualize feature correlations (heatmap), target distribution (histogram), and BMI vs. progression (scatter plot).
   - Insights: BMI and s5 are strongly correlated with progression; no redundant features.
4. **Data Cleaning**: No cleaning needed (no missing values, standardized features).
5. **Data Splitting**: Split into train (75%), validation (10%), and test (15%) sets using `train_test_split`.

### Part 2: Univariate Polynomial Regression
- **Models**: Train polynomial regression models (degrees 0–5) on BMI vs. progression.
- **Comparison**: Evaluate models on training and validation sets using R-squared, MAE, and MAPE. Summarize in a table.
- **Best Model**: Degree 1 (linear) model, selected for highest validation R-squared and lowest MAE/MAPE.
- **Test Results**: Report R-squared, MAE, and MAPE on the test set.
- **Visualization**: Plot train, validation, and test data with the degree 1 model fit.
- **Equation**: Extract the linear equation (e.g., `y = 975.28 * bmi + 152.08`).
- **Prediction**: Predict progression for a specific BMI (e.g., BMI=0.05).
- **Parameters**: Calculate trainable parameters (e.g., degree 1: 3 parameters).
- **Conclusion**: Discuss model performance, failure modes (e.g., high variance), and limitations (e.g., single feature).

### Part 3: Multivariate Models
- **Features**: Use all 10 features (no features dropped, justified by EDA).
- **Models**:
  - Polynomial regression: Degrees 2 and 3.
  - Decision trees: max_depth 3 and 5.
  - kNN: k=5 and k=10.
- **Comparison**: Evaluate models on train, validation, and test sets using R-squared, MAE, and MAPE. Summarize in a table.
- **Best Model**: kNN (k=5), selected for highest test R-squared (~0.4765) and lowest MAE (~41.7194).
- **Test Results**: Report metrics for all models.
- **Equation**: Note that kNN has no explicit equation; provide polynomial equation structure for completeness.
- **Prediction**: Predict progression for a sample input vector (e.g., mean feature values).
- **Parameters**: Calculate parameters (e.g., polynomial degree 2: 67, kNN: 0).
- **Conclusion**: Discuss performance, failure modes (e.g., polynomial overfitting), and limitations (e.g., small dataset).