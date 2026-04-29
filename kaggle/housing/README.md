## Introduction
With the goal of a career shift, I've decided to tackle this [Kaggle Competition House Prices Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competence as a project intended to get hands-on experience in Machine Learning.
This README contains an overview of the whole process, issues faced and lessons learned.

Note: Within this document, I'll refer to Kaggle evaluation as "Production", because I worked on it as a real project.


## Project
Train a model to predict House Prices for unknown data using the Ames dataset, which contains tabular data about houses and their respective sale prices (labels).
There is also a test set without labels, which is used to generate predictions to upload to production.

## Key challenge
The main challenge here is to make the model generalize. That means, performing well in CV and also in unseen data or production.

## Summary
1. Initial Exploratory Data Analysis (EDA)
- Verify missing and invalid values on features to decide the best approach to fill them (some fixed value, mode, mean, etc).
- Know target distribution (Real and Log1p) to spot outliers and start thinking about type of model selection.
- Analyze data structure by looking for correlations and feature distribution to detect candidates to apply log1p, sqrt or more powerful transformations.
- Select important features to avoid the ones having no signal.

2. Robust and reusable pipeline creation focused on:
- Fix invalid and null values.
- Remove outliers.
- Transform skewed distributions detected in EDA, mainly focused on Linear Models performance.
- Handle imbalanced categorical distributions, to avoid break performance models by unique or scattered non representative values.
- Engineer feature interactions to increase predictive signal by combining related features.
- Avoid data leakage in train and CV strategies by ensuring that transformers and models are always fitted using only training data.
- Train, tune and compare different model performances to select the best candidate for final estimations.

3. Model performance analysis
- Inspect prediction residuals to detect possible outliers causing poor generalization in production.
- Compare training and unseen data distribution to fix observed model generalization issues using tools like XGBClassifier, Kolmogorov-Smirnov (KS)test, category distribution analysis, etc.
- Testing CV folding strategies including grouping, repeating and simple randomization to find the best way to approximate the distribution to production.

4. Model tuning - Performance comparison
- Linear Regression, Ridge, Lasso, ElasticNet, XGBRegressor, model stacking.
- Tuned models reducing complexity and strengthening regularization values to avoid overtting and improve generalization.
- Using GridSearch for models with small numbers of hyperparameters and Random search on the other case to optimize the searching process.

4. Iterate the whole process after seeing results in production.

## Key takeaways
- A model performing well in CV and local test set doesn't guarantee a good generalization. This "unexpected" behavior can be related to several issues, like missing or invalid data in production, data leakage, imbalanced categories, unstable features, difference between test / prod data distribution and others.
- Choosing the right CV strategy is very important to improve model generalization.
- Creating a robust pipeline is crucial to get reproducible results and to avoid data leakage on training.
- Engineered features can help but they also degrade performance badly.
- Extreme outliers can change the model behavior dramatically.
- Inspecting residuals from a simple model can be useful to detect data that degrades performance.

## Project Structure
In a production project I would organize most of the code in libraries, to make the code clearer and easy to reuse. But, for this particular case I considered that having many files would make the navigation harder. For that reason, I've included everything in one file and I tried to create an easy to understand structure.

### Imports
Grouping imports to handle them easily.

EDA
Covers some Exploratory Data Analysis and some preliminary functions to fix data issues needed at this point.

### Preprocessing 
Contains the custom transformers used in the pipeline to process data correctly avoiding leakage, even in CV process.

### Experiment
Final pipeline creation, including preprocessing transformer and model. Here, some common functions are run to evaluate and store results to compare models performance through the whole process.

### Analysis
Focused in find the best CV strategy that make CV/test and production score similar, log1p over engineered features, residuals study, etc.

### Modeling
Model tuning for Linear Regression, Ridge, Lasso, ElasticNet and XGBRegressor.

### Submission
Generic submission process to easily generate the predictions file to upload to production.

