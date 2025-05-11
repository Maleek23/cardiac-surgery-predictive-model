# %% [markdown]
# # Predictive Model for Clinical Excellence in Cardiac Surgery
# 
# ## Overview
# 
# This project aims to develop a predictive model to estimate the likelihood of operative mortality and postoperative complications in cardiac surgery. By leveraging the Society of Thoracic Surgeons (STS) dataset, we intend to provide healthcare professionals with valuable predictive insights that will improve patient care and surgical decision-making.
# 
# ### Project Objectives:
# 
# 1. **Predictive Analytics**: Accurately predict the likelihood of operative mortality and complications, such as renal failure, stroke, and prolonged ventilation.
# 2. **Data-Driven Decision-Making**: Use predictions to enhance surgical decision-making and improve clinical outcomes.
# 
# This project uses advanced machine learning techniques, specifically **Gradient Boosting**, to develop reliable predictive models.
# 

# %% [markdown]
# ## 1. Import Libraries and Load Data
# 
# **Description**: The first step is to import the necessary libraries for data manipulation, visualization, and machine learning. We will also load the dataset to start our analysis.
# 
# - **Pandas and NumPy** are used for data manipulation.
# - **Matplotlib and Seaborn** are used for visualization to better understand our dataset.
# - **Scikit-Learn** is used to build, train, and evaluate our predictive model.
# 
# Let's begin by importing the required libraries and loading the dataset.
# 

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Aesthetic styling for visuals
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Load the dataset
file_path = 'ModelData_2425.xlsx'  # Adjust path accordingly
data = pd.read_excel(file_path)

# Displaying the first 5 rows of data to understand its structure
data.head()


# %% [markdown]
# ## 2. Data Cleaning and Preprocessing
# 
# **Description**: This step focuses on cleaning and preprocessing the data to ensure it is suitable for modeling.
# 
# ### Data Cleaning Steps:
# 
# 1. Remove missing values to ensure the model is trained on complete data.
# 2. Convert categorical data (`COMPLICATION`) into numerical values using encoding.
# 3. Generate summary statistics to understand the distribution of features.
# 
# Preprocessing is crucial for improving model performance and avoiding issues during training.
# 

# %%
# Data cleaning - Drop rows with missing values
data_cleaned = data.dropna().copy()

# Encode categorical variables to numerical data
if 'COMPLICATION' in data_cleaned.columns:
    data_cleaned['complication_encoded'] = data_cleaned['COMPLICATION'].astype('category').cat.codes

# List of numerical features for analysis
numerical_features = [
    'MORBIDITY & MORTALITY', 'STROKE', 'RENAL FAILURE', 
    'REOPERATION', 'PROLONGED VENTILATION', 'LONG HOSPITAL STAY', 'SHORT HOSPITAL STAY'
]

# Display cleaned data summary statistics
data_cleaned.describe()


# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)
# 
# ### 3.1 Pairwise Relationships
# 
# **Description**: Visualize relationships between different features and the target variable to assess correlations.
# 
# - Pairwise plots help us understand the relationships between the numerical features and the outcome (`OPERATIVE MORTALITY`).
# - This is important for identifying potential predictive relationships in our data.
# 

# %%
# Pairwise relationships visualization for numerical features
sns.pairplot(data_cleaned[numerical_features + ['OPERATIVE MORTALITY']])
plt.suptitle('Pairwise Plot of Numerical Features and Operative Mortality', y=1.02)
plt.show()


# %% [markdown]
# ### 3.2 Correlation Heatmap
# 
# **Description**: Create a heatmap to visualize correlations between features and the target variable.
# 
# - The heatmap helps us identify strongly correlated features, which can assist in selecting the most relevant features for our model.
# - Features with high correlations might lead to redundancy or overfitting, which needs to be avoided.
# 
# Let's now plot the heatmap.
# 

# %%
# Selecting only numeric columns for the correlation heatmap
numeric_data = data_cleaned.select_dtypes(include=[np.number])

# Creating the correlation heatmap with only numeric data
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_data.corr(), annot=True, cmap='viridis', linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# %% [markdown]
# ## 4. Feature Selection and Train-Test Split
# 
# **Description**: Define the features and target variable, and split the dataset into training and testing sets.
# 
# - Selecting key features ensures the model uses only relevant information, improving efficiency and reducing computation time.
# - Splitting the dataset into training and testing subsets is essential to evaluate the model's ability to generalize to unseen data.
# 

# %%
# Define features and target variable
features = [
    'complication_encoded', 'MORBIDITY & MORTALITY', 'STROKE', 
    'RENAL FAILURE', 'REOPERATION', 'PROLONGED VENTILATION', 
    'LONG HOSPITAL STAY', 'SHORT HOSPITAL STAY'
]
target = 'OPERATIVE MORTALITY'

# Split data into training and testing sets
X = data_cleaned[features]
y = data_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# ## 5. Model Building and Hyperparameter Tuning
# 
# **Description**: We will build a **Gradient Boosting Regressor** and use **Randomized Search** to optimize hyperparameters.
# 
# - **Gradient Boosting** is a powerful technique that combines weak models to create a strong learner.
# - **Hyperparameter tuning** is crucial to improve model performance by identifying the best values for parameters like `learning_rate` and `n_estimators`.
# 

# %%
# Hyperparameter tuning for Gradient Boosting Regressor
param_dist_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}

gb_random_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42), 
    param_distributions=param_dist_gb, 
    n_iter=20, 
    cv=5, 
    scoring='r2', 
    verbose=2, 
    n_jobs=-1,
    random_state=42
)
gb_random_search.fit(X_train, y_train)

# Use the best estimator from Randomized Search
best_gb_model = gb_random_search.best_estimator_

# Predict using the best model
y_pred_gb = best_gb_model.predict(X_test)

# Evaluation metrics
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f'Best Parameters for Gradient Boosting: {gb_random_search.best_params_}')
print(f'Mean Squared Error (MSE) for Gradient Boosting: {mse_gb:.2f}')
print(f'R-squared (R2) Score for Gradient Boosting: {r2_gb:.2f}')


# %% [markdown]
# ## 6. Visualizing Model Performance
# 
# ### 6.1 Actual vs Predicted Plot
# 
# **Description**: Visualize the relationship between actual and predicted values to evaluate model accuracy.
# 
# - An **Actual vs Predicted Plot** helps in understanding how well the model's predictions align with actual outcomes.
# - Ideally, points should fall along the diagonal line, indicating accurate predictions.
# 

# %%
from sklearn.metrics import mean_squared_error, r2_score

# Making predictions with the model
y_pred = best_gb_model.predict(X_test)

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, ci=None, line_kws={"color": "red"}, scatter_kws={'alpha': 0.5, 's': 50})
plt.xlabel('Actual Operative Mortality')
plt.ylabel('Predicted Operative Mortality')
plt.title('Actual vs Predicted Operative Mortality for Gradient Boosting Model')
plt.show()

# Metrics summary
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")


# %%
# Feature importance from the best gradient boosting model
importances = best_gb_model.feature_importances_
feature_names = X.columns

# Creating a DataFrame to summarize feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance in Predicting Operative Mortality')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# %%
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(best_gb_model, X, y, cv=5, scoring='r2')
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()


# %% [markdown]
# ### 6.2 Residual Plot
# 
# **Description**: Plot residuals to evaluate any bias in the predictions.
# 
# - **Residuals** represent the difference between actual and predicted values.
# - A symmetric residual plot centered around zero indicates minimal bias in the model.
# 

# %%
# Calculating residuals
residuals = y_test - y_pred

# Plotting residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution for Gradient Boosting Model')
plt.show()


# %% [markdown]
# ## 7. Save the Final Model
# 
# **Description**: Save the trained Gradient Boosting model for future use or deployment.
# 
# - Saving the model ensures that it can be used again without retraining, saving time and computational resources.
# 

# %%
# Save the best Gradient Boosting model to a file
joblib.dump(best_gb_model, 'best_gradient_boosting_model.pkl')

# Load the model for future predictions
loaded_gb_model = joblib.load('best_gradient_boosting_model.pkl')
future_predictions = loaded_gb_model.predict(X_test)


# %% [markdown]
# ## 8. Summary and Insights
# 
# After hyperparameter tuning using **Randomized Search**, the **Gradient Boosting Regressor** achieved:
# 
# - **Mean Squared Error (MSE)**: 0.04
# - **R-squared (R²) Score**: 0.78
# 
# ### Key Insights:
# - The **Gradient Boosting model** provided good predictive power with minimal error, making it suitable for predicting cardiac surgery outcomes.
# - The **Residual Plot** indicates minimal bias, suggesting that the model’s predictions are generally accurate.
# 
# ### Next Steps:
# 1. **Further Tuning**: Consider using **Grid Search** for finer hyperparameter tuning.
# 2. **Ensemble Learning**: Combine with other models like **Random Forest** to enhance generalizability.
# 3. **Deployment**: Deploy the model in a clinical setting and validate its performance with real-time data.
# 

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Assuming X_test and y_test are already defined

# Predicting using the best model
y_pred = best_gb_model.predict(X_test)

# Calculating Mean Squared Error (MSE) and R-squared (R²) score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the findings for MSE and R² score
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

# Analysis for the feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_gb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nKey Features Influencing Operative Mortality:")
print(feature_importances)

# Creating a simplified findings dictionary for easier analysis
findings = {
    "Predictive Accuracy": {
        "MSE": mse,
        "R-squared Score": r2
    },
    "Top Risk Factors": feature_importances['Feature'].head(3).values.tolist()
}

# Displaying the simplified findings
print("\n=== Findings Based on Predictive Model ===")
print(f"1. Predictive Accuracy: \n   - Mean Squared Error (MSE): {mse:.2f}\n   - R-squared Score: {r2:.2f}")
print(f"\n2. Top 3 Key Risk Factors Affecting Operative Mortality: \n   - {findings['Top Risk Factors'][0]}\n   - {findings['Top Risk Factors'][1]}\n   - {findings['Top Risk Factors'][2]}\n")

# Creating recommendations based on the findings
recommendations = """
Recommendations:
1. **Focus on Key Risk Factors**: Based on feature importance, efforts to reduce risks associated with **Renal Failure**, **Prolonged Ventilation**, and **Short Hospital Stay** are likely to have the highest impact in reducing operative mortality.
2. **Improve Data Collection**: Increasing the diversity and amount of data used will likely improve model performance and generalizability.
3. **Periodic Model Updates**: Retraining the model regularly with new patient data will ensure that predictions remain accurate over time.
"""

print(recommendations)



