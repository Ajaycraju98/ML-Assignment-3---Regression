# ML-Assignment-3---Regression
This  Assignment explains about the regressions in machine learning
# Key Components to be Fulfilled:
# 1.Loading and Preprocessing :

*Load the California Housing dataset using the fetch_california_housing function from sklearn.
*Convert the dataset into a pandas DataFrame for easier handling.
*Handle missing values (if any) and perform necessary feature scaling (e.g., standardization).
*Explain the preprocessing steps you performed and justify why they are necessary for this dataset.

import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression

# Load the California housing dataset
data = fetch_california_housing()

# Convert to Pandas DataFrame
#The dataset was converted to a DataFrame for easier handling and manipulation.
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
df

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each colum:\n", missing_values)

#No missing values were found, so no imputation was required. 
#If missing values existed, we could have used methods like mean/mode imputation or dropped them based on severity.

df.duplicated().sum()
# In the dataset,there is no duplicate values

# Differentiating Columns
#To understand the categorical column and numerical columns 
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Print Results
print("Categorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)

# In the dataset,there is no categorical column.so need not implementing Encoding categorical data

# Correlation with target variable
#To understand the correlation of columns with respect to Target column 
print(df.corr()['Target'].sort_values(ascending=False))

#To find the best features from the dataset using Kbest feture selection
from sklearn.feature_selection import SelectKBest, f_classif
# SelectKBest for feature selection
X = df.drop(columns=['Target'])  # Features
y = df['Target']  # Target

select_k = SelectKBest(score_func=f_classif, k=4)  # Selecting Top 1 feature, depends on the person
X_selected = select_k.fit_transform(X, y)

# Get the names and scores of the selected features
selected_features = X.columns[select_k.get_support()]
selected_scores = select_k.scores_[select_k.get_support()] # to find scores of all features

print("Selected Features:", selected_features)
print("Feature Scores based on select_k:", selected_scores)


# Create a DataFrame to display feature names and scores
feature_scores_df = pd.DataFrame({'Feature': selected_features, 'Score': selected_scores})


# Sort by scores in ascending order
feature_scores_df = feature_scores_df.sort_values(by="Score", ascending=False)

# Print results
print("Selected Features:\n", feature_scores_df)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Data (Features):")
print(X_train)
print("\nTesting Data (Features):")
print(X_test)

#This step is crucial to evaluate the model’s performance on unseen data and prevent overfitting.


X_train
X_train.shape
y_train

X_test
y_test
# Display the first few rows of the scaled DataFrame
print(df.head())

# Feature Scaling

#Standardization ensures all features have a mean of 0 and a standard deviation of 1, making them comparable.
# Scaling using StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

X_train_scaled
print(len(X_train_scaled[0]))
X_train_scaled.shape
X_train.shape

from sklearn.preprocessing import StandardScaler

y_train_df = pd.DataFrame(y_train) #converting to data frame from series
scaler = StandardScaler()
scaler.fit(y_train_df)
y_train_scaled = scaler.transform(y_train_df)
y_train_scaled

y_test

# 2.Regression Algorithm Implementation :
Implement the following regression algorithms:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Support Vector Regressor (SVR)
For each algorithm:
Provide a brief explanation of how it works.
Explain why it might be suitable for this dataset.

# Linear regression
Linear Regression models the relationship between independent variables (features) and the dependent variable (Target) using a linear equation
In the dataset,relationship between features and the target is approximately linear, Linear Regression can perform well.

# Model training using Linear regression

model = LinearRegression()
# Train the model using the dataset
model.fit(X_train_scaled, y_train_scaled)

from sklearn.preprocessing import StandardScaler

y_train_df = pd.DataFrame(y_train) #converting to data frame from series
scaler = StandardScaler()
scaler.fit(y_train_df)
y_train_scaled = scaler.transform(y_train_df)

# Scaling using StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)

from sklearn.preprocessing import StandardScaler

y_test_df = pd.DataFrame(y_test) #converting to data frame from series
scaler = StandardScaler()
scaler.fit(y_test_df)
y_test_scaled = scaler.transform(y_test_df)

# DecisionTreeRegressor
A Decision Tree splits the dataset into branches based on feature values, using a tree-like structure.
Decision Trees can capture non-linear relationships better than Linear Regression

# 3.Model Evaluation and Comparison:
Evaluate the performance of each algorithm using the following metrics:
Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R-squared Score (R²)
Compare the results of all models and identify:
The best-performing algorithm with justification.
The worst-performing algorithm with reasoning.

# multiple ML models
# Initialize models
models = {
    "LinearRegression":LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor()
}

# Train and evaluate models
model_results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_scaled)
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test_scaled, predictions)
    mse = mean_squared_error(y_test_scaled, predictions)
    rmse = mean_squared_error(y_test_scaled, predictions, squared=False)
    r2 = r2_score(y_test_scaled, predictions)
    model_results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}

    # Display model performance
for name, metrics in model_results.items():
    print(f"\n{name} Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Cross validation
from sklearn.model_selection import cross_val_score, KFold

# Initialize Linear Regression model
model2 = LinearRegression()

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 folds

# Perform cross-validation and get scores (default scoring is R^2)
scores = cross_val_score(model2, X_train_scaled, y_train_scaled, cv=kf, scoring='r2')

print("Cross-Validation R² Scores:", scores)
print("Mean R² Score:", np.mean(scores))

# If Mean Squared Error (MSE) as the evaluation metric
mse_scores = -cross_val_score(model2, X_train_scaled, y_train_scaled, cv=kf, scoring='neg_mean_squared_error')
print("Cross-Validation MSE Scores:", mse_scores)
print("Mean MSE Score:", np.mean(mse_scores))

# Train and evaluate models
model_results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_scaled)
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test_scaled, predictions)
    mse = mean_squared_error(y_test_scaled, predictions)
    rmse = mean_squared_error(y_test_scaled, predictions, squared=False)
    r2 = r2_score(y_test_scaled, predictions)
    model_results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}
# Display model performance
for name, metrics in model_results.items():
    print(f"\n{name} Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Identify best and worst models
best_model = max(model_results, key=lambda x: model_results[x]["R2 Score"])
worst_model = min(model_results, key=lambda x: model_results[x]["R2 Score"])

print("Best Model:", best_model, model_results[best_model])
print("Worst Model:", worst_model, model_results[worst_model])

The dataset trains and evaluates Linear Regression and Decision Tree Regressor  models using metrics like MSE, MAE, RMSE, and R².
The best model based on the highest R² Score and the worst model based on the lowest R² Score.

Linear Regression performed best because it provides the lowest error metrics (MSE, RMSE, MAE) and the highest R² score, 
indicating that it explains a significant portion of the variance in the target variable.
Since the California housing dataset has a continuous target variable and linear relationships between some features and the target, Linear Regression is a suitable choice.

Decision Tree Regressor performed the worst due to overfitting. 
Decision trees tend to capture noise and work well for small datasets but fail on larger datasets where relationships are more complex. 
The high error values suggest poor generalization.

--The End--
