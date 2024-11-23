### **Linear Regression: An Overview**

Linear regression is a fundamental and widely used algorithm for predicting a continuous target variable based on one or more input features. It assumes a linear relationship between the dependent variable (target) and independent variables (features).

### **Linear Regression Formula**
For a single feature:
\[
y = \beta_0 + \beta_1 \cdot X + \epsilon
\]
Where:
- \( y \) = Target variable (dependent variable).
- \( X \) = Independent variable (feature).
- \( \beta_0 \) = Intercept (constant).
- \( \beta_1 \) = Slope (coefficient).
- \( \epsilon \) = Error term.

For multiple features:
\[
y = \beta_0 + \beta_1 \cdot X_1 + \beta_2 \cdot X_2 + \dots + \beta_n \cdot X_n + \epsilon
\]

### **Types of Linear Regression**
There are several variations of linear regression based on the number of features and the form of regularization used:

1. **Simple Linear Regression**: 
   - Involves one independent variable (feature).
   - Used to predict the target based on a single feature.

2. **Multiple Linear Regression**:
   - Involves multiple independent variables (features).
   - Used to predict the target based on multiple features.
   - The model equation is an extension of the simple linear regression equation.

3. **Ridge Regression (L2 Regularization)**:
   - A variant of linear regression where a penalty term (L2) is added to the cost function.
   - Helps prevent overfitting by shrinking the coefficients.

4. **Lasso Regression (L1 Regularization)**:
   - Similar to Ridge, but with L1 regularization.
   - Helps in feature selection by driving some coefficients to zero.

5. **ElasticNet**:
   - Combines L1 and L2 regularization techniques.
   - Balances both penalties to improve performance.

---

### **Sample Code for Linear Regression (Simple Linear and Multiple Linear)**

#### **1. Simple Linear Regression**
Here, we predict a single target variable based on one feature.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data (X: Feature, y: Target variable)
X = np.array([[1], [2], [3], [4], [5]])  # Feature
y = np.array([1, 2, 3, 4, 5])            # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualizing the results
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, model.predict(X), color='red', label='Fitted Line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()
```

#### **2. Multiple Linear Regression**
Here, we predict a target variable based on multiple features.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data (X: Multiple features, y: Target variable)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # Features
y = np.array([1, 2, 3, 4, 5])                            # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualizing the results (for the first feature vs. target)
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], y, color='blue', label='Original Data')  # Scatter plot for the first feature
plt.plot(X[:, 0], model.predict(X), color='red', label='Fitted Line')  # Fitted line
plt.xlabel('First Feature')
plt.ylabel('Target')
plt.legend()
plt.title('Multiple Linear Regression')
plt.show()
```

---

### **Explanation of the Code**

1. **Data Preparation**:
   - `X` is the input feature(s), and `y` is the target variable.
   - The dataset is split into training and testing sets using `train_test_split` to evaluate model performance.
   
2. **Model Training**:
   - A `LinearRegression` object is created.
   - `model.fit(X_train, y_train)` trains the model using the training data.

3. **Prediction**:
   - After training, the model is used to predict target values on the test data using `model.predict(X_test)`.

4. **Evaluation**:
   - The **Mean Squared Error (MSE)** is calculated to assess the model's performance, where a lower MSE indicates a better fit.

5. **Visualization**:
   - The original data points and the regression line are plotted to visualize how well the model fits the data.

---

### **Types of Linear Regression: Summary**

| Type                      | Use Case                           | Key Feature                           |
|---------------------------|------------------------------------|----------------------------------------|
| **Simple Linear Regression** | One feature, one target           | One independent variable               |
| **Multiple Linear Regression** | Multiple features, one target    | Multiple independent variables         |
| **Ridge Regression**        | Reduces overfitting               | L2 regularization (penalty term)      |
| **Lasso Regression**        | Feature selection, reduces overfitting | L1 regularization (sparsity)          |
| **ElasticNet**              | Combination of Ridge and Lasso     | Combines L1 and L2 regularization     |

Linear regression provides an excellent baseline for many regression tasks and is simple to implement with `sklearn`. The ability to include regularization techniques like **Ridge** and **Lasso** helps prevent overfitting when working with complex datasets.


### **Evaluation Metrics for Regression Models**

When assessing the performance of a regression model, we use various metrics to understand how well the model fits the data. Below are some of the most commonly used evaluation metrics:

---

### **1. Mean Absolute Error (MAE)**

#### **Definition**
- The **Mean Absolute Error (MAE)** is the average of the absolute differences between the predicted and actual values. It gives us an idea of how far off the predictions are, on average.
- **Formula**:
\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]
Where:
- \(y_i\) = Actual values
- \(\hat{y}_i\) = Predicted values
- \(n\) = Number of data points

#### **Interpretation**
- MAE is easy to understand and interpret, but it does not penalize larger errors as much as other metrics.
- Lower MAE indicates better model performance.

#### **Python Code**:
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
```

---

### **2. Mean Squared Error (MSE)**

#### **Definition**
- **Mean Squared Error (MSE)** is the average of the squared differences between the predicted and actual values. It is more sensitive to outliers than MAE because the errors are squared.
- **Formula**:
\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

#### **Interpretation**
- MSE penalizes larger errors more heavily due to squaring the residuals.
- A lower MSE indicates better performance, but the scale of MSE is dependent on the units of the target variable.

#### **Python Code**:
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

---

### **3. Root Mean Squared Error (RMSE)**

#### **Definition**
- **Root Mean Squared Error (RMSE)** is the square root of the mean squared error (MSE). It brings the error back to the same unit as the target variable, making it easier to interpret.
- **Formula**:
\[
RMSE = \sqrt{MSE}
\]

#### **Interpretation**
- RMSE is more sensitive to large errors than MAE.
- A lower RMSE indicates a better model fit, and it is easier to interpret because it is in the same units as the target variable.

#### **Python Code**:
```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")
```

---

### **4. R-squared (\(R^2\) Score)**

#### **Definition**
- **R-squared** is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
- **Formula**:
\[
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
\]
Where:
- \(y_i\) = Actual values
- \(\hat{y}_i\) = Predicted values
- \(\bar{y}\) = Mean of actual values

#### **Interpretation**
- \(R^2\) ranges from 0 to 1, where:
  - \( R^2 = 1 \) means the model perfectly fits the data.
  - \( R^2 = 0 \) means the model does not explain the variance in the target variable.
- Negative values of \(R^2\) indicate that the model performs worse than a simple mean-based model.

#### **Python Code**:
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")
```

---

### **5. Adjusted R-squared (Adj \(R^2\))**

#### **Definition**
- **Adjusted R-squared** adjusts the \(R^2\) score based on the number of features in the model. It penalizes the model for adding irrelevant features, making it a better metric for multiple linear regression.
- **Formula**:
\[
Adj R^2 = 1 - \left(1 - R^2\right) \cdot \frac{n - 1}{n - p - 1}
\]
Where:
- \(n\) = Number of data points
- \(p\) = Number of independent variables (features)

#### **Interpretation**
- Unlike \(R^2\), the adjusted \(R^2\) decreases when irrelevant features are added to the model.
- It is useful for comparing models with a different number of predictors.

#### **Python Code**:
```python
# Number of data points (n) and features (p)
n = len(y_test)
p = X_test.shape[1]

# Calculate Adjusted R-squared
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Adjusted R-squared: {adj_r2}")
```

---

### **Summary of Metrics**

| Metric                | Formula                                            | Interpretation                                    |
|-----------------------|----------------------------------------------------|--------------------------------------------------|
| **MAE**               | \( \frac{1}{n} \sum |y_i - \hat{y}_i| \)         | Average of absolute errors (in same units as target) |
| **MSE**               | \( \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \)         | Penalizes larger errors (squared units of target) |
| **RMSE**              | \( \sqrt{MSE} \)                                   | Square root of MSE (in same units as target)     |
| **R-squared (\(R^2\))** | \( 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} \) | Proportion of variance explained (0 to 1)       |
| **Adjusted \(R^2\)**   | \( 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1} \) | Adjusted for number of features (penalizes excess features) |

---

These metrics help you evaluate and compare regression models, and selecting the right one depends on your specific goals and the type of model you're using.