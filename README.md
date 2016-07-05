# Stochastic Gradient Descent for Linear Models
Gradient descent is a first-order optimization algorithm. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point. Stochastic gradient descent (SGD) is a stochastic approximation of the gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions, where each summand function is tipically associated with the i-th example in the data set used for training. To economize on the computational cost at every iteration, stochastic gradient descent samples a subset of summand functions at every step. This is very effective in the case of large-scale machine learning problems.

This package implements the SGD algorithm in Python for the following linear models:
* Linear regression
* Ridge regression
* Logistic regression
* Logistic regression with L2 regularization

Gradient descent is suitable to these models because the error functions to be minimized are convex, so any local minimum is also a global minimum.

## Dependencies
The recommended Python version is 2.7.5 and the NumPy version used for testing is 1.11.0.

## Usage

### Instantiate a linear model
```python
from pysgd import linear_models

# Linear regression
linreg = linear_models.SGDLinearRegression(n_epochs=50, learning_rate=0.01, batch_size=1)

# Ridge regression
ridgereg = linear_models.SGDRidgeRegression(n_epochs=50, learning_rate=0.01, batch_size=1, regularization_strength=0.1)

# Logistic regression
logreg = linear_models.SGDLogisticRegression(n_epochs=50, learning_rate=0.01, batch_size=1)

# Logistic regression with L2 regularization
l2logreg = linear_models.SGDL2RegularizedLogisticRegression(n_epochs=50, learning_rate=0.01, batch_size=1, regularization_strength=0.1)
```
It's also possible to use non-fixed learning rates. Léon Bottou shows in his paper ["Stochastic Gradient Tricks"](https://www.microsoft.com/en-us/research/publication/stochastic-gradient-tricks/) an optimal learning rate for L2-regularized models that is a function of iteration count and regularization strength. Such learning rate can be used by passing a lambda function for the `learning_rate` parameter of the constructors. This function is called every iteration with one argument `t`, where `t` is the iteration counter.
```python
ridgereg = linear_models.SGDRidgeRegression(n_epochs=50, learning_rate=lambda t: gamma0 / (1 + gamma0 * regularization_strength * t), batch_size=1, regularization_strength=regularization_strength)
```

### Train the model
`X` is a NumPy `array` of shape `[n_samples, n_features]` and `y` is a NumPy array of shape `[n_samples]`.
```python
linreg.fit(X_training, y_training)
ridgereg.fit(X_training, y_training)
logreg.fit(X_training, y_training)
l2logreg.fit(X_training, y_training)
```

### Make predictions
The result is a NumPy `array` of shape `[n_samples]`.
```python
linreg_predictions = linreg.predict(X_test)
ridgereg_predictions = ridgereg.predict(X_test)
logreg_predictions = logreg.predict(X_test)
l2logreg_predictions = l2logreg.predict(X_test)
```
