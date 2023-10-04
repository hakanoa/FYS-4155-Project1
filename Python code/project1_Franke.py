import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def FrankeFunction(x, y):
    """
    Generate the Franke Function model with added noise.

    Parameters:
    x (numpy.ndarray): Array of x-coordinates.
    y (numpy.ndarray): Array of y-coordinates.

    Returns:
    numpy.ndarray: Array of data points.
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, 0.2)

def create_X(x, y, n):
    """
    Create the design matrix for 2D polynomial regression.

    Parameters:
    x (numpy.ndarray): Array of x-coordinates.
    y (numpy.ndarray): Array of y-coordinates.
    n (int): The polynomial degree.

    Returns:
    numpy.ndarray: The design matrix.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)  # Number of data points
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))  # Initialize the design matrix with ones

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)  # Number of elements for current degree
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y ** k)

    return X

def fit_beta(X, y):
    """
    Fit a linear regression model and estimate coefficients.

    Parameters:
    X (numpy.ndarray): The design matrix.
    y (numpy.ndarray): Target values.

    Returns:
    numpy.ndarray: Estimated coefficients.
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def R2(y_data, y_model):
    """
    Calculate the R-squared value for a model's predictions.

    Parameters:
    y_data (numpy.ndarray): Actual target values.
    y_model (numpy.ndarray): Predicted values.

    Returns:
    float: R-squared value.
    """
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data, y_model):
    """
    Calculate the Mean Squared Error (MSE) for a model's predictions.

    Parameters:
    y_data (numpy.ndarray): Actual target values.
    y_model (numpy.ndarray): Predicted values.

    Returns:
    float: Mean Squared Error.
    """
    n = np.size(y_model)  # Number of data points
    return np.sum((y_data - y_model) ** 2) / n

def OLS(n):
    """
    Perform Ordinary Least Squares (OLS) regression.

    Parameters:
    n (int): Polynomial degree.

    Returns:
    tuple: A tuple containing MSE_train, MSE_test, R2_train, R2_test, and beta.
    """
    x = np.arange(0, 1, 0.05)  # Array of x-coordinates
    y = np.arange(0, 1, 0.05)  # Array of y-coordinates
    z = FrankeFunction(x, y)  # Generate data points using FrankeFunction
    X = create_X(x, y, n=n)  # Create the design matrix for given degree

    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2)  # Split data into train and test sets
    scaler = StandardScaler(with_std=False).fit(X_train)  # Initialize the StandardScaler
    X_train = scaler.transform(X_train)  # Transform the training data
    X_test = scaler.transform(X_test)  # Transform the testing data
    
    beta = fit_beta(X_train, y_train)  # Fit the model and estimate beta coefficients
    
    ytilde = X_train @ beta  # Predictions on the training data
    MSE_train = MSE(y_train, ytilde)  # Calculate MSE on training data
    R2_train = R2(y_train, ytilde)  # Calculate R-squared on training data

    ypredict = X_test @ beta  # Predictions on the test data
    R2_test = R2(y_test, ypredict)  # Calculate R-squared on test data
    MSE_test = MSE(y_test, ypredict)  # Calculate MSE on test data
    
    return MSE_train, MSE_test, R2_train, R2_test, beta

def Ridge(n, nlambdas, lambdas):
    """
    Perform Ridge regression.

    Parameters:
    n (int): Polynomial degree.
    nlambdas (int): Number of lambda values to test.
    lambdas (numpy.ndarray): Array of lambda values.

    Returns:
    tuple: A tuple containing MSE_train, MSE_test, R2_train, R2_test, and beta_ridge.
    """
    x = np.arange(0, 1, 0.05)  
    y = np.arange(0, 1, 0.05)  
    z = FrankeFunction(x, y)  
    X = create_X(x, y, n=n)  

    MSE_train = np.zeros(nlambdas) 
    MSE_test = np.zeros(nlambdas) 
    R2_train = np.zeros(nlambdas) 
    R2_test = np.zeros(nlambdas)  
    beta_ridge = np.zeros((nlambdas, X.shape[1]))  # Array to store beta values for different lambdas

    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2)  
    scaler = StandardScaler(with_std=False).fit(X_train)  
    X_train = scaler.transform(X_train) 
    X_test = scaler.transform(X_test) 
    I = np.eye(X_train.shape[1])  # Identity matrix

    for i in range(nlambdas):
        lmb = lambdas[i]  # Current lambda value
        beta_ridge[i, :] = np.linalg.pinv(X_train.T @ X_train + lmb * I) @ X_train.T @ y_train  # Ridge regression
        ytilde_ridge = X_train @ beta_ridge[i, :]  # Predictions on the training data
        ypredict_ridge = X_test @ beta_ridge[i, :]  # Predictions on the test data

        R2_train[i] = R2(y_train, ytilde_ridge)  # Calculate R-squared on training data
        MSE_train[i] = MSE(y_train, ytilde_ridge)  # Calculate MSE on training data
        R2_test[i] = R2(y_test, ypredict_ridge)  # Calculate R-squared on test data
        MSE_test[i] = MSE(y_test, ypredict_ridge)  # Calculate MSE on test data

    return MSE_train, MSE_test, R2_train, R2_test, beta_ridge

def Lasso(n, nlambdas, lambdas):
    """
    Perform Lasso regression.

    Parameters:
    n (int): Polynomial degree.
    nlambdas (int): Number of lambda values to test.
    lambdas (numpy.ndarray): Array of lambda values.

    Returns:
    tuple: A tuple containing MSE_train, MSE_test, R2_train, R2_test, and beta.
    """
    x = np.arange(0, 1, 0.05)  
    y = np.arange(0, 1, 0.05)  
    z = FrankeFunction(x, y)  
    X = create_X(x, y, n=n)  
    lambdas = np.logspace(-3, 2, nlambdas)  # Array of lambda values

    MSE_train = np.zeros(nlambdas)  
    MSE_test = np.zeros(nlambdas)  
    R2_train = np.zeros(nlambdas)  
    R2_test = np.zeros(nlambdas)  
    beta = np.zeros((nlambdas, X.shape[1])) 

    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2) 
    scaler = StandardScaler(with_std=False).fit(X_train) 
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test) 

    for i in range(nlambdas):
        lmb = lambdas[i]  # Current lambda value
        RegLasso = linear_model.Lasso(alpha=lmb, max_iter=10000, fit_intercept=True)  # Lasso regression model
        RegLasso.fit(X_train, y_train)  # Fit the model
        ytilde = RegLasso.predict(X_train)  # Predictions on the training data
        ypredict = RegLasso.predict(X_test)  # Predictions on the test data
        beta[i, :] = RegLasso.coef_  # Coefficients

        MSE_train[i] = MSE(y_train, ytilde)  # Calculate MSE on training data
        MSE_test[i] = MSE(y_test, ypredict)  # Calculate MSE on test data
        R2_train[i] = R2(y_train, ytilde)  # Calculate R-squared on training data
        R2_test[i] = R2(y_test, ypredict)  # Calculate R-squared on test data

    return MSE_train, MSE_test, R2_train, R2_test, beta

def bias_variance_OLS(degree, N, n_bootstraps, noise=0.1, test_size=0.2):
    """
    Perform OLS regression with bias-variance analysis.

    Parameters:
    degree (int): Polynomial degree.
    N (int): Total data points.
    n_bootstraps (int): Number of bootstrap iterations.
    noise (float): Noise level.
    test_size (float): Fraction of data used for testing.

    Returns:
    tuple: A tuple containing MSE, bias^2, and variance.
    """
    x = np.arange(0, 1, 0.05)  
    y = np.arange(0, 1, 0.05)  
    z = FrankeFunction(x, y)  
    X = create_X(x, y, n=degree)  
    z = FrankeFunction(x, y)
    X = create_X(x, y, n=degree)

    
    x_train, x_test, y_train, y_test = train_test_split(x.ravel(), y.ravel(), test_size=test_size)

    # Initialize lists to store predictions
    y_pred = []

    # Perform bootstrap resampling
    for _ in range(n_bootstraps):
        x_, y_ = resample(x_train, y_train)

        # Fit and predict
        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
        y_pred.append(model.fit(x_.reshape(-1, 1), y_).predict(x_test.reshape(-1, 1)))

    y_pred = np.array(y_pred)

    # Calculate MSE, bias, and variance
    mse = np.mean(np.mean((y_test - y_pred) ** 2, axis=0))
    bias = np.mean((y_test - np.mean(y_pred, axis=0)) ** 2)
    variance = np.mean(np.var(y_pred, axis=0))

    return mse, bias, variance

# Define polynomial degree, lambda values, and other parameters
n = 5  # polynomial degree
n_list = list(range(1, n + 1))  # List of polynomial degrees
nlambdas = 10
lambdas_Ridge = np.logspace(-6, 1, nlambdas)
lambdas_Lasso = np.logspace(-7, 4, nlambdas)
N = 10000  # Total data points

# OLS
mse_train_OLS = np.zeros(n)
mse_test_OLS = np.zeros(n)
r2_train_OLS = np.zeros(n)
r2_test_OLS = np.zeros(n)

# Ridge
mse_train_Ridge = np.zeros((n, nlambdas))
mse_test_Ridge = np.zeros((n, nlambdas))
r2_train_Ridge = np.zeros((n, nlambdas))
r2_test_Ridge = np.zeros((n, nlambdas))

# Lasso
mse_train_Lasso = np.zeros((n, nlambdas))
mse_test_Lasso = np.zeros((n, nlambdas))
r2_train_Lasso = np.zeros((n, nlambdas))
r2_test_Lasso = np.zeros((n, nlambdas))

# Loop over polynomial degrees and compute performance metrics for each method
for i in range(n):
    mse_train_OLS[i], mse_test_OLS[i], r2_train_OLS[i], r2_test_OLS[i], _ = OLS(i)
    mse_train_Ridge[i, :], mse_test_Ridge[i, :], r2_train_Ridge[i, :], r2_test_Ridge[i, :], _ = Ridge(i, nlambdas, lambdas_Ridge)
    mse_train_Lasso[i, :], mse_test_Lasso[i, :], r2_train_Lasso[i, :], r2_test_Lasso[i, :], _ = Lasso(i, nlambdas, lambdas_Lasso)

# OLS plots
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(n_list, mse_train_OLS, label='Train')
plt.plot(n_list, mse_test_OLS, label='Test')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE')
plt.title('MSE OLS')
plt.xticks(n_list)
plt.legend()
plt.subplot(122)
plt.plot(n_list, r2_train_OLS, label='Train')
plt.plot(n_list, r2_test_OLS, label='Test')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('R2')
plt.xticks(n_list)
plt.title('R2 OLS')
plt.legend()
plt.tight_layout()

# Ridge plots MSE
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Ridge regression - MSE train for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE')
plt.xticks(n_list)
for j in range(nlambdas):
    plt.plot(n_list, mse_train_Ridge[:, j], label = f'λ={lambdas_Ridge[j]:.1e}')
plt.subplot(122)
plt.title('Ridge regression - MSE test for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE')
plt.xticks(n_list)
for j in range(nlambdas):
    plt.plot(n_list, mse_test_Ridge[:, j], label = f'λ={lambdas_Ridge[j]:.1e}')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Ridge plots R2
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Ridge regression - R2 train for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('R2')
plt.xticks(n_list)
for j in range(nlambdas):
    plt.plot(n_list, r2_train_Ridge[:, j], label = f'λ={lambdas_Ridge[j]:.1e}')
plt.subplot(122)
plt.title('Ridge regression - R2 test for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('R2')
plt.xticks(n_list)
for j in range(nlambdas):
    plt.plot(n_list, r2_test_Ridge[:, j], label = f'λ={lambdas_Ridge[j]:.1e}')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Lasso plots MSE
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Lasso regression - MSE train for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE')
plt.xticks(n_list)
for j in range(nlambdas):
    plt.plot(n_list, mse_train_Lasso[:, j], label = f'λ={lambdas_Lasso[j]:.1e}')
plt.subplot(122)
plt.title('Lasso regression - MSE test for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE')
plt.xticks(n_list)
for j in range(nlambdas):
    plt.plot(n_list, mse_test_Lasso[:, j], label = f'λ={lambdas_Lasso[j]:.1e}')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Lasso plots R2
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Lasso regression - R2 train for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('R2')
plt.xticks(n_list)
for j in range(nlambdas):
    plt.plot(n_list, r2_train_Lasso[:, j], label = f'λ={lambdas_Lasso[j]:.1e}')
plt.subplot(122)
plt.title('Lasso regression - R2 test for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('R2')
plt.xticks(n_list)
for j in range(nlambdas):
    plt.plot(n_list, r2_test_Lasso[:, j], label = f'λ={lambdas_Lasso[j]:.1e}')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()





# Bias variance trade-off
if __name__ == "__main__":
    degree_range = list(range(1, 5+1))  # List of polynomial degrees from 1 to 5
    n_bootstraps = 100
    N = 500  # Total data points
    
    mse_results = []
    bias_results = []
    variance_results = []

    for degree in degree_range:
        mse, bias, variance = bias_variance_OLS(degree, N, n_bootstraps)
        mse_results.append(mse)
        bias_results.append(bias)
        variance_results.append(variance)

        print(f"Degree: {degree}, MSE: {mse}, Bias^2: {bias}, Variance: {variance}")

    # Plot MSE, Bias^2, and Variance
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.title("MSE as a function of polynomial degree")
    plt.xlabel('Degree')
    plt.ylabel('MSE')
    plt.xticks(degree_range)
    plt.plot(degree_range, mse_results)

    plt.subplot(132)
    plt.title("Bias^2 as a function of polynomial degree")
    plt.xlabel('Degree')
    plt.ylabel('Bias^2')
    plt.xticks(degree_range)
    plt.plot(degree_range, bias_results)

    plt.subplot(133)
    plt.title("Variance as a function of polynomial degree")
    plt.xlabel('Degree')
    plt.ylabel('Variance')
    plt.xticks(degree_range)
    plt.plot(degree_range, variance_results)

    plt.tight_layout()
    plt.show()

