# -*- coding: utf-8 -*-


import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio.v3 import imread
from sklearn import linear_model



def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((15000, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x[:15000] ** (i - k)) * (y[:15000] ** k)

    return X


def fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y



def R2(y_data, y_model):
    """
   Computed R2 score of model.

   Params:
   y_data: Array
       array real data points
   y_model: Array
       array of predicted response by trained model

   Returns:
   R2 score of model
   """
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    """
   Computed MSE of model.

   Params:
   y_data: Array
       array real data points
   y_model: Array
       array of predicted response by trained model

   Returns:
      """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n



def OLS(n):
    """
   Perform Ordinary Least Squares (OLS) regression.

   Parameters:
   n (int): Polynomial order.

   Returns:
   MSE_train (float): Training Mean Squared Error (MSE).
   MSE_test (float): Testing Mean Squared Error (MSE).
   R2_train (float): Training R2 score.
   R2_test (float): Testing R2 score.
   beta (ndarray): Fitted beta parameters.
   """
   
    # Load the terrain image
    terrain = imread('SRTM_data_Norway_1.tif')

    # Ensure that the terrain array has the desired size (N x N)
    N = 10000  # Desired size of the terrain
    size = 15000  # Number of data points to keep
    terrain = terrain[:N, :N]

    # Create a mesh of image pixels
    x = np.linspace(0, 1, terrain.shape[0])
    y = np.linspace(0, 1, terrain.shape[1])
    x_mesh, y_mesh = np.meshgrid(x, y)
    z_mesh = terrain.T  # Transpose for consistent shape

    # Reduce the data to 15000 points
    N = np.array(range(x_mesh.size))
    kept_data = np.random.permutation(N)[:size]
    x_red, y_red, z_red = x_mesh.flatten()[kept_data], y_mesh.flatten()[kept_data], z_mesh.flatten()[kept_data]

    # Create the design matrix X
    X = create_X(x_red, y_red, n)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, z_red, test_size=0.2)

    # Scale the data using StandardScaler
    scaler = StandardScaler(with_std=False).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the beta parameters using the training data
    beta = fit_beta(X_train, y_train)
    
    # Compute the Training R2 and MSE
    ytilde = X_train @ beta
    MSE_train = MSE(y_train, ytilde)
    R2_train = R2(y_train, ytilde)
   
    # Compute the Testing R2 and MSE
    ypredict = X_test @ beta
    R2_test = R2(y_test, ypredict)
    MSE_test = MSE(y_test, ypredict)
    
    return MSE_train, MSE_test, R2_train, R2_test, beta


def Ridge(n, nlambdas, lambdas):
    """
    Perform Ridge regression.

    Parameters:
    n (int): Polynomial order.
    nlambdas (int): Number of lambda values.
    lambdas (ndarray): Array of lambda values.

    Returns:
    MSE_train (ndarray): Training Mean Squared Errors (MSE) for different lambdas.
    MSE_test (ndarray): Testing Mean Squared Errors (MSE) for different lambdas.
    R2_train (ndarray): Training R2 scores for different lambdas.
    R2_test (ndarray): Testing R2 scores for different lambdas.
    beta_ridge (ndarray): Fitted beta parameters for different lambdas.
    """
    

    terrain = imread('SRTM_data_Norway_1.tif')

    # Ensure that the terrain array has the desired size (N x N)
    N = 10000
    size = 15000
    terrain = terrain[:N, :N]

    # Creates mesh of image pixels
    x = np.linspace(0, 1, terrain.shape[0])
    y = np.linspace(0, 1, terrain.shape[1])
    x_mesh, y_mesh = np.meshgrid(x, y)
    z_mesh = terrain.T  # Transpose for consistent shape

    # Reduce to 15000 points
    N = np.array(range(x_mesh.size))
    kept_data = np.random.permutation(N)[:size]
    x_red, y_red, z_red = x_mesh.flatten()[kept_data], y_mesh.flatten()[kept_data], z_mesh.flatten()[kept_data]

    X = create_X(x_red, y_red, n)

    # split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, z_red, test_size=0.2)

    # Scale data
    scaler = StandardScaler(with_std=False).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    MSE_train = np.zeros(nlambdas)
    MSE_test = np.zeros(nlambdas)
    R2_train = np.zeros(nlambdas)
    R2_test = np.zeros(nlambdas)
    beta_ridge = np.zeros((nlambdas, X.shape[1]))

    I = np.eye(X_train.shape[1])  # Adjusted for the shape of the identity matrix
    for i in range(nlambdas):
        lmb = lambdas[i]
        beta_ridge[i, :] = np.linalg.pinv(X_train.T @ X_train + lmb * I) @ X_train.T @ (y_train)
        ytilde_ridge = X_train @ beta_ridge[i, :]
        ypredict_ridge = X_test @ beta_ridge[i, :]

        # Training R2 and MSE
        R2_train[i] = R2(y_train, ytilde_ridge)
        MSE_train[i] = MSE(y_train, ytilde_ridge)

        # Testing R2 and MSE
        R2_test[i] = R2(y_test, ypredict_ridge)
        MSE_test[i] = MSE(y_test, ypredict_ridge)

    return MSE_train, MSE_test, R2_train, R2_test, beta_ridge



def Lasso(n, nlambdas, lambdas):
    """
   Perform Lasso regression.

   Parameters:
   n (int): Polynomial order.
   nlambdas (int): Number of lambda values.
   lambdas (ndarray): Array of lambda values.

   Returns:
   MSE_train (ndarray): Training Mean Squared Errors (MSE) for different lambdas.
   MSE_test (ndarray): Testing Mean Squared Errors (MSE) for different lambdas.
   R2_train (ndarray): Training R2 scores for different lambdas.
   R2_test (ndarray): Testing R2 scores for different lambdas.
   beta (ndarray): Fitted beta parameters for different lambdas.
   """
    lambdas = np.logspace(-6, 1, nlambdas)

    terrain = imread('SRTM_data_Norway_1.tif')

    # Ensure that the terrain array has the desired size (N x N)
    N = 10000
    size = 15000
    terrain = terrain[:N, :N]

    # Creates mesh of image pixels
    x = np.linspace(0, 1, terrain.shape[0])
    y = np.linspace(0, 1, terrain.shape[1])
    x_mesh, y_mesh = np.meshgrid(x, y)
    z_mesh = terrain.T  # Transpose for consistent shape

    # Reduce to 15000 points
    N = np.array(range(x_mesh.size))
    kept_data = np.random.permutation(N)[:size]
    x_red, y_red, z_red = x_mesh.flatten()[kept_data], y_mesh.flatten()[kept_data], z_mesh.flatten()[kept_data]

    X = create_X(x_red, y_red, n)

    # split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, z_red, test_size=0.2)
    # Scale data
    scaler = StandardScaler(with_std=False).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    MSE_train = np.zeros(nlambdas)
    MSE_test = np.zeros(nlambdas)
    R2_train = np.zeros(nlambdas)
    R2_test = np.zeros(nlambdas)
    beta = np.zeros((nlambdas, X.shape[1]))  # Initialize betaLasso matrix

    for i in range(nlambdas):
        lmb = lambdas[i]
        RegLasso = linear_model.Lasso(alpha=lmb,max_iter=10000, fit_intercept=False)
        RegLasso.fit(X_train, y_train)
        ytilde = RegLasso.predict(X_train)
        ypredict = RegLasso.predict(X_test)
        beta[i, :] = RegLasso.coef_
        MSE_train[i] = MSE(y_train, ytilde)
        MSE_test[i] = MSE(y_test, ypredict)
        R2_train[i] = R2(y_train, ytilde)
        R2_test[i] = R2(y_test, ypredict)
       
        
        
    
    return MSE_train,MSE_test, R2_train, R2_test,beta




# Load the terrain and make mesh grid and design matrix
terrain = imread('SRTM_data_Norway_1.tif')

N = 10000
n = 5 # polynomial order
terrain = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)
z = terrain
X = create_X(x_mesh, y_mesh,n)


n_list = list(range(1, n + 1))  # List of polynomial degrees
nlambdas = 10
lambdas_Ridge = np.logspace(-10, 10, nlambdas)
lambdas_Lasso = np.logspace(-10, 10, nlambdas)


#OLS
mse_train_OLS = np.zeros(n)
mse_test_OLS = np.zeros(n)
r2_train_OLS = np.zeros(n)
r2_test_OLS = np.zeros(n)
beta_OLS = np.zeros((n,n))


#Ridge
mse_train_Ridge = np.zeros((n, nlambdas))
mse_test_Ridge = np.zeros((n, nlambdas))
r2_train_Ridge = np.zeros((n, nlambdas))
r2_test_Ridge = np.zeros((n, nlambdas))

#Lasso

mse_train_Lasso = np.zeros((n, nlambdas))
mse_test_Lasso = np.zeros((n, nlambdas))
r2_train_Lasso = np.zeros((n, nlambdas))
r2_test_Lasso = np.zeros((n, nlambdas))




for i in range(n):
    mse_train_OLS[i], mse_test_OLS[i], r2_train_OLS[i], r2_test_OLS[i],_ = OLS(i)
    mse_train_Ridge[i,:],mse_test_Ridge[i,:],r2_train_Ridge[i,:],r2_test_Ridge[i,:],_ = Ridge(i, nlambdas, lambdas_Ridge )
    mse_train_Lasso[i,:],mse_test_Lasso[i,:],r2_train_Lasso[i,:],r2_test_Lasso[i,:],_ = Lasso(i, nlambdas, lambdas_Lasso)
   

#OLS

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(n_list, mse_train_OLS, label ='Train' )
plt.plot(n_list, mse_test_OLS, label = 'Test' )
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE ')
plt.title('MSE OLS')
plt.legend()
plt.subplot(122)
plt.plot(n_list, r2_train_OLS, label ='Train' )
plt.plot(n_list, r2_test_OLS, label = 'Test' )
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('R2 ')
plt.title('R2 OLS')
plt.legend()
plt.tight_layout()


#Ridge
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Ridge regression - MSE train for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE')
for j in range(nlambdas):
    plt.plot(n_list, mse_train_Ridge[:, j], label=f'$\lambda$={lambdas_Ridge[j]:.4f}')
plt.subplot(122)
plt.title('Ridge regression - MSE test for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE')
for j in range(nlambdas):
    plt.plot(n_list, mse_test_Ridge[:, j], label=f'$\lambda$={lambdas_Ridge[j]:.4f}')
    


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


#Lasso
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Lasso regression - MSE train for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE')
for j in range(nlambdas):
    plt.plot(n_list, mse_train_Lasso[:, j], label=f'$\lambda$={lambdas_Lasso[j]:.4f}')
plt.subplot(122)
plt.title('Lasso regression - MSE test for Different $\lambda$s')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('MSE')
for j in range(nlambdas):
    plt.plot(n_list, mse_test_Lasso[:, j], label=f'$\lambda$={lambdas_Lasso[j]:.4f}')
    


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
        