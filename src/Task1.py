# 1st STEP: Import packages and classes
# import joblib
import numpy as np
from math import sin
from utils import load_data, save_sklearn_model, evaluate_predictions
from sklearn.linear_model import LinearRegression


# 2ed STEP: Provide data through my path(my data are in a binary file, that i cannot see)
# Define the inputs(regressors,x) and outputs(predictor,y) that should be arrays(class numpy.ndarray)
# x = np.array([.., .., .., .., .., ..]).reshape((-1, 1))
# y = np.array([.., .., .., .., .., ..])
my_data_folder = '../data/data.npz'
x = load_data(my_data_folder)
y = load_data(my_data_folder)


# 3ed STEP: Create model - (model = LinearRegression()) - and fit it
# 4 functions of LinearRegression()'s method
# fit_intercept() --> calculate the optimal values of the weights b0,b1,b2,b3,b4, using the x and y as arguments
# normalize()
# copy_X()
# n_jobs()
model = LinearRegression().fit(x,y)


# 4th STEP: Get the results (with: .score())
r_sq = model.score(x, y)
# print('coefficient of determination:', r_sq)
# Use of .intercept_ which is scalar and coef_ that which is array
# print('intercept:', model.intercept_)
# print('slope:', model.coef_)


# 5th STEP: MAKE Y WITH TWO DIMENTIONS AS X
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))               


# 6th STEP: Get the parameters
# I multiply each element of x with model.coef_ and add model.intercept_ to the product
# f(x,θ) = θ0 + θ1*x1 +θ2*x2 + θ3*cos(x2) + θ4*x1ˆ2
theta0 = new_model.intercept_
theta1, theta2, theta3, theta4 = new_model.coef_
print("\nThe parameter values are: theta0 = {}, theta1 = {}, theta2 = {}, theta3 = {}, theta4 = {}.".format(theta0, theta1, theta2, theta3, theta4))


# 7th STEP: Predict of the response
# y_pred = model.predict(x)                                               # eg
# y_pred = model.intercept_ + model.coef_ * x                             # eg
# MINE 
y_pred = new_model.predict(x)
print('predicted response:', y_pred, sep='\n')


# 8th: Save the model in the Linear_Regression.pickle file
save_sklearn_model(LinearRegression, '../deliverable/Linear_Regression.pickle')
