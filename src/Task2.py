# 1st STEP: Import packages and classes
# import joblib
import numpy as np
from math import sin
from utils import load_data, save_sklearn_model, evaluate_predictions
from sklearn.linear_model import LinearRegression
# Maybe need more Libraries

# 1st STEP: Load the data
my_data_folder = '../data/data.npz'
x = load_data(my_data_folder)
y = load_data(my_data_folder)

