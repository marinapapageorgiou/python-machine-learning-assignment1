# 1st step
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2ed step
# read the data(dataset) from the file 
dataset = pd.read_csv('../data/data.npz') 

# 3ed step
# extracts features from the dataset
X = dataset.iloc[:, 1:-1].values 
# extracts the labels from the dataset
y = dataset.iloc[:, -1].values 

