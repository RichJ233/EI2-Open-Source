#################################
# Name: ARX.py
# Function: Class of autoregressive with an exogenous input prediction model
# Author: Shaoxiong Zeng
# Date: 02/03/2024
# Environment:  Python - 3.11.5
#               Numpy - 1.24.3
#               Pandas - 2.1.4
#################################
import numpy as np
import pandas as pd


class ARX:
    def __init__(self, p=10, exogenous_input_dimension=0, exogenous_input_size=0):
        self.p= p
        self.exogenous_input_dimension= exogenous_input_dimension
        self.exogenous_input_size =exogenous_input_size
        self.w = np.zeros((1, self.p + self.exogenous_input_size * self.exogenous_input_dimension))

    def train(self, input_data, out_put, reg=1e-3):
        W = np.linalg.inv(np.dot(input_data.T, input_data) + reg * np.eye(input_data.shape[1]))
        self.w = np.dot(np.dot(W, input_data.T), out_put).T

    def predict(self, input_data, var=0.5):
        pre = np.zeros((input_data.shape[0], 1))
        for i in range(input_data.shape[0]):
            pre[i] = np.dot(self.w, input_data[i])
        gaussnoise = np.random.normal(0, var ** 0.5, input_data.shape[0]).reshape(input_data.shape[0], -1)
        return (pre + gaussnoise).squeeze()
