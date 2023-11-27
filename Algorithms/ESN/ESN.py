# -*- coding: utf-8 -*-
###
##############################################################
# @file: ESN.py
# @author: Sun Yuyang
# @version: V1.0
# @date: 2021/4
# @brief：
##############################################################
# @Revised file
# @author: Rich Jiang
# @version: V1.1
# @date: 2021/4/11
# @brief：Rewrite/Rename and Fix the problem of the ESN module partly and optimize the structure
#         and update the chinese remark
# @addition: Based on the previous code and Mantas code
# Distributed under MIT license https://opensource.org/licenses/MIT
##############################################################
# @notice: This module based on numpy
# @brief: This is an Echo State Network (ESN) class.
#         input_size: input data size, default: 2
#         output_size: output data size, default: 1
#         reservoir_number: reservoir units number, default: 100
#         uniform_range: connection matrix init range, example [-x, x], default: 0.01
#         leaky_decay: leaking rate, default: 0.7
#         spectral_radius: spectral radius, default: 0.9
#         sparse_rate: sparse rate, default: 0.02
##############################################################
# 注意: 这个类的运行依赖于numpy和matplotlib库，调用本类之前，确保已经安装了这两个python库。
# 这是一个回声状态网络(Echo State Network, ESN)的基本类，其基本参数介绍如下:
# input_size: 输入数据的维度，默认为2
# output_size: 输出数据的维度，默认为1，当output_size为1时，属于单步预测；不为1时，属于多步预测
# reservoir_number: 神经网络隐含层的神经元个数，在ESN中隐含层又叫做储备池，所以该参数也可表示为储备池的神经元个数，默认为100个
# uniform_range: 输入层和隐含层之间的连接矩阵，以及隐含层之中的自连接矩阵，这两个矩阵在初始化时的初始范围[-x,x]，默认为0.01
# sparse_rate: 稀疏系数（connectivity）
# spectral_radius: 谱半径
##############################################################
###

import numpy as np


def dropout(reservoir_number, sparse_rate):
    ##############################################################
    # @brief: drop out data in matrix
    # @param: reservoir_number
    #         sparse_rate: dropout rate
    # @retval: sparse_matrix: after dropout matrix
    ##############################################################
    number_of_zero = int(reservoir_number ** 2 * (1 - sparse_rate))

    zero_matrix = np.zeros(number_of_zero)
    one_matrix = np.ones(reservoir_number ** 2 - number_of_zero)
    one_zero_matrix = np.hstack((zero_matrix, one_matrix))
    np.random.shuffle(one_zero_matrix)

    sparse_matrix = one_zero_matrix.reshape(reservoir_number, reservoir_number)
    return sparse_matrix


class ESN:
    ##############################################################
    # @brief: This is a single ESN module
    # @param: as introduction
    # @retval: null
    ##############################################################
    def __init__(self, input_size=1, output_size=1, reservoir_number=100, leaky_decay=0.7, uniform_range=0.01,
                 spectral_radius=0.9, sparse_rate=0.05):
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_number = reservoir_number
        self.reservoir_state = np.zeros((self.reservoir_number, 1))
        self.leaky_decay = leaky_decay
        self.uniform_range = uniform_range
        # calc W_in
        self.W_in = (np.random.rand(reservoir_number, input_size) - 0.5) * 2 * uniform_range  # w_in=-1~1*uniform_range
        # calc W_res
        self.W_res = (np.random.rand(reservoir_number, reservoir_number) - 0.5) * 2  # w_res = -1~1
        # init w_res with sparse_rate connectivity:
        sparse_matrix = dropout(self.reservoir_number, sparse_rate)
        self.W_res *= sparse_matrix
        # normalizing and setting spectral radius:
        prv_sp_r = np.max(np.abs(np.linalg.eig(self.W_res)[0]))
        self.W_res = self.W_res / prv_sp_r
        self.W_res *= spectral_radius  # update w_res with radius as spectral_radius
        # 计算 W_out 大小
        self.W_out = np.zeros((output_size, self.reservoir_number))  # add output_size for multi predict
        self.W_out_self = np.zeros((output_size, self.reservoir_number))  # add output_size for multi predict

    def __repr__(self):
        return '**输入层维数为{}，隐含层维数为{}**'.format(
            self.input_size, self.reservoir_number)

    def train(self, input_data, output, reg=1e-3):
        ##############################################################
        # @brief: Train of ESN, update the weight matrix W_out
        # @param: input_data: in shape [a, b], a-length of training set, b-input size
        #         output: in shape [a, b], a-length of training set, b-output size
        #         reg: the regularization parameter
        # @retval: null
        ##############################################################

        if input_data.shape[0] != output.shape[0]:
            print('数据输入格式错误，输入数据和输出数据的行数不相等！')
            return
        X = np.zeros((input_data.shape[0], self.reservoir_number))
        Y = output

        for epoch in range(input_data.shape[0]):
            u = input_data[epoch, np.newaxis].T
            self.reservoir_state = self.__update(self.reservoir_state, u, self.leaky_decay)
            X[epoch, :] = np.squeeze(self.reservoir_state)
        W_out_part = np.linalg.inv(np.dot(X.T, X) + reg * np.eye(self.reservoir_number))
        self.W_out = np.dot(np.dot(W_out_part, X.T), Y).T

    def predict(self, input_data):
        ##############################################################
        # @brief: predict the result
        # @param: input_data: in shape [a, b], a-length of input, b-input size
        # @retval: result: in shape [a, b], a-length of input, b-output size
        ##############################################################
        result = np.zeros((input_data.shape[0], self.output_size))

        for i in range(input_data.shape[0]):
            u = input_data[i, np.newaxis].T
            self.reservoir_state = self.__update(self.reservoir_state, u, self.leaky_decay)
            result[i] = np.dot(self.W_out, self.reservoir_state).squeeze()
        return result

    def __update(self, x, u, a):  # x:储备池状态，u:输入数据，a:leaky decay rate，即漏衰减率
        return (1 - a) * x + a * np.tanh(np.dot(self.W_in, u) + np.dot(self.W_res, x))
    pass
