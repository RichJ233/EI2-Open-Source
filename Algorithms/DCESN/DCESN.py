# -*- coding: utf-8 -*-
###
##############################################################
# @file: DESN.py
# @author: Rich Jiang
# @version: V1.2
# @date: 2021/5/13
# @brief：Create/Rename/Fix the problem of the DESN module and optimize the structure
# @addition: Null
# Distributed under MIT license https://opensource.org/licenses/MIT
##############################################################
# @notice: This module relay on numpy and ESN.py
# @brief: This is an Deep-chain Echo State Network (DESN) class.
#         esn_number: ESN module number in CESN, default: 2
#         input_size: input data size, default: 1
#         output_size: output data size, default: 1
#         reservoir_number: reservoir units number, default: 100
#         uniform_range: connection matrix init range, example [-x, x], default: 0.01
#         leaky_decay: leaking rate, default: 0.7
#         spectral_radius: spectral radius, default: 0.9
#         sparse_rate: sparse rate, default: 0.05
#         equal_w: make each ESN module have same reservoir weight
##############################################################
# 注意: 这个类的运行依赖于numpy和matplotlib库，调用本类之前，确保已经安装了这两个python库。
# 这是一个深度链式回声状态网络(Deep-chain Echo State Network, DCESN)的基本类，其基本参数介绍如下:
# esn_number: 深度链式回声网络中ESN的数量，默认为2
# input_size: 输入数据的维度，默认为1
# output_size: 输出数据的维度，默认为1，当output_size为1时，属于单步预测；不为1时，属于多步预测
# reservoir_number: 神经网络隐含层的神经元个数，在ESN中隐含层又叫做储备池，所以该参数也可表示为储备池的神经元个数，默认为100个
# uniform_range: 输入层和隐含层之间的连接矩阵，以及隐含层之中的自连接矩阵，这两个矩阵在初始化时的初始范围[-x,x]，默认为0.01
# sparse_rate: 稀疏系数(connectivity)
# spectral_radius: 谱半径
# equal_w: 回声状态网络储备池权重相等，默认为 不等
##############################################################
###
import numpy as np
from ESN import ESN


def update(x, u, a, w_in, w_res):  # x:储备池状态，u:输入数据，a:leaky decay rate，即漏衰减率
    return (1 - a) * x + a * np.tanh(np.dot(w_in, u) + np.dot(w_res, x))


class DCESN:
    def __init__(self, esn_number=2, input_size=1, output_size=1, reservoir_number=(100,),
                 leaky_decay=(0.7,), uniform_range=(0.01,), spectral_radius=(0.9,), sparse_rate=(0.05,),equal_w=False):
        self.sum_reservoir=[]
        self.unit_out_temp_train = []
        self.esn_number = esn_number
        self.parts = []

        self.input_size = input_size
        self.output_size = output_size

        reservoir_number = list(reservoir_number)
        leaky_decay = list(leaky_decay)
        uniform_range = list(uniform_range)
        spectral_radius = list(spectral_radius)
        sparse_rate = list(sparse_rate)

        self.reservoir_number = reservoir_number + reservoir_number[-1:] * (esn_number - len(reservoir_number))
        self.leaky_decay = leaky_decay + leaky_decay[-1:] * (esn_number - len(leaky_decay))
        self.uniform_range = uniform_range + uniform_range[-1:] * (esn_number - len(uniform_range))
        self.spectral_radius = spectral_radius + spectral_radius[-1:] * (esn_number - len(spectral_radius))
        self.sparse_rate = sparse_rate + sparse_rate[-1:] * (esn_number - len(sparse_rate))
        if equal_w:
            for i in range(self.esn_number):
                if i == 0:

                    self.parts.append(ESN(input_size=self.input_size,
                                          output_size=self.output_size,
                                          reservoir_number=self.reservoir_number[i],
                                          leaky_decay=self.leaky_decay[i],
                                          uniform_range=self.uniform_range[i],
                                          spectral_radius=self.spectral_radius[i],
                                          sparse_rate=self.sparse_rate[i]))
                else:
                    if i==1:
                        self.parts.append(ESN(input_size=self.input_size + self.output_size,
                                              output_size=self.output_size,
                                              reservoir_number=self.reservoir_number[i],
                                              leaky_decay=self.leaky_decay[i],
                                              uniform_range=self.uniform_range[i],
                                              spectral_radius=self.spectral_radius[i],
                                              sparse_rate=self.sparse_rate[i],copyWres=True,Wres=self.parts[0].W_res))
                    else:
                        self.parts.append(ESN(input_size=self.input_size + self.output_size,
                                              output_size=self.output_size,
                                              reservoir_number=self.reservoir_number[i],
                                              leaky_decay=self.leaky_decay[i],
                                              uniform_range=self.uniform_range[i],
                                              spectral_radius=self.spectral_radius[i],
                                              sparse_rate=self.sparse_rate[i], copyWres=True, Wres=self.parts[1].W_res,copyWin=True,Win=self.parts[1].W_in))
        else:
            for i in range(self.esn_number):
                if i == 0:

                    self.parts.append(ESN(input_size=self.input_size,
                                          output_size=self.output_size,
                                          reservoir_number=self.reservoir_number[i],
                                          leaky_decay=self.leaky_decay[i],
                                          uniform_range=self.uniform_range[i],
                                          spectral_radius=self.spectral_radius[i],
                                          sparse_rate=self.sparse_rate[i]))
                else:
                    self.parts.append(ESN(input_size=self.input_size + self.output_size,
                                          output_size=self.output_size,
                                          reservoir_number=self.reservoir_number[i],
                                          leaky_decay=self.leaky_decay[i],
                                          uniform_range=self.uniform_range[i],
                                          spectral_radius=self.spectral_radius[i],
                                          sparse_rate=self.sparse_rate[i]))

    def __repr__(self):
        str_info = '*' * 65 + '\n'
        str_info += '深度链式回声状态网络，共有{}个ESN单元，其中每个ESN单元的结构如下:\n'.format(self.esn_number)
        for i in range(self.esn_number):
            str_info += '第{}层: '.format(str(i + 1))
            str_info += str(self.parts[i])[2:-2]
            str_info += '\n'
        str_info += '*' * 65 + '\n'
        return str_info

    def train(self, input, output, reg=1e-3):
        # input: [a,b]  a-data length; b-input size
        # output: [a,b,c]  a-esn_module; b-data length; c-output size
        if input.shape[1] != self.input_size:
            print("input size not equal!")
            return
        if input.shape[0] != output.shape[1]:
            print("data length not equal!")
            return
        if output.shape[0] != self.esn_number:
            print("data length not equal_1!")
            return

        data_length = input.shape[0]
        print('data_length:', data_length)
        unit_out_temp = np.zeros((data_length, output.shape[2]))
        sum_reservoir = np.zeros((data_length, self.parts[0].reservoir_number))

        for i in range(self.esn_number):
            X = np.zeros((data_length, self.parts[0].reservoir_number))
            Y = output[i, :]
            if i == 0:
                in_dt_temp = input
            else:
                in_dt_temp = np.hstack((unit_out_temp, input))
            for epoch in range(data_length):
                u = in_dt_temp[epoch, np.newaxis].T
                self.parts[i].reservoir_state = update(self.parts[i].reservoir_state, u,
                                                       self.parts[i].leaky_decay,
                                                       self.parts[i].W_in,
                                                       self.parts[i].W_res)
                X[epoch, :] = np.squeeze(self.parts[i].reservoir_state)
            self.sum_reservoir.append(X)
            if i == 0:
                sum_reservoir = X
            else:

                sum_reservoir = np.hstack((sum_reservoir, X))
            output_weight_part = np.dot(sum_reservoir.T, sum_reservoir)  # [reservoir_number_sum, reservoir_number_sum]
            Eye_array = np.eye(sum_reservoir.shape[1])  # [reservoir_number_sum, reservoir_number_sum]
            W_out_part = np.linalg.inv(output_weight_part + reg * Eye_array)  # [reservoir_sum, reservoir_sum]
            self.parts[i].W_out_self = np.dot(np.dot(W_out_part, sum_reservoir.T), Y).T

            for tim in range(data_length):
                sum_reservoir_temp = sum_reservoir[tim, :]
                unit_out_temp[tim, :] = np.dot(self.parts[i].W_out_self, sum_reservoir_temp)
            self.unit_out_temp_train.append(unit_out_temp)

            #print('train', i, f.MAE(unit_out_temp, Y))

    def predict(self, input):
        # input: [a,b]  a-data length; b-input size
        if input.shape[1] != self.input_size:
            print("size not equal!")
            return
        data_len = input.shape[0]
        print("input data length:", data_len)

        result = np.zeros((self.esn_number, data_len, self.output_size))
        X = np.zeros((data_len, self.parts[0].reservoir_number))
        sum_reservoir = np.zeros((data_len, self.parts[0].reservoir_number))
        result_temp = np.zeros((data_len, self.output_size))

        for i in range(self.esn_number):
            X = np.zeros((data_len, self.parts[0].reservoir_number))
            if i == 0:
                in_dt_temp = input
            else:
                in_dt_temp = np.hstack((result_temp, input))
            for epoch in range(data_len):
                u = in_dt_temp[epoch, np.newaxis].T
                self.parts[i].reservoir_state = update(self.parts[i].reservoir_state, u,
                                                       self.parts[i].leaky_decay,
                                                       self.parts[i].W_in,
                                                       self.parts[i].W_res)
                X[epoch, :] = np.squeeze(self.parts[i].reservoir_state)

            if i == 0:
                sum_reservoir = X
            else:
                sum_reservoir = np.hstack((sum_reservoir, X))

            for tim in range(data_len):
                sum_reservoir_temp = sum_reservoir[tim, :]
                result_temp[tim, :] = np.dot(self.parts[i].W_out_self, sum_reservoir_temp)

            result[i, :] = result_temp
        return result, sum_reservoir

    pass