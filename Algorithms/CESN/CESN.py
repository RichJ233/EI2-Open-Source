# -*- coding: utf-8 -*-
###
##############################################################
# @file: CESN.py
# @author: Rich Jiang
# @version: V1.0
# @date: 2021/4/13
# @brief：Rewrite/Rename/Fix the problem of the CESN module partly and optimize the structure
#         and update the chinese remark
# @addition: Based on the previous code
# Distributed under MIT license https://opensource.org/licenses/MIT
##############################################################
# @notice: This module based on numpy and ESN.py
# @brief: This is an Chain Echo State Network (ChainESN) class.
#         esn_number: ESN module number in CESN, default: 2
#         input_size: input data size, default: 1
#         output_size: output data size, default: 1
#         reservoir_number: reservoir units number, default: 100
#         uniform_range: connection matrix init range, example [-x, x], default: 0.01
#         leaky_decay: leaking rate, default: 0.7
#         spectral_radius: spectral radius, default: 0.9
#         sparse_rate: sparse rate, default: 0.05
##############################################################
# 注意: 这个类的运行依赖于numpy库和ESN.py，调用本类之前，确保已经包括了这个python库。
# 这是一个回声状态网络(Chain-Echo State Network, ESN)的基本类，其基本参数介绍如下:
# esn_number: 链式回声网络中ESN的数量，默认为2
# input_size: 输入数据的维度，默认为1
# output_size: 输出数据的维度，默认为1
# reservoir_number: 神经网络隐含层的神经元个数，在ESN中隐含层又叫做储备池，所以该参数也可表示为储备池的神经元个数，默认为100个
# uniform_range: 输入层和隐含层之间的连接矩阵，以及隐含层之中的自连接矩阵，这两个矩阵在初始化时的初始范围[-x,x]，默认为0.01
# sparse_rate: 稀疏系数(connectivity)
# spectral_radius: 谱半径
##############################################################
###
import numpy as np
from ESN_C import ESN


class ChainESN:
    ##############################################################
    # @brief: This is a Chain ESN class
    # @param: as introduction
    # @retval: null
    ##############################################################
    def __init__(self, esn_number=2, input_size=1, output_size=1, reservoir_number=(100,),
                 leaky_decay=(0.7,), uniform_range=(0.01,), spectral_radius=(0.9,), sparse_rate=(0.05,)):
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
        str_info += '链式回声状态网络，共有{}个ESN单元，其中每个ESN单元的结构如下:\n'.format(self.esn_number)
        for i in range(self.esn_number):
            str_info += '第{}层: '.format(str(i + 1))
            str_info += str(self.parts[i])[2:-2]
            str_info += '\n'
        str_info += '*' * 65 + '\n'
        return str_info

    def train(self, input, output, reg=1e-3):
        ##############################################################
        # @brief: Train of CESN, update the weight matrix parts[i].W_out
        # @param: input_data: in shape [a, b, c], a-esn number, b-length of training set, c-input size
        #         output: in shape [a, b, c], a-esn number, b-length of training set, c-output size
        #         reg: the regularization parameter
        # @retval: null
        ##############################################################
        for i in range(self.esn_number):
            if i == 0:
                u = input[i]
            else:
                u = np.hstack((output[i - 1], input[i]))
            self.parts[i].train(u, output[i], reg=reg)

    def predict(self, input_data):
        ##############################################################
        # @brief: predict the result
        # @param: input_data: in shape [a, b, c], a-esn number, b-length of input, c-input size
        # @retval: result: in shape [a, b], a-length of input, b-output size
        ##############################################################
        result = np.zeros((input_data[0].shape[0], self.output_size))
        for i in range(self.esn_number):
            if i == 0:
                u = input_data[i]
                result = self.parts[i].predict(u)
            else:
                u = np.hstack((result, input_data[i]))
                result = self.parts[i].predict(u)
        return result

    pass




