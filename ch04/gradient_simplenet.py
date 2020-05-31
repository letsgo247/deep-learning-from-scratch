# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


# np.random.seed(1)

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화
        self.y = []

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        # print('z = ', z)
        y = softmax(z)
        self.y = y
        loss = cross_entropy_error(y, t)

        return loss




net = simpleNet()
print('W = ', net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print('p = ', p)
print('argmax = ', np.argmax(p))


t = np.array([0, 0, 1])
print('t = ', t)
net.loss(x,t)
print('y = ', net.y)
print('loss = ', net.loss(x,t))

f = lambda D: net.loss(x,t)

dW = numerical_gradient(f, net.W)
print('dW = ', dW)


# f = lambda _: net.loss(x,t)
# # print(net.loss(x,t))
# # print(type(net.loss(x,t)))
# dW = numerical_gradient(f, net.W)
#
# # print(f)
# print(dW)
