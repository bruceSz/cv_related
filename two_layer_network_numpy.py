#!/usr/bin/env python
# coding=utf-8




import numpy as np
import matplotlib.pyplot as plt

# N : batch size.
N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H);
w2 = np.random.randn(H, D_out)

l_rate = 1e-6
X = [];
Y = [];

for t in range(500):
    h = x.dot(w1);
    h_relu = np.maximum(h,0);
    y_pred = h_relu.dot(w2);

    loss = np.square(y_pred - y).sum()
    print(t, loss)
    X.append(t);
    Y.append(loss);
    grad_y_pred = 2.0* (y_pred  - y);
    grad_w2 = h_relu.T.dot(grad_y_pred);
    grad_h_relu = grad_y_pred.dot(w2.T)
    
    # here the gred of maximum is 1 (when input is positive) and 0 (when input is negative or 0)
    # -> grad of the relu is 1 or 0 , 
    # below two lines are for compute the grad_h
    grad_h = grad_h_relu.copy() * 1;
    grad_h[h<0] = 0;
    #grad_h[h>0] = 1;
    print( t,   " is grad_h is : " + str(grad_h))
    grad_w1 = x.T.dot(grad_h);
    w1 -= l_rate * grad_w1;
    w2 -= l_rate * grad_w2;



plt.plot(X,Y)
plt.show();