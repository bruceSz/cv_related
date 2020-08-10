
#!/usr/bin/env python
# coding=utf-8


import torch


"""
    In pytorch, each time forward excution, we build  a dag, instead of a static
    dag like which is used in tensorflow.
        
    Static dag, we can optimize the dag and make early stage compute resource 
    allocation.
    Dynamic dag, we can implement control logic here.    
"""

class MyReLu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x<0]  = 0
        return grad_x



device = torch.device("cpu")
N, D_in, H, D_out  = 64, 1000, 100, 10


x = torch.randn(N, D_in, device = device)
y = torch.randn(N, D_out, device = device)

w1 = torch.randn(D_in, H, device = device, requires_grad = True)
w2 = torch.randn(H, D_out, device = device, requires_grad = True)

learning_rate = 1e-6

for t in range(500):
    y_pred = MyReLu.apply(x.mm(w1)).mm(w2)
    
    loss = (y_pred - y).pow(2).sum()

    print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad;
        w2 -= learning_rate * w2.grad;

        w1.grad.zero_()
        w2.grad.zero_()