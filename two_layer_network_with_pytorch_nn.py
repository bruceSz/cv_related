#!/usr/bin/env python
# coding=utf-8


import torch

device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N,D_in, device = device)
y = torch.randn(N,D_out, device=device)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
    ).to(device)

loss_fn = torch.nn.MSELoss(size_average=False)

# here the learning_rate of `1e-6` is too small for nn
learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

optimizer.zero_grad()

for t in range(500):
    

    y_pred = model(x)
    
    loss = loss_fn(y_pred, y)
    
    print(t, loss.item())
    #optimizer.zero_grad()
    model.zero_grad()

    loss.backward()
    optimizer.step()


    #with torch.no_grad():
    #    for p in model.parameters():
    #        p.data = p.data -  learning_rate * p.grad

    