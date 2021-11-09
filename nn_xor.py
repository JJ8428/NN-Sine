import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
import random as r
from tqdm import tqdm

# Define NN
class custom_NN(nn.Module):

    def __init__(self):
        super(custom_NN, self).__init__()
        self.f1 = nn.Linear(2, 2)
        self.F = nn.ReLU()
        self.f2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.f1(x)
        x = self.F(x)
        x = self.f2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).data.tolist()

# Init NN and some parameters
model = custom_NN()
optimizer = optim.SGD(model.parameters(), lr=0.02)
epochs = 1000
steps = 100

# Generate the data
data = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
result = torch.Tensor([0,1,1,0]).view(-1,1)

# Define Loss function
criterion = nn.MSELoss()

print('Begin Training')
start_time = time.time()

for epoch in tqdm(range(0, epochs)):
    '''
    if epoch % 100 == 0:
        print('Epoch:', epoch)
    '''
    model.train()
    for i in range(0, steps):
        rand = r.randint(0, 3)
        input = data[rand]
        output_exp = result[rand] # Expected output
        # print(input, output_exp)
    
        optimizer.zero_grad()
        output_pred = model(input) # Predicted output
        loss = criterion(output_pred, output_exp)
        '''
        if epoch == epochs-1 and i == steps-1:
            print('Final Loss:' loss)
        '''
        loss.backward()
        optimizer.step()
    model.eval() # == model.train(False)

total_time = int(time.time() - start_time)
print('End Training')
print("Time to train:", total_time, "[s]")

print(model.predict(torch.Tensor([0, 0]))) # 0
print(model.predict(torch.Tensor([0, 1]))) # 1
print(model.predict(torch.Tensor([1, 0]))) # 1
print(model.predict(torch.Tensor([1, 1]))) # 0

print(model.f1.weight)
print(model.f2.weight)