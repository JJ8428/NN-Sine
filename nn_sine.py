import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

# Define NN
class custom_NN(nn.Module):

    def __init__(self):
        super(custom_NN, self).__init__()
        '''
        self.f = nn.ReLU()
        self.h1 = nn.Linear(1, 20)
        self.h2 = nn.Linear(20, 20)
        self.h3 = nn.Linear(20, 1)
        '''
        self.layers = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        # This is done by sequential already
        '''
        ah1 = self.f(self.h1(x))
        ah2 = self.f(self.h2(ah1))
        return self.h3(ah2)
        '''
        x = torch.from_numpy(x.astype(np.float32).reshape(x.shape[0],1))
        y = self.layers(x)
        return y

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).data.tolist()
            
# Parameters to control learning
model = custom_NN()
optimizer = optim.SGD(model.parameters(), lr=0.02)
size = 1500
epochs = 100

# Generate the data
domain = (0, 2*np.pi)
x = np.linspace(domain[0], domain[1], size)
y = np.sin(x)

# Plot the data
# Not needed since we do this later
'''
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('Angle [radians]')
plt.ylabel('y = sin(x)')
plt.axis('tight')
plt.show()
'''

# Define Custom Loss function
l1_loss = nn.L1Loss()
def criterion(pred, target):
    with torch.no_grad():
        return l1_loss(pred, target)**2 + \
               (model.forward(0).data - \
                model.forward(2*np.pi).data)**2

# Train network
shuffle = np.random.permutation(size)
step = int(size/epochs)
loss_x = list(range(0, epochs))
loss_de = [] # Loss for every epoch

print('Begin Training')
start_time = time.time()

for epoch in tqdm(range(0, epochs)):
    model.train()
    for i in range(0, size, step):
        data_x = x[shuffle[i:i+step]]
        data_y = y[shuffle[i:i+step]]

        # Reset gradients
        optimizer.zero_grad()

        # print(type(data_x))
        # print(type(data_y))

        y_pred = model(data_x)

        correct_shape = (data_y.__len__(), 1)
        data_y_t = torch.from_numpy(data_y).view(correct_shape)
        # print(type(data_y), data_y.__len__())
        # print(type(y_pred), y_pred.size())
        # print(type(data_y_t), data_y_t.size())
        '''
        nn.L1Loss() requires tensors of same exact dim
        <class 'torch.Tensor'> torch.Size(_)
        <class 'torch.Tensor'> torch.Size(_)
        '''
        loss = l1_loss(y_pred, data_y_t)
        loss_de.append(loss.data)
        loss.backward() # Back propogation
        optimizer.step()
    model.eval() # == model.train(False)

total_time = int(time.time() - start_time)
print('End Training')
print("Time to train:", total_time, "[s]")

# Generate data to test
plot_x = np.linspace(domain[0], domain[1], 20)
plot_y = model.predict(plot_x)

# Plot comparison of sin(x) and N(x, theta)
print('sin(x) Vs. NN plot')
plt.plot(plot_x, plot_y, label="Predicted")
plt.plot(x, y, label="Training Data")
plt.title('NN Sine Wave')
plt.legend()
plt.grid(True)
plt.xlabel('Angle [radians]')
plt.ylabel('y = sin(x)')
plt.axis('tight')
plt.show()