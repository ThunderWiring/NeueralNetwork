import numpy as np
from network import Network
from data import xor_data, Batch, get_batch

network = Network(2, hidden_layers_sizes=[2, 2,1])

for _ in range(15000):
    network.train_batch(get_batch())


print('======================================================================')
print('[0,0]', network.calc(np.array([[0], [0]])))
print('[0,1]', network.calc(np.array([[0], [1]])))
print('[1,0]', network.calc(np.array([[1], [0]])))
print('[1,1]', network.calc(np.array([[1], [1]])))