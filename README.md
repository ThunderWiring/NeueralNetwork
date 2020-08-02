# NeueralNetwork
implementation of a neural network from scratch with backpropagation as learning algo

This project takes a different approach for implementing a neural network from scratch, as it's heavily relys on OOP.

The neural network implementation is spread over the following 4 files:
* network: Manages the feedforward and backpropagation
* layer: represents a single layer in the network. This module on its own is decoupled from other layers in the network
* neuron: The smallest unit in the network which actually does the calculations. Each Neuron is un-aware of other neurons in the network, 
as well as it's unaware of the layer it belongs to
* activation_function: Defines both Sigmoid and ReLu functions that are used byt the neurons to calc their activation.

To run this network, simply run the `main.py` file.
Training data is found on the `data.py` file
