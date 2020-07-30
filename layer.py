
import numpy as np
from neuron import Neuron
from activation_func import ReLu, Sigmoid
from typing import List
from enum import Enum

LayerType = Enum('Type', 'INPUT HIDDEN OUTPUT')

class Layer:

    def __init__(self, num_of_nodes: int, input_size: int, layer_type: LayerType):
        '''
        @param num_of_nodes: number of nodes in the layers
        @param input_size: number of inputs per neuron in this layer
        '''
        is_output = layer_type == LayerType.OUTPUT
        activation_function = Sigmoid()  if is_output else ReLu()

        self.error = 0
        self.type = layer_type
        self._input_size = input_size
        self._layer_size = num_of_nodes
        self.neurons: List[Neuron] = []
        for _ in range(num_of_nodes):
            self.neurons.append(Neuron(input_size, activation_function))

    def get_output(self, input_sample: np.array(np.array(float))) -> np.array(np.array(float)):
        if self.type is LayerType.INPUT:
            return input_sample.T

        a = np.zeros(shape=(self._layer_size, 1))
        for i in range(self._layer_size):
            a[i] = self.neurons[i].calc_output(input_sample[0])
        return a.T

    def get_weights(self) -> np.array(np.array(float)):
        weights = np.zeros(shape=(self._layer_size, self._input_size))
        for i in range(self._layer_size):
            weights[i] = self.neurons[i].weights
        return weights

    def get_activations(self) -> np.array(float):
        if self.type is LayerType.INPUT:
            raise('input layer has no activations')
        a = np.zeros(shape=(self._layer_size, 1))
        for i in range(self._layer_size):
            a[i] = self.neurons[i].get_activation()
        return a

    def _get_derivatives(self) -> np.array(float):
        dt = np.zeros(shape=(self._layer_size, 1))
        for i in range(len(self.neurons)):
            dt[i] = self.neurons[i].calc_derivative()
        return dt

    def get_cost_gradient(self,
                          W_next: np.array(float), Err_next: np.array(float),
                          A_prev: np.array(float), Y_current: np.array(float)):
        '''
        calculates the cost function gradient for this layer.
        '''
        weights = None
        biases = None
        derivatives = self._get_derivatives()
        A = self.get_activations()

        if self.type is LayerType.OUTPUT:
            delta = np.multiply((A - Y_current), derivatives.T)
            weights = np.dot(delta, A_prev.T)
            biases = delta
        else:
            e = np.dot(W_next.transpose(), Err_next)
            delta = np.multiply(e, derivatives)
            biases = delta
            weights = np.dot(delta, A_prev.T)

        self.error = biases
        return weights, biases

    def update_knobs(self, delta_weights, delta_biases):
        for i in range(self._layer_size):
            self.neurons[i].update_weights_by(delta_weights[i])
            self.neurons[i].update_bias_by(delta_biases[i])
