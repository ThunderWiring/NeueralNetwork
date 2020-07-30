
from data import  Batch
import numpy as np
from layer import Layer, LayerType
from typing import List


class Network:
    def __init__(self, input_size: int, hidden_layers_sizes: List[int]):
        '''
        @param input_size: Size of the input layer (i.e. how many inputs does
                            the network receive)
        @param hidden_layers_sizes: list of integers where each number at index 
                            i specifies how many nodes does the ith hidden layer contains.
        '''
        self._learning_rate = 0.4
        self._layers: List[Layer] = [Layer(input_size, 1, LayerType.INPUT)]
        for i in range(len(hidden_layers_sizes)):
            in_size = input_size if i == 0 else hidden_layers_sizes[i - 1]
            layer_type = LayerType.OUTPUT if (i == len(hidden_layers_sizes) - 1) \
                else LayerType.HIDDEN
            self._layers.append(
                Layer(hidden_layers_sizes[i], in_size, layer_type))

    def _feedforward(self, input_sample):
        next_input = np.array(input_sample.T)
        for i in range(1, len(self._layers)):
            next_input = self._layers[i].get_output(next_input)
        return next_input

    def calc(self, inputs: np.array(float)) -> np.array(float):
        '''
        Returns the calculation result per inputs vector. This function should 
        be called after the network is trained.
        '''
        return self._feedforward(inputs)

    def train_batch(self, batch: Batch):
        input_vec = batch.inputs
        output_vec = batch.outputs
        self._feedforward(input_vec)
        self._train(input_vec, output_vec, self._learning_rate)

    def _train(self, input_vec, output_vec, eta):
        delta_biase = []
        delta_weights = []
        next_err = None,
        for l in range(1, len(self._layers)):
            if self._layers[-l].type is LayerType.INPUT:
                continue
            w_next = None if self._layers[-l].type is LayerType.OUTPUT \
                else self._layers[-l+1].get_weights()
            err_next = None if self._layers[-l].type is LayerType.OUTPUT \
                else self._layers[-l+1].error
            a_prev = input_vec if self._layers[-l - 1].type is LayerType.INPUT \
                else self._layers[-l-1].get_activations()

            delta_weights_l, delta_biase_l = self._layers[-l].get_cost_gradient(
                w_next, err_next, a_prev, output_vec)
            delta_biase.append(delta_biase_l * self._learning_rate)
            delta_weights.append(delta_weights_l * self._learning_rate)

        # update weights and biases
        for l in range(1, len(self._layers)):
            self._layers[-l].update_knobs(delta_weights[l-1], delta_biase[l-1])

