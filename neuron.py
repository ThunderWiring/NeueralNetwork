
import numpy as np
from activation_func import ActivationFunc


class Neuron:
    def __init__(self, input_size: int, activiation_func: ActivationFunc):
        self._activation_func = activiation_func
        self._input_size = input_size
        self._error = 0
        self._z = None
        self._a = None

        self.bias = 0
        self.weights = np.random.uniform(size=(1, input_size))

    def calc_output(self, inputs):
        self._z = np.dot(self.weights, inputs.T) + self.bias
        self._a = self._activation_func.activate(self._z)
        return self._a

    def get_activation(self):
        if self._a is None:
            raise('Cannot get activation value: Neuron has not been activated yet.')
        return self._a

    def calc_derivative(self):
        if self._z is None:
            raise(
                'Cannot calculate derivative for neoron: Neoron has not been activated.')
        return self._activation_func.derive(self._z)

    def update_bias_by(self, delta_b: float):
        self.bias -= delta_b

    def update_weights_by(self, delta_weights: np.array(float)):
        self.weights -= delta_weights
