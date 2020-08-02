import numpy as np
from abc import ABC, abstractmethod


class ActivationFunc(ABC):
    @abstractmethod
    def activate(self, val):
        pass

    @abstractmethod
    def derive(self, val):
        pass


class Sigmoid(ActivationFunc):
    def _sigmoid(self, val):
        return 1 / (1 + np.exp(-val))

    def activate(self, val):
        return self._sigmoid(val)

    def derive(self, val):
        sig = self._sigmoid(val)
        return sig * (1-sig)


class ReLu(ActivationFunc):
    def activate(self, val):
        return max(0, val)

    def derive(self, val):
        #! the derivative is undefined at 0
        return 0 if val <= 0 else 1
