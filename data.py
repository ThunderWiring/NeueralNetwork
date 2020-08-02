
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Batch:
    inputs: np.array(np.array(float))
    outputs: np.array(float)


xor_data: List[Batch] = [
    Batch(np.array([[0], [0]]), np.array([0])),
    Batch(np.array([[1], [1]]), np.array([0])),
    Batch(np.array([[0], [1]]), np.array([1])),
    Batch(np.array([[1], [0]]), np.array([1])),
]

nand_data: List[Batch] = [
    Batch(np.array([[0], [0]]), np.array([1])),
    Batch(np.array([[1], [1]]), np.array([0])),
    Batch(np.array([[0], [1]]), np.array([1])),
    Batch(np.array([[1], [0]]), np.array([1])),
]

def get_batch():
    '''
    return random input from data
    '''
    ind = np.random.randint(0, high=4)
    return xor_data[ind]
