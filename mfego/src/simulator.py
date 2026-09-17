from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    """
    Abstract base class for simulators.
    The user of the python package MUST inherit from this class
    """
    def __init__(self, L):
        self.L = L

    @abstractmethod
    def evaluate(self, x, level):
        """
        Abstract method to evaluate the simulator at a given point and fidelity level.
        Must be implemented by subclasses. Could run Neuralfoil, Xfoil, CFD, RANS ...
        """
        pass

