"""
This module defines the abstract base class for
simulators in the multifidelity optimization framework.
"""

from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    """
    Abstract base class for simulators.
    The user of the python package MUST inherit from this class
    """

    def __init__(self, num_levels: int) -> None:
        """
        Initialize the simulator with the number of fidelity levels.
        """
        self.num_levels = num_levels

    @abstractmethod
    def evaluate(self, design_point: list, level: int) -> float:
        """
        Abstract method to evaluate the simulator at a given point and fidelity level.
        Must be implemented by subclasses. Could run Neuralfoil, Xfoil, CFD, RANS ...
        """
        pass
