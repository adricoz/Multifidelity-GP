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
    def evaluate(self, design_point: list, level: int) -> tuple[float, dict]:
        """
        Abstract method to evaluate the simulator at a given point and fidelity level.
        Must be implemented by subclasses. Could run Neuralfoil, Xfoil, CFD, RANS ...

        MUST return:
        1. The value to be minimized by the GP (ex: log10(drag))
        2. A dictionary of real valkues
        """
        pass
