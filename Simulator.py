import logging

import Aerosandbox as asb
import neuralfoil as nf
import numpy as np
from scipy.optimize import root_scalar

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, L, target_cl = 1.0):
        self.L = L
        self.target_cl = target_cl

    def evaluate(self, x_normalized, level):
        """ Generates airfoil, finds alpha for target Cl and return pure Cd"""
        m_camber = 0.02 + x_normalized[0] * 0.08
        p_camber = 0.3
        t_thickness = 0.08 + x_normalized[2] * (0.17 - 0.08)

        exact_name = f"naca_{m_camber:.2f}_p{p_camber*10}_t{t_thickness*100:.2f}"

        try:
            airfoil = asb.Airfoil(name = exact_name)
        except Exception:
            return 1e6  # Return a high drag value if airfoil generation fails
        
        def lift_error(alpha_test):
            try:
                aero = nf.get_aero_from_airfoil(airfoil, alpha_test, Re = 5e5, mach = 0.0,  model_size = "xxxlarge")
                return float(np.squeeze(aero["cl"])) - self.target_cl
            except: 
                return 100.0 # Error state
        try:
            res = root_scalar(lift_error, bracket=[-5, 15], method='brentq')
            alpha_opt = res.root
            aero_final = nf.get_aero_from_airfoil(airfoil, alpha_opt, Re = 5e5, mach = 0.0, model_size = "xxxlarge")
            cd = float(np.squeeze(aero_final["cd"]))
            if cd <= 0 or np.isnan(cd):
                logger.warning(f"Invalid Cd value for airfoil {exact_name} at level {level}. Returning high drag.")
                return 1e6

            # add a sligt fidelity bias to simulate fidelity levels 
            fidelity_bias = (level / (100.0 * self.L))
            return cd + fidelity_bias
        except ValueError:
            logger.warning(f"Root finding failed for airfoil {exact_name} at level {level}. Returning high drag.")
            return 1e6  # Return a high drag value if root finding fails