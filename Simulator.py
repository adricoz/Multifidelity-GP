import logging

import aerosandbox as asb
import neuralfoil as nf
import numpy as np
from scipy.optimize import root_scalar

logger = logging.getLogger(__name__)

def generate_continuous_naca4(m_camber, p_position, t_thickness, n_points=100):
    """
    Generates mathematical coordinates for the naca profile.
    """
    x = np.linspace(0, 1, n_points)
    
    # camber
    y_c = np.where(x < p_position,
                   m_camber / p_position**2 * (2 * p_position * x - x**2),
                   m_camber / (1 - p_position)**2 * ((1 - 2 * p_position) + 2 * p_position * x - x**2))
    
    # angle derivative of camber line
    dyc_dx = np.where(x < p_position,
                      2 * m_camber / p_position**2 * (p_position - x),
                      2 * m_camber / (1 - p_position)**2 * (p_position - x))
    theta = np.arctan(dyc_dx)
    
    # thickness
    y_t = 5 * t_thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # outer / inner
    x_u = x - y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    x_l = x + y_t * np.sin(theta)
    y_l = y_c - y_t * np.cos(theta)
    
    # leading edge and trailing edge
    x_coords = np.concatenate((x_u[::-1], x_l[1:]))
    y_coords = np.concatenate((y_u[::-1], y_l[1:]))
    
    return np.column_stack((x_coords, y_coords))

class Simulator:
    def __init__(self, L, target_cl = 1.0):
        self.L = L
        self.target_cl = target_cl

    def evaluate(self, x_normalized, level):
        """ Generates airfoil, finds alpha for target Cl and return pure Cd"""
        m_camber = 0.02 + x_normalized[0] * 0.08
        p_camber = 0.3
        t_thickness = 0.08 + x_normalized[1] * (0.17 - 0.08)

        exact_name = f"naca_{m_camber:.2f}{p_camber*10}{t_thickness*100:.2f}"
        camber_int = int(round(m_camber * 100))      # ex 0.0423 -> 4
        pos_int = int(round(p_camber * 10))        # ex., 0.3 -> 3
        thick_int = int(round(t_thickness * 100))    # ex, 0.128 -> 13
            
        # Formatting: if thickness < 10, add a leading zero (ex, 09)
        naca_string = f"naca{camber_int}{pos_int}{thick_int:02d}"
        print("naca_string : ", naca_string)

        try:
            airfoil = asb.Airfoil(name = naca_string)
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
                logger.warning(f"Invalid Cd value for airfoil {naca_string} at level {level}. Returning high drag.")
                return 1e6

            # add a sligt fidelity bias to simulate fidelity levels 
            fidelity_bias = (level / (100.0 * self.L))
            return cd + fidelity_bias
        except ValueError:
            logger.warning(f"Root finding failed for airfoil {naca_string} at level {level}. Returning high drag.")
            return 1e6  # Return a high drag value if root finding fails