"""
This module provides functions for optimizing hydrofoil designs 
using neural network-based aerodynamic simulations.
"""
import aerosandbox as asb
import neuralfoil as nf
import numpy as np
from scipy.optimize import root_scalar


def foil_mid_fidelity(alpha, naca_string : str ="naca4412",
                      string_modelclass : str ="xxsmall", Re: float =5e5) -> tuple[float, float]:
    """
    Computes the drag coefficient for a given airfoil 
    at a specific angle of attack and fidelity level.
    Parameters:
    - alpha: float
        Angle of attack in degrees.
    - naca_string: str
        NACA airfoil designation (default is "naca4412").
    - string_modelclass: str
        Model class for the neural network used in the simulation (default is "xxsmall").
    Returns:
    - Cd: float
        The computed drag coefficient.
    """
    aero = nf.get_aero_from_airfoil(
        airfoil = asb.Airfoil(naca_string),
        alpha = alpha,
        Re = Re,
        model_size = string_modelclass,
        n_crit=1,
        xtr_upper=0.1,
        xtr_lower=0.1
    )
    return np.squeeze(aero['CD']), np.squeeze(aero['CL'])

def generate_continuous_naca4(m_camber, p_position, t_thickness, n_points=100):
    """
    Generates mathematical coordinates for the naca profile.
    """
    x = np.linspace(0, 1, n_points)

    # camber
    y_c = np.where(x < p_position,
                   m_camber / p_position**2 * (2 * p_position * x - x**2),
                   m_camber / (1 - p_position)**2 * ((1 - 2 * p_position)
                                             + 2 * p_position * x - x**2))

    # angle derivative of camber line
    dyc_dx = np.where(x < p_position,
                      2 * m_camber / p_position**2 * (p_position - x),
                      2 * m_camber / (1 - p_position)**2 * (p_position - x))
    theta = np.arctan(dyc_dx)

    # thickness
    y_t = 5 * t_thickness * (0.2969 * np.sqrt(x) - 0.1260 * x\
                              - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    # outer / inner
    x_u = x - y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    x_l = x + y_t * np.sin(theta)
    y_l = y_c - y_t * np.cos(theta)

    # leading edge and trailing edge
    x_coords = np.concatenate((x_u[::-1], x_l[1:]))
    y_coords = np.concatenate((y_u[::-1], y_l[1:]))

    return np.column_stack((x_coords, y_coords))



def objective_function(airfoil_obj, target_cl, level, L):
    """
    Computes the drag coefficient for a given 
    airfoil at a specific angle of attack and fidelity level.

    The angle of attack giving the target lift coefficient is always computed with the
    highest NeuralFoil model ("xxxlarge"), for every level: all the levels are evaluated at
    the same operating point (this is also what allows a single-fidelity test with L = 1).
    The drag coefficient is then computed with the model of the requested level.

    Returns:
    - (cd, cl, alpha): cd is NaN if the computation failed ([FIX-E2]).
    """
    def cl_error(alpha_test):
        aero = nf.get_aero_from_airfoil(
            airfoil=airfoil_obj, alpha=alpha_test, Re=5e5,
            model_size="xxxlarge", n_crit=1, xtr_upper=0.1, xtr_lower=0.1
        )
        cl_current = float(np.squeeze(aero["CL"]))
        return cl_current - target_cl


    modelclasses = ["xxsmall","xsmall","small","medium","large","xlarge","xxlarge","xxxlarge"]
    modelclass = modelclasses[-1] # Default to the highest fidelity model class
    if L >= 1 and L <= len(modelclasses):
        #definition of the modelclass based on the fidelity level
        if level < L :
            # [FIX-E2] level l -> l-th model (as in legacy/custom_fluid_functions.py):
            # int(level/L)*len(modelclasses) was always 0 ("xxsmall") for every level < L
            modelclass = modelclasses[level - 1]
        elif level == L:
            modelclass = modelclasses[-1]
        else:
            raise ValueError(
                f"Error: Fidelity level {level} is not defined. "
                f"Must be between 1 and {L}."
            )

    try:
        result = root_scalar(cl_error, bracket=[-5.0, 15.0], method='brentq')
        alpha_perfect = result.root

        # NeuralFoil with personalized object!
        aero = nf.get_aero_from_airfoil(
            airfoil=airfoil_obj,
            alpha=alpha_perfect, 
            Re=5e5,
            model_size = modelclass,
            n_crit=1,
            xtr_upper=0.1,
            xtr_lower=0.1
        )

        # secure extraction
        cl = float(np.squeeze(aero["CL"]))
        cd = float(np.squeeze(aero["CD"]))
        
        # filter for negative or NaN drag coefficients
        if np.isnan(cd) or cd <= 0:
            # [FIX-E2/R4] NaN instead of a 1e6 penalty: the optimizer stores the point as a
            # failed evaluation (excluded from the GP). A 1e6 value destroyed the surrogate.
            print("Cd is NaN or negative. The evaluation is flagged as failed (NaN).")
            return np.nan, np.nan, alpha_perfect

        return cd, cl, alpha_perfect

    except Exception as e:  # noqa: BLE001
        print(f"An error occurred during the aerodynamic computation: {e}")
        # [FIX-E2/R4] failed evaluation (was a 1e6 penalty)
        return np.nan, np.nan, np.nan

def find_optimal_foil(alpha: float = 5.0) -> tuple[list, list, list]:
    """
    Finds the optimal NACA 4-digit airfoil for a given angle of attack (alpha) 
    by evaluating a range of camber and thickness values.
    """
    inst_cl = []
    inst_cd = []
    naca_profile = []
    # naca profile generation
    pos_camber = 3

    for camber in range(4, 10, 1):
        camber = int(camber)
        for thickness in range(8, 18, 1):
            thickness = int(thickness)
            if thickness < 10:
                naca_string = f"naca{camber:.0f}{pos_camber:.0f}0{thickness:.0f}"
            else:
                naca_string = f"naca{camber:.0f}{pos_camber:.0f}{thickness:.0f}"
            # example naca string: "naca6412"
            cd, cl = foil_mid_fidelity(alpha = alpha, naca_string = naca_string, 
                                       string_modelclass = "xxlarge")
            inst_cl.append(cl)
            inst_cd.append(cd)
            naca_profile.append(naca_string)

    return inst_cl, inst_cd, naca_profile
