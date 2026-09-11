import numpy as np 
import neuralfoil as nf
import aerosandbox as asb

# implementation of a typical fluid solver function call...
# Here we use neuralfoil to compute drag coefficients

def foil_mid_fidelity(alpha, naca_string="naca4412", string_modelclass="xxsmall"):

    """
    Computes the drag coefficient for a given airfoil at a specific angle of attack and fidelity level.
    Parameters:
    - alpha: float
        Angle of attack in degrees.
    - naca_string: str
        NACA airfoil designation (default is "naca6412").
    - string_modelclass: str
        Model class for the neural network used in the simulation (default is "xxsmall").
    Returns:
    - Cd: float
        The computed drag coefficient.
    """
    aero = nf.get_aero_from_airfoil(
        airfoil = asb.Airfoil(naca_string),
        alpha = alpha,
        Re = 50000,
        model_size = string_modelclass
    )
    return np.squeeze(aero['CD'])

def objective_function(x_params, level, L):
    """    
    Evaluates the custom Fluid function using physical inputs.
    params:
    - x_params: array-like, shape (1,)
        The input parameters in physical scale.
    - level: int
        The fidelity level (1, 2, or L).
    - L: int
        The total number of fidelity levels.
    """
    modelclass = ["xxsmall","xsmall","small","medium","large","xlarge","xxlarge","xxxlarge"]

    alpha  = float(x_params[0])

    # simple implementation of a 3 levels fidelity function where the fidelity is controlled by
    # the size of the neural network used to compute the drag coefficient

    if L >= 1 and L <= 8:
       if level < L :
              modelclass = modelclass[level-1]
              Cd = foil_mid_fidelity(alpha*180/np.pi, string_modelclass=modelclass)
              if float(Cd) < 0:
                      print(f"Warning: Drag coefficient {Cd:.4f} is negative. Returning a high penalty value.")
                      return 1e9
              else:
                  return float(Cd)
       elif level == L:
            modelclass = modelclass[-1]
            Cd = foil_mid_fidelity(alpha*180/np.pi, string_modelclass=modelclass)

            if float(Cd) < 0:
                print(f"Warning: Drag coefficient {Cd:.4f} is negative. Returning a high penalty value.")
                return 1e9
            else:
                return float(Cd)
       else:
            raise ValueError(f"Error: Fidelity level {level} is not defined. Must be between 1 and {L}.")
    if L<1 or L>8:
        raise ValueError(f"Error: Fidelity level {L} is not defined. Must be between 1 and 8.")

def generate_continuous_naca4(m_camber, p_position, t_thickness, n_points=100):
    """
    Generates mathematicl coordinates for the naca profile.
    """
    x = np.linspace(0, 1, n_points)
    
    # camber
    y_c = np.where(x < p_position,
                   m_camber / p_position**2 * (2 * p_position * x - x**2),
                   m_camber / (1 - p_position)**2 * ((1 - 2 * p_position) + 2 * p_position * x - x**2))
    
    # 2. Dérivée pour l'angle
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