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
