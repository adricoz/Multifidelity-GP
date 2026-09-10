import numpy as np 
import neuralfoil as nf
import aerosandbox as asb

def foil_mid_fidelity(alpha, naca_string="naca4412", string_modelclass="xxsmall", Re=5e5):

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
        Re = Re,
        model_size = string_modelclass,
        n_crit=1,
        xtr_upper=0.1,
        xtr_lower=0.1
    )
    return np.squeeze(aero['CD']), np.squeeze(aero['CL'])

def objective_function(naca_string, level, L, alpha_min=0, alpha_max=10, alpha_step=0.02, target_cl=1.0):
    """    
    """
    modelclass = ["xxsmall","xsmall","small","medium","large","xlarge","xxlarge","xxxlarge"]

    alpha_min, alpha_max, alpha_step   = alpha_min, alpha_max, alpha_step
    naca_string = str(naca_string)

    if L >= 1 and L <= len(modelclass):
       #definition of the modelclass based on the fidelity level
       if level < L :
           modelclass = modelclass[int(level/L)*len(modelclass)]
       elif level == L:
            modelclass = modelclass[-1]
       else:
            raise ValueError(f"Error: Fidelity level {level} is not defined. Must be between 1 and {L}.")

       best_Cl = 0
       Cd = 0
       alpha = 0
       # Loop to find the angle of attack corresponding to the target lift coefficient (CL)
       for a in np.arange(alpha_min, alpha_max, alpha_step):
        diff_best_Cl = np.abs(best_Cl - target_cl)
        Cd_tmp, Cl = foil_mid_fidelity(a, naca_string=naca_string, string_modelclass=modelclass)
        if float(Cd_tmp) < 0:
            print(f"Warning: Drag coefficient {Cd:.4f} is negative. Returning a high penalty value.")
            return 1e9
        else:
            if np.abs(Cl - target_cl) < diff_best_Cl:
                best_Cl = Cl
                alpha = a
                Cd = Cd_tmp
       
       return float(Cd), float(best_Cl), float(alpha)
       
    if L<1 or L>8:
        raise ValueError(f"Error: Fidelity level {L} is not defined. Must be between 1 and 8.")
