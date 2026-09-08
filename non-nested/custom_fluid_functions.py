import numpy as np 

# Equation and relations in this file are based on Faltinsen - Hydrodynamics of High-Speed Marine Vehicles, 
# Cambridge University Press, 2005.
def drag_Cd(Cl, Lambda):
    """
    Calculate the drag coefficient (Cd) based on the lift coefficient (Cl) and aspect ratio (Lambda).
    
    Parameters:
    Cl : float
        Lift coefficient.
    Lambda : float
        Aspect ratio of the wing or airfoil.
        
    Returns:
    Cd : float
        Drag coefficient.
    """
    # Empirical relation for drag coefficient based on lift coefficient and aspect ratio
    Cd = (Cl**2) / (np.pi * Lambda)
    return Cd

def Lambda_6130(alpha, lift_coefficient):
    """
    Calculate the aspect ratio (Lambda) based on the angle of attack (alpha) and lift coefficient.
    
    Parameters:
    alpha : float
        Angle of attack.
    lift_coefficient : float
        Lift coefficient.
        
    Returns:
    Lambda : float
        Aspect ratio of the wing or airfoil.
    """
    Lambda = 2 * lift_coefficient / (2 * np.pi * alpha - lift_coefficient)
    return Lambda

def Lambda_6131(alpha, lift_coefficient):
    """
    Calculate the aspect ratio (Lambda) based on the angle of attack (alpha) and lift coefficient.
    
    Parameters:
    alpha : float
        Angle of attack.
    lift_coefficient : float
        Lift coefficient.
        
    Returns:
    Lambda : float
        Aspect ratio of the wing or airfoil.
    """
    Lambda = 8 * alpha * np.pi * lift_coefficient / (-1.0 * lift_coefficient**2 + 4 * alpha**2 * np.pi**2)
    return Lambda

def Lambda_6132(alpha, lift_coefficient):
    """
    Calculate the aspect ratio (Lambda) based on the angle of attack (alpha) and lift coefficient.
    
    Parameters:
    alpha : float
        Angle of attack.
    lift_coefficient : float
        Lift coefficient.
        
    Returns:
    Lambda : float
        Aspect ratio of the wing or airfoil.
    """
    Lambda = ((2 * lift_coefficient-alpha * np.pi) + np.sqrt(4 * np.pi * alpha * lift_coefficient + alpha * np.pi**2)) \
        / (2 * alpha * np.pi - lift_coefficient)
    return Lambda

def objective_function(x_params, lift_coefficient, level, L):
    """
    Calculate the objective function based on the angle of attack (alpha), lift coefficient, level, and total levels.
    Lift coefficient is usually fixed and goal is to minimize the drag coefficent.
    
    Parameters:
    x_params : tuple
        A tuple containing (alpha).
    lift_coefficient : float
        Lift coefficient.
    level : int
        Current fidelity level.
    L : int
        Total number of fidelity levels.
        
    Returns:
    objective_value : float
        Value of the objective function.
    """
    alpha = x_params
    if level == 1:
        Lambda = Lambda_6130(alpha, lift_coefficient) 

    elif level == 2:
        Lambda = Lambda_6131(alpha, lift_coefficient)

    elif level == 3:
        Lambda = Lambda_6132(alpha, lift_coefficient)

    else:

        raise ValueError("Level must be 1, 2, or 3.")
    
    # Calculate drag coefficient using the drag_Cd function
    Cd = drag_Cd(lift_coefficient, Lambda)
    
    # Objective function could be a combination of drag and lift coefficients, adjusted by level
    objective_value = Cd + (level / L) * lift_coefficient  # Example formulation
    
    return float(objective_value)