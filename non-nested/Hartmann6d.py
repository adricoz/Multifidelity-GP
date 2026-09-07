import numpy as np
#OK
def Hartmann6D(x):
    """
    Hartmann 6D function.

    Parameters
    ----------
    x : array-like, shape (6,)
        Input point in 6D space.

    Returns
    -------
    float
        Function value at the input point.
    """
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = [[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
         [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
         [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
         [17.0, 8.0, 0.05, 10.0, 14.0, 3.5]]
    P = [[1312, 1696, 5569, 1244, 8283, 5886],
         [2329, 4135, 8307, 3736, 1004, 9991],
         [2348, 1451, 3522, 2883, 3047, 6650],
         [4047, 8828, 8732, 5743, 1091, 381]]

    x = np.asarray(x)
    if x.shape != (6,):
        raise ValueError("Input must be a point in R^6.")

    total = sum(alpha[i] * np.exp(-sum(A[i][j] * (x[j] - P[i][j] / 10000) ** 2 for j in range(6))) for i in range(4))
    
    return -total


    # defining a multi-fidelity Hartmann approximation function
def f_l(x, deg=6, k=1, delta=0.1):
    if deg not in [6]:
        raise ValueError("deg must be either 2 or 6.")

    if k == np.inf or k is None:
        return Hartmann6D(x)

    u_0 = -5.0 if deg == 6 else -7.0
    if k == 0:
        return u_0

    delta_x_k = (delta / k) * np.ones(deg)
    x_shifted = np.array(x) + delta_x_k

    f_true = Hartmann6D(x_shifted)

    u_prev = u_0
    for _ in range(int(k)):
        u_prev = 0.5 * ((f_true**2) / u_prev + u_prev)

    return u_prev

def evaluate_fidelity(x, level, L):
    """
    Evaluates the Hartmann function at the desired fidelity level. 
    Auto adapts to the max fidelity level L
    
    Arguments:
    - x : array-like, shape (6,)
        Input point in 6D space.
    - level : int
        Fidelity level (1 to L). Level 1 is the lowest fidelity, and level L is the highest fidelity (true Hartmann function).
    - L : int
        Total number of fidelity levels. The highest fidelity corresponds to level L.

    Returns:
    - float
        Function value at the input point for the specified fidelity level.
    """
    # security, verifies that the security level exists 
    if not (1 <= level <= L):
        raise ValueError(f"Erreur : Le niveau {level} n'est pas défini. Il doit être entre 1 et {L}.")
    
    # High fidelity case, true Hartmann function
    if level == L:
        
        return f_l(x, deg=6, k=np.inf)
        
    # Low fidelity case 
    else:
        # use level as parameter k
        return f_l(x, deg=6, k=level, delta=0.05)