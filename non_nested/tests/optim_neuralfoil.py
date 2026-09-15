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

# def objective_function(naca_string, level, L, alpha_min=0, alpha_max=10, alpha_step=0.02, target_cl=1.0):
#     """    
#     """
#     modelclass = ["xxsmall","xsmall","small","medium","large","xlarge","xxlarge","xxxlarge"]

#     alpha_min, alpha_max, alpha_step   = alpha_min, alpha_max, alpha_step
#     naca_string = str(naca_string)

#     if L >= 1 and L <= len(modelclass):
#        #definition of the modelclass based on the fidelity level
#        if level < L :
#            modelclass = modelclass[int(level/L)*len(modelclass)]
#        elif level == L:
#             modelclass = modelclass[-1]
#        else:
#             raise ValueError(f"Error: Fidelity level {level} is not defined. Must be between 1 and {L}.")

#        best_Cl = 0
#        Cd = 0
#        alpha = 0
#        # Loop to find the angle of attack corresponding to the target lift coefficient (CL)
#        for a in np.arange(alpha_min, alpha_max, alpha_step):
#         diff_best_Cl = np.abs(best_Cl - target_cl)
#         Cd_tmp, Cl = foil_mid_fidelity(a, naca_string=naca_string, string_modelclass=modelclass)
#         if float(Cd_tmp) < 0:
#             print(f"Warning: Drag coefficient {Cd:.4f} is negative. Returning a high penalty value.")
#             return 1e9
#         else:
#             if np.abs(Cl - target_cl) < diff_best_Cl:
#                 best_Cl = Cl
#                 alpha = a
#                 Cd = Cd_tmp
       
#        return float(Cd), float(best_Cl), float(alpha)
       
#     if L<1 or L>8:
#         raise ValueError(f"Error: Fidelity level {L} is not defined. Must be between 1 and 8.")
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

from scipy.optimize import root_scalar

def objective_function(airfoil_obj, target_cl, level, L):
    """
    
    """
    def erreur_cl(alpha_test):
            aero = nf.get_aero_from_airfoil(
                airfoil=airfoil_obj, alpha=alpha_test, Re=5e5, 
                model_size="xxxlarge", n_crit=1, xtr_upper=0.1, xtr_lower=0.1
            )
            cl_actuel = float(np.squeeze(aero["CL"]))
            return cl_actuel - target_cl

    
    modelclasses = ["xxsmall","xsmall","small","medium","large","xlarge","xxlarge","xxxlarge"]
    if L >= 1 and L <= len(modelclasses):
        #definition of the modelclass based on the fidelity level
        if level < L :
            modelclass = modelclasses[int(level/L)*len(modelclasses)]
        elif level == L:
             modelclass = modelclasses[-1]
        else:
             raise ValueError(f"Error: Fidelity level {level} is not defined. Must be between 1 and {L}.")

    try:
        result = root_scalar(erreur_cl, bracket=[-5.0, 15.0], method='brentq')      
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
            print("Cd is NaN or negative. Returning a high penalty value.")
            return 1e6, 0.0, alpha_perfect
            
        return cd, cl, alpha_perfect
        
    except Exception as e:
        print(f"An error occurred during the aerodynamic computation: {e}")
        return 1e6, 0.0, 0.0

def find_optimal_foil(alpha= 5.0):
    inst_cl = []
    inst_cd = []
    naca_profile = [] 
    # naca profile generation
    pos_camber = int(3)

    for camber in range(4, 10, 1):
        camber = int(camber)
        for thickness in range(8, 18, 1):
            thickness = int(thickness)
            if thickness < 10:
                naca_string = f"naca{camber:.0f}{pos_camber:.0f}0{thickness:.0f}"
            else:
                naca_string = f"naca{camber:.0f}{pos_camber:.0f}{thickness:.0f}"
            # example naca string: "naca6412"
            Cd, Cl = foil_mid_fidelity(alpha = alpha, naca_string = naca_string, string_modelclass = "xxlarge")
            inst_cl.append(Cl)
            inst_cd.append(Cd)
            naca_profile.append(naca_string)

    return inst_cl, inst_cd, naca_profile

""" cl, cd, naca =  find_optimal_foil(alpha = 5.0)
import matplotlib.pyplot as plt
plt.figure(figsize=(35, 25))

plt.scatter(cd, cl, color='royalblue', edgecolors='k', s=80, zorder=3)
for i in range(len(naca)):
    plt.annotate(
        naca[i], 
        (cd[i], cl[i]),              # position of the point
        textcoords="offset points",  # how to position the text
        xytext=(8, -3),              # pixel shift: 8 to the right 3 down
        ha='left',                   # horizontal alignement
        va='center',                 # vertical alignment
        fontsize=12,
        alpha=0.85
    )

plt.title("$alpha = 5°$")
plt.xlabel("$C_d$", fontsize=25)
plt.ylabel("$C_l)", fontsize=25)
plt.grid(True, linestyle='--', alpha=0.5, zorder=1)

#plt.xlim(0.013, 0.018)
#plt.ylim(0.85, 1.2)
plt.tight_layout() 
plt.savefig("cl_vs_cd_naca.png", dpi=500) """