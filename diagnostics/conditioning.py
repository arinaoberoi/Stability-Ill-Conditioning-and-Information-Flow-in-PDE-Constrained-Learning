import numpy as np

def amplification_factor(u_true, u_recon):
    """
    Measures sensitivity of reconstruction to perturbations.
    """
    return np.linalg.norm(u_recon - u_true) / np.linalg.norm(u_true)
