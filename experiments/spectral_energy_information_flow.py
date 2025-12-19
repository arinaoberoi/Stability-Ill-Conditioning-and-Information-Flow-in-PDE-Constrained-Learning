"""
This experiment analyzes how reconstruction energy is distributed across
spectral modes, and how instability manifests as high-frequency amplification.

Core idea:
  - Stable reconstructions concentrate energy in low-frequency modes
  - Ill-conditioned inverses inject spurious high-frequency energy
  - Regularization reshapes information flow across scales

Provides a proxy for 'information flow' grounded in PDE structure.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

N = 50
h = 1.0 / (N + 1)
noise_level = 1e-3
lambda_reg = 1e-2

def build_laplacian(N):
    e = np.ones(N)
    T = sp.diags([e, -4*e, e], [-1, 0, 1], shape=(N, N))
    I = sp.eye(N)
    return (sp.kron(I, T) + sp.kron(T, I)) / (h**2)

L = build_laplacian(N)

x = np.linspace(h, 1 - h, N)
X, Y = np.meshgrid(x, x)
u_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
u_true = u_true.flatten()
f = -2 * np.pi**2 * u_true
f_noisy = f + noise_level * np.random.randn(*f.shape)
u_direct = spla.spsolve(L, f_noisy)
A = L.T @ L + lambda_reg * sp.eye(L.shape[0])
b = L.T @ f_noisy
u_reg = spla.spsolve(A, b)

# -----------------------------
# Spectral decomposition
# -----------------------------
num_modes = 300
eigvals, eigvecs = spla.eigs(L, k=num_modes, which="SM")
idx = np.argsort(np.abs(eigvals))
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

def spectral_energy(u, eigvecs):
    coeffs = eigvecs.T @ u
    return np.abs(coeffs)**2

E_true = spectral_energy(u_true, eigvecs)
E_direct = spectral_energy(u_direct, eigvecs)
E_reg = spectral_energy(u_reg, eigvecs)
E_true /= E_true.sum()
E_direct /= E_direct.sum()
E_reg /= E_reg.sum()

plt.figure(figsize=(8, 5))
plt.semilogy(E_true, label="True", linewidth=2)
plt.semilogy(E_direct, label="Direct inverse", alpha=0.8)
plt.semilogy(E_reg, label="Regularized inverse", alpha=0.8)
plt.xlabel("Spectral mode index (low → high frequency)")
plt.ylabel("Normalized energy")
plt.title("Spectral energy distribution and information flow")
plt.legend()
plt.tight_layout()
plt.show()
