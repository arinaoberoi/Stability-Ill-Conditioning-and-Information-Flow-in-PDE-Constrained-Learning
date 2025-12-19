"""
This script studies how small perturbations in boundary measurements
propagate through the inverse problem for Laplace's equation.

I compare:
  (1) Direct inversion
  (2) Tikhonov-regularized inversion

Key questions:
  - When does low residual error fail to imply stable recovery?
  - How does ill-conditioning manifest as noise amplification?

Designed to mirror imaging-style inverse problems.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

N = 50                  # grid resolution
noise_level = 1e-3      # measurement noise
lambda_reg = 1e-2       # regularization strength

h = 1.0 / (N + 1)
grid = np.linspace(h, 1 - h, N)

def build_laplacian(N):
    """2D finite-difference Laplacian with Dirichlet BCs."""
    e = np.ones(N)
    T = sp.diags([e, -4*e, e], [-1, 0, 1], shape=(N, N))
    I = sp.eye(N)
    L = (sp.kron(I, T) + sp.kron(T, I)) / (h**2)
    return L

L = build_laplacian(N)

X, Y = np.meshgrid(grid, grid)
u_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
u_true = u_true.flatten()

# Forward model (Laplace u = f)
f = -2 * np.pi**2 * u_true

# -----------------------------
# Simulated noisy measurements
# -----------------------------
noise = noise_level * np.random.randn(*f.shape)
f_noisy = f + noise

# -----------------------------
# Direct inversion (ill-conditioned)
# -----------------------------
u_direct = spla.spsolve(L, f_noisy)

# -----------------------------
# Tikhonov-regularized inversion
# -----------------------------
A = L.T @ L + lambda_reg * sp.eye(L.shape[0])
b = L.T @ f_noisy
u_reg = spla.spsolve(A, b)

def rel_error(u_hat, u_true):
    return np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true)

err_direct = rel_error(u_direct, u_true)
err_reg = rel_error(u_reg, u_true)

print("Relative reconstruction error:")
print(f"  Direct inversion:     {err_direct:.3e}")
print(f"  Regularized inversion:{err_reg:.3e}")

def plot_field(u, title):
    plt.imshow(u.reshape(N, N), cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plot_field(u_true, "True solution")

plt.subplot(1, 3, 2)
plot_field(u_direct, "Direct inversion")

plt.subplot(1, 3, 3)
plot_field(u_reg, "Tikhonov-regularized")

plt.tight_layout()
plt.show()
