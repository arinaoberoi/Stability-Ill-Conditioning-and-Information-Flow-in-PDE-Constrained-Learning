"""
This script analyzes the singular value spectrum of the discrete Laplacian
operator and demonstrates how ill-conditioning drives instability in
inverse reconstruction tasks.

Key focus:
  - Singular value decay
  - Condition number growth
  - Noise amplification mechanisms

This mirrors stability issues in imaging and inverse PDE problems.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# -----------------------------
# Discretization parameters
# -----------------------------
N = 40
h = 1.0 / (N + 1)

def build_laplacian(N):
    e = np.ones(N)
    T = sp.diags([e, -4*e, e], [-1, 0, 1], shape=(N, N))
    I = sp.eye(N)
    L = (sp.kron(I, T) + sp.kron(T, I)) / (h**2)
    return L

L = build_laplacian(N)

# -----------------------------
# Spectral analysis
# -----------------------------
print("Computing singular values...")
num_sv = 200  # partial spectrum
svals = spla.svds(L, k=num_sv, return_singular_vectors=False)
svals = np.sort(svals)
cond_est = svals.max() / svals.min()
print(f"Estimated condition number: {cond_est:.2e}")

def noise_amplification(L, noise_level=1e-4):
    """Measure amplification of noise through inverse."""
    n = L.shape[0]
    noise = noise_level * np.random.randn(n)
    u_noise = spla.spsolve(L, noise)
    return np.linalg.norm(u_noise) / np.linalg.norm(noise)

noise_levels = np.logspace(-6, -2, 8)
amps = [noise_amplification(L, nl) for nl in noise_levels]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.semilogy(svals, linewidth=2)
plt.xlabel("Index")
plt.ylabel("Singular value")
plt.title("Singular value decay of Laplacian")

plt.subplot(1, 2, 2)
plt.loglog(noise_levels, amps, marker="o")
plt.xlabel("Noise level")
plt.ylabel("Amplification factor")
plt.title("Noise amplification through inverse")

plt.tight_layout()
plt.show()
