import numpy as np
import matplotlib.pyplot as plt

"""
1D Laplace inverse problem:
    u''(x) = 0 on (0, 1)
with noisy boundary measurements.

We study instability of reconstruction under small noise.
"""

N = 100
x = np.linspace(0, 1, N)
u_true = x
noise_level = 1e-3
u0_noisy = u_true[0] + noise_level * np.random.randn()
u1_noisy = u_true[-1] + noise_level * np.random.randn()
u_recon = u0_noisy + (u1_noisy - u0_noisy) * x
error = np.linalg.norm(u_recon - u_true) / np.linalg.norm(u_true)

plt.figure(figsize=(6, 4))
plt.plot(x, u_true, label="True solution", linewidth=2)
plt.plot(x, u_recon, "--", label="Reconstruction (noisy BCs)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(f"Laplace inverse problem | relative error = {error:.2e}")
plt.legend()
plt.tight_layout()
plt.show()
