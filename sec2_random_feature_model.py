"""
Section 2 — Random Feature Model & Kernel Spectrum Learning Dynamics
=====================================================================
Implements f(x) = φ(x)^T θ, builds the kernel K, simulates gradient
descent, and shows eigenmode-resolved learning curves.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42)

# ── Parameters ──────────────────────────────────────────────────────
N = 200          # number of data points
P = 500          # number of random features
d = 10           # input dimension
lr = 0.01        # learning rate for discrete GD
n_steps = 3000   # training steps

# ── Generate synthetic data ────────────────────────────────────────
X = np.random.randn(N, d)                        # data matrix  N×d
W_feat = np.random.randn(d, P) / np.sqrt(d)      # random weights d×P
Phi = np.maximum(X @ W_feat, 0)                   # ReLU features N×P

# Target: a smooth + noisy function
w_true = np.random.randn(P) / np.sqrt(P)
y = Phi @ w_true + 0.1 * np.random.randn(N)

# ── Build kernel K = (1/N) Φ Φ^T ──────────────────────────────────
K = (Phi @ Phi.T) / N                             # N×N kernel matrix

# ── Eigen-decomposition ───────────────────────────────────────────
eigenvalues, eigenvectors = np.linalg.eigh(K)
# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"  Kernel matrix K: {K.shape}")
print(f"  Top-5 eigenvalues:    {eigenvalues[:5]}")
print(f"  Bottom-5 eigenvalues: {eigenvalues[-5:]}")
print(f"  Condition number:     {eigenvalues[0]/eigenvalues[-1]:.1f}")

# ── Project targets onto eigenbasis ────────────────────────────────
y_eigen = eigenvectors.T @ y   # coefficients in eigenbasis

# ── Analytical solution: f_k(t) = (1 - exp(-λ_k t)) y_k ──────────
times = np.arange(0, n_steps) * lr
n_modes = N

learned_frac = np.zeros((len(times), n_modes))
for i, t in enumerate(times):
    learned_frac[i, :] = 1.0 - np.exp(-eigenvalues * t)

# ── PLOT 1: Kernel Eigenvalue Spectrum ─────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.bar(range(N), eigenvalues, color='steelblue', alpha=0.8, width=1.0)
ax.set_xlabel('Eigenvalue index k', fontsize=13)
ax.set_ylabel('λ_k', fontsize=13)
ax.set_title('Kernel Eigenvalue Spectrum  K = (1/N) Φ Φ^T', fontsize=14)
ax.set_yscale('log')
ax.axhline(y=eigenvalues[-1], color='red', ls='--', alpha=0.7,
           label=f'λ_min = {eigenvalues[-1]:.4f}')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "kernel_eigenvalue_spectrum.png"), dpi=150)
plt.close()
print("  ✓ Saved: kernel_eigenvalue_spectrum.png")

# ── PLOT 2: Eigenmode Learning Curves ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Choose representative modes
modes_fast = [0, 1, 2, 4, 9]          # large eigenvalues
modes_slow = [N-1, N-2, N-3, N-5, N-10]  # small eigenvalues

colors_fast = plt.cm.Reds(np.linspace(0.4, 0.9, len(modes_fast)))
colors_slow = plt.cm.Blues(np.linspace(0.4, 0.9, len(modes_slow)))

ax = axes[0]
for j, k in enumerate(modes_fast):
    ax.plot(times, learned_frac[:, k],
            color=colors_fast[j], lw=2,
            label=f'mode {k}  (λ={eigenvalues[k]:.2f})')
ax.set_xlabel('Effective time (lr × step)', fontsize=12)
ax.set_ylabel('Fraction learned  1−exp(−λ_k t)', fontsize=12)
ax.set_title('Fast Modes (Large Eigenvalues)', fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.95, ls=':', color='gray', alpha=0.5)

ax = axes[1]
for j, k in enumerate(modes_slow):
    ax.plot(times, learned_frac[:, k],
            color=colors_slow[j], lw=2,
            label=f'mode {k}  (λ={eigenvalues[k]:.4f})')
ax.set_xlabel('Effective time (lr × step)', fontsize=12)
ax.set_ylabel('Fraction learned  1−exp(−λ_k t)', fontsize=12)
ax.set_title('Slow Modes (Small Eigenvalues) — Memorization', fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.95, ls=':', color='gray', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eigenmode_learning_curves.png"), dpi=150)
plt.close()
print("  ✓ Saved: eigenmode_learning_curves.png")

# ── PLOT 3: Training step vs sample quality & memorization ────────
# "Sample quality" ∝ contribution of top eigenmodes
# "Memorization"  ∝ contribution of bottom eigenmodes
n_top = 20
n_bot = 20

quality = np.zeros(len(times))
memorization = np.zeros(len(times))

energy_top = np.sum(y_eigen[:n_top] ** 2)
energy_bot = np.sum(y_eigen[-n_bot:] ** 2)

for i, t in enumerate(times):
    frac = 1.0 - np.exp(-eigenvalues * t)
    quality[i] = np.sum((frac[:n_top] * y_eigen[:n_top]) ** 2) / max(energy_top, 1e-12)
    memorization[i] = np.sum((frac[-n_bot:] * y_eigen[-n_bot:]) ** 2) / max(energy_bot, 1e-12)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times, quality, color='green', lw=2.5, label='Sample Quality (top eigenmodes)')
ax.plot(times, memorization, color='red', lw=2.5, label='Memorization (bottom eigenmodes)')
# Mark τ_gen and τ_mem
tau_gen_idx = np.argmax(quality > 0.9)
tau_mem_idx = np.argmax(memorization > 0.9)
tau_gen = times[tau_gen_idx] if tau_gen_idx > 0 else times[-1]
tau_mem = times[tau_mem_idx] if tau_mem_idx > 0 else times[-1]

ax.axvline(tau_gen, color='green', ls='--', alpha=0.6, label=f'τ_gen ≈ {tau_gen:.1f}')
ax.axvline(tau_mem, color='red', ls='--', alpha=0.6, label=f'τ_mem ≈ {tau_mem:.1f}')
ax.axvspan(tau_gen, tau_mem, alpha=0.12, color='gold', label='Generalization Window')
ax.set_xlabel('Effective time', fontsize=13)
ax.set_ylabel('Fraction of energy learned', fontsize=13)
ax.set_title('Generalization Window: τ_gen < t < τ_mem', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "generalization_window.png"), dpi=150)
plt.close()
print("  ✓ Saved: generalization_window.png")

print("""
  INTERPRETATION
  ──────────────
  • Large eigenvalues correspond to smooth, population-level structure.
    These modes are learned FAST → good sample quality early.
  • Small eigenvalues correspond to per-sample memorization directions.
    These modes are learned SLOW → memorization appears late.
  • The gap between τ_gen and τ_mem is the "generalization window."
  • Analogy: Imagine balls rolling in valleys.  Deep valleys (large λ)
    are reached quickly; shallow grooves (small λ) take forever.
""")
