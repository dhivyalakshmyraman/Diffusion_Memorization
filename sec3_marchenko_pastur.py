"""
Section 3 — Marchenko–Pastur Spectrum
=======================================
Generate a random feature matrix, compute K = (1/N)ΦΦ^T, compare the
empirical eigenvalue histogram with the theoretical MP distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(123)

# ── Parameters ──────────────────────────────────────────────────────
N = 400         # samples
P = 1600        # features
sigma2 = 1.0    # variance of entries
gamma = N / P   # aspect ratio = 0.25

print(f"  N = {N},  P = {P},  γ = N/P = {gamma:.2f}")

# ── Random feature matrix & kernel ─────────────────────────────────
Phi = np.random.randn(N, P) * np.sqrt(sigma2 / P)
K = Phi @ Phi.T    # N×N (note: no 1/N factor so bulk ~ σ²)

eigenvalues = np.linalg.eigvalsh(K)

# ── Theoretical Marchenko–Pastur density ───────────────────────────
lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
lam_plus  = sigma2 * (1 + np.sqrt(gamma)) ** 2

x = np.linspace(max(lam_minus * 0.8, 1e-6), lam_plus * 1.2, 2000)
mp_density = np.zeros_like(x)
mask = (x >= lam_minus) & (x <= lam_plus)
mp_density[mask] = (1 / (2 * np.pi * sigma2 * gamma * x[mask])) * \
                   np.sqrt((lam_plus - x[mask]) * (x[mask] - lam_minus))

print(f"  λ₋ = {lam_minus:.4f}")
print(f"  λ₊ = {lam_plus:.4f}")
print(f"  Empirical range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

# ── PLOT: Marchenko–Pastur ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(eigenvalues, bins=80, density=True, alpha=0.6,
        color='steelblue', edgecolor='white', label='Empirical eigenvalues')
ax.plot(x, mp_density, 'r-', lw=2.5, label='Marchenko–Pastur theory')
ax.axvline(lam_minus, ls='--', color='orange', lw=1.5, label=f'λ₋ = {lam_minus:.3f}')
ax.axvline(lam_plus,  ls='--', color='purple', lw=1.5, label=f'λ₊ = {lam_plus:.3f}')
ax.set_xlabel('Eigenvalue  λ', fontsize=13)
ax.set_ylabel('Density', fontsize=13)
ax.set_title(f'Marchenko–Pastur Distribution  (N={N}, P={P}, γ={gamma:.2f})',
             fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "marchenko_pastur.png"), dpi=150)
plt.close()
print("  ✓ Saved: marchenko_pastur.png")

# ── PLOT: Eigenvalue CDF — shows pile-up at small eigenvalues ──────
fig, ax = plt.subplots(figsize=(10, 5))
sorted_eigs = np.sort(eigenvalues)
cdf = np.arange(1, N + 1) / N
ax.plot(sorted_eigs, cdf, 'b-', lw=2)
ax.axvline(lam_minus, ls='--', color='orange', alpha=0.7)
ax.set_xlabel('Eigenvalue λ', fontsize=13)
ax.set_ylabel('CDF', fontsize=13)
ax.set_title('Cumulative Distribution of Eigenvalues', fontsize=14)
ax.annotate('Pile-up of small eigenvalues\n→ Slow modes → Memorization',
            xy=(lam_minus * 1.2, 0.15), fontsize=11,
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eigenvalue_cdf.png"), dpi=150)
plt.close()
print("  ✓ Saved: eigenvalue_cdf.png")

print("""
  INTERPRETATION
  ──────────────
  • The Marchenko–Pastur law predicts the bulk spectral density of
    K = Φ Φ^T for random Φ with i.i.d. entries.
  • The density has a SHARP LOWER EDGE at λ₋ and UPPER EDGE at λ₊.
  • Near λ₋ the density diverges as ~ 1/√(λ−λ₋), creating a
    PILE-UP of small eigenvalues.
  • These small eigenvalues correspond to memorization directions:
      τ_k = 1/λ_k → very long learning times.
  • As N grows, the smallest eigenvalues shrink ~ 1/N, so
    τ_mem ~ N — memorization is delayed proportionally.
  • Physical analogy: Think of a rubber sheet stretched over many
    small bumps. The sheet easily follows the large bumps (large λ)
    but takes a long time to settle into the tiny crevices (small λ).
""")
