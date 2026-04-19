"""
Section 7 — Glassy Dynamics Analogy
=====================================
Construct a toy multi-minima loss landscape, visualize the energy
surface and gradient descent trajectory. Connect to spin glass physics.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(99)

# ── Glassy loss landscape ─────────────────────────────────────────
# A smooth quadratic + many small random bumps (like a spin glass)
def glassy_loss(x, y, centers, amplitudes, widths, global_scale=0.3):
    """Loss = global bowl + sum of Gaussian bumps (local minima)."""
    L = global_scale * (x**2 + y**2)
    for (cx, cy), a, w in zip(centers, amplitudes, widths):
        L -= a * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * w**2))
    return L

def glassy_grad(x, y, centers, amplitudes, widths, global_scale=0.3):
    """Gradient of the glassy loss."""
    gx = global_scale * 2 * x
    gy = global_scale * 2 * y
    for (cx, cy), a, w in zip(centers, amplitudes, widths):
        e = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * w**2))
        gx += a * (x - cx) / (w**2) * e
        gy += a * (y - cy) / (w**2) * e
    return gx, gy

# ── Random bump parameters ────────────────────────────────────────
n_bumps = 40
centers = np.random.randn(n_bumps, 2) * 2.0
amplitudes = np.random.exponential(0.5, n_bumps)
widths = np.random.uniform(0.3, 0.8, n_bumps)

# ── Gradient descent trajectory ───────────────────────────────────
x0, y0 = 3.5, 3.0
lr = 0.02
n_steps = 800
traj = [(x0, y0)]
x, y = x0, y0
for _ in range(n_steps):
    gx, gy = glassy_grad(x, y, centers, amplitudes, widths)
    x -= lr * gx
    y -= lr * gy
    traj.append((x, y))
traj = np.array(traj)

# ── PLOT 1: 3D surface ────────────────────────────────────────────
xx = np.linspace(-4.5, 4.5, 300)
yy = np.linspace(-4.5, 4.5, 300)
XX, YY = np.meshgrid(xx, yy)
ZZ = glassy_loss(XX, YY, centers, amplitudes, widths)

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XX, YY, ZZ, cmap='inferno', alpha=0.85,
                rstride=3, cstride=3, edgecolor='none')
# Plot trajectory on surface
z_traj = glassy_loss(traj[:, 0], traj[:, 1], centers, amplitudes, widths)
ax.plot(traj[:, 0], traj[:, 1], z_traj + 0.05, 'c-', lw=1.5, alpha=0.9)
ax.scatter([traj[0, 0]], [traj[0, 1]], [z_traj[0] + 0.1],
           c='lime', s=80, marker='*', zorder=10)
ax.scatter([traj[-1, 0]], [traj[-1, 1]], [z_traj[-1] + 0.1],
           c='red', s=80, marker='o', zorder=10)
ax.set_xlabel('θ₁', fontsize=12)
ax.set_ylabel('θ₂', fontsize=12)
ax.set_zlabel('Loss', fontsize=12)
ax.set_title('Glassy Loss Landscape (Spin-Glass Analogy)', fontsize=14)
ax.view_init(elev=35, azim=-60)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "glassy_landscape_3d.png"), dpi=150)
plt.close()
print("  ✓ Saved: glassy_landscape_3d.png")

# ── PLOT 2: Contour + trajectory ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
contour = ax.contourf(XX, YY, ZZ, levels=50, cmap='inferno', alpha=0.9)
plt.colorbar(contour, ax=ax, label='Loss')

# Trajectory with color = time
n = len(traj)
for i in range(n - 1):
    frac = i / n
    color = plt.cm.cool(frac)
    ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-', color=color, lw=1.5, alpha=0.8)

ax.plot(traj[0, 0], traj[0, 1], '*', color='lime', markersize=15,
        markeredgecolor='white', markeredgewidth=1.5, label='Start')
ax.plot(traj[-1, 0], traj[-1, 1], 'o', color='red', markersize=10,
        markeredgecolor='white', markeredgewidth=1.5, label='End')
ax.set_xlabel('θ₁', fontsize=13)
ax.set_ylabel('θ₂', fontsize=13)
ax.set_title('GD Trajectory on Glassy Landscape', fontsize=14)
ax.legend(fontsize=11)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "glassy_trajectory.png"), dpi=150)
plt.close()
print("  ✓ Saved: glassy_trajectory.png")

# ── PLOT 3: Loss along trajectory ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
z_traj = glassy_loss(traj[:, 0], traj[:, 1], centers, amplitudes, widths)
ax.plot(z_traj, color='darkorange', lw=2)
ax.set_xlabel('GD step', fontsize=13)
ax.set_ylabel('Loss', fontsize=13)
ax.set_title('Loss Along GD Trajectory — Slow Relaxation in Glassy Landscape',
             fontsize=14)

# Annotate phases
n4 = n_steps // 4
ax.axvspan(0, n4, alpha=0.1, color='green')
ax.axvspan(n4, 3 * n4, alpha=0.1, color='gold')
ax.axvspan(3 * n4, n_steps, alpha=0.1, color='red')
ax.text(n4 // 2, ax.get_ylim()[1] * 0.9, 'Fast\ndescent', ha='center',
        fontsize=10, color='green', fontweight='bold')
ax.text(2 * n4, ax.get_ylim()[1] * 0.9, 'Slow relaxation\n(glassy regime)',
        ha='center', fontsize=10, color='goldenrod', fontweight='bold')
ax.text(3.5 * n4, ax.get_ylim()[1] * 0.9, 'Trapping\n(memorization)',
        ha='center', fontsize=10, color='red', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "glassy_loss_trajectory.png"), dpi=150)
plt.close()
print("  ✓ Saved: glassy_loss_trajectory.png")

# ── PLOT 4: Eigenvalue connection illustration ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Deep valley = large eigenvalue = fast mode
x_v = np.linspace(-3, 3, 200)
axes[0].plot(x_v, 2.0 * x_v**2, 'b-', lw=3)
axes[0].set_title('Large λ → Deep Valley\n(Fast Mode = Generalization)', fontsize=11)
axes[0].set_xlabel('Mode direction', fontsize=10)
axes[0].set_ylabel('Energy', fontsize=10)
axes[0].annotate('Ball rolls quickly\nto bottom', xy=(0, 0), xytext=(1.5, 8),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='blue'),
                 color='blue')

# Panel 2: Shallow valley = small eigenvalue = slow mode
axes[1].plot(x_v, 0.05 * x_v**2, 'r-', lw=3)
axes[1].set_title('Small λ → Shallow Valley\n(Slow Mode = Memorization)', fontsize=11)
axes[1].set_xlabel('Mode direction', fontsize=10)
axes[1].set_ylabel('Energy', fontsize=10)
axes[1].set_ylim(*axes[0].get_ylim())
axes[1].annotate('Ball barely moves,\ntakes forever', xy=(0, 0), xytext=(1.5, 8),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='red'),
                 color='red')

# Panel 3: Combined — the generalization window
t = np.linspace(0, 50, 500)
lam_large = 2.0
lam_small = 0.05
axes[2].plot(t, 1 - np.exp(-lam_large * t), 'b-', lw=2.5, label=f'Fast mode (λ={lam_large})')
axes[2].plot(t, 1 - np.exp(-lam_small * t), 'r-', lw=2.5, label=f'Slow mode (λ={lam_small})')
tg = 1 / lam_large
tm = 1 / lam_small
axes[2].axvline(tg, ls='--', color='blue', alpha=0.5)
axes[2].axvline(tm, ls='--', color='red', alpha=0.5)
axes[2].axvspan(tg, tm, alpha=0.15, color='gold', label='Generalization Window')
axes[2].set_xlabel('Time t', fontsize=10)
axes[2].set_ylabel('Fraction learned', fontsize=10)
axes[2].set_title('τ_gen < t < τ_mem\n(The Sweet Spot)', fontsize=11)
axes[2].legend(fontsize=9, loc='center right')

plt.suptitle('Physical Analogy: Balls in Valleys → Spectral Relaxation',
             fontsize=13, y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "glassy_analogy.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: glassy_analogy.png")

print("""
  INTERPRETATION
  ──────────────
  • The glassy landscape contains one DEEP global bowl (smooth
    population-level structure) and many SHALLOW bumps (individual
    sample memorization).
  • Gradient descent quickly descends the global bowl — this
    corresponds to learning the smooth distribution (τ_gen).
  • But getting trapped in and escaping tiny local minima is
    incredibly slow — this corresponds to memorization (τ_mem).
  • CONNECTION TO SPIN GLASSES: In spin glass physics, the energy
    landscape has exponentially many near-degenerate local minima.
    Relaxation to the ground state takes time exponential in system
    size. Similarly, memorization in diffusion models takes time
    proportional to N because the "barriers" (small eigenvalues)
    scale as 1/N.
  • SPECTRAL RELAXATION: Each eigenmode relaxes independently.
    Large eigenvalues = fast relaxation (≈ learning the distribution).
    Small eigenvalues = slow relaxation (≈ memorizing individual points).
    Early stopping acts as a spectral filter, cutting off slow modes
    — providing IMPLICIT REGULARIZATION.

  ═══════════════════════════════════════════════════════════════════
           FINAL CONCLUSION
  ═══════════════════════════════════════════════════════════════════
  • Diffusion models learn smooth structure FIRST (large eigenvalues)
  • Memorization directions correspond to SMALL eigenvalues of the
    NTK kernel, which take time 1/λ_min ~ N to be learned
  • Dataset size DELAYS memorization:  τ_mem ∝ N
  • There exists a natural generalization window:
        τ_gen  <  training time  <  τ_mem
    In this window the model generates high-quality samples
    WITHOUT memorizing — enabled by implicit dynamical regularization.
  ═══════════════════════════════════════════════════════════════════
""")
