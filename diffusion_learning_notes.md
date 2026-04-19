# Diffusion Memorization: Learning Notes

This document provides a comprehensive summary of our Q&A sessions exploring the theoretical and empirical mechanics of how diffusion models learn smooth structures before memorizing datasets.

## Section 2: Random Feature Model
### 1. The Three Plots
* **Kernel Eigenvalue Spectrum:** Displays the eigenvalues of the kernel matrix $K = \frac{1}{N} \Phi \Phi^T$ on a log scale, highlighting the large order of magnitude difference between large eigenvalues (population features) and small eigenvalues (sample-specific noise).
* **Eigenmode Learning Curves:** Contrasts the fast learning trajectories of top "fast" modes vs. the extremely slow convergence of bottom "slow" modes.
* **Generalization Window:** Tracks overall "Sample Quality" vs. "Memorization" over time, shading the "Generalization Window" ($\tau_{gen} < t < \tau_{mem}$) where the model generates high-quality samples without memorizing.

### 2. Analytical vs. "Real" Training
* **$\tau_{gen}$ and $\tau_{mem}$:** Explicit checkpoints calculated based on when the top 20 eigenmodes (generalization) or bottom 20 eigenmodes (memorization) cross a 90% completion threshold.
* **Is there real training?** No, there is no actual gradient descent loop. The script leverages the exact analytical solution for gradient flow ($1 - e^{-\lambda_k t}$) to confidently bypass computational overhead while perfectly simulating continuous-time training physics.

### 3. Understanding the Core Math Shortcut
```python
learned_frac[i, :] = 1.0 - np.exp(-eigenvalues * t)
```
This line calculates exactly what percentage of the pattern has been learned by time `t`. Because `eigenvalues` is an array, it simultaneously computes the fraction learned for *all* modes at once. Modes with huge eigenvalues shrink $e^{-\lambda_k t}$ to 0 rapidly, causing the "fraction learned" to hit 1.0 extremely fast.

---

## Section 3: Marchenko-Pastur
### 4. "Density" in the Marchenko-Pastur Plot
The "Density" label on the y-axis represents the true mathematical **probability density** of finding an eigenvalue at that value on the x-axis. The empirical histogram is normalized (total area = 1.0) so that it acts as a Probability Density Function (PDF) and can be directly visually compared to the theoretical Marchenko-Pastur mathematical equation.

---

## Section 4: Score Spikes
### 5. What is $\sigma_t$?
$\sigma_t$ represents the specific **noise level** added to data at a specific timestep in the forward diffusion process. Large $\sigma_t$ means high noise (heavily blurry dataset), and small $\sigma_t$ means low noise (distinct, sharp, recognizable data points).

### 6. Score Vectors at $\sigma_t = 1.5$ vs $0.2$
* **$\sigma_t = 1.5$ (High Noise):** Vectors point smoothly inward toward the overall center of mass of the dataset. They only see the blurry global shape, guiding generation toward broad, generalizable structure.
* **$\sigma_t = 0.2$ (Low Noise):** The broad vectors shatter into extreme, localized "gravity wells" pointing exactly toward nearest individual training samples. These sharp gradient pulls are explicitly the mathematical mechanism for model memorization.

---

## Section 5: Diffusion Training
### 7. The Two-Moons Dataset
A synthetic 2D toy dataset built purely from trigonometry. It consists of 500 coordinate points mapped into two intertwining crescent moons, with random Gaussian noise added to give the specific moons realistic physical "thickness."

### 8. Quality vs. Memorization Plot
* **Coverage (Y-axis):** The fraction of real training points that have at least one generated sample landing nearby ($\leq 0.15$ distance). High coverage guarantees the model avoids "mode-collapse".
* **Average Distance to Nearest Point:** At Epoch 100, the network generates random noise, placing points far from the true shape, resulting in high average distance (~1.0). Once the shape is learned, every point natively lands accurately on the 2D moons, making the average distance plunge to near 0.
* **Minimal Distance:** Plunges to 0 instantly. Simply because 500 entirely random points are scattered in a small space, pure luck dictates that the absolute "luckiest" single generated point will accidentally land next to a true point.

### 9. Score Matching Loss
The foundational mathematical objective. The network receives a noisy point and must predict an arrow (vector) pointing toward the clean data. The loss is simply the **Mean Squared Error (MSE)** between the network's predicted vector and the mathematically perfect target vector (`-noise / sigma`).

### 10. The Loss Curve "Waves"
* **Light Blue Waves:** The raw, unsmoothed error per-step. It looks like jagged chaos because every training batch utilizes wildly different random data points and noise levels.
* **Dark Blue Line:** A 50-epoch rolling moving average that forcefully isolates the underlying downward optimization trend by removing the raw step-to-step noise.

### 11. Diffusion Sample Evolution
* **Epoch 100:** Scattered random dots.
* **Epoch 500-1000 ($\tau_{gen}$):** Dots smoothly trace out new, original, and continuous coordinates across both crescent patterns perfectly matching the concept (Generalization).
* **Epoch 2000-3000 ($\tau_{mem}$):** Blue dots abandon creating "new" points and collapse exactly on top of the original red training coordinates (Memorization).

---

## Section 6: Dataset Scaling
### 12. What is $N$?
$N$ is the **Dataset Size** (the total number of Two-Moons training points). The core plots confirm the central paper's thesis: linearly scaling up dataset size greatly delays memorization ($\tau_{mem} \propto N$) but barely shifts the time required to generalize ($\tau_{gen}$). 

### 13. The Massive Spike (~400) at Epoch 2250 for N=2000
A visual representation of an accidental **numerical instability**—often called a "flyaway sample." A momentary destabilizing bad gradient shifted network weights; the Langevin sampling mechanism subsequently blindly followed astronomical hallucinated score gradients and aggressively launched a generated point thousands of coordinates out of bounds. This completely destroyed the simple mathematical `average` temporarily. 

### 14. Why is the Generalization Window exactly 4750?
It's an **artificial script ceiling**. The python script has a hardcoded break at `max_epochs=5000`. Huge dataset sizes ($N$) take immensely longer to memorize data, and the script simply runs out of time over 5000 cycles before crossing the required threshold (`average distance < 0.08`). Thus, solving the default math: `tau_mem (Timeout ceiling: 5000) - tau_gen (Near immediate: 250) = 4750`.
