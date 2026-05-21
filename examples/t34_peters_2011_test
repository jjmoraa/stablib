import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# parameters
# -----------------------------
epsilons = np.linspace(0.01, 5.0, 100)
n_harmonics = 10

# time grid (one period 2π since Ω = 1)
Nt = 4096
t = np.linspace(0, 2*np.pi, Nt, endpoint=False)

# storage: P[m, eps]
P = np.zeros((len(epsilons), n_harmonics))

# -----------------------------
# loop over epsilon
# -----------------------------
for i, eps in enumerate(epsilons):

    # A(t) = exp(eps cos t)
    A = np.exp(eps * np.cos(t))

    # Fourier coefficients via FFT
    Ak = np.fft.fft(A) / Nt

    # take only cosine part (real symmetric signal)
    Ak = np.real(Ak)

    # shift: k=0 center
    Ak = np.fft.fftshift(Ak)

    # harmonic index center
    center = Nt // 2

    # extract harmonics m = 0..M
    for m in range(n_harmonics):
        P[i, m] = np.abs(Ak[center + m])

# -----------------------------
# normalization (Peters-style)
# -----------------------------
P = P / np.sum(P, axis=1, keepdims=True)

# -----------------------------
# plot
# -----------------------------
plt.figure(figsize=(10, 6))

for m in range(n_harmonics):
    plt.plot(epsilons, P[:, m], label=f"m={m}")

plt.xlabel("ε")
plt.ylabel("Modal participation φ_m")
plt.title(r"Peters Case A: $e^{\epsilon \cos t}$ harmonic content")
plt.grid(True)
plt.legend()
plt.show()