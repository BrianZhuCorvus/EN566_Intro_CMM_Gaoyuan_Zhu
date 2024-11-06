import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parameters
D = 2  # Diffusion constant
L = 100  # Length
dx = 1.0  # step size
dt = 0.1  # Time step
nx = int(L / dx)  # Number of spatial points
nt = 100  # Number of time steps to evolve

u = np.zeros(nx)
u[nx//2 - 2 : nx//2 + 2] = 1.0  # From 48 to 52


snapshots = []
snapshot_times = []

# Diffusion loop
for n in range(nt):
    u_new = u.copy()
    for i in range(1, nx - 1):
        u_new[i] = u[i] + (D * dt / dx**2) * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new.copy()

    if n % (nt // 10) == 0:
        snapshots.append(u.copy())
        snapshot_times.append(n * dt)


# Plot
for i, u_snapshot in enumerate(snapshots):
    plt.plot(u_snapshot, label=f'Time = {snapshot_times[i]:.2f}')

plt.legend()
plt.xlabel('Position')
plt.ylabel('Density')
plt.show()

# Gaussian function
def gaussian(x, A, sigma):
    return A * np.exp(-(x)**2 / (2 * sigma**2))

def plot_gaussian(x, A, sigma):
    plt.plot()

# fitting
x = np.linspace(-L/2, L/2, nx)
for i, u_snapshot in enumerate(snapshots):
    popt, _ = curve_fit(gaussian, x, u_snapshot)
    A, sigma = popt
    expected_sigma = np.sqrt(2 * D * snapshot_times[i])
    print(f"Time: {snapshot_times[i]:.2f}, Fitted sigma: {sigma:.2f}, Expected sigma: {expected_sigma:.2f}")


