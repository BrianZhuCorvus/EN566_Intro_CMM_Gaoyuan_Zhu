import numpy as np
import matplotlib.pyplot as plt

a = 0.6  # Distance dipole charges
R = 10  # Radius spherical boundary
grid_size = 100  # Number of grid points
tolerance = 1e-5
Q = 1  # Magnitude of the dipole charge

# Create grid
V = np.zeros((grid_size, grid_size))
x = np.linspace(-R, R, grid_size)
y = np.linspace(-R, R, grid_size)
X, Y = np.meshgrid(x, y)  # Create a mesh grid of (x, y) values


# Place charges at (a/2, 0) and (-a/2, 0)
V[grid_size//2, grid_size//2 + int(a/(2*R)*grid_size)] = Q  # Positive charge at (a/2, 0)
V[grid_size//2, grid_size//2 - int(a/(2*R)*grid_size)] = -Q  # Negative charge at (-a/2, 0)

def jacobi_method(V, tolerance):
    V_new = V.copy()
    error = np.inf  # Initialize error
    iterations = 0  # number of iterations

    # Iterate until the potential converges within the specified tolerance
    while error > tolerance:
        V_new[1:-1, 1:-1] = 0.25 * (V[1:-1, :-2] + V[1:-1, 2:] + V[:-2, 1:-1] + V[2:, 1:-1])  # Update interior points
        error = np.max(np.abs(V_new - V))
        V = V_new.copy()
        iterations += 1
    return V, iterations

# Jacobi method
V_final, num_iterations = jacobi_method(V, tolerance)

# Plot Jacobi
plt.figure(figsize=(6, 6))  # Create a square figure for the plot
plt.contour(X, Y, V_final, levels=20)  # Plot the equipotential lines with 20 levels
plt.xlabel('x (units)')
plt.ylabel('y (units)')
plt.title('Equipotential Lines for Electric Dipole using Jacobi Method')
plt.colorbar(label='Potential (V)')
plt.show()

print(f"Number of iterations: {num_iterations}")

# Define a list of tolerances to test
tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
iterations_list = []

# Run Jacobi method for each tolerance and record the number of iterations
for tol in tolerances:
    _, num_iterations = jacobi_method(V, tol)
    iterations_list.append(num_iterations)

# Plot tolerance vs iterations
plt.figure()
plt.plot(tolerances, iterations_list, marker='o')
plt.xscale('log')  # Use logarithmic scale for tolerance
plt.xlabel('Tolerance')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations vs Tolerance for Jacobi Method')
plt.grid(True)
plt.show()


def sor_method(V, tolerance, omega):
    """SOR method for solving Poisson's equation with relaxation factor omega."""
    V_new = V.copy()
    error = np.inf
    iterations = 0

    # Iterate until the potential converges within the specified tolerance
    while error > tolerance:
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                V_new[i, j] = (1 - omega) * V[i, j] + omega * 0.25 * (V[i-1, j] + V[i+1, j] + V[i, j-1] + V[i, j+1])
        error = np.max(np.abs(V_new - V))
        V = V_new.copy()
        iterations += 1
    return V, iterations

# Test SOR method with omega = 1.1
omega = 1.01
V_sor, num_iterations_sor = sor_method(V, tolerance, omega)

# Plot SOR method
plt.figure(figsize=(6, 6))
plt.contour(X, Y, V_sor, levels=20)
plt.xlabel('x (units)')
plt.ylabel('y (units)')
plt.title(f'Equipotential Lines using SOR Method (ω = {omega})')
plt.colorbar(label='Potential (V)')
plt.show()

print(f"Number of iterations (SOR): {num_iterations_sor}")

grid_sizes = [50, 100, 150, 200]  # Different grid sizes to test
iterations_sor = []

for size in grid_sizes:
    V = np.zeros((size, size))  # Reinitialize potential for each grid size
    _, num_iter = sor_method(V, tolerance, omega)
    iterations_sor.append(num_iter)

# Plot grid size vs iterations
plt.figure()
plt.plot(grid_sizes, iterations_sor, marker='o')
plt.xlabel('Grid Size (n)')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations vs Grid Size for SOR Method')
plt.grid(True)
plt.show()
