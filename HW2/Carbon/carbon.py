import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Carbon-14 decay simulation.')
    parser.add_argument('--plot', type=int, required=True)
    return parser.parse_args()

# Numbers
T = 5700
tau = T / np.log(2)  # decay constant
N0 = 1e-12  # initial (Kg)
duration = 20000

# analytical solution
def an_sol(t, N0, tau):
    return N0 * np.exp(-t / tau)

#numerical solution, Euler
def nu_sol(N0, tau, delta_t, duration):
    num_steps = int(duration / delta_t) + 1
    time_points = np.linspace(0, duration, num_steps)
    N = np.zeros(num_steps)
    N[0] = N0

    for i in range(1, num_steps):
        a = -N[i-1] / tau  # a is dN over dt
        N[i] = N[i-1] + a * delta_t  # Euler's method

    return time_points, N

# Main function
if __name__ == "__main__":
    args = parse_arguments()
    delta_t = args.plot

    # Calculate numerical solution with the provided time-step width
    time_points, N_nu = nu_sol(N0, tau, delta_t, duration)

    # Analytical solution
    time_points_exact = np.linspace(0, duration, 1000)
    N_an = an_sol(time_points_exact, N0, tau)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points_exact, N_an, label="Analytical Solution", color="black", linewidth=2)
    plt.plot(time_points, N_nu, 'o-', label=f"Numerical Solution (delta_t={delta_t} years)", markersize=4)

    plt.title("Carbon-14 Decay in Ancient Artifacts ")
    plt.xlabel("Time (years)")
    plt.ylabel("Amount of Carbon-14 (kg)")
    plt.legend()
    plt.grid(True)
    plt.show()