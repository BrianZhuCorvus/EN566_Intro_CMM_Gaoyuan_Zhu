import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.8  # (m/s^2)
l = 9.8  # Length (m)
gamma = 0.25  # (s^-1)
alpha_D = 0.2  # Driving force(rad/s^2)
theta_0 = 0.1  # Initial (rad)
omega_0 = 0.0  # angular velocity (rad/s)
t_max = 200  #  (s)
dt = 0.01  # Time step (s)


def euler_meth(omega_D):
    t_values = np.arange(0, t_max, dt)
    theta = np.zeros_like(t_values)
    omega = np.zeros_like(t_values)

    theta[0] = theta_0
    omega[0] = omega_0

    for i in range(1, len(t_values)):
        omega[i] = omega[i - 1] - (
                    g / l * theta[i - 1] + 2 * gamma * omega[i - 1] - alpha_D * np.sin(omega_D * t_values[i - 1])) * dt
        theta[i] = theta[i - 1] + omega[i] * dt

    return t_values, theta, omega


def runge_kutta_4(omega_D):
    t_values = np.arange(0, t_max, dt)
    theta = np.zeros_like(t_values)
    omega = np.zeros_like(t_values)

    theta[0] = theta_0
    omega[0] = omega_0

    for i in range(1, len(t_values)):
        t = t_values[i - 1]

        k1_theta = omega[i - 1]
        k1_omega = -g / l * theta[i - 1] - 2 * gamma * omega[i - 1] + alpha_D * np.sin(omega_D * t)

        k2_theta = omega[i - 1] + 0.5 * dt * k1_omega
        k2_omega = -g / l * (theta[i - 1] + 0.5 * dt * k1_theta) - 2 * gamma * (
                    omega[i - 1] + 0.5 * dt * k1_omega) + alpha_D * np.sin(omega_D * (t + 0.5 * dt))

        k3_theta = omega[i - 1] + 0.5 * dt * k2_omega
        k3_omega = -g / l * (theta[i - 1] + 0.5 * dt * k2_theta) - 2 * gamma * (
                    omega[i - 1] + 0.5 * dt * k2_omega) + alpha_D * np.sin(omega_D * (t + 0.5 * dt))

        k4_theta = omega[i - 1] + dt * k3_omega
        k4_omega = -g / l * (theta[i - 1] + dt * k3_theta) - 2 * gamma * (
                    omega[i - 1] + dt * k3_omega) + alpha_D * np.sin(omega_D * (t + dt))

        theta[i] = theta[i - 1] + dt / 6.0 * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
        omega[i] = omega[i - 1] + dt / 6.0 * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)

    return t_values, theta, omega


# Function to extract amplitude and phase shift for different Omega_D values
def get_amplitude_phase(method, omega_D_values):
    amplitudes = []
    phases = []

    for omega_D in omega_D_values:
        t_values, theta, omega = method(omega_D)
        steady_state_theta = theta[int(0.75 * len(theta)):]  # Use last quarter of the data for steady-state
        steady_state_t = t_values[int(0.75 * len(t_values)):]

        amplitude = np.max(steady_state_theta) - np.min(steady_state_theta)  # Approximate amplitude
        phase_shift = np.angle(np.fft.fft(steady_state_theta)[1])  # Approximate phase shift using FFT

        amplitudes.append(amplitude)
        phases.append(phase_shift)

    return amplitudes, phases


# Values of Omega_D
omega_D_values = np.linspace(0.5, 1.5, 10)

# amplitude and phase shift for Euler and Runge-Kutta 4th method
euler_amplitudes, euler_phases = get_amplitude_phase(euler_meth, omega_D_values)
rk_amplitudes, rk_phases = get_amplitude_phase(runge_kutta_4, omega_D_values)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(omega_D_values, euler_amplitudes, label="Euler Amplitude", marker='o')
plt.plot(omega_D_values, rk_amplitudes, label="Runge-Kutta 4 Amplitude", marker='x')
plt.xlabel('Driving Frequency $\\Omega_D$ (rad/s)')
plt.ylabel('Amplitude $\\theta_0(\\Omega_D)$ (rad)')
plt.title('Resonance Curve: Amplitude vs Driving Frequency')
plt.legend()
plt.grid(True)


# Plotting phase shift curves (Phase shift vs Omega_D)
plt.figure(figsize=(10, 6))
plt.plot(omega_D_values, euler_phases, label="Euler Phase Shift", marker='o')
plt.plot(omega_D_values, rk_phases, label="Runge-Kutta 4 Phase Shift", marker='x')
plt.xlabel('Driving Frequency $\\Omega_D$ (rad/s)')
plt.ylabel('Phase Shift $\\phi(\\Omega_D)$ (rad)')
plt.title('Phase Shift vs Driving Frequency')
plt.legend()
plt.grid(True)
plt.show()

