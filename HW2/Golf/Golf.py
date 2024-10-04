import numpy as np
import matplotlib.pyplot as plt
import argparse

# Numbers
g = 9.81  # gravity (m/s^2)
rho = 1.29  # air density (kg/m^3)
A = 0.0014  # cross section, golf ball (m^2)
m = 0.046  # m, golf ball (kg)
v0 = 70  # initial v (m/s)
S0 = 0.25  # Magnus constant
dt = 0.001  # time gap(s)


# Function, trajectory
def cal_trajectory(theta_du, C,dimped=False, magnus=False):
    theta = np.radians(theta_du)  # degree to radians
    v_x, v_y = v0 * np.cos(theta), v0 * np.sin(theta)
    x, y = 0, 0  # initial
    trajectory = []

    v_max = v0
    v_final = v_max  # this will hold the velocity right before hitting the ground

    while y >= 0:
        v = np.sqrt(v_x ** 2 + v_y ** 2)

        if dimped:
            if v >= 14:
                C = 7 / v
            else: C = 0.5

        # Drag force
        F_drag = C * rho * A *( v ** 2 )/ 2
        a_x = -F_drag * v_x / v / m
        a_y = -F_drag * v_y / v / m - g


        # Magnus force
        if magnus:
            F_magnus = S0 * m * v
            a_x -= F_magnus * v_y / v / m
            a_y += F_magnus * v_x / v / m

        # re-give
        v_x += a_x * dt
        v_y += a_y * dt
        x += v_x * dt
        y += v_y * dt
        trajectory.append([x, y])

        if v > v_max:
            v_max = v

        v_final = v

    return np.array(trajectory), v_max, v_final,x


# Plot trajectory for given angles
def plot_trajectory(theta):
    angles = [9, 15, 30, 45] if theta == "all" else [float(theta)]
    plt.figure(figsize=(10, 6))

    for angle in angles:
        # Ideal
        trajectory,v_max,v_final,x = cal_trajectory(angle, 0,dimped=False, magnus=False)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Ideal {angle}°')
        print("v max =", v_max, " ", " v final = ", v_final, " x max = ", x)

        # Smooth, drag
        trajectory,v_max,v_final,x = cal_trajectory(angle, 0.5, dimped=False, magnus=False)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Smooth Drag {angle}°')
        print("v max =", v_max, " ", " v final = ", v_final, " x max = ", x)

        # Dimped, drag (C=1/2 for v<14m/s)
        trajectory,v_max,v_final,x = cal_trajectory(angle, 0.5,dimped=True, magnus=False)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Dimped Drag {angle}°')
        print("v max =", v_max, " ", " v final = ", v_final, " x max = ", x)

        # Dimped, drag, spin
        trajectory,v_max,v_final,x = cal_trajectory(angle, 0.5,dimped=True, magnus=True)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Dimped Drag + Spin {angle}°')
        print("v max =", v_max, " ", " v final = ", v_final, " x max = ", x)

    #plt.title("Golf Ball Trajectories")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=str, default="all")
    args = parser.parse_args()

    plot_trajectory(args.plot)

