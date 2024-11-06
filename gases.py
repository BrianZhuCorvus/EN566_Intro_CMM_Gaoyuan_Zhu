import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

from matplotlib.colors import ListedColormap
from scipy.ndimage import label

# Grid size
width, height = 60, 40
grid = np.zeros((height, width), dtype=int)

#0 is emp -1 A 1 B

# Initialize
for y in range(height):
    for x in range(width // 3):
        grid[y, x] = -1  # Species A

for y in range(height):
    for x in range(2 * width // 3, width):
        grid[y, x] = 1  # Species B

#positions of A, B
a_positions = [(y, x) for y in range(height) for x in range(width // 3)]
b_positions = [(y, x) for y in range(height) for x in range(2 * width // 3, width)]

def choose_particle():
    if random.random() < 0.5 :
        y, x = random.choice(a_positions)
        particle_type = -1
    else:
        y, x = random.choice(b_positions)
        particle_type = 1
    return  y, x, particle_type

def random_walk():
    # Choose randomly between A and B particles
    y, x, particle_type = choose_particle()
    # Choose a random dir
    direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
    new_y, new_x = y + direction[0], x + direction[1]

    # Check if the move is within bounds and to an empty cell
    if 0 <= new_y < height and 0 <= new_x < width and grid[new_y, new_x] == 0:
        # Update the grid
        grid[new_y, new_x] = particle_type
        grid[y, x] = 0

        # Update the position list
        if particle_type == -1:
            a_positions.remove((y, x))
            a_positions.append((new_y, new_x))
        else:
            b_positions.remove((y, x))
            b_positions.append((new_y, new_x))
        return
    else:
        random_walk()
n_iterations = 50000

def part1(n = n_iterations, n_snapshot = 5):
    color = [(1, 0, 0, 0.7), 'white', (0, 0, 1, 0.7)]
    cmap = ListedColormap(color)
    for i in range(n + 1):
        random_walk()

        # Visualize at certain intervals
        if i % (n // n_snapshot) == 0:


            plt.imshow(grid, cmap = cmap, origin="lower")
            plt.title(f"Iteration: {i}")
            plt.xlabel("x")
            plt.ylabel("y")

            red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Partical A')
            blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Particel B')
            plt.legend(handles=[red_patch, blue_patch],loc='upper right')

            plt.show()
            plt.pause(0.4)
            plt.cla()
    i += 1
    plt.imshow(grid, cmap=cmap, origin="lower")
    plt.title(f"Iteration: %d" % i )
    plt.xlabel("x")
    plt.ylabel("y")

    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Partical A')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Particel B')
    plt.legend(handles=[red_patch, blue_patch], loc='upper right')

    plt.show()


def calculate_linear_densities():
    n_A = np.sum(grid == -1, axis=0)/800  # Count particles A along each column
    n_B = np.sum(grid == 1, axis=0) /800 # Count particles B along each column
    return n_A, n_B


def part2(n=n_iterations):
    #color = [(1, 0, 0, 0.7), 'white', (0, 0, 1, 0.7)]
    #cmap = ListedColormap(color)
    snapshots = []
    temp = 0
    for i in range(n+1):
        random_walk()

        # Take snapshots and densities at regular intervals
        if i % (n // 5) == 0:

            # Calculate linear densities and store them
            n_A, n_B = calculate_linear_densities()
            snapshots.append((n_A, n_B, i))

    for i, snapshots_n in enumerate(snapshots):
        plt.plot(snapshots_n[0], color='r',label='Density of Particle A' )
        plt.plot(snapshots_n[1], color='b',label='Density of Particle B' )
        plt.title(f"Density of A and B After {snapshots_n[2]} iteration(s)")
        plt.legend()
        plt.show()
    # Plot linear population densities for selected snapshots

def part3(n_trials = 100, n = n_iterations):
    n_A, n_B = 0, 0
    for j in range(n_trials):
        for i in range(n):
            random_walk()
        A, B = calculate_linear_densities()
        n_A += A
        n_B += B
        print(j)
    n_A /= n_trials
    n_B /= n_trials

    #n_A, n_B = calculate_linear_densities()
    plt.plot(n_A,color='r',label="Particle A")
    plt.plot(n_B,color='b',label="Particle B")
    plt.title(f"More Accurate linear density of A and B after {n_trials} trials with {n} iterations")
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, required=True)
    args = parser.parse_args()

    if args.part == 1:
        part1(500000,5)
    elif args.part == 2:
        part2()
    elif args.part == 3:
        part3()
    else:
        print("Please input '--part==1, 2 or 3'")

if __name__ == "__main__":
    main()

