import argparse
import numpy as np
import matplotlib.pyplot as plt
import random


def plot_distribution(n, subdivisons):
    random_float = [0] * n
    for i in range(n):
        random_float[i] = random.uniform(0.0, 1.0)
    plt.hist(random_float, bins = subdivisons, density = True, alpha = 0.4, color = 'b', edgecolor = 'b')
    plt.title(f'Probability Distribution of {n} Random Numbers\n'
              f'With {subdivisons} Subdivisions')
    plt.xlabel('value')
    plt.ylabel('probability')
    plt.show()

def plot_box_muller(n, subdivisons, mu = 0 , sigma = 1, x_range=(-5, 5)):
    U1 = [0] * n
    U2 = [0] * n
    X = [0] * n
    Z = [0] * n

    for i in range(n):
        U1[i] = random.uniform(0.0, 1.0)
        U2[i] = random.uniform(0.0, 1.0)
        Z[i] = np.sqrt(-2 * np.log(U1[i])) * np.cos(2 * np.pi * U2[i])
        X[i] = mu + sigma * Z[i]

    x_Gauss = np.linspace(x_range[0], x_range[1], 500)
    y_Gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x_Gauss - mu) / sigma) ** 2)

    plt.plot(x_Gauss, y_Gauss)
    plt.hist(X, bins=subdivisons, density=True, alpha=0.4, color='b', edgecolor='b')
    plt.title(f'Probability Distribution of {n} Random Numbers\n'
              f'With {subdivisons} Subdivisions')
    plt.xlabel('value')
    plt.ylabel('probability')
    plt.show()


def part1():
    for sub in [10, 20, 50, 100]:
        plot_distribution(1000, sub)

    for sub in [10, 20, 50, 100]:
        plot_distribution(1000000, sub)
def part2():
    for sub in [10, 20, 50, 100]:
        plot_box_muller(1000, sub)
    for sub in [10, 20, 50, 100]:
        plot_box_muller(1000000, sub)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type = int, required = True)
    args = parser.parse_args()

    if args.part == 1:
        part1()
    else:
        part2()

if __name__ == "__main__":
    main()
