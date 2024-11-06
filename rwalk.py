import numpy as np
import matplotlib.pyplot as plt
import argparse


def TwoD_rwalk_plot_walkpath(n):
    x, y = 0, 0
    x_positions = [x]
    y_positions = [y]
    for _ in range(n):
        direction = np.random.choice(4)

        if direction == 0:
            x += 1  # Move +x
        elif direction == 1:
            x -= 1  # Move -x
        elif direction == 2:
            y += 1  # Move +y
        else:
            y -= 1  # Move -y

        # Store the new position
        x_positions.append(x)
        y_positions.append(y)

    plt.figure(figsize=(8, 8))
    plt.plot(x_positions, y_positions, marker="o", color="b", markersize=4, linewidth=1, label="Random Walk Path")
    plt.plot(0, 0, marker="o", color="green", markersize=10, label="Start (0, 0)")
    plt.plot(x, y, marker="o", color="red", markersize=10, label="End Position")

    plt.title("2D Random Walk path with %d step" % n)
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()


def TwoD_rwalk_plot_walkdestination(n, N):
    x, y = 0, 0
    x_positions = [x]
    y_positions = [y]
    for _ in range(N):
        x, y = 0, 0
        for _ in range(n):
            direction = np.random.choice(4)

            if direction == 0:
                x += 1  # Move +x
            elif direction == 1:
                x -= 1  # Move -x
            elif direction == 2:
                y += 1  # Move +y
            else:
                y -= 1  # Move -y

            # Store the new position
            x_positions.append(x)
            y_positions.append(y)
        plt.plot(x, y, marker="o", color=(1, 0, 0, 0.2), markersize=7)
    plt.plot(x, y, marker="o", color=(1, 0, 0, 0.1), markersize=7, label="End Position")
    plt.plot(0, 0, marker="o", color="green", markersize=8, label="Start (0, 0)")
    plt.legend()
    plt.axis('equal')
    plt.title("Destinations after walking %d steps" % n)
    plt.show()

def TwoD_rwalk_Distancex(n_step,N):
    x, y = 0, 0
    x_positions = [x]
    y_positions = [y]
    for _ in range(N):
        x, y = 0, 0
        for _ in range(n_step):
            direction = np.random.choice(4)

            if direction == 0:
                x += 1  # Move +x
            elif direction == 1:
                x -= 1  # Move -x
            elif direction == 2:
                y += 1  # Move +y
            else:
                y -= 1  # Move -y
        x_positions.append(x)
        y_positions.append(y)
    average_x = [sum(x_positions)/(len(x_positions)-1), n_step]
    average_x2 = [sum([x ** 2 for x in x_positions])/(len(x_positions)-1), n_step]
    print(average_x,average_x2)
    return average_x, average_x2

def TwoD_rwalk_Distancexy(n_step,N):
    x, y = 0, 0
    a,b,c,d = 0,0,0,0
    x_positions = [x]
    y_positions = [y]
    r2 = []
    for _ in range(N):
        x, y = 0, 0
        for _ in range(n_step):
            direction = np.random.choice(4)

            if direction == 0:
                x += 1  # Move +x
                a += 1
            elif direction == 1:
                x -= 1  # Move -x
                b += 1
            elif direction == 2:
                y += 1  # Move +y
                c += 1
            else:
                y -= 1  # Move -y
                d += 1
        x_positions.append(x)
        y_positions.append(y)
    #print(a/(a + b + c + d), b/(a + b + c + d), c/(a + b + c + d), d/(a + b + c + d))
    r2 = [sum(x ** 2 + y ** 2 for x, y in zip(x_positions, y_positions))/(len(x_positions) + 1), n_step]
    print(r2)
    return r2

def TwoD_rawalk_Dis_Plot(n_start, n_stop, N_loop):
    x = []
    x2 = []
    y = []
    y2 = []

    for i in range(n_start, n_stop + 1):
        average_x, average_x2 = TwoD_rwalk_Distancex(i, N_loop)
        x.append(average_x[1])
        y.append(average_x[0])
        x2.append(average_x2[1])
        y2.append((average_x2[0]))

    plt.figure()
    plt.scatter(x,y,color='b')
    plt.grid(True)
    plt.xlabel("Number of steps")
    plt.ylabel("⟨$x_n$⟩")
    plt.title("⟨$x_n$⟩ at different steps")

    plt.figure()
    plt.scatter(x2, y2, color='r')
    plt.grid(True)
    plt.xlabel("Number of steps")
    plt.ylabel("⟨$x_n^2$⟩")
    plt.title("⟨$x_n^2$⟩ at different steps")
    plt.show()

def TwoD_rawalk_Dis_Plotxy(n_start, n_stop, N_loop):
    for i in range(n_start, n_stop + 1):
        r2 = TwoD_rwalk_Distancexy(i, N_loop)
        plt.scatter(r2[1], r2[0], color = 'b')
    plt.grid(True)
    plt.xlabel("Number of steps")
    plt.ylabel("⟨$r^2$⟩")
    plt.title("⟨$r^2$⟩ at different steps")
    plt.show()


def part1():
    TwoD_rawalk_Dis_Plot(3, 100, 10000)

def part2():
    TwoD_rawalk_Dis_Plotxy(3, 100, 10000)

def part3(n):
    TwoD_rwalk_plot_walkpath(n)

def part4(step, n_destination):
    TwoD_rwalk_plot_walkdestination(step, n_destination)




# TwoD_rawalk_Dis_Plot(3, 100, 10000)
# print(TwoD_rwalk_Distancexy(5,10))
# TwoD_rawalk_Dis_Plotxy(3,100,10000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type = int, required = True)
    args = parser.parse_args()

    if args.part == 1:
        part1()
    elif args.part == 2:
        part2()
    elif args.part == 3:
        num = int(input("how many steps?"))
        part3(num)
    elif args.part == 4:
        step = int(input("number of steps?"))
        n_destination = int(input("number of destinations shown?"))
        part4(step, n_destination)
    else:
        print("please put in --part=1,2,3 or 4")


if __name__ == "__main__":
    main()