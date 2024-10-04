import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Plot trigonometric functions.")

    # 添加命令行参数
    parser.add_argument("--function", type=str, required=True,
                        help="Comma-separated list of functions with no space in between: cos,sin,sinc")
    parser.add_argument("--write", type=str, help="Output filename to write data.")
    parser.add_argument("--read", type=str, help="Input filename to read data from.")
    parser.add_argument("--print", type=str, choices=['jpeg', 'eps', 'pdf'],
                        help="Format to save the plot.")

    args = parser.parse_args()

    if args.read:
        # 从文件读取数据
        data = np.loadtxt(args.read, delimiter='\t', skiprows=1)
        x = data[:, 0]
        results = data[:, 1:]
        functions = args.function.split(',')
    else:
        # 计算函数值
        x = np.arange(-10, 10.05, 0.05)
        functions = args.function.split(',')
        results = []

        for func in functions:
            if func == 'cos':
                results.append(np.cos(x))
            elif func == 'sin':
                results.append(np.sin(x))
            elif func == 'sinc':
                results.append(np.sinc(x / np.pi))

        results = np.array(results).T

        if args.write:
            # 将结果写入文件
            with open(args.write, 'w') as f:
                header = "x\t" + "\t".join(functions) + "\n"
                f.write(header)
                for i in range(len(x)):
                    line = f"{x[i]:.4f}\t" + "\t".join(f"{results[i, j]:.4f}" for j in range(len(functions))) + "\n"
                    f.write(line)

    # 绘图
    plt.figure()
    for i, func in enumerate(functions):
        plt.plot(x, results[:, i], label=func)

    plt.title("Trigonometric Functions")
    plt.xlabel("x")
    plt.ylabel("Function Value")
    plt.legend()
    plt.grid()

    if args.print:
        plt.savefig(f'plot.{args.print}', format=args.print)
    else:
        plt.show()


if __name__ == "__main__":
    main()
