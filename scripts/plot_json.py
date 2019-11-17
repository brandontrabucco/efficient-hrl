import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot Data")
    parser.add_argument("--json_files", type=str, nargs="+")
    parser.add_argument("--names", type=str, nargs="+")
    parser.add_argument("--output_file", type=str, default="result.png")
    parser.add_argument("--title", type=str, default="Learning Curve")
    parser.add_argument("--xlabel", type=str, default="Iteration")
    parser.add_argument("--ylabel", type=str, default="Return")
    args = parser.parse_args()

    plt.clf()
    f = plt.figure(figsize=(10, 5))
    ax = f.add_subplot(111)

    outer_values = {x: defaultdict(list) for x in set(args.names)}
    for i, path_to_file in enumerate(args.json_files):
        with open(path_to_file, "r") as f:
            data = np.array(json.load(f))
            for x, y in zip(data[:, 1], data[:, 2]):
                outer_values[args.names[i]][x].append(y)

    for name, data in outer_values.items():
        steps = np.array(sorted(data.keys()))
        means = np.array([np.mean(data[s]) for s in steps])
        stds = np.array([np.std(data[s]) for s in steps])
        rgb = np.random.uniform(0.0, 1.0, size=3)
        ax.plot(
            steps,
            means,
            label=name,
            color=np.append(rgb, 1.0))
        ax.fill_between(
            steps,
            means - stds,
            means + stds,
            color=np.append(rgb, 0.2))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)

    ax.set_title(args.title)
    ax.legend()
    plt.savefig(args.output_file)
