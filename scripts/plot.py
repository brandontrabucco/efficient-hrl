import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot Data")
    parser.add_argument("--event_files", type=str, nargs="+")
    parser.add_argument("--names", type=str, nargs="+")
    parser.add_argument("--output_file", type=str, default="result.png")
    parser.add_argument("--tag", type=str, default="average_eval1_hrl_success", nargs="+")
    parser.add_argument("--title", type=str, default="Learning Curve")
    parser.add_argument("--xlabel", type=str, default="Iteration")
    parser.add_argument("--ylabel", type=str, default="Return")
    args = parser.parse_args()

    plt.clf()
    f = plt.figure(figsize=(10, 5))
    ax = f.add_subplot(111)

    for tag in args.tag:
        outer_values = {x: [[], []] for x in set(args.names)}
        for i, path_to_file in enumerate(args.event_files):
            values = []
            steps = []
            outer_values[args.names[i]][0].append(values)
            outer_values[args.names[i]][1] = steps

            for e in tf.train.summary_iterator(path_to_file):
                for v in e.summary.value:
                    if tag == v.tag:
                        values.append(v.simple_value)
                        steps.append(e.step)

        for name, data in outer_values.items():
            mean = np.mean(data[0], axis=0)
            std = np.std(data[0], axis=0)
            rgb = np.random.uniform(0.0, 1.0, size=3)
            ax.plot(
                data[1],
                mean,
                label=name + " " + tag,
                color=np.append(rgb, 1.0))
            ax.fill_between(
                data[1],
                mean - std,
                mean + std,
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
