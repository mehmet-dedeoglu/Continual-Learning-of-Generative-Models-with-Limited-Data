from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from config import parse_args_fid


def smooth(input_vec, window=5):
    input_len = len(input_vec)
    output_vec = []
    j = 0
    for i in range(input_len):
        output_vec.append(np.sum(input_vec[j:i + 1]) / (i + 1 - j))
        if i > window - 2:
            j = j + 1
    return np.array(output_vec)


def plot_from_txt(data_path, iters_path, fig_texts):
    y = []
    x = []
    colors_array = list(mat.colors.cnames.keys())
    lines_array = list(mat.lines.lineStyles.keys())
    markers_array = list(mat.markers.MarkerStyle.markers.keys())
    markers_array[7] = 'x'
    for i in range(0, len(data_path)):
        data = []
        iters = []
        data_file = open(data_path[i], "r")
        iter_file = open(iters_path[i], "r")
        ins = []
        for data_line in data_file:
            data.append(float(data_line[:-1]))
            if data[-1] != data[-1]:
                data[-1] = 2000
        for iter_line in iter_file:
            iters.append(float(iter_line[:-1]))
        data_file.close()
        iter_file.close()
        y.append(np.asarray(data, dtype='float64'))
        x.append(np.asarray(iters, dtype='float64'))

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 30}

    mat.rc('font', **font)

    fig, ax = plt.subplots()
    for j in range(0, len(y)):
        y[j] = smooth(y[j], window=5)
        ax.plot(x[j], y[j], linestyle=lines_array[0], marker=markers_array[j+2], linewidth=6, markersize=16)
    ax.set(xlabel=fig_texts[0], ylabel=fig_texts[1], title=fig_texts[2])
    ax.grid()
    fig.savefig(fig_texts[4])
    fig.savefig(fig_texts[5])
    plt.legend(fig_texts[3], fontsize=30)
    plt.show(block=True)


if __name__ == '__main__':
    args = parse_args_fid()

    fig_text = [args.labels[0], args.labels[1], args.labels[2], args.legend_text,
                'score.png', 'score.svg']
    iters = args.iteration_files
    filename_fid = args.fid_files
    plot_from_txt(filename_fid, iters, fig_text)
