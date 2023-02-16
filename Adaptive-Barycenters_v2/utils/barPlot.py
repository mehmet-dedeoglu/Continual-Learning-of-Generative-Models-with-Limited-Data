from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from Implementation.utils.config import parse_args_barplot


def list_smooth(input_vec, window=5):
    list_len = len(input_vec)
    output_ = []
    for _i in range(list_len):
        input_len = len(input_vec[_i])
        output_vec = []
        j = 0
        for _k in range(input_len):
            output_vec.append(np.sum(input_vec[_i][j:_k + 1]) / (_k + 1 - j))
            if _k > window - 2:
                j = j + 1
        output_.append(np.array(output_vec, dtype='float64'))
    return output_


def compute_vars(data_in):
    data_np = np.array(data_in, dtype='float64')
    m, n = data_np.shape
    min_vals, max_vals, mean_vals = np.ones(n), np.ones(n), np.ones(n)
    for i_ in range(n):
        min_vals[i_], max_vals[i_], mean_vals[i_] = data_np[:, i_].min(), data_np[:, i_].max(), data_np[:, i_].mean()

    return min_vals, max_vals, mean_vals


def barPlot_generator(fig_texts, args_):
    y = []
    x = []
    colors_array = list(mat.colors.cnames.keys())
    linecolor_idxs = [7, 9, 120, 54, 105, 51]
    fillcolor_idxs = [49, 0, 98, 57, 6, 66]
    hatchcolor_idxs = [129, 130, 68, 109, 99, 146]
    line_colors = [colors_array[xx] for xx in linecolor_idxs]
    fill_colors = [colors_array[xx] for xx in fillcolor_idxs]
    hatch_colors = [colors_array[xx] for xx in hatchcolor_idxs]
    lines_array = list(mat.lines.lineStyles.keys())
    hatch_array = ['...', '///', '\\\\\\', '|||', '---', '+++', 'xxx', 'o', 'O', '*']
    markers_array = list(mat.markers.MarkerStyle.markers.keys())
    markers_array[7] = 'x'

    for i in range(len(args_.exp_number)):
        sample_iters, sample_data, iters = [], [], []
        for j in range(args_.exp_number[i]):
            data_file = open(args_.score_files[i][j], "r")
            data_file = data_file.read().splitlines()
            iter_file = open(args_.iteration_files[i][j], "r")
            iter_file = iter_file.read().splitlines()
            assert len(data_file) == len(iter_file)
            extracted = np.array([[float(iter_file[a]), float(data_file[a])] for a in range(len(data_file))
                                  if a % args_.sample_freq == 0])
            iters, data = extracted[10:, 0], extracted[10:, 1]
            sample_iters.append(iters)
            sample_data.append(data)
        # Compute the mean, max and min values
        min_, max_, mean_ = compute_vars(list_smooth(sample_data, window=args_.smooth_samples))
        y.append([min_, max_, mean_])
        x.append(iters)

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 26}

    mat.rc('font', **font)

    fig = plt.figure(figsize=(12, 8))
    for k in range(len(y)):
        plt.plot(x[k], y[k][2], linestyle=lines_array[0], marker=markers_array[k + 2], linewidth=3,
                 markersize=9, color=line_colors[k])
    for k in range(len(y)):
        plt.plot(x[k], y[k][0], linestyle=lines_array[k+1], linewidth=1, color=hatch_colors[k])
        plt.plot(x[k], y[k][1], linestyle=lines_array[k+1], linewidth=1, color=hatch_colors[k])
        # plt.fill_between(x[k], y[k][0], y[k][1], facecolor=hatch_colors[k])
        plt.fill_between(x[k], y[k][0], y[k][1], facecolor='None', hatch=hatch_array[k], edgecolor=hatch_colors[k])
    plt.xlabel(fig_texts[0])
    plt.ylabel(fig_texts[1])
    plt.title(fig_texts[2])
    plt.grid()
    plt.legend(fig_texts[3], fontsize=26)
    plt.show(block=True)
    fig.savefig(fig_texts[4])
    fig.savefig(fig_texts[5])

    fldrs = args_.input_file.split('/')[:-1]
    name_ = fldrs[0]
    for kk in range(1, len(fldrs)):
        name_ = name_ + '/' + fldrs[kk]
    name_ = name_ + '/' + args_.input_file.split('/')[-1].split('.')[-2] + '_metrics.txt'
    metrics = []
    for k in range(len(y)):
        argmaxmean_ = np.argmax(y[k][2])
        max_mean = y[k][2][argmaxmean_]
        max_max = y[k][1][argmaxmean_]
        max_min = y[k][0][argmaxmean_]
        argminmean_ = np.argmin(y[k][2])
        min_mean = y[k][2][argminmean_]
        min_max = y[k][1][argminmean_]
        min_min = y[k][0][argminmean_]
        max_max_max = np.max(y[k][1])
        min_min_min = np.min(y[k][0])
        metrics.append([max_mean, max_max, max_min, min_mean, min_max, min_min, max_max_max, min_min_min,
                        y[k][2][-1], y[k][1][-1], y[k][0][-1]])
    np.savetxt(name_, metrics, delimiter=', ')


if __name__ == '__main__':
    args = parse_args_barplot()

    fig_text = [args.labels[0], args.labels[1], args.labels[2], args.legend_text,
                'score.png', 'score.svg']
    barPlot_generator(fig_text, args)