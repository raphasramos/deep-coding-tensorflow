import numpy as np
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def save_metric_plot(path, psnr_l, ssim_l, msssim_l, bpp_l, legend, markers):
    metrics_names = ['psnr', 'ssim', 'msssim']
    all_metrics = [psnr_l, ssim_l, msssim_l]

    for cont, metric in enumerate(all_metrics):
        plt.xlabel('bpp')
        plt.ylabel(metrics_names[cont])
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()
        for c_bpp, c_metric, cnt in zip(bpp_l, metric, range(len(psnr_l))):
            plt.plot(c_bpp, c_metric, marker=markers[cnt], markersize=6,
                        linewidth=2)
        plt.legend(legend, loc='center right')
        plot_name = path + '/' + '_mean_plot_' + str(metrics_names[cont]) + '.pdf'
        plt.savefig(str(plot_name), dpi=360)
        plt.close()

num_metrics = 4
codecs = 4
legend = ['modelo_5', 'modelo_6', 'jpeg', 'jpeg2k']
markers = {0: 'o', 1: 'p', 2: 'P', 3: '*'}


all_dfs = []

all_dfs.append(pd.read_csv('test.csv'))
all_dfs.append(pd.read_csv('_analysis_network.csv'))
all_dfs.append(pd.read_csv('_analysis_jpeg.csv'))
all_dfs.append(pd.read_csv('_analysis_jpeg2k.csv'))

assert(len(all_dfs) == codecs)

levels = all_dfs[0].shape[1]//num_metrics

psnr_l = [list() for x in range(codecs)]
ssim_l = [list() for x in range(codecs)]
msssim_l = [list() for x in range(codecs)]
bpp_l = [list() for x in range(codecs)]

for l in range(levels):
    for d, c in zip(all_dfs, range(codecs)):
        bpp_l[c].append(d['bpp' + str(l)].iloc[-1])
        psnr_l[c].append(d['psnr' + str(l)].iloc[-1])
        ssim_l[c].append(d['ssim' + str(l)].iloc[-1])
        msssim_l[c].append(d['msssim' + str(l)].iloc[-1])

save_metric_plot('..', psnr_l, ssim_l, msssim_l, bpp_l, legend, markers)
