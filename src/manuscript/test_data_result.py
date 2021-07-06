import csv
import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, filtfilt, butter, medfilt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

from utils.visualization.signal import show_signal, signal_to_frame
from utils.visualization.common import draw_fps
from utils.processing.dft import get_fft, get_ifft, band_pass_filtering, get_fft_freq
from utils.processing.normalize import zero_centered_normalize

from PyEMD import EMD

female = ['jiwon', 'mikyung', 'nahye', 'sugyeong', 'chaewon', 'nayeon', 'soeui']
male = ['hyeonsang', 'kyungwon', 'woohyuk', 'kunha', 'kunyoung', 'seunggun', 'siwon']

bright = ['jiwon', 'mikyung', 'nahye', 'sugyeong', 'hyeonsang', 'kyungwon', 'woohyuk']
dark = ['chaewon', 'nayeon', 'soeui', 'kunha', 'kunyoung', 'seunggun', 'siwon']


def calc_bpm(signal, name, visualize):
    normed = zero_centered_normalize(signal)
    bpms = []
    for i in range(3):
        conv_size = 1 + (i * 2)
        conv = np.convolve(normed, np.ones((conv_size,)) / conv_size, 'valid')

        # processing
        peaks = find_peaks(conv, height=0, distance=5)[0]

        if visualize and i == 0:
            frame, scale = signal_to_frame(conv, 300, sel_points=peaks, ret_scale=True, background=(255,255,255), foreground=(0, 0, 0), padding=5)
            cv2.imshow(name, frame)

        if peaks.shape[0] < 2:
            return []

        peak1 = peaks[1:]
        peak2 = peaks[:-1]
        ppi = (peak1 - peak2).mean()
        bpm = 60 / (ppi/ 5)
        bpms.append(bpm)

    return bpms

def get_bpms(target_path, visualize=False):
    paths = glob.glob(target_path + '*.npy')

    bpms = []
    names = []
    corrs = []
    for path in paths:
        name = os.path.split(path)[-1][:-4]
        names.append(name)
        bpms.append([])
        corrs.append([])

        data = np.load(path)
        estimated = data[0]
        reference = data[1]

        for x, y in zip(estimated, reference):
            # correlation
            coef = np.corrcoef(x, y)[0][1]
            corrs[-1].append(coef)

            # calc bpm
            x_bpms = calc_bpm(x, 'x', visualize)
            y_bpms = calc_bpm(y, 'y', visualize)

            diff_min = 9999
            bpm = None
            for x_bpm, y_bpm in zip(x_bpms, y_bpms):
                if x_bpm != 0 and y_bpm != 0:
                    diff = np.fabs(x_bpm-y_bpm)
                    if diff_min > diff:
                        diff_min = diff
                        bpm = (x_bpm, y_bpm)

            if bpm is not None:
                bpms[-1].append(bpm)

            # show
            if visualize:
                if len(corrs[-1]) == 1:
                    print(path)
                show_signal('estimated', x, 300, background=(255, 255, 255), foreground=(0, 0, 0), padding=5)
                show_signal('reference', y, 300, background=(255, 255, 255), foreground=(0, 0, 0), padding=5)

                key = cv2.waitKey(0)
                if key == 27:
                    break
                elif key == 's':
                    np.save('sample.npy', [x, y])
    return bpms, names, corrs

def show_bpms(name, x_bpm, y_bpm):
    scale_max = max(x_bpm.max(), y_bpm.max())
    scale_min = min(x_bpm.min(), y_bpm.min())
    frame = signal_to_frame(y_bpm, 1000, scale=(scale_min, scale_max))
    frame = signal_to_frame(x_bpm, frame=frame, scale=(scale_min, scale_max), foreground=(0, 255, 255))
    cv2.imshow(name, frame)

def bpm_process(bpms, names, visualize=False):
    new_bpms = []
    for bpm_list, name in zip(bpms, names):
        np_bpm = np.array(bpm_list)
        x_bpm = np_bpm[:, 0]
        y_bpm = np_bpm[:, 1]

        med_size = 15

        x_med = medfilt(x_bpm, med_size)
        y_med = medfilt(y_bpm, med_size)

        meds = np.concatenate(([x_med], [y_med]), axis=0).T
        new_bpms.append(meds)

        if visualize:
            show_bpms('before', x_bpm, y_bpm)
            show_bpms('after', x_med, y_med)
            key = cv2.waitKey(0)
            if key == 27:
                break

    return new_bpms



def bland_altman_plot(data1, data2, ax=None, fig=None):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    # Heatmap
    xy = np.vstack([mean, diff])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = mean[idx], diff[idx], z[idx]

    if ax is not None:
        ax.scatter(x, y, c=z, s=point_size, cmap=plt.cm.jet)
        ax.axhline(md, color='gray', linestyle='-')
        ax.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        ax.axhline(md - 1.96 * sd, color='gray', linestyle='--')

    else:
        plt.ylim([-6, 6])
        plt.scatter(x, y, c=z, s=point_size, cmap=plt.cm.jet)
        plt.axhline(md, color='gray', linestyle='-')
        plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

        q = np.polyfit(x, y, 1)
        print('bland altman r2', r2_score(x, y), q)
        p = np.poly1d(q)
        plt.plot(x, p(x), "k-")

        plt.xlabel('(REFERENCE+PROPOSED)/2 (bpm)', fontsize=font_size)
        plt.ylabel('PROPOSED-REFERENCE (bpm)', fontsize=font_size)
        plt.colorbar()

        print('mean: ', md, 'std: ', sd)

def scatter_plot(data1, data2, ax=None, fig=None):
    xy = np.vstack([data1, data2])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = data1[idx], data2[idx], z[idx]

    if ax is None:
        plt.scatter(x, y, c=z, s=point_size, cmap=plt.cm.jet)
        plt.xlabel('PROPOSED (bpm)', fontsize=font_size)
        plt.ylabel('REFERENCE (bpm)', fontsize=font_size)

        plt.colorbar()

        q = np.polyfit(x, y, 1)
        p = np.poly1d(q)
        plt.plot(x, p(x), "k-")
        print("y=%.6fx+(%.6f)"%(q[0],q[1]))
    else:
        ax.scatter(x, y, c=z, s=point_size, cmap=plt.cm.jet)


def total_analysis(bpms, corrs, names, exps=(1, 2, 3, 4)):
    total_mse = np.array([])
    total_corr = np.array([])

    total_bpm1 = np.array([])
    total_bpm2 = np.array([])

    for bpm_list, name, corr in zip(bpms, names, corrs):
        exp_num = name[0]
        if int(exp_num) not in exps:
            continue

        print(name)
        np_bpms = np.array(bpm_list)
        total_bpm1 = np.append(total_bpm1, np_bpms[:, 0])
        total_bpm2 = np.append(total_bpm2, np_bpms[:, 1])

        mses = np.fabs(np_bpms[:, 0] - np_bpms[:, 1])
        np_corr = np.array(corr)

        total_mse = np.append(total_mse, mses)
        total_corr = np.append(total_corr, np_corr)

        print("[COR] Count: %d\tMEAN: %.03f\tSTD: %.03f" % (np_corr.shape[0], np_corr.mean(), np_corr.std()))
        print("[MSE] Count: %d\tMEAN: %.03f\tSTD: %.03f" % (mses.shape[0], mses.mean(), mses.std()))
        print()

    print('Total result')
    print("[COR] Count: %d\tMEAN: %.03f\tSTD: %.03f" % (total_corr.shape[0], total_corr.mean(), total_corr.std()))
    print("[MSE] Count: %d\tMEAN: %.03f\tSTD: %.03f" % (total_mse.shape[0], total_mse.mean(), total_mse.std()))

    print('R2 : ', r2_score(total_bpm1, total_bpm2))

    x_data = total_bpm1
    y_data = total_bpm2

    plt.subplot(1, 2, 1)
    bland_altman_plot(x_data, y_data)

    plt.subplot(1, 2, 2)
    scatter_plot(x_data, y_data)


    # plt.tight_layout()

    plt.show()

def exp_analysis(bpms, corrs, names, exps=(1, 2, 3, 4)):
    exp_mse = [np.array([]), np.array([]), np.array([]), np.array([])]
    exp_corr = [np.array([]), np.array([]), np.array([]), np.array([])]
    exp_bpm_x = [np.array([]), np.array([]), np.array([]), np.array([])]
    exp_bpm_y = [np.array([]), np.array([]), np.array([]), np.array([])]
    exp_bpm_list = [[], [], [], []]

    for bpm_list, name, corr in zip(bpms, names, corrs):
        exp_num = name[0]
        if int(exp_num) not in exps:
            continue

        np_corr = np.array(corr)
        np_bpms = np.array(bpm_list)
        mses = np.fabs(np_bpms[:, 0] - np_bpms[:, 1])
        # if 'kunha' in name:
        #     continue
        # if 'kyungwon' in name:
        #     continue
        if '1' in name:
            exp_mse[0] = np.append(exp_mse[0], mses)
            exp_corr[0] = np.append(exp_corr[0], np_corr)
            exp_bpm_x[0] = np.append(exp_bpm_x[0], np_bpms[:, 0])
            exp_bpm_y[0] = np.append(exp_bpm_y[0], np_bpms[:, 1])
            exp_bpm_list[0].append([np_bpms[:, 0], np_bpms[:, 1]])
        elif '2' in name:
            exp_mse[1] = np.append(exp_mse[1], mses)
            exp_corr[1] = np.append(exp_corr[1], np_corr)
            exp_bpm_x[1] = np.append(exp_bpm_x[1], np_bpms[:, 0])
            exp_bpm_y[1] = np.append(exp_bpm_y[1], np_bpms[:, 1])
            exp_bpm_list[1].append([np_bpms[:, 0], np_bpms[:, 1]])
        elif '3' in name:
            exp_mse[2] = np.append(exp_mse[2], mses)
            exp_corr[2] = np.append(exp_corr[2], np_corr)
            exp_bpm_x[2] = np.append(exp_bpm_x[2], np_bpms[:, 0])
            exp_bpm_y[2] = np.append(exp_bpm_y[2], np_bpms[:, 1])
            exp_bpm_list[2].append([np_bpms[:, 0], np_bpms[:, 1]])
        elif '4' in name:
            exp_mse[3] = np.append(exp_mse[3], mses)
            exp_corr[3] = np.append(exp_corr[3], np_corr)
            exp_bpm_x[3] = np.append(exp_bpm_x[3], np_bpms[:, 0])
            exp_bpm_y[3] = np.append(exp_bpm_y[3], np_bpms[:, 1])
            exp_bpm_list[3].append([np_bpms[:, 0], np_bpms[:, 1]])

    for i in range(4):
        print('EXP'+str(i+1)+' result')
        print("[COR] Count: %d\tMEAN: %.03f\tSTD: %.03f" % (exp_corr[i].shape[0], exp_corr[i].mean(), exp_corr[i].std()))
        print("[MSE] Count: %d\tMEAN: %.03f\tSTD: %.03f" % (exp_mse[i].shape[0], exp_mse[i].mean(), exp_mse[i].std()))
        print()

    for i in range(4):
        x_data = exp_bpm_x[i]
        y_data = exp_bpm_y[i]

        plt.subplot(4, 2, i * 2 + 1)
        bland_altman_plot(x_data, y_data)

        plt.subplot(4, 2, (i + 1) * 2)
        scatter_plot(x_data, y_data)

        print('R2 : ', r2_score(x_data, y_data))
        print()
    plt.show()

    plt.rcParams.update({'font.size': 22})
    for i in range(1):
        i = 3
        for j, (s1, s2) in enumerate(exp_bpm_list[i]):
            if j > 8:
                break
            # plt.subplot(14, 4, j * 4 + i + 1)
            plt.subplot(3, 3, j + 1)
            plt.ylim(1, 45)
            plt.xlabel('Time')
            plt.ylabel('bpm')
            plt.plot(s2, linewidth=5, color='b')
            plt.plot(s1, linewidth=5, color='r')
    plt.show()

def gender_analysis(bpms, corrs, names, exps=(1, 2, 3, 4)):
    gender_mse = [[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]
    gender_corr = [[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]
    gender_bpm_x = [[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]
    gender_bpm_y = [[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]

    for bpm_list, name, corr in zip(bpms, names, corrs):
        exp_num = name[0]
        if int(exp_num) not in exps:
            continue

        np_corr = np.array(corr)
        np_bpms = np.array(bpm_list)
        mses = np.fabs(np_bpms[:, 0] - np_bpms[:, 1])

        exp_i = 0
        for e in exps:
            if e == int(exp_num):
                break
            exp_i += 1
        if name[2:] in female:
            gender_mse[0][exp_i] = np.append(gender_mse[0][exp_i], mses)
            gender_corr[0][exp_i] = np.append(gender_corr[0][exp_i], np_corr)
            gender_bpm_x[0][exp_i] = np.append(gender_bpm_x[0][exp_i], np_bpms[:, 0])
            gender_bpm_y[0][exp_i] = np.append(gender_bpm_y[0][exp_i], np_bpms[:, 1])
        elif name[2:] in male:
            gender_mse[1][exp_i] = np.append(gender_mse[1][exp_i], mses)
            gender_corr[1][exp_i] = np.append(gender_corr[1][exp_i], np_corr)
            gender_bpm_x[1][exp_i] = np.append(gender_bpm_x[1][exp_i], np_bpms[:, 0])
            gender_bpm_y[1][exp_i] = np.append(gender_bpm_y[1][exp_i], np_bpms[:, 1])

    print('[Total result]')
    for i in range(2):
        if i == 0:
            print('Female result')
        else:
            print('Male result')
        total_corr = np.hstack(gender_corr[i])
        total_mse = np.hstack(gender_mse[i])
        print("[COR] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_corr.shape[0], total_corr.mean(), total_corr.std()))
        print("[MSE] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_mse.shape[0], total_mse.mean(), total_mse.std()))
        print()

        x_data = np.hstack(gender_bpm_x[i])
        y_data = np.hstack(gender_bpm_y[i])

        plt.subplot(2, 2, i * 2 + 1)
        bland_altman_plot(x_data, y_data)

        plt.subplot(2, 2, (i + 1) * 2)
        scatter_plot(x_data, y_data)

        print('R2 : ', r2_score(x_data, y_data))
        print()
    plt.show()

    print('\n[Exp result]')
    for i in range(2):
        if i == 0:
            print('Female result')
        else:
            print('Male result')
        for j, (x_data, y_data, total_corr, total_mse) in enumerate(
                zip(gender_bpm_x[i], gender_bpm_y[i], gender_corr[i], gender_mse[i])):
            print(
                "[COR] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_corr.shape[0], total_corr.mean(), total_corr.std()))
            print("[MSE] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_mse.shape[0], total_mse.mean(), total_mse.std()))
            print()

            plt.subplot(6, 2, (i * 6) + j * 2 + 1)
            plt.ylim(-6, 6)
            bland_altman_plot(x_data, y_data)

            plt.subplot(6, 2, (i * 6) + (j + 1) * 2)
            scatter_plot(x_data, y_data)

            print('R2 : ', r2_score(x_data, y_data))
            print()

    plt.show()

def bright_analysis(bpms, corrs, names, exps=(1, 2, 3, 4)):
    bright_mse = [[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]
    bright_corr = [[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]
    bright_bpm_x = [[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]
    bright_bpm_y = [[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]

    for bpm_list, name, corr in zip(bpms, names, corrs):
        exp_num = name[0]
        if int(exp_num) not in exps:
            continue

        np_corr = np.array(corr)
        np_bpms = np.array(bpm_list)
        mses = np.fabs(np_bpms[:, 0] - np_bpms[:, 1])

        exp_i = 0
        for e in exps:
            if e == int(exp_num):
                break
            exp_i += 1
        if name[2:] in bright:
            bright_mse[0][exp_i] = np.append(bright_mse[0][exp_i], mses)
            bright_corr[0][exp_i] = np.append(bright_corr[0][exp_i], np_corr)
            bright_bpm_x[0][exp_i] = np.append(bright_bpm_x[0][exp_i], np_bpms[:, 0])
            bright_bpm_y[0][exp_i] = np.append(bright_bpm_y[0][exp_i], np_bpms[:, 1])
        elif name[2:] in dark:
            bright_mse[1][exp_i] = np.append(bright_mse[1][exp_i], mses)
            bright_corr[1][exp_i] = np.append(bright_corr[1][exp_i], np_corr)
            bright_bpm_x[1][exp_i] = np.append(bright_bpm_x[1][exp_i], np_bpms[:, 0])
            bright_bpm_y[1][exp_i] = np.append(bright_bpm_y[1][exp_i], np_bpms[:, 1])

    print('[Total result]')
    for i in range(2):
        if i == 0:
            print('Bright result')
        else:
            print('Dark result')
        total_corr = np.hstack(bright_corr[i])
        total_mse = np.hstack(bright_mse[i])
        print("[COR] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_corr.shape[0], total_corr.mean(), total_corr.std()))
        print("[MSE] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_mse.shape[0], total_mse.mean(), total_mse.std()))
        print()

        x_data = np.hstack(bright_bpm_x[i])
        y_data = np.hstack(bright_bpm_y[i])

        plt.subplot(2, 2, i * 2 + 1)
        bland_altman_plot(x_data, y_data)

        plt.subplot(2, 2, (i + 1) * 2)
        scatter_plot(x_data, y_data)

        print('R2 : ', r2_score(x_data, y_data))
        print()
    plt.show()

    print('\n[Exp result]')
    for i in range(2):
        if i == 0:
            print('Bright result')
        else:
            print('Dark result')
        for j, (x_data, y_data, total_corr, total_mse) in enumerate(zip(bright_bpm_x[i], bright_bpm_y[i], bright_corr[i], bright_mse[i])):
            print("[COR] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_corr.shape[0], total_corr.mean(), total_corr.std()))
            print("[MSE] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_mse.shape[0], total_mse.mean(), total_mse.std()))
            print()

            plt.subplot(6, 2, (i*6) + j * 2 + 1)
            plt.ylim(-6, 6)
            bland_altman_plot(x_data, y_data)

            plt.subplot(6, 2, (i*6) + (j + 1) * 2)
            scatter_plot(x_data, y_data)

            print('R2 : ', r2_score(x_data, y_data))
            print()

    plt.show()


def both_analysis(bpms, corrs, names, exps=(1, 2, 3, 4)):
    both_mse = [[np.array([]), np.array([]), np.array([])],[np.array([]), np.array([]), np.array([])],[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]
    both_corr = [[np.array([]), np.array([]), np.array([])],[np.array([]), np.array([]), np.array([])],[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]
    both_bpm_x = [[np.array([]), np.array([]), np.array([])],[np.array([]), np.array([]), np.array([])],[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]
    both_bpm_y = [[np.array([]), np.array([]), np.array([])],[np.array([]), np.array([]), np.array([])],[np.array([]), np.array([]), np.array([])], [np.array([]), np.array([]), np.array([])]]

    for bpm_list, name, corr in zip(bpms, names, corrs):
        exp_num = name[0]
        if int(exp_num) not in exps:
            continue

        np_corr = np.array(corr)
        np_bpms = np.array(bpm_list)
        mses = np.fabs(np_bpms[:, 0] - np_bpms[:, 1])

        exp_i = 0
        for e in exps:
            if e == int(exp_num):
                break
            exp_i += 1
        if name[2:] in bright and name[2:] in female:
            option_idx = 0
        elif name[2:] in bright and name[2:] in male:
            option_idx = 1
        elif name[2:] in dark and name[2:] in female:
            option_idx = 2
        elif name[2:] in dark and name[2:] in male:
            option_idx = 3
        both_mse[option_idx][exp_i] = np.append(both_mse[option_idx][exp_i], mses)
        both_corr[option_idx][exp_i] = np.append(both_corr[option_idx][exp_i], np_corr)
        both_bpm_x[option_idx][exp_i] = np.append(both_bpm_x[option_idx][exp_i], np_bpms[:, 0])
        both_bpm_y[option_idx][exp_i] = np.append(both_bpm_y[option_idx][exp_i], np_bpms[:, 1])

    print('[Total result]')
    for i in range(4):
        if i == 0:
            print('Bright /  Female')
        elif i == 1:
            print('Bright /  Male')
        elif i == 2:
            print('Dark / Female')
        elif i == 3:
            print('Dark / Male')

        total_corr = np.hstack(both_corr[i])
        total_mse = np.hstack(both_mse[i])
        print("[COR] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_corr.shape[0], total_corr.mean(), total_corr.std()))
        print("[MSE] Count: %d\tMEAN: %.02f\tSTD: %.02f" % (total_mse.shape[0], total_mse.mean(), total_mse.std()))
        print()

        x_data = np.hstack(both_bpm_x[i])
        y_data = np.hstack(both_bpm_y[i])

        plt.subplot(4, 2, i * 2 + 1)
        plt.ylim(-6, 6)
        bland_altman_plot(x_data, y_data)

        plt.subplot(4, 2, (i + 1) * 2)
        scatter_plot(x_data, y_data)

        print('R2 : ', r2_score(x_data, y_data))
        print()
    plt.show()


    plt.show()


if __name__ == '__main__':
    target_path = 'D:/respiration/npy/gui2/estimated/'
    if os.path.isfile(target_path + 'analysis/names.csv'):
        names = []
        processed = []
        corrs = []

        with open(target_path+'analysis/names.csv', 'r') as f:
            rd = csv.reader(f)
            for line in rd:
                name = line[0]
                names.append(name)
                processed.append(np.load(target_path+'analysis/'+name+'_bpm.npy'))
                corrs.append(np.load(target_path + 'analysis/' + name + '_corr.npy'))
    else:
        bpms, names, corrs = get_bpms(target_path=target_path, visualize=True)
        processed = bpm_process(bpms=bpms, names=names, visualize=False)

        with open(target_path+'analysis/names.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            for bpm, name, corr in zip(processed, names, corrs):
                wr.writerow([name])
                np.save(target_path+'analysis/'+name+'_bpm.npy', bpm)
                np.save(target_path + 'analysis/' + name + '_corr.npy', corr)

    point_size = 30
    font_size = 14
    plt.figure(figsize=(13, 23))
    total_analysis(processed, corrs, names, exps=(1, 3, 4))
    # exp_analysis(processed, corrs, names, exps=(1, 2, 3, 4))
    # bright_analysis(processed, corrs, names, exps=(1, 3, 4))
    # gender_analysis(processed, corrs, names, exps=(1, 3, 4))
    # both_analysis(processed, corrs, names, exps=(1, 3, 4))
    # show_results(processed, names, corrs)

    # correlation_mean = [0.93, 0.94, 0.89, 0.93, 0.94, 0.92, 0.92, 0.93]
    # correlation_std = [0.1, 0.06, 0.16, 0.08, 0.08, 0.12, 0.11, 0.09]
    # mse_mean = [0.09, 0.05, 0.09, 0.1, 0.1, 0.07, 0.13, 0.05]
    # mse_std = [0.33, 0.13, 0.36, 0.36, 0.4, 0.24, 0.4, 0.23]
    #
    # plt.errorbar(list(range(len(correlation_mean))), correlation_mean, yerr=correlation_std, fmt='o')
    # plt.errorbar(list(range(len(mse_mean))), mse_mean, yerr=mse_std, fmt='o')
    # plt.show()
