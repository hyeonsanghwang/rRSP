import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from utils.visualization.signal import show_signal, signal_to_frame
from utils.processing.normalize import zero_centered_normalize

def peaks(s):
    normed = zero_centered_normalize(s)
    peaks = find_peaks(normed, height=0, distance=5)[0]
    frame, scale = signal_to_frame(normed, 300, sel_points=peaks, ret_scale=True, background=(255, 255, 255),
                                   foreground=(0, 0, 0), padding=5)
    return peaks, frame


def show_plot(s1, s2, peak1, peak2):
    line_size = 4
    point_size = 150
    font_size = 15

    s1 = zero_centered_normalize(s1)
    s2 = zero_centered_normalize(s2)

    plt.figure(figsize=(5, 3))
    plt.plot(s1, linewidth=line_size, zorder=1, label='Reference')
    plt.plot(s2, linewidth=line_size, zorder=3, label='Proposed')

    # plt.scatter(peak1, s1[peak1], c='b', s=point_size, zorder=2, label='Reference peak')
    # plt.scatter(peak2, s2[peak2], c='r', s=point_size, zorder=4, label='Proposed peak')

    plt.legend(loc='lower right', prop={'size': 13})

    plt.xlabel('Frame #', fontsize=font_size)
    plt.ylabel('Normalized amplitude', fontsize=font_size)

    plt.tight_layout()
    plt.show()


path = 'D:/respiration/npy/gui2/estimated/*'
paths = glob.glob(path)

for path in paths:
    print(path)
    if '4' not in path:
        continue
    ests, refs = np.load(path)
    for ref, est in zip(refs, ests):
        ref_peaks, ref_frame = peaks(ref)
        est_peaks, est_frame = peaks(est)
        cv2.imshow('ref', ref_frame)
        cv2.imshow('est', est_frame)
        key = cv2.waitKey(0)

        if key == 27:
            break
        if key == ord(' '):
            break
        if key == 13:
            show_plot(ref, est, ref_peaks, est_peaks)
    if key == 27:
        break

cv2.destroyAllWindows()
