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
    return peaks



path = 'D:/respiration/npy/gui2/estimated/*'
paths = glob.glob(path)

expa = np.array([0])
exp1 = np.array([0])
exp2 = np.array([0])
exp3 = np.array([0])

for path in paths:
    print(path)
    if '2' in path:
        continue
    if not ('1' in path or '3' in path or '4' in path):
        continue
    ests, refs = np.load(path)
    for ref, est in zip(refs, ests):
        ref_peaks = peaks(ref)
        est_peaks = peaks(est)

        if ref_peaks.shape[0] == est_peaks.shape[0]:
            p1 = ref_peaks[:-1]
            p2 = ref_peaks[1:]
            ref_ppi = p1 - p2

            p3 = est_peaks[:-1]
            p4 = est_peaks[1:]
            est_ppi = p3 - p4

            diff_ppi = ref_ppi - est_ppi
            if '1' in path:
                exp3 = np.append(exp3, diff_ppi)
            elif '3' in path:
                exp1 = np.append(exp1, diff_ppi)
            elif '4' in path:
                exp2 = np.append(exp2, diff_ppi)
            expa = np.append(expa, diff_ppi)

exp1 = exp1[1:]
exp2 = exp2[1:]
exp3 = exp3[1:]
expa = expa[1:]

print(exp1.shape, exp2.shape, exp3.shape, expa.shape)
print(np.abs(expa).mean()*200, np.abs(exp1).mean()*200, np.abs(exp2).mean()*200, np.abs(exp3).mean()*200)
print(np.abs(expa).std(), np.abs(exp1).std(), np.abs(exp2).std(), np.abs(exp3).std())



