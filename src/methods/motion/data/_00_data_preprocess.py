import numpy as np
import cv2

from utils.processing.dft import get_fft


class ColorDomain:
    RGB = 0
    GRAY = 1
    YCrCb = 2
    HSV = 3


def frame_process(frame, color, data_size=8):
    if color == ColorDomain.RGB:
        converted = frame
    elif color == ColorDomain.GRAY:
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif color == ColorDomain.YCrCb:
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    elif color == ColorDomain.HSV:
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    else:
        print('Undefined color type.')
        return None
    resized = cv2.resize(converted, dsize=None, fx=1/data_size, fy=1/data_size)
    return resized


def get_normalized_fft(data, axis=0):
    fft = np.fabs(get_fft(data, axis=axis))
    fft_max = fft.max(axis=0)
    if len(fft_max.shape) == 0:
        fft_max = 1 if fft_max == 0.0 else fft_max
    else:
        fft_max[fft_max==0] = 1
    normed = fft / fft_max
    return normed