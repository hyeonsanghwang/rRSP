import numpy as np
import cv2
import os

from path import data_path
from utils.processing.normalize import zero_centered_normalize

from methods.motion.data._00_data_preprocess import ColorDomain, frame_process
from methods.motion.data._00_similarity_method import *


if __name__ == '__main__':
    # Set parameters
    FPS = 10
    TARGET_FPS = 5
    SKIP_RATIO = FPS // TARGET_FPS
    STRIDE = 10
    WINDOW_SIZE = 64
    DATA_SIZE = 8
    COLOR = ColorDomain.RGB

    # Save paths
    x_path = data_path('train/xs.npy')
    signal_path = data_path('train/signal.npy')
    label_path = data_path('train/label.npy')

    # Set data
    version = '3_3'
    frames = np.load(data_path('npy/'+version+'_frames.npy'))
    signal = np.load(data_path('npy/'+version+'_signal.npy'))
    index = np.array(list(range(WINDOW_SIZE*SKIP_RATIO))) % 2 == 0
    print('Now version :', version)

    save_xs = []
    save_signal = []
    save_label = []

    for i in range(0, frames.shape[0] - SKIP_RATIO * WINDOW_SIZE, 10):
        # Set frame data
        xs = frames[i: i+WINDOW_SIZE*SKIP_RATIO]
        xs = xs[index]
        x_data = []
        for frame in xs:
            x_data.append(frame_process(frame, COLOR, DATA_SIZE))
        x_data = np.array(x_data)

        # Set signal data
        ys = signal[i: i+WINDOW_SIZE*SKIP_RATIO]
        y_data = ys[index]

        # Normalize
        normed_xs = zero_centered_normalize(x_data, axis=0)
        normed_ys = zero_centered_normalize(y_data)

        # Get label
        label = get_pearson_correlation_score(normed_xs, normed_ys)
        thres = 0.6
        label[label >= thres] = 1
        label[label < thres] = 0
        h, w = label.shape
        label = label.reshape((h, w, 1))

        # Set data to save
        save_xs.append(x_data)
        save_signal.append(normed_ys)
        save_label.append(label)

    np_xs = np.array(save_xs).astype(np.uint8)
    np_signal = np.array(save_signal)
    np_label = np.array(save_label)

    print('> Captured data shape')
    print('Xs :', np_xs.shape, '/ Signals :', np_signal.shape, '/ Labels :', np_label.shape)
    print()

    if os.path.exists(x_path) and os.path.exists(signal_path) and os.path.exists(label_path):
        exist_xs = np.load(x_path)
        exist_signal = np.load(signal_path)
        exist_label = np.load(label_path)
        print('> Existing data shape')
        print('Xs :', exist_xs.shape, '/ Signals :', exist_signal.shape, '/ Labels :', exist_label.shape)
        print()

        np_xs = np.append(exist_xs, np_xs, axis=0)
        np_signal = np.append(exist_signal, save_signal, axis=0)
        np_label = np.append(exist_label, save_label, axis=0)
        print('> Total data shape')
        print('Xs :', np_xs.shape, '/ Signals :', np_signal.shape, '/ Labels :', np_label.shape)
        print()

    np.save(x_path, np_xs)
    np.save(signal_path, np_signal)
    np.save(label_path, np_label)
