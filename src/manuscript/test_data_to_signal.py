import cv2
import numpy as np
import os
import glob

from keras.models import load_model

from utils.processing.clustering import clustering_hierarchy
from utils.processing.normalize import zero_centered_normalize
from utils.visualization.signal import show_signal


def process(input_data, roi):
    # Extract signal from input data
    input_data = input_data[0]
    signals = input_data.transpose((3, 0, 1, 2))

    signals = signals[..., roi == 1.0].transpose(1, 2, 0)[..., 0]
    signals = signals.T

    # Zero-centered normalization
    signals = zero_centered_normalize(signals, axis=-1)

    # Set all elements 0 signal to 1
    signal_sum = np.fabs(signals).sum(axis=-1)
    signals[signal_sum == 0.0] += 1.0

    # Clustering
    n_cluster = 4
    if signals.shape[0] < n_cluster:
        print('ERROR!!!')
        return None

    reverse_signals = -signals
    input_signals = np.append(signals, reverse_signals, axis=0)
    cluster = clustering_hierarchy(input_signals, n_cluster, affinity='cosine', linkage='complete')

    # Divide cluster
    origin_cluster = cluster[: signals.shape[0]]
    reverse_cluster = cluster[signals.shape[0]:]

    # Count number of crossed cluster elements
    cluster_counter = []
    counter = []

    for i in range(n_cluster):
        cluster_counter.append(max((origin_cluster == i).astype(np.int).sum(), 1))
        arr = []
        for j in range(n_cluster):
            mask = (origin_cluster == i) & (reverse_cluster == j)
            arr.append(mask.astype(np.int).sum())
        counter.append(arr)
    cluster_counter = np.array(cluster_counter)
    counter = np.array(counter)

    cluster_ratio = (counter.T / cluster_counter).T
    cluster_max_index = cluster_ratio.argmax(axis=-1)
    cluster_empty_index = cluster_ratio.max(axis=-1) == 0
    cluster_max_index[cluster_empty_index] = -1

    pairs = []
    for i in range(n_cluster):
        if not cluster_empty_index[i]:
            if cluster_max_index[cluster_max_index[i]] == i:
                pairs.append((i, cluster_max_index[i]))
                cluster_empty_index[cluster_max_index[i]] = True

    if pairs:
        cluster_size = []
        for pair in pairs:
            cluster_size.append(cluster_counter[pair[0]] + cluster_counter[pair[1]])
        idx = np.argmax(cluster_size)

        signal1 = signals[origin_cluster == pairs[idx][0]].mean(axis=0)
        signal2 = signals[origin_cluster == pairs[idx][1]].mean(axis=0)

        ret_signal = signal1 - signal2
        return ret_signal

    print('end point')
    return None


data_path = 'D:/respiration/npy/gui2/'
save_path = 'D:/respiration/npy/gui2/estimated/'
model_path = '../../model/detect_roi/model_fcn.h5'

model = load_model(model_path, compile=False)


frame_paths = glob.glob(data_path + '*_frame_*.npy')
signal_paths = glob.glob(data_path + '*_signal_*.npy')


FPS = 10
PROCESS_FPS = 5
WINDOW_SIZE = 64
RESIZE_RATIO = 8

fps_ratio = (FPS / PROCESS_FPS)
buffer_size = int(fps_ratio * WINDOW_SIZE)
index = (np.array(list(range(buffer_size))) / fps_ratio).astype(np.int)
target_index = (index != np.insert(index, 0, -1)[:-1])

for frame_path, signal_path in zip(frame_paths, signal_paths):
    print(frame_path)
    print(signal_path)

    splitted = os.path.split(frame_path)
    name = splitted[-1][8:-4]
    exp_num = splitted[-1][0]
    save_file_name = save_path + exp_num + '_' + name + '.npy'

    if os.path.isfile(save_file_name):
        continue

    frames = np.load(frame_path)
    signal = np.load(signal_path)

    # 대표 이미지
    if exp_num == '1':
        cv2.imwrite(save_path + 'thumbnail/' + name + '.png', frames[0])

    save_datas = [[], []]
    length = frames.shape[0]
    resized = []
    for i, frame in enumerate(frames):
        if i == (length - buffer_size):
            break
        if i >= (200 - buffer_size):
            resized.append(cv2.resize(frame, (640 // RESIZE_RATIO, 480 // RESIZE_RATIO)) / 255.0)
            if len(resized) > buffer_size:
                del resized[0]
            if len(resized) == buffer_size:
                xs = np.array([resized], np.float32)
                xs = xs[:, target_index]
                ref_signal = signal[i-buffer_size: i]
                pred = model.predict(xs)[0][..., 0]
                roi = (pred >= 0.5).astype(np.float32)

                estimated = process(xs, roi)
                if estimated is None:
                    continue
                elif np.fabs(estimated).sum() == 0.0:
                    continue

                resized_ref = cv2.resize(np.array([ref_signal], np.float32), (64, 1))[0]

                coef = np.corrcoef(resized_ref, estimated)[0][1]
                if coef < 0:
                    estimated *= -1

                save_datas[0].append(estimated)
                save_datas[1].append(resized_ref)

    save_np_array = np.array(save_datas)
    print(save_np_array.shape)
    # np.save(save_file_name, save_np_array)
