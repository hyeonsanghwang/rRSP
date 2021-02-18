import numpy as np
import cv2

from utils.processing.clustering import clustering_hierarchy
from utils.processing.normalize import zero_centered_normalize
from utils.visualization.signal import show_signal


class ClusteringTester:

    # 설정 가능한 값
    # Min signal num
    # n_cluster

    def __init__(self):
        self.estimated_signal = None

    def process(self, input_data, roi, minimum_signal_num=100):
        # Extract signal from input data
        input_data = input_data[0]
        signals = input_data.transpose((3, 0, 1, 2))
        signals = signals[..., roi == 1.0].transpose(1, 2, 0)[..., 0]
        signals = signals.T

        # Random sampling when too many signals
        idxs = np.random.choice(signals.shape[0], min(minimum_signal_num, signals.shape[0]), replace=False)
        signals = signals[idxs]

        # Zero-centered normalization
        signals = zero_centered_normalize(signals, axis=-1)

        # Set all elements 0 signal to 1
        signal_sum = np.fabs(signals).sum(axis=-1)
        signals[signal_sum == 0.0] += 1.0

        # Clustering
        n_cluster = 4
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

            if self.estimated_signal is None:
                self.estimated_signal = signal1
            else:
                self.estimated_signal[: -1] = self.estimated_signal[1:]
                self.estimated_signal[-1] = signal1[-1]

            show_signal('signal1', signal1, 500)
            show_signal('signal2', signal2, 500)

            return self.estimated_signal
        else:
            return np.zeros((input_data.shape[0], ))