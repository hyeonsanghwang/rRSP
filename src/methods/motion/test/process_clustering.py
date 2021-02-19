import numpy as np
import cv2

from utils.processing.clustering import clustering_hierarchy
from utils.processing.normalize import zero_centered_normalize
from utils.processing.pca import get_pca, get_pca_frame
from utils.visualization.signal import show_signal


class ClusteringTester:

    FRAME_NAME_CLUTER = "Clustering"
    FRAME_NAME_SYMMETRY = "Symmetry"
    # 설정 가능한 값
    # Min signal num
    # n_cluster

    def __init__(self, params):
        self.params = params
        self.estimated_signal = None

    def process(self, input_data, roi, minimum_signal_num=50):
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
        if signals.shape[0] < n_cluster:
            return self.estimated_signal
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
                self.estimated_signal = signal1 - signal2
            else:
                coef1 = np.corrcoef(self.estimated_signal[1:], signal1[:-1])[0][1]
                coef2 = np.corrcoef(self.estimated_signal[1:], signal2[:-1])[0][1]
                self.estimated_signal = signal1 - signal2 if coef1 > coef2 else signal2 - signal1

        if self.params.process_clustering_show_cluster:
            pca_signal = np.append(signals, [[0] * 64], axis=0)
            pca = get_pca(pca_signal, 2)
            frame_pca = get_pca_frame(pca, cluster=cluster[:len(cluster) // 2])
            cv2.imshow(self.FRAME_NAME_CLUTER, frame_pca)

        if self.params.process_clustering_show_symmetry_cluster:
            pca_signal = np.append(input_signals, [[0] * 64], axis=0)
            pca = get_pca(pca_signal, 2)
            frame_pca = get_pca_frame(pca, cluster=cluster)
            cv2.imshow(self.FRAME_NAME_SYMMETRY, frame_pca)

        return self.estimated_signal

    def stop(self):
        self.params.process_clustering_show_cluster = False
        self.params.process_clustering_show_symmetry_cluster = False
        cv2.destroyWindow(self.FRAME_NAME_CLUTER)
        cv2.destroyWindow(self.FRAME_NAME_SYMMETRY)