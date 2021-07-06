import numpy as np
import cv2
from numba import cuda

from keras.models import load_model

# from gui2.module_frame import frame_process
from methods.roi_detection.data._00_data_preprocess import frame_process
from utils.processing.normalize import zero_centered_normalize
from utils.processing.clustering import clustering_hierarchy
from utils.processing.pca import get_pca, get_pca_frame
from utils.visualization.signal import show_signal
from gui.tab_cluster import ClusterParameter


class ClusteringBased:
    FRAME_NAME_CLUSTER = "Clustering"
    FRAME_NAME_SYMMETRY = "Symmetry"

    def __init__(self, params: ClusterParameter, capture_fps):
        self.params = params

        self.capture_fps = capture_fps
        self.model_path = params.model_path
        self.maximum_point = 100

        self.set_parameters(params.window_size,
                            params.process_fps,
                            params.color_domain,
                            params.resize_ratio,
                            params.threshold)


        # ROI detection model
        self.roi_model = None

        # buffers
        self.buffer = []
        self.signal = None
        self.return_signal = np.array([0] * self.window_size, np.float32)

        # Single image buffer
        self.frame = None
        self.processed = None
        self.score = None
        self.roi = None
        self.estimated_signal = None

        # ETC
        self.show_mode = 0
        self.flip = False
        self.show_cluster = False
        self.show_sym_cluster = False

    # -----------------------------------------------------------------------------------------------
    #                                          Set parameters
    # -----------------------------------------------------------------------------------------------
    def set_parameters(self, window_size, process_fps, color_domain, resize_ratio, roi_threshold):
        self.window_size = window_size
        self.process_fps = process_fps
        self.color_domain = color_domain
        self.resize_ratio = resize_ratio
        self.roi_threshold = roi_threshold

        # Select index
        fps_ratio = (self.capture_fps / process_fps)
        self.buffer_size = int(fps_ratio * window_size)
        index = (np.array(list(range(self.buffer_size))) / fps_ratio).astype(np.int)
        self.target_index = (index != np.insert(index, 0, -1)[:-1])

    def set_model_path(self, model_path):
        self.model_path = model_path
        # cuda.select_device(0)
        # cuda.close()
        self.roi_model = None

    # -----------------------------------------------------------------------------------------------
    #                                   Reset memory and buffers
    # -----------------------------------------------------------------------------------------------
    def reset(self):
        # Release model
        # cuda.select_device(0)
        # cuda.close()
        self.roi_model = None

        # Reset buffers
        self.buffer = []
        self.signal = None
        self.frame = None
        self.processed = None
        self.score = None
        self.roi = None

    # -----------------------------------------------------------------------------------------------
    #                                        Main processing
    # -----------------------------------------------------------------------------------------------
    def get_respiration(self, frame):
        # Save as member variable
        self.frame = frame

        # Load ROI detection model
        if self.roi_model is None:
            self.roi_model = load_model(self.model_path, compile=False)

        # Calculate ROI
        frames = self._get_frames(frame)
        if frames is None:
            return self.return_signal
        self.score, self.roi = self._get_roi(frames)

        # Extract ROI signals (# of roi pixels (or maximum_point), window_size)
        signals = self._get_signals_with_roi(frames)

        # Zero-centered normalization
        signals = zero_centered_normalize(signals, axis=-1)

        # Clustering with symmetry data
        n_cluster = 4
        if signals.shape[0] < n_cluster:
            return self.return_signal
        origin_cluster, reverse_cluster = self._get_cluster(signals, n_cluster)

        # Post process
        extracted_signals = self._get_extracted_signals(signals, n_cluster, origin_cluster, reverse_cluster)
        if extracted_signals is None:
            return self.return_signal
        if self.signal is None:
            self.signal = extracted_signals[0] - extracted_signals[1]
        else:
            coef1 = np.corrcoef(self.signal[1:], extracted_signals[0][:-1])[0][1]
            coef2 = np.corrcoef(self.signal[1:], extracted_signals[0][:-1])[0][1]
            self.signal = extracted_signals[0] - extracted_signals[1] if coef1 > coef2 else extracted_signals[1] - extracted_signals[0]

        # Visualize
        if self.show_cluster:
            self._show_cluster_plane(signals, origin_cluster)
        if self.show_sym_cluster:
            self._show_symmetry_cluster_plane(signals, np.append(origin_cluster, reverse_cluster, axis=0))
        return -self.signal if self.flip else self.signal

    def get_show_frame(self):
        if self.show_mode == ClusterParameter.SHOW_MODE_ORIGINAL:
            return self.frame

        elif self.show_mode == ClusterParameter.SHOW_MODE_RESIZED:
            return (self.processed * 255).astype(np.uint8)

        elif self.show_mode == ClusterParameter.SHOW_MODE_SCORE:
            return self.frame if self.score is None else (self.score * 255).astype(np.uint8)

        elif self.show_mode == ClusterParameter.SHOW_MODE_ROI:
            return self.frame if self.roi is None else (self.roi * 255).astype(np.uint8)

    # -----------------------------------------------------------------------------------------------
    #                                 Methods for extracting signal
    # -----------------------------------------------------------------------------------------------
    def _get_frames(self, frame):
        # Process frame
        self.processed = frame_process(frame, self.color_domain, self.resize_ratio) / 255.0
        self.buffer.append(self.processed)

        # Check buffer length
        if len(self.buffer) < self.buffer_size:
            self.params.signal_change_progress.emit(len(self.buffer), self.buffer_size)
            return None
        elif len(self.buffer) == self.buffer_size + 1:
            del self.buffer[0]
        elif len(self.buffer) > self.buffer_size + 1:
            self.buffer = self.buffer[-self.buffer_size:]

        # Get target frames
        np_buffer = np.array(self.buffer, np.float32)
        frames = np_buffer[self.target_index]
        return frames

    def _get_roi(self, frames):
        pred = self.roi_model.predict(np.expand_dims(frames, 0))[0, ...]
        score = pred[..., 0]
        roi = (score >= self.roi_threshold).astype(np.float32)
        return score, roi

    def _get_signals_with_roi(self, frames):
        # Extract ROI signals (# of roi pixels, window_size)
        signals = frames.transpose((3, 0, 1, 2))
        signals = signals[..., self.roi == 1.0].transpose(1, 2, 0)[..., 0]
        signals = signals.T

        # Random sampling when too many signals
        idxs = np.random.choice(signals.shape[0], min(self.maximum_point, signals.shape[0]), replace=False)
        signals = signals[idxs]
        return signals

    def _get_cluster(self, signals, n_cluster):
        # Set all elements 0 signal to 1
        signal_sum = np.fabs(signals).sum(axis=-1)
        signals[signal_sum == 0.0] += 1.0

        # Clustering with symmetry data
        reverse_signals = -signals
        input_signals = np.append(signals, reverse_signals, axis=0)
        cluster = clustering_hierarchy(input_signals, n_cluster, affinity='cosine', linkage='complete')

        # Divide cluster
        origin_cluster = cluster[: signals.shape[0]]
        reverse_cluster = cluster[signals.shape[0]:]
        return origin_cluster, reverse_cluster

    def _get_extracted_signals(self, signals, n_cluster, origin_cluster, reverse_cluster):
        num_mat = np.zeros((n_cluster, n_cluster), np.float32)
        for i in range(n_cluster):
            for j in range(n_cluster):
                num_mat[i][j] = ((origin_cluster == i) & (reverse_cluster == j)).astype(np.float32).sum()

        num_mat[num_mat.T == 0] = 0

        symmetry_mat = np.triu(num_mat + num_mat.T, 1)
        symmetry_max = symmetry_mat.max()
        if symmetry_max == 0:
            return None
        i, j = np.where(symmetry_mat == symmetry_max)

        i, j = (i, j) if i.shape[0] == 1 else (i[0], j[0])
        signal1 = signals[origin_cluster == i].mean(axis=0)
        signal2 = signals[origin_cluster == j].mean(axis=0)

        return signal1, signal2

    # -----------------------------------------------------------------------------------------------
    #                                    Methods for visualization
    # -----------------------------------------------------------------------------------------------
    def _show_cluster_plane(self, signals, cluster):
        pca_signal = np.append(signals, [[0] * 64], axis=0)
        pca = get_pca(pca_signal, 2)
        frame_pca = get_pca_frame(pca, cluster=cluster)
        cv2.imshow(self.FRAME_NAME_CLUSTER, frame_pca)

    def _show_symmetry_cluster_plane(self, signals, cluster):
        pca_signal = np.append(np.append(signals, -signals, axis=0), [[0] * 64], axis=0)
        pca = get_pca(pca_signal, 2)
        frame_pca = get_pca_frame(pca, cluster=cluster)
        cv2.imshow(self.FRAME_NAME_SYMMETRY, frame_pca)


if __name__ == '__main__':
    model = ClusteringBased(64, 20, 5, 0, 8, 0.5, '../../../model/detect_roi/model_fcn.h5')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Frame', frame)

        signal = model.get_respiration(frame)
        show_signal('signal', signal, 500)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
