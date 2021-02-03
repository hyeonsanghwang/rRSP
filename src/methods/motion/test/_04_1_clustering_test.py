import cv2
import numpy as np

from keras.models import load_model

from motion.data._00_data_preprocess import frame_process, convolution, normalize_zero_centered
from utils.clustering import clustering_hierarchy
from utils.pca import get_pca, get_pca_frame
from utils.visualize import show_signal, get_demo_frame, calc_peak2peak_interval


class ClusteringTester:
    SHOW_NORMAL_FRAME = 1
    SHOW_RESIZED_FRAME = 2
    SHOW_PREDICT_SCORE = 3
    SHOW_ROI = 4
    SHOW_MODES = [SHOW_NORMAL_FRAME, SHOW_RESIZED_FRAME, SHOW_PREDICT_SCORE, SHOW_ROI]

    def __init__(self, fps, process_fps, window_size, resize_ratio, detect_roi_path, color):
        self.fps = fps
        self.process_fps = process_fps
        self.window_size = window_size
        self.resize_ratio = resize_ratio
        self.detect_roi_model = load_model(detect_roi_path, compile=False)
        self.color = color
        self.init()
        print('1: Normal frame\n2: Resized frame\n3: Predict score\n4: ROI')

    def init(self):
        # Frame name
        self.frame_name = 'Demo'
        self.init_window = True

        # Set buffer size
        fps_ratio = self.fps / self.process_fps
        self.buffer_size = int(fps_ratio * self.window_size)

        # Set buffers
        self.image_buffer = []
        self.estimated_signal = [0]
        self.estimated_bpm = 0
        self.sensor_bpm = 0

        # Index of image in the buffer to process
        index = (np.array(list(range(self.buffer_size))) / fps_ratio).astype(np.int)
        self.target_index = (index != np.insert(index, 0, -1)[:-1])

        # Show selected pixel's signal
        self.px = 0
        self.py = 0

        # Process outputs
        self.processed = None
        self.input_data = None
        self.pred_score = None
        self.roi = None
        self.pca = None
        self.pred_results = None
        self.signals = None
        self.prev_signal = None
        self.signal1 = None
        self.signal2 = None


    def process(self,
                frame,
                signal,
                fps,
                roi_threshold=0.5,
                show_mode=1,
                show_resized_frame=False,
                show_model_predicts=False,
                show_roi=False,
                show_method_result=False,
                show_selected_pixel=False):

        if self.set_input_data(frame, show_resized_frame, show_selected_pixel, show_mode):
            if self.set_roi(roi_threshold, show_model_predicts, show_roi, show_mode):
                if self.set_cluster(show_method_result, show_mode):
                    signal_value = self.get_signal_value()
                else:
                    signal_value = 0
            else:
                signal_value = 0
        else:
            signal_value = 0

        self.set_estimated_signal(signal_value)

        show_frame = self.get_show_frame(frame, show_mode)
        self.show_demo_frame(show_frame, signal, fps)

    def set_input_data(self, frame, show_resized_frame=True, show_selected_pixel=True, show_mode=1):
        self.processed = frame_process(frame, self.color, self.resize_ratio)
        self.image_buffer.append(self.processed)

        if show_resized_frame:
            self._show_resized('Resized frame', self.processed)

        if len(self.image_buffer) > self.buffer_size:
            del self.image_buffer[0]
        if len(self.image_buffer) == self.buffer_size:
            self.input_data = (np.array(self.image_buffer) / 255.0)[self.target_index]

            if show_selected_pixel:
                if self.input_data.shape[-1] == 1:
                    show_signal('Selected signal', self.input_data[:, self.py, self.px, 0], 500)
                else:
                    for i in range(self.input_data.shape[-1]):
                        show_signal('Selected signal (Channel %d)'%i, self.input_data[:, self.py, self.px, 0], 500)
            return True
        else:
            return False

    def set_roi(self, threshold, show_model_predicts=True, show_roi=True, show_mode=1):
        pred = self.detect_roi_model.predict(np.expand_dims(self.input_data, axis=0))[0, ...]
        self.pred_score = pred[..., 0]
        self.pred_results = []
        for i in range(1, pred.shape[-1]):
            self.pred_results.append(pred[..., i])
        self.roi = (self.pred_score > threshold).astype(np.float32)

        if show_roi:
            self._show_resized('Predict score', self.pred_score)
            self._show_resized('ROI', self.roi)
        if show_model_predicts:
            for i, result in enumerate(self.pred_results):
                self._show_resized('Result %02d' % i, result)

        if np.sum(self.roi) == 0.0:
            return False
        else:
            return True

    def set_cluster(self, show_clustering_result=True, show_mode=1):
        # Select pixels
        self.signals = self.input_data.transpose((3, 0, 1, 2))
        self.signals = self.signals[..., self.roi == 1.0].transpose(1, 2, 0)[..., 0]
        self.signals = self.signals.T
        idxs = np.random.choice(self.signals.shape[0], min(100, self.signals.shape[0]), replace=False)
        self.signals = normalize_zero_centered(self.signals[idxs].T).T
        signal_sum = self.signals.sum(axis=-1)
        self.signals[signal_sum == 0.0] += 1.0

        # Clustering
        n_cluster = 4
        reverse_signals = -self.signals
        input_signals = np.append(self.signals, reverse_signals, axis=0)

        cluster = clustering_hierarchy(input_signals, n_cluster, affinity='cosine', linkage='complete')
        if show_clustering_result:
            PCA_FRAME_NAME = 'PCA'
            if cv2.getWindowProperty(PCA_FRAME_NAME, 0) < 0:
                cv2.namedWindow(PCA_FRAME_NAME)
                cv2.setMouseCallback(PCA_FRAME_NAME, self._pca_mouse_callback, param=(PCA_FRAME_NAME,))


            pca_signal = np.append(self.signals, [[0]*64], axis=0)
            self.pca = get_pca(pca_signal, 2)
            frame_pca = get_pca_frame(self.pca, cluster=cluster[:len(cluster) // 2])
            cv2.imshow(PCA_FRAME_NAME, frame_pca)

            aaa = np.append(input_signals, [[0]*64], axis=0)
            ppp = get_pca(aaa, 2)

            fff = get_pca_frame(ppp, cluster=cluster)
            cv2.imshow('aaaaaaaa', fff)



        origin_cluster = cluster[: self.signals.shape[0]]
        reverse_cluster = cluster[self.signals.shape[0]:]

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

        self.pairs = []
        for i in range(n_cluster):
            if not cluster_empty_index[i]:
                if cluster_max_index[cluster_max_index[i]] == i:
                    self.pairs.append((i, cluster_max_index[i]))
                    cluster_empty_index[cluster_max_index[i]] = True

        if self.pairs:
            cluster_size = []
            for pair in self.pairs:
                cluster_size.append(cluster_counter[pair[0]] + cluster_counter[pair[1]])
            idx = np.argmax(cluster_size)

            self.signal1 = self.signals[origin_cluster == self.pairs[idx][0]].mean(axis=0)
            self.signal2 = self.signals[origin_cluster == self.pairs[idx][1]].mean(axis=0)

            if show_clustering_result:
                show_signal('Cluster1 signal', self.signal1, 500)
                show_signal('Cluster2 signal', self.signal2, 500)

            return True
        else:
            return False

    def get_signal_value(self):
        if self.prev_signal is None:
            signal = self.signal1 - self.signal2
        else:
            align_signal_list = np.array([self.prev_signal[1:],
                                          -self.prev_signal[1:],
                                          self.signal1[:-1],
                                          self.signal2[:-1]])
            align_cluster = clustering_hierarchy(align_signal_list, n=2, affinity='cosine', linkage='complete')

            if align_cluster[0] == align_cluster[2]:
                signal = self.signal1 - self.signal2
            else:
                signal = self.signal2 - self.signal1

        signal = (signal - signal.min()) / (signal.max() - signal.min())
        self.prev_signal = signal
        diff_val = signal[-1] - signal[-2]
        return diff_val

    def set_estimated_signal(self, signal_value):
        self.estimated_signal.append(self.estimated_signal[-1] - signal_value)
        if len(self.estimated_signal) > self.buffer_size:
            del self.estimated_signal[0]

    def get_show_frame(self, frame, show_mode):
        if show_mode == self.SHOW_NORMAL_FRAME:
            return frame
        elif show_mode == self.SHOW_RESIZED_FRAME:
            return self._resize_image(self.processed)
        elif show_mode == self.SHOW_PREDICT_SCORE:
            pred_score = self.pred_score[:, :, None] * np.ones(3)[None, None, :] * 255
            return self._resize_image(pred_score.astype(np.uint8))
        elif show_mode == self.SHOW_ROI:
            roi = self.roi[:, :, None] * np.ones(3)[None, None, :] * 255
            return self._resize_image(roi.astype(np.uint8))

    def show_demo_frame(self, frame, signal, fps):
        bpm = self.calculate_bpm(normalize_zero_centered(np.array(self.estimated_signal)), self.fps, 0.1)
        self.estimated_bpm = self.estimated_bpm if bpm == 0.0 else bpm

        bpm = self.calculate_bpm(normalize_zero_centered(np.array(signal)), 20, 0.5)
        self.sensor_bpm = self.sensor_bpm if bpm == 0.0 else bpm

        res_frame = get_demo_frame(frame, self.estimated_signal, signal, self.estimated_bpm, self.sensor_bpm,
                                   round(fps))

        if self.init_window:
            self.init_window = False
            cv2.namedWindow(self.frame_name)
            cv2.setMouseCallback(self.frame_name, self._demo_mouse_callback, param=frame.shape)
        cv2.imshow(self.frame_name, res_frame)

    def calculate_bpm(self, signal, fps, min_amplitude):
        conv = convolution([signal], kernel_size=15)[0]
        _, bpm = calc_peak2peak_interval(conv, fps, band=(0.1, 0.7), min_amplitude=min_amplitude, ret_bpm=True)
        return bpm

    def _show_resized(self, name, image):
        img = self._resize_image(image)
        cv2.imshow(name, img)

    def _resize_image(self, image):
        return cv2.resize(image, dsize=None, fx=self.resize_ratio, fy=self.resize_ratio,
                          interpolation=cv2.INTER_AREA)

    def _draw_optical_flow_result(self, frame):
        show_frame = frame.copy()
        mask = np.zeros_like(show_frame)
        for i, (curr, prev) in enumerate(zip(self.curr_points, self.prev_points)):
            x1, y1 = prev.ravel()
            x2, y2 = curr.ravel()
            mask = cv2.line(mask, (x2, y2), (x1, y1), (0, 255, 255), 2)
            show_frame = cv2.circle(show_frame, (x2, y2), 3, (0, 255, 255), -1)
        img = cv2.add(show_frame, mask)
        return img

    def _demo_mouse_callback(self, e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONUP:
            gap = 2
            x -= gap
            y -= gap
            if 0 < y < p[0] and 0 < x < p[1]:
                self.px = x // self.resize_ratio
                self.py = y // self.resize_ratio

    def _pca_mouse_callback(self, e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONUP:
            frame_name = p[0]
            pca_x, pca_y = self.pca[:, 0], self.pca[:, 1]
            norm_x = (((pca_x - pca_x.min()) / (pca_x.max() - pca_x.min())) * 500).astype(np.int32)
            norm_y = (((pca_y - pca_y.min()) / (pca_y.max() - pca_y.min())) * 500).astype(np.int32)

            diff = np.square(norm_x - x) + np.square(norm_y - y)
            found_idx = np.argmin(diff)

            if found_idx != -1:
                frame = get_pca_frame(self.pca, selected_idx=found_idx, bi_norm=False)
                cv2.imshow(frame_name, frame)
                show_signal('Selected PCA signal', self.signals[found_idx], 500, thickness=3)