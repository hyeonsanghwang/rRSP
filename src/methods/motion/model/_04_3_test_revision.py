import cv2
import numpy as np
import csv
from datetime import datetime

from keras.models import load_model

from motion.data._00_data_preprocess import frame_process, convolution, normalize_zero_centered

from utils.visualize import get_signal_frame, show_signal, get_demo_frame, calc_peak2peak_interval


class OpticalFlowTester:
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    SHOW_NORMAL_FRAME = 1
    SHOW_RESIZED_FRAME = 2
    SHOW_PREDICT_SCORE = 3
    SHOW_ROI = 4
    SHOW_OPTICAL_FLOW = 5
    SHOW_MODES = [SHOW_NORMAL_FRAME, SHOW_RESIZED_FRAME, SHOW_PREDICT_SCORE, SHOW_ROI, SHOW_OPTICAL_FLOW]

    def __init__(self, fps, process_fps, window_size, resize_ratio, detect_roi_path, color):
        self.fps = fps
        self.process_fps = process_fps
        self.window_size = window_size
        self.resize_ratio = resize_ratio
        self.detect_roi_model = load_model(detect_roi_path, compile=False)
        self.color = color
        self.init()
        print('1: Normal frame\n2: Resized frame\n3: Predict score\n4: ROI\n5: Optical flow')

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
        self.gray = None
        self.prev_gray = None
        self.input_data = None
        self.pred_score = None
        self.roi = None
        self.pred_results = None
        self.prev_points = None
        self.curr_points = None
        self.point_status = None


        self.video_writer = None


        self.file = open('log.csv', 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['timestamp', 'remote resp(bpm)', 'contact resp(bpm)'])


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

        if self.set_input_data(frame, show_resized_frame, show_mode):
            if self.set_roi(roi_threshold, show_model_predicts, show_roi, show_selected_pixel, show_mode):
                if self.set_optical_flow(frame, show_method_result, show_mode):
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

    def set_input_data(self, frame, show_resized_frame=True, show_mode=1):
        if self.gray is not None:
            self.prev_gray = self.gray
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.processed = frame_process(frame, self.color, self.resize_ratio)
        self.image_buffer.append(self.processed)

        if show_resized_frame:
            self._show_resized('Resized frame', self.processed)

        if len(self.image_buffer) > self.buffer_size:
            del self.image_buffer[0]
        if len(self.image_buffer) == self.buffer_size:
            self.input_data = (np.array(self.image_buffer) / 255.0)[self.target_index]
            return True
        else:
            return False

    # def ____________TEST_ROI_SNR(self):
    #
    #     pixel_signals = self.input_data.reshape((64, -1))
    #     pixel_signals -= pixel_signals.mean(axis=0)
    #     pixel_signals = pixel_signals.T
    #
    #     from scipy.fftpack import rfft, rfftfreq
    #     fft = np.fabs(rfft(pixel_signals, axis=-1))
    #     freq = rfftfreq(64, d=1.0 / 5.0)
    #     target_mask = np.logical_and(freq >= 0.1, freq <= 0.75)
    #     noise_mask = np.logical_or(freq < 0.1, freq > 0.75)
    #
    #     target_amp = fft[:, target_mask].sum(axis=-1)
    #     noise_amp = fft[:, noise_mask].sum(axis=-1)
    #
    #     snr = target_amp / noise_amp
    #     snr = snr.reshape((60, 80, 3)).min(axis=-1)
    #     snr[self.roi==0] = 0
    #     snr = (snr - snr.min()) / (snr.max() - snr.min())
    #     cv2.imshow('snr', cv2.resize(snr, dsize=(640, 480), interpolation=cv2.INTER_AREA))






    def set_roi(self, threshold, show_model_predicts=True, show_roi=True, show_selected_pixel=True, show_mode=1):
        pred = self.detect_roi_model.predict(np.expand_dims(self.input_data, axis=0))[0, ...]
        self.pred_score = pred[..., 0]
        self.pred_results = []
        for i in range(1, pred.shape[-1]):
            self.pred_results.append(pred[..., i])
        self.roi = (self.pred_score > threshold).astype(np.float32)

        # self.____________TEST_ROI_SNR()

        if show_roi:
            self._show_resized('Predict score', self.pred_score)
            self._show_resized('ROI', self.roi)
        if show_model_predicts:
            for i, result in enumerate(self.pred_results):
                self._show_resized('Result %02d' % i, result * self.roi)

        if show_selected_pixel:
            if self.input_data.shape[-1] == 1:
                show_signal('Selected signal', self.input_data[:, self.py, self.px, 0], 500)
            else:
                for i in range(self.input_data.shape[-1]):
                    signal = self.input_data[:, self.py, self.px, i]
                    frame_width = 500
                    signal_frame = get_signal_frame(signal, frame_width)

                    noise_length = int(self.pred_results[0][self.py, self.px] * frame_width)
                    noise_start = int(self.pred_results[1][self.py, self.px] * frame_width)
                    noise_end = int(self.pred_results[2][self.py, self.px] * frame_width)

                    modi = signal_frame[:, noise_start: noise_end, :]
                    modi[:, :, 1] = modi[:, :, 1] + modi[:, :, 2]

                    modi = signal_frame[:, noise_start: noise_start+noise_length, :]
                    modi[:, :, 0] = modi[:, :, 0] + modi[:, :, 2]

                    cv2.imshow('Selected signal (Channel %d)' % i, signal_frame)

        if np.sum(self.roi) == 0.0:
            return False
        else:
            return True

    def set_optical_flow(self, frame, show_optical_flow_result=True, show_mode=1):
        ys, xs = np.where(self.roi == 1.0)
        xs = np.expand_dims(xs, axis=-1)
        ys = np.expand_dims(ys, axis=-1)
        ps = np.append(xs, ys, axis=-1).astype(np.float32) * self.resize_ratio + self.resize_ratio / 2
        self.prev_points = np.expand_dims(ps, axis=1)
        self.curr_points, self.point_status, e = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                                          self.gray,
                                                                          self.prev_points,
                                                                          None,
                                                                          **self.lk_params)
        self.prev_points = self.prev_points[self.point_status == 1.0]
        if self.curr_points is None:
            self.curr_points = self.prev_points
        else:
            self.curr_points = self.curr_points[self.point_status == 1.0]

        if show_optical_flow_result or show_mode == self.SHOW_OPTICAL_FLOW:
            self.optical_flow_frame = self._draw_optical_flow_result(frame)
            if show_optical_flow_result:
                cv2.imshow('Optical flow result', self.optical_flow_frame)

        if self.point_status.sum() == 0.0:
            return False
        else:
            return True

    def get_signal_value(self):
        movement = self.curr_points - self.prev_points
        binary_movement = ((movement > 0).astype(np.float32) * 2) - 1
        mean_movement = binary_movement.mean(axis=0)

        if mean_movement.size == 0:
            return 0
        else:
            return mean_movement[1] # Y movement

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
        elif show_mode == self.SHOW_OPTICAL_FLOW:
            return self.optical_flow_frame

    def show_demo_frame(self, frame, signal, fps):
        bpm = self.calculate_bpm(normalize_zero_centered(np.array(self.estimated_signal)), self.fps, 0.1)
        self.estimated_bpm = self.estimated_bpm if bpm == 0.0 else bpm

        bpm = self.calculate_bpm(normalize_zero_centered(np.array(signal)), 20, 0.5)
        self.sensor_bpm = self.sensor_bpm if bpm == 0.0 else bpm

        self.writer.writerow([datetime.now().strftime('%Y/%m/%d %H:%M:%S'), self.estimated_bpm, self.sensor_bpm])

        res_frame = get_demo_frame(frame, self.estimated_signal, signal, self.estimated_bpm, self.sensor_bpm, round(fps))
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter('output.avi', fourcc, 60.0, (res_frame.shape[1], res_frame.shape[0]))


        if self.init_window:
            self.init_window = False
            cv2.namedWindow(self.frame_name)
            cv2.setMouseCallback(self.frame_name, self._demo_mouse_callback, param=frame.shape)
        # cv2.imshow(self.frame_name, res_frame)
        self.video_writer.write(res_frame)

    def calculate_bpm(self, signal, fps, min_amplitude):
        conv = convolution([signal], kernel_size=15)[0]
        _, bpm = calc_peak2peak_interval(conv, fps, band=(0.1, 0.7), min_amplitude=min_amplitude, ret_bpm=True)
        return bpm

    def _show_resized(self, name, image):
        img = self._resize_image(image)
        cv2.imshow(name, img)

    def _resize_image(self, image):
        return cv2.resize(image, dsize=None, fx=self.resize_ratio, fy=self.resize_ratio, interpolation=cv2.INTER_AREA)

    def _draw_optical_flow_result(self, frame):
        show_frame = frame.copy()
        mask = np.zeros_like(show_frame)
        for i, (curr, prev) in enumerate(zip(self.curr_points, self.prev_points)):
            x1, y1 = prev.ravel()
            x2, y2 = curr.ravel()
            if y1 < 240:
                continue
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


