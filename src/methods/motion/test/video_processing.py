import numpy as np
import cv2

from PyQt5.QtCore import QThread, pyqtSignal, QMutex

from resp_utils.resp_signal import SignalStream
from utils.camera.video_stream import VideoStream
from utils.visualization.signal import show_sin_signals


class ProcessingThread(QThread):
    INPUT_TYPE_WEBCAM = 0
    INPUT_TYPE_NUMPY = 1
    INPUT_TYPE_VIDEO_FILE = 2

    PROCESS_MODE_CLUSTER = 0
    PROCESS_MODE_OPTICAL_FLOW = 1

    signal_get_image = pyqtSignal(np.ndarray)
    signal_get_signal = pyqtSignal(np.ndarray)
    signal_end_process = pyqtSignal()

    is_processing = False
    process_mode = 0
    video_stream = None
    signal_stream = None

    is_show_bpms = False
    init_bpms = False
    bpms = None
    bpm_mutex = QMutex()

    def set_parameters(self, fps, process_fps, window_size, resize_ratio, color_domain, detect_threshold):
        self.fps = fps
        self.process_fps = process_fps
        self.window_size = window_size
        self.resize_ratio = resize_ratio
        self.color_domain = color_domain
        self.detect_threshold = detect_threshold

        if self.video_stream is not None:
            self.video_stream.set(cv2.CAP_PROP_FPS, fps)

    def set_camera_source(self, input_type, args):
        if input_type == self.INPUT_TYPE_WEBCAM:
            camera_num, use_sensor = args
            self.video_stream = VideoStream(camera_num, self.fps)
            if use_sensor:
                self.signal_stream = SignalStream(None, window_period=self.window_size / self.process_fps, show=False)
        elif input_type == self.INPUT_TYPE_NUMPY:
            video_path, signal_path = args
            self.video_stream = VideoStream(video_path, self.fps)
            if signal_path is not None:
                self.signal_stream = SignalStream(signal_path, window_period=self.window_size / self.process_fps, show=False)
        elif input_type == self.INPUT_TYPE_VIDEO_FILE:
            video_path = args
            self.video_stream = VideoStream(video_path, self.fps)
            self.signal_stream = None

    def run(self):
        self.is_processing = True
        while self.is_processing:
            ret, frame = self.video_stream.read()
            if not ret:
                break
            print(self.process_mode)
            # Processing
            self.signal_get_image.emit(frame)
            if self.signal_stream is not None:
                signal = self.signal_stream.get_signal()
                self.signal_get_signal.emit(np.array(signal))
            self.show_bpms()

            # Key events
            key = cv2.waitKey(self.video_stream.delay())
            if key == 27:
                self.reset_bpms()
        # Release resource
        self.video_stream.release()
        if self.signal_stream is not None:
            self.signal_stream.close()
        cv2.destroyAllWindows()
        self.signal_end_process.emit()

    def set_process_mode(self, mode):
        self.process_mode = mode

    def start_show_bpms(self, bpms):
        self.bpm_mutex.lock()
        self.is_show_bpms = True
        self.init_bpms = True
        self.bpms = bpms
        self.bpm_mutex.unlock()

    def show_bpms(self):
        self.bpm_mutex.lock()
        if self.is_show_bpms:
            if self.bpms is not None:
                show_sin_signals(bpms=self.bpms, init_data=self.init_bpms)
                self.init_bpms = False
        self.bpm_mutex.unlock()

    def reset_bpms(self):
        self.is_show_bpms = False
        cv2.destroyWindow('BPMs')
