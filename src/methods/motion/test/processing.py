import numpy as np
import cv2

from PyQt5.QtCore import QThread, pyqtSignal, QMutex

from methods.motion.test.module_frame import FrameManager
from methods.motion.test.module_roi import ROIManager
from methods.motion.test.module_bpm import BPMManager
from methods.motion.test.process_clustering import ClusteringTester

from resp_utils.resp_signal import SignalStream
from utils.camera.video_stream import VideoStream



class ProcessingThread(QThread):
    INPUT_TYPE_WEBCAM = 0
    INPUT_TYPE_NUMPY = 1
    INPUT_TYPE_VIDEO_FILE = 2

    PROCESS_MODE_CLUSTER = 0
    PROCESS_MODE_OPTICAL_FLOW = 1

    SHOW_MODE_ORIGIANL = 0
    SHOW_MODE_RESIZED = 1
    SHOW_MODE_SCORE = 2
    SHOW_MODE_ROI = 3

    signal_get_image = pyqtSignal(np.ndarray)
    signal_set_fps = pyqtSignal()
    signal_get_estimated_signal = pyqtSignal(np.ndarray, int)
    signal_get_reference_signal = pyqtSignal(np.ndarray, int)
    signal_end_process = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Resource
        self.is_processing = False
        self.video_stream = None
        self.signal_stream = None

        # Processing
        self.process_mode = 0
        self.show_mode = 0

        # modules
        self.frame_manager = FrameManager(QMutex())
        self.bpm_manager = BPMManager(QMutex())
        self.roi_manager = ROIManager(QMutex())
        self.process_clustering = ClusteringTester()

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

    def set_parameters(self, parent):
        self.fps = parent.spin_fps.value()
        self.process_fps = parent.spin_process_fps.value()
        self.window_size = parent.spin_window_size.value()
        self.resize_ratio = parent.spin_resize_ratio.value()
        self.color_domain = parent.combo_color_domain.currentIndex()
        self.detect_threshold = parent.spin_threshold.value()

        self.frame_manager.set_parameters(self.resize_ratio, self.color_domain, self.fps, self.process_fps, self.window_size)
        parent.progress_bar.setRange(0, self.frame_manager.buffer_size)
        parent.progress_bar.reset()
        if self.video_stream is not None:
            self.video_stream.set(cv2.CAP_PROP_FPS, self.fps)

    def set_model(self, path):
        self.roi_manager.set_model(path)

    def show_bpms(self, bpms):
        self.bpm_manager.set_parameters(bpms)

    def set_show_mode(self, mode):
        if mode == self.SHOW_MODE_SCORE:
            ret = self.roi_manager.get_score()
            if ret is not None:
                self.show_mode = mode
        elif mode == self.SHOW_MODE_ROI:
            ret = self.roi_manager.get_roi()
            if ret is not None:
                self.show_mode = mode
        else:
            self.show_mode = mode

    def set_process_mode(self, mode):
        self.process_mode = mode

    def run(self):
        self.is_processing = True
        while self.is_processing:
            ret, frame = self.video_stream.read()
            if not ret:
                break

            # Processing
            self.bpm_manager.show_bpms()
            self.frame_manager.set_frame(frame)
            frames = self.frame_manager.get_frame()
            signal = None

            if frames is not None:
                score, roi = self.roi_manager.calculate_roi(frames, self.detect_threshold)
                # if roi is not None:
                #     signal = self.process_clustering.process(frames, roi)

            # Show
            self.show_frame(frame)
            self.show_signal(signal)
            self.show_reference()

            # Key events
            self.process_key_event()

        # Release resource
        self.video_stream.release()
        if self.signal_stream is not None:
            self.signal_stream.close()
        cv2.destroyAllWindows()
        self.signal_end_process.emit()


    def show_frame(self, frame):
        if self.show_mode == self.SHOW_MODE_ORIGIANL:
            image = frame
        elif self.show_mode == self.SHOW_MODE_RESIZED:
            image = self.frame_manager.processed
        elif self.show_mode == self.SHOW_MODE_SCORE:
            image = self.roi_manager.get_score()
        elif self.show_mode == self.SHOW_MODE_ROI:
            image = self.roi_manager.get_roi()

        self.signal_get_image.emit(image)
        self.signal_set_fps.emit()

    def show_signal(self, signal):
        if signal is None:
            signal = np.zeros((100, ))
        self.signal_get_estimated_signal.emit(signal, 0)

    def show_reference(self):
        if self.signal_stream is not None:
            signal = self.signal_stream.get_signal()
            self.signal_get_reference_signal.emit(np.array(signal), 1)

    def process_key_event(self):
        key = cv2.waitKey(self.video_stream.delay())
        if key == 27:
            self.bpm_manager.stop()
