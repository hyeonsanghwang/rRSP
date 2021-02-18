from time import sleep

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QMutex

from methods.motion.test.module_bpm import BPMManager
from methods.motion.test.module_frame import FrameManager
from methods.motion.test.module_roi import ROIManager
from methods.motion.test.process_clustering import ClusteringTester
from resp_utils.resp_signal import SignalStream
from utils.camera.video_stream import VideoStream


class ProcessManager:
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

    def __init__(self, params):
        self.params = params

        # Initialize managers
        self.video_stream = None
        self.signal_stream = None
        self.frame_manager = FrameManager(QMutex())
        self.roi_manager = ROIManager(QMutex())
        self.bpm_manager = BPMManager(QMutex())

        # Initialize parameters
        self.set_parameters()

        # Create processor
        self.process_clustering = ClusteringTester()

    # Main loop --------------------------------------------------------------------------------------------------------
    def start(self):
        while self.params.is_started:
            self.frame_manager.clear_buffer()
            self.init_variables()

            is_stopping = False
            while self.params.is_processing:
                is_stopping = True

                # Check change
                self.check_change_source()
                self.check_change_parameters()
                self.check_change_model()
                self.check_change_mode_show()
                self.check_change_show_bpm()

                # Set frame
                ret, frame = self.video_stream.read()
                if not ret:
                    break
                self.frame_manager.set_frame(frame)
                frames = self.frame_manager.get_frames()

                # Processing
                if frames is not None:
                    score, roi = self.roi_manager.calculate_roi(frames, self.detect_threshold)
                    if roi is not None:
                        if self.params.mode_process == self.PROCESS_MODE_CLUSTER:
                            self.signal = self.process_clustering.process(frames, roi)

                # Show
                try:
                    self.show_frame(frame)
                    self.show_signal()
                    self.show_reference()
                    self.bpm_manager.show_bpms()
                except:
                    break

                # Key events
                self.process_key_event()

            # Release resource
            if is_stopping:
                try:
                    self.params.signal_dict[self.params.SIGNAL_STOP_PROCESS].emit()
                except:
                    pass
            if self.video_stream is not None:
                self.video_stream.release()
            if self.signal_stream is not None:
                self.signal_stream.close()
            cv2.destroyAllWindows()
        sleep(0.1)

    # Set methods ------------------------------------------------------------------------------------------------------
    def init_variables(self):
        self.signal = None
        self.model_path = None
        self.mode_show = 0

    def set_parameters(self):
        self.fps = self.params.fps
        self.process_fps = self.params.process_fps
        self.window_size = self.params.window_size
        self.resize_ratio = self.params.resize_ratio
        self.color_domain = self.params.color_domain
        self.detect_threshold = self.params.detect_threshold

    # Check state method -----------------------------------------------------------------------------------------------
    def check_change_parameters(self):
        if self.params.is_changed_parameters:
            self.set_parameters()
            self.params.is_changed_parameters = False

            self.frame_manager.set_parameters(self.resize_ratio, self.color_domain, self.fps, self.process_fps, self.window_size)
            if self.video_stream is not None:
                print(self.video_stream)
                self.video_stream.set(cv2.CAP_PROP_FPS, self.fps)

            self.params.signal_dict[self.params.SIGNAL_START_PROCESS].emit(self.frame_manager.buffer_size)

    def check_change_source(self):
        if self.params.is_changed_source:
            self.video_stream = VideoStream(self.params.src_video, self.fps)

            if self.params.src_type == self.INPUT_TYPE_WEBCAM and self.params.src_signal:
                self.signal_stream = SignalStream(None, window_period=self.window_size / self.process_fps, show=False)
            elif self.params.src_type == self.INPUT_TYPE_NUMPY and self.params.src_signal is not None:
                self.signal_stream = SignalStream(self.params.src_signal, window_period=self.window_size / self.process_fps, show=False)
            else:
                self.signal_stream = None
            self.params.is_changed_source = False

    def check_change_model(self):
        if self.params.is_changed_model:
            if self.model_path != self.params.src_model:
                self.model_path = self.params.src_model
                self.roi_manager.set_model(self.model_path)
            self.params.is_changed_model = False

    def check_change_mode_show(self):
        if self.params.mode_show != self.mode_show:
            if self.params.mode_show == self.SHOW_MODE_SCORE:
                if self.roi_manager.get_score() is not None:
                    self.mode_show = self.params.mode_show
            elif self.params.mode_show == self.SHOW_MODE_ROI:
                if self.roi_manager.get_roi() is not None:
                    self.mode_show = self.params.mode_show
            else:
                self.mode_show = self.params.mode_show

    def check_change_show_bpm(self):
        if self.params.is_show_bpm:
            self.bpm_manager.set_parameters(self.params.bpm_list)
            self.params.is_show_bpm = False

    # Visualize methods ------------------------------------------------------------------------------------------------
    def show_frame(self, frame):
        if self.mode_show == self.SHOW_MODE_ORIGIANL:
            image = frame
        elif self.mode_show == self.SHOW_MODE_RESIZED:
            image = (self.frame_manager.processed * 255).astype(np.uint8)
        elif self.mode_show == self.SHOW_MODE_SCORE:
            image = self.roi_manager.get_score()
        elif self.mode_show == self.SHOW_MODE_ROI:
            image = self.roi_manager.get_roi()

        self.params.signal(self.params.SIGNAL_CHANGED_FRAME).emit(image)
        self.params.signal(self.params.SIGNAL_CHANGED_FPS).emit()

    def show_signal(self):
        if self.signal is None:
            self.params.signal(self.params.SIGNAL_CHANGED_ESTIMATED_SIGNAL).emit(np.zeros((100, )), 0)
        else:
            self.params.signal(self.params.SIGNAL_CHANGED_ESTIMATED_SIGNAL).emit(self.signal, 0)

    def show_reference(self):
        if self.signal_stream is not None:
            signal = self.signal_stream.get_signal()
            self.params.signal(self.params.SIGNAL_CHANGED_REFERENCE_SIGNAL).emit(np.array(signal), 1)

    # Opencv window key event ------------------------------------------------------------------------------------------
    def process_key_event(self):
        key = cv2.waitKey(self.video_stream.delay())
        if key == 27:
            self.bpm_manager.stop()
