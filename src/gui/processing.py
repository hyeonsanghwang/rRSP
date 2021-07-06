from time import sleep

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal

from methods.roi_detection.clustering_based import ClusteringBased
from methods.roi_detection.optical_flow_based import OpticalFlowBased
from methods.instance_segmentation.segmentation_based import SegmentationBased

from project_utils.resp_signal import SignalStream
from utils.camera.video_stream import VideoStream

from gui.window import Parameter
from gui.tab_cluster import ClusterParameter
from gui.tab_optical import OpticalFlowParameter
from gui.tab_segmentation import SegmentationParameter

from scipy.signal import find_peaks


class ProcessManager:

    def __init__(self, params):
        self.params: Parameter = params[0]
        self.cluster_params: ClusterParameter = params[1]
        self.optical_params: OpticalFlowParameter = params[2]
        self.segment_params: SegmentationParameter = params[3]

        # Init managers
        self.video_stream: VideoStream = None
        self.signal_stream: SignalStream = None

        # Processing method
        self.process_method = 0
        self.process = None

        # UI components
        self.frame = None
        self.estimated_signal = None
        self.reference_signal = None

    # -----------------------------------------------------------------------------------------
    #                                     Main process loop
    # -----------------------------------------------------------------------------------------
    def start(self):

        # Main loop
        while not self.params.main_loop_closed:
            sleep(0.1)

            # rRSP process loop
            while self.params.process_started:
                # Detect change of parameters
                self.set_source()
                self.set_method()
                self.set_parameters()

                # Read video
                ret, frame = self.video_stream.read()
                if not ret:
                    break

                # Read signal
                self.reference_signal = None if self.signal_stream is None else self.signal_stream.get_signal()

                # Process extract respiration
                if self.process is not None:
                    self.estimated_signal = self.process.get_respiration(frame)

                # Update ui from processing result
                self.update_ui()

                # Sleep until the UI update is over
                cv2.waitKey(self.video_stream.delay())

            # Release resource and wait
            self.params.process_started = False
            self.release_source()
            if self.process is not None:
                self.process.reset()

    # -----------------------------------------------------------------------------------------
    #                                      Set input source
    # -----------------------------------------------------------------------------------------
    def set_source(self):
        # Set changed source
        if self.params.src_changed:
            self.params.src_changed = False
            self.video_stream = VideoStream(self.params.src_info["video"])
            if self.params.src_info["signal"] is not None:
                if self.params.src_info["type"] == Parameter.INPUT_WEBCAM:
                    self.signal_stream = SignalStream(window_period=10, show=False)
                    self.signal_stream = self.signal_stream if self.signal_stream.data_available else None
                elif self.params.src_info["type"] == Parameter.INPUT_NUMPY:
                    self.signal_stream = SignalStream(self.params.src_info["signal"], show=False)
                    self.signal_stream = self.signal_stream if self.signal_stream.data_available else None

        # Set changed fps
        if self.params.fps_changed:
            self.params.fps_changed = False
            self.video_stream.set(cv2.CAP_PROP_FPS, self.params.fps)

    def release_source(self):
        if self.video_stream is not None:
            self.video_stream.release()
            self.video_stream = None

        if self.signal_stream is not None:
            self.signal_stream.close()
            self.signal_stream = None

        self.frame = None
        self.estimated_signal = None
        self.reference_signal = None

    # -----------------------------------------------------------------------------------------
    #                                 Detect change of parameters
    # -----------------------------------------------------------------------------------------
    def set_parameters(self):
        # Case clustering
        if self.process_method == Parameter.METHOD_CLUSTERING:
            if self.cluster_params.parameter_changed:
                self.cluster_params.parameter_changed = False
                self.process.set_parameters(self.cluster_params.window_size,
                                            self.cluster_params.process_fps,
                                            self.cluster_params.color_domain,
                                            self.cluster_params.resize_ratio,
                                            self.cluster_params.threshold)

            if self.cluster_params.model_changed:
                self.cluster_params.model_changed = False
                self.process.set_model_path(self.cluster_params.model_path)

            if self.cluster_params.show_mode_changed:
                self.cluster_params.show_mode_changed = False
                self.process.show_mode = self.cluster_params.show_mode

            if self.cluster_params.function_changed:
                self.cluster_params.function_changed = False
                self.process.show_cluster = self.cluster_params.show_cluster
                self.process.show_sym_cluster = self.cluster_params.show_sym_cluster
                self.process.flip = self.cluster_params.flip


        # Case optical flow
        elif self.process_method == Parameter.METHOD_OPTICAL_FLOW:
            if self.optical_params.parameter_changed:
                self.optical_params.parameter_changed = False
                self.process.set_parameters(self.optical_params.window_size,
                                            self.optical_params.process_fps,
                                            self.optical_params.color_domain,
                                            self.optical_params.resize_ratio,
                                            self.optical_params.threshold)

            if self.optical_params.model_changed:
                self.optical_params.model_changed = False
                self.process.set_model_path(self.optical_params.model_path)

            if self.optical_params.show_mode_changed:
                self.optical_params.show_mode_changed = False
                self.process.show_mode = self.optical_params.show_mode

        # Case segmentation
        elif self.process_method == Parameter.METHOD_SEGMENTATION:
            if self.segment_params.show_mode_change:
                self.segment_params.show_mode_change = False
                self.process.show_mode = self.segment_params.show_mode

    def set_method(self):
        if self.params.method_changed:
            self.params.method_changed = False
            self.process_method = self.params.method

            # Release previous method
            if self.process is not None:
                self.process.reset()
                self.process = None

            # Create new method
            if self.process_method == Parameter.METHOD_CLUSTERING:
                self.process = ClusteringBased(self.cluster_params, self.params.fps)
            elif self.process_method == Parameter.METHOD_OPTICAL_FLOW:
                self.process = OpticalFlowBased(self.optical_params, self.params.fps)
            elif self.process_method == Parameter.METHOD_SEGMENTATION:
                self.process = SegmentationBased(None)

    # -----------------------------------------------------------------------------------------
    #                                       Update window UI
    # -----------------------------------------------------------------------------------------
    def update_ui(self):
        try:
            if self.process is not None:
                self.params.signal_change_frame.emit(self.process.get_show_frame())
            if self.estimated_signal is not None:
                if len(self.estimated_signal) > 1:
                    self.params.signal_change_estimated.emit(self.estimated_signal)
            if self.reference_signal is not None:
                if len(self.reference_signal) > 1:
                    np_signal = np.array(self.reference_signal, np.float32)
                    self.params.signal_change_reference.emit(np_signal)
        except:
            print('Window already closed.')

