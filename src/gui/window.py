import os
import cv2
import numpy as np
from time import perf_counter
from scipy.signal import find_peaks

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QCheckBox
from PyQt5 import QtGui

# from .processing import ProcessManager
from .window_form import WindowForm
from utils.visualization.signal import signal_to_frame, show_sin_signals


class Parameter:
    INPUT_WEBCAM = 0
    INPUT_NUMPY = 1
    INPUT_VIDEO = 2

    METHOD_CLUSTERING = 0
    METHOD_OPTICAL_FLOW = 1
    METHOD_SEGMENTATION = 2

    SIGNAL_CHANGE_FRAME = 0
    SIGNAL_CHANGE_ESTIMATED = 1
    SIGNAL_CHANGE_REFERENCE = 2

    def __init__(self):
        # Process loop flag
        self.main_loop_closed = False
        self.process_started = False

        # Fps related parameters
        self.fps_changed = True
        self.fps = 20

        # Source related parameters
        self.src_changed = True
        self.src_info = None

        # Processing method related parameters
        self.method_changed = True
        self.method = 0

        # Shared signals
        self.signal_change_frame = None
        self.signal_change_estimated = None
        self.signal_change_reference = None

    def set_signals(self, signal_type, signal):
        if signal_type == self.SIGNAL_CHANGE_FRAME:
            self.signal_change_frame = signal
        elif signal_type == self.SIGNAL_CHANGE_ESTIMATED:
            self.signal_change_estimated = signal
        elif signal_type == self.SIGNAL_CHANGE_REFERENCE:
            self.signal_change_reference = signal


class MainWindow(WindowForm):
    signal_changed_frame = pyqtSignal(np.ndarray)
    signal_changed_estimated = pyqtSignal(np.ndarray)
    signal_changed_reference = pyqtSignal(np.ndarray)

    signal_stop_process = pyqtSignal()
    signal_changed_fps = pyqtSignal(int)
    signal_reset_progress = pyqtSignal(int)

    def __init__(self, params):
        super().__init__(params)
        self.params: Parameter = params[0]

        # Showing bpm related parameters
        self.is_show_bpm = False
        self.is_init_bpm = True
        self.bpms = []

        # FPS related parameters
        self.time_stamp = []

        # Connect event
        self.b_start.clicked.connect(self.on_clicked_process_started)
        self.b_show_bpm.clicked.connect(self.on_clicked_show_bpm)

        self.b_np_load_video.clicked.connect(lambda: self.on_clicked_load_file(self.lbl_np_video_name))
        self.b_np_load_signal.clicked.connect(lambda: self.on_clicked_load_file(self.lbl_np_signal_name))
        self.b_file_load_video.clicked.connect(lambda: self.on_clicked_load_file(self.lbl_file_video_name))

        self.cb_sensor.clicked.connect(lambda: self.on_checked_show_signal(self.cb_sensor))
        self.cb_reference.clicked.connect(lambda: self.on_checked_show_signal(self.cb_reference))

        self.tab_src.currentChanged.connect(self.on_changed_src_tab)
        self.tab_mode.currentChanged.connect(self.on_changed_method_tab)


        # Connect custom signal
        self.signal_changed_frame.connect(self.show_frame)
        self.signal_changed_estimated.connect(self.show_estimated_signal)
        self.signal_changed_reference.connect(self.show_reference_signal)

        # Set signal to parameters
        self.params.set_signals(Parameter.SIGNAL_CHANGE_FRAME, self.signal_changed_frame)
        self.params.set_signals(Parameter.SIGNAL_CHANGE_ESTIMATED, self.signal_changed_estimated)
        self.params.set_signals(Parameter.SIGNAL_CHANGE_REFERENCE, self.signal_changed_reference)

    # -----------------------------------------------------------------------------------------
    #                                   Parameters setting
    # -----------------------------------------------------------------------------------------
    def set_parameter(self):
        # Set fps
        self.params.fps = self.spin_fps.value()
        self.params.fps_changed = True

        # Set source information
        self.params.src_info = self.get_source_information()
        self.params.src_changed = True

    def get_source_information(self):
        # Init source information
        curr_idx = self.tab_src.currentIndex()
        src = {"type": curr_idx}

        # Camera setting
        if curr_idx == 0:  # camera
            src["video"] = self.spin_camera_num.value()
            src["signal"] = True if self.cb_sensor.isChecked() else None

        # Numpy setting
        elif curr_idx == 1:
            if self.np_video_path is None:
                QMessageBox.critical(self, "Error", "비디오 파일을 선택하세요.")
                return None

            src["video"] = self.np_video_path
            src["signal"] = None
            if self.cb_reference.isChecked():
                if self.np_signal_path is None:
                    QMessageBox.critical(self, "Error", "신호 파일을 선택하세요.")
                    return None
                src["signal"] = self.np_signal_path

        # Video setting
        else:
            if self.file_video_path is None:
                QMessageBox.critical(self, "Error", "비디오 파일을 선택하세요.")
                return None
            src["video"] = self.file_video_path
            src["signal"] = None

        return src

    # -----------------------------------------------------------------------------------------
    #                                       Event listener
    # -----------------------------------------------------------------------------------------

    # Start button clicked
    def on_clicked_process_started(self):
        if self.b_start.text() == "Start":
            self.set_parameter()
            if self.params.src_info is not None:
                self.start_process()
        else:
            self.stop_process()

    def start_process(self):
        self.params.process_started = True
        self.b_start.setText('Stop')
        self.time_stamp = []

    def stop_process(self):
        self.params.process_started = False
        self.b_start.setText('Start')
        self.time_stamp = []

    # Show signal checkbox clicked
    def on_checked_show_signal(self, widget):
        if widget.isChecked():
            self.resize_window_long()
        else:
            self.resize_window_short()

    # Load buttons clicked
    def on_clicked_load_file(self, target_label):
        if target_label == self.lbl_file_video_name:
            path = QFileDialog.getOpenFileName(self, 'Load video')[0]
        else:
            path = QFileDialog.getOpenFileName(self, 'Load npy', 'D:/respiration/npy/', filter='Numpy file (*.npy)')[0]

        if path:
            if target_label == self.lbl_file_video_name:
                self.file_video_path = path
            elif target_label == self.lbl_np_video_name:
                self.np_video_path = path
            elif target_label == self.lbl_np_signal_name:
                self.np_signal_path = path
            name = os.path.split(path)[-1]
            target_label.setText(name)

    # Source tab changed
    def on_changed_src_tab(self, index):
        self.stop_process()
        if index == 0:
            self.on_checked_show_signal(self.cb_sensor)
        elif index == 1:
            self.on_checked_show_signal(self.cb_reference)
        else:
            self.resize_window_short()

    # Method tab changed
    def on_changed_method_tab(self, index):
        if index == 0:
            self.params.method = Parameter.METHOD_CLUSTERING
        elif index == 1:
            self.params.method = Parameter.METHOD_OPTICAL_FLOW
        elif index == 2:
            self.params.method = Parameter.METHOD_SEGMENTATION
        self.params.method_changed = True

    # Show bpm button
    def on_clicked_show_bpm(self):
        if self.is_show_bpm:

            self.is_show_bpm = False
            self.b_show_bpm.setText('Show')
        else:
            self.bpms = []
            for cb in self.cb_bpms:
                if cb.isChecked():
                    self.bpms.append(int(cb.text()))

            self.is_init_bpm = True
            self.is_show_bpm = True
            self.b_show_bpm.setText('Stop')

    # -----------------------------------------------------------------------------------------
    #                                      Signal processing
    # -----------------------------------------------------------------------------------------
    @pyqtSlot(np.ndarray)
    def show_frame(self, frame):
        # Calculate FPS
        self.time_stamp.append(perf_counter())
        time_length = len(self.time_stamp)
        if time_length > self.params.fps * 1:
            del self.time_stamp[0]
            time_length -= 1
        if time_length > 1:
            processing_time = self.time_stamp[-1] - self.time_stamp[0]
            fps = float(time_length-1) / processing_time
        else:
            fps = 0.0
        self.lbl_fps.setText("%02d" % round(fps))

        # Show frame
        image = cv2.resize(frame, (self.lbl_frame.width(), self.lbl_frame.height()), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]

        if len(image.shape) == 2:
            image = QtGui.QImage(image, w, h, w, QtGui.QImage.Format_Grayscale8)
        else:
            if image.shape[-1] == 1:
                image = QtGui.QImage(image, w, h, w, QtGui.QImage.Format_Grayscale8)
            else:
                image = QtGui.QImage(image, w, h, w * 3, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap(image)
        self.lbl_frame.setPixmap(pixmap)

        # Show sine signals
        if self.is_show_bpm and self.params.process_started:
            show_sin_signals(name="BPMs", bpms=self.bpms, init_data=self.is_init_bpm)
            self.is_init_bpm = False


    @pyqtSlot(np.ndarray)
    def show_estimated_signal(self, signal):
        target_widget = self.lbl_signal_estimated
        w = target_widget.width()
        h = target_widget.height()

        frame = signal_to_frame(signal, width=w, height=h, foreground=(0, 255, 0), background=(0, 0, 0))
        image = QtGui.QImage(frame, w, h, w * 3, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap(image)
        target_widget.setPixmap(pixmap)

    @pyqtSlot(np.ndarray)
    def show_reference_signal(self, signal):
        target_widget = self.lbl_signal_reference
        w = target_widget.width()
        h = target_widget.height()

        frame = signal_to_frame(signal, width=w, height=h, foreground=(0, 255, 0), background=(0, 0, 0))
        image = QtGui.QImage(frame, w, h, w * 3, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap(image)
        target_widget.setPixmap(pixmap)

    # -----------------------------------------------------------------------------------------
    #                                       Event overriding
    # -----------------------------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.params.main_loop_closed = True
        self.params.process_started = False
        event.accept()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(e)

        if e.key() == Qt.Key_1:
            print(1)
