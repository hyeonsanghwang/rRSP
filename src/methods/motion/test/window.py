import os
import cv2
import numpy as np
from time import perf_counter

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5 import QtGui

from .processing import ProcessManager
from .window_form import WindowForm
from utils.visualization.signal import signal_to_frame


class MainWindow(WindowForm):
    signal_start_process = pyqtSignal(int)
    signal_stop_process = pyqtSignal()
    signal_changed_frame = pyqtSignal(np.ndarray)
    signal_changed_estimated_signal = pyqtSignal(np.ndarray, int)
    signal_changed_reference_signal = pyqtSignal(np.ndarray, int)
    signal_changed_fps = pyqtSignal()

    def __init__(self, params):
        super().__init__()
        self.params = params

        # Init parameters
        self.np_video_path = None
        self.np_signal_path = None
        self.file_video_path = None
        self.time_stamp = []

        # Processing thread
        self.set_signals()

        # Events that require communication with processing thread
        self.b_start.clicked.connect(lambda: self.on_button_event(self.b_start))
        self.b_set_parameter.clicked.connect(lambda: self.on_button_event(self.b_set_parameter))
        self.b_set_model.clicked.connect(lambda: self.on_button_event(self.b_set_model))
        self.b_show_bpm.clicked.connect(lambda: self.on_button_event(self.b_show_bpm))
        self.b_show_original.clicked.connect(lambda: self.on_click_show_mode(self.b_show_original))
        self.b_show_resized.clicked.connect(lambda: self.on_click_show_mode(self.b_show_resized))
        self.b_show_score.clicked.connect(lambda: self.on_click_show_mode(self.b_show_score))
        self.b_show_roi.clicked.connect(lambda: self.on_click_show_mode(self.b_show_roi))

        # Tab change events
        self.tab_src.currentChanged.connect(self.on_change_data_source)
        self.tab_mode.currentChanged.connect(self.on_change_process_mode)

        # Data source events
        self.b_np_load_video.clicked.connect(lambda: self.on_click_load_file(self.lbl_np_video_name))
        self.b_np_load_signal.clicked.connect(lambda: self.on_click_load_file(self.lbl_np_signal_name))
        self.b_file_load_video.clicked.connect(lambda: self.on_click_load_file(self.lbl_file_video_name))

        # Signal events
        self.cb_sensor.clicked.connect(lambda: self.on_click_show_signal(self.cb_sensor))
        self.cb_reference.clicked.connect(lambda: self.on_click_show_signal(self.cb_reference))

        # Set default model path
        self.set_model('../../../model/detect_roi/model.h5')

    # Set methods ------------------------------------------------------------------------------------------------------
    def set_signals(self):
        self.signal_start_process.connect(self.start_processing)
        self.signal_stop_process.connect(self.stop_processing)
        self.signal_changed_frame.connect(self.show_image)
        self.signal_changed_estimated_signal.connect(self.show_reference_signal)
        self.signal_changed_reference_signal.connect(self.show_reference_signal)
        self.signal_changed_fps.connect(self.set_fps)

        self.params.set_signal(self.params.SIGNAL_START_PROCESS, self.signal_start_process)
        self.params.set_signal(self.params.SIGNAL_STOP_PROCESS, self.signal_stop_process)
        self.params.set_signal(self.params.SIGNAL_CHANGED_FRAME, self.signal_changed_frame)
        self.params.set_signal(self.params.SIGNAL_CHANGED_ESTIMATED_SIGNAL, self.signal_changed_estimated_signal)
        self.params.set_signal(self.params.SIGNAL_CHANGED_REFERENCE_SIGNAL, self.signal_changed_reference_signal)
        self.params.set_signal(self.params.SIGNAL_CHANGED_FPS, self.signal_changed_fps)

    def set_parameters(self):
        self.params.fps = self.spin_fps.value()
        self.params.process_fps = self.spin_process_fps.value()
        self.params.window_size = self.spin_window_size.value()
        self.params.resize_ratio = self.spin_resize_ratio.value()
        self.params.color_domain = self.combo_color_domain.currentIndex()
        self.params.detect_threshold = self.spin_threshold.value()
        self.params.is_changed_parameters = True

    def set_source(self):
        curr_idx = self.tab_src.currentIndex()

        if curr_idx == 0:  # camera
            self.params.src_type = ProcessManager.INPUT_TYPE_WEBCAM
            self.params.src_video = self.spin_camera_num.value()
            self.params.src_signal = self.cb_sensor.isChecked()

        elif curr_idx == 1:  # numpy
            self.params.src_type = ProcessManager.INPUT_TYPE_NUMPY

            if self.np_video_path is None:
                QMessageBox.critical(self, "Error", "비디오 파일을 선택하세요.")
                return False
            self.params.src_video = self.np_video_path

            if self.cb_reference.isChecked():
                if self.np_signal_path is None:
                    QMessageBox.critical(self, "Error", "신호 파일을 선택하세요.")
                    return False
                self.params.src_signal = self.np_signal_path
            else:
                self.params.src_signal = None

        else:  # video file
            if self.file_video_path is None:
                QMessageBox.critical(self, "Error", "비디오 파일을 선택하세요.")
                return False

            self.params.src_type = ProcessManager.INPUT_TYPE_VIDEO_FILE
            self.params.src_video = self.file_video_path
            self.params.src_signal = None
        self.params.is_changed_source = True
        return True

    def set_model(self, path):
        self.model_path = path
        self.params.src_model = path
        self.params.is_changed_model = True
        name = os.path.split(path)[-1]
        self.lbl_model_name.setText(name)

    # Widget signal events ---------------------------------------------------------------------------------------------
    def on_button_event(self, button):
        # Start button
        if button == self.b_start:
            if button.text() == "Start":
                if not self.set_source():
                    return
                self.set_parameters()
                self.on_change_process_mode(None)
                self.params.is_processing = True
                self.b_start.setText('Stop')
            else:
                self.stop_processing()

        # Set parameter button
        elif button == self.b_set_parameter:
            self.set_parameters()

        # Load model button
        elif button == self.b_set_model:
            path = QFileDialog.getOpenFileName(self, 'Load model path', '../../../model/detect_roi/', filter='Keras model (*.h5)')[0]
            if path:
                self.set_model(path)

        # Show bpm button
        elif button == self.b_show_bpm:
            bpms = []
            for i, cb in enumerate(self.cb_bpms):
                if cb.isChecked():
                    bpms.append(int(cb.text()))
            self.params.bpm_list = bpms
            self.params.is_show_bpm = True

    def on_change_process_mode(self, index):
        self.params.mode_process = self.tab_mode.currentIndex()

    def on_change_data_source(self, index):
        self.stop_processing()
        if index == 0:
            self.on_click_show_signal(self.cb_sensor)
        elif index == 1:
            self.on_click_show_signal(self.cb_reference)
        else:
            self.resize_window_short()

    def on_click_show_signal(self, widget):
        if widget.isChecked():
            self.resize_window_long()
        else:
            self.resize_window_short()

    def on_click_load_file(self, target_label):
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

    def on_click_show_mode(self, button):
        if button == self.b_show_original:
            self.params.mode_show = ProcessManager.SHOW_MODE_ORIGIANL
        elif button == self.b_show_resized:
            self.params.mode_show = ProcessManager.SHOW_MODE_RESIZED
        elif button == self.b_show_score:
            self.params.mode_show = ProcessManager.SHOW_MODE_SCORE
        elif button == self.b_show_roi:
            self.params.mode_show = ProcessManager.SHOW_MODE_ROI

    # Custom signal methods --------------------------------------------------------------------------------------------
    @pyqtSlot(int)
    def start_processing(self, value):
        self.progress_bar.setRange(0, value)
        self.progress_bar.reset()

    @pyqtSlot()
    def stop_processing(self):
        self.params.is_processing = False
        self.b_start.setText('Start')

    @pyqtSlot(np.ndarray)
    def show_image(self, frame):
        width = self.lbl_frame.width()
        height = self.lbl_frame.height()
        image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        if len(image.shape) == 2:
            image = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1], QtGui.QImage.Format_Grayscale8)
        else:
            if image.shape[-1] == 1:
                image = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1], QtGui.QImage.Format_Grayscale8)
            else:
                image = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap(image)
        self.lbl_frame.setPixmap(pixmap)

        val = self.progress_bar.value()
        self.progress_bar.setValue(val+1)

    @pyqtSlot(np.ndarray, int)
    def show_reference_signal(self, signal, target):
        if target == 0:
            widget = self.lbl_signal_estimated
        else:
            widget = self.lbl_signal_reference
        w = widget.width()
        h = widget.height()

        frame = signal_to_frame(signal, width=w, height=h, foreground=(0, 255, 0))
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap(image)
        widget.setPixmap(pixmap)

    @pyqtSlot()
    def set_fps(self):
        time = perf_counter()
        self.time_stamp.append(time)
        time_length = len(self.time_stamp)
        if time_length > self.params.fps * 1:
            del self.time_stamp[0]
            time_length -= 1

        if time_length > 1:
            processing_time = self.time_stamp[-1] - self.time_stamp[0]
            fps = float(time_length - 1) / processing_time
        else:
            fps = 0.0
        self.lbl_fps.setText('%02d' % round(fps))

    # Event methods ----------------------------------------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.params.is_started = False
        self.stop_processing()
        event.accept()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(e)

        if e.key() == Qt.Key_1:
            self.params.mode_show = ProcessManager.SHOW_MODE_ORIGIANL
        elif e.key() == Qt.Key_2:
            self.params.mode_show = ProcessManager.SHOW_MODE_RESIZED
        elif e.key() == Qt.Key_3:
            self.params.mode_show = ProcessManager.SHOW_MODE_SCORE
        elif e.key() == Qt.Key_4:
            self.params.mode_show = ProcessManager.SHOW_MODE_ROI
