import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import cv2
import numpy as np
from time import perf_counter

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QSize, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QWidget, QListWidget, QListWidgetItem, QVBoxLayout, QCheckBox, QPushButton, \
    QTabWidget, QSpinBox, QFileDialog, QMessageBox, QLabel, QProgressBar
from PyQt5 import uic, QtGui

from path import ui_path
from .processing import ProcessingThread
from .window_form import WindowForm
from utils.visualization.signal import signal_to_frame


class MainWindow(WindowForm):
    def __init__(self):
        super().__init__()

        # Init parameters
        self.np_video_path = None
        self.np_signal_path = None
        self.file_video_path = None
        self.time_stamp = []

        # Processing thread
        self.processing_thread = ProcessingThread()
        self.processing_thread.signal_get_image.connect(self.show_image)
        self.processing_thread.signal_set_fps.connect(self.set_fps)
        self.processing_thread.signal_get_estimated_signal.connect(self.show_reference_signal)
        self.processing_thread.signal_get_reference_signal.connect(self.show_reference_signal)
        self.processing_thread.signal_end_process.connect(self.stop_processing)

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
        path = '../../../model/detect_roi/model.h5'
        name = os.path.split(path)[-1]
        self.lbl_model_name.setText(name)
        self.processing_thread.set_model(path)

    def on_button_event(self, button):
        if button == self.b_start:
            if button.text() == "Start":
                input_type, args = self.get_data_source()
                if input_type is not None:
                    self.processing_thread.set_parameters(self)
                    self.processing_thread.set_camera_source(input_type, args)
                    self.processing_thread.set_process_mode(self.tab_mode.currentIndex())
                    # 더 추가
                    self.processing_thread.start()
                    self.b_start.setText('Stop')
            else:
                self.stop_processing()

        elif button == self.b_set_parameter:
            self.processing_thread.set_parameters(self)

        elif button == self.b_set_model:
            path = QFileDialog.getOpenFileName(self, 'Load model path', '../../../model/detect_roi/', filter='Keras model (*.h5)')[0]
            if path:
                name = os.path.split(path)[-1]
                self.lbl_model_name.setText(name)
                self.processing_thread.set_model(path)

        elif button == self.b_show_bpm:
            bpms = []
            for i, cb in enumerate(self.cb_bpms):
                if cb.isChecked():
                    bpms.append(int(cb.text()))
            self.processing_thread.show_bpms(bpms)

    def stop_processing(self):
        self.processing_thread.is_processing = False
        self.processing_thread.wait()
        self.b_start.setText('Start')

    def get_data_source(self):
        curr_idx = self.tab_src.currentIndex()
        if curr_idx == 0:  # camera
            camera_num = self.spin_camera_num.value()
            use_sensor = self.cb_sensor.isChecked()
            input_type = ProcessingThread.INPUT_TYPE_WEBCAM
            args = (camera_num, use_sensor)

        elif curr_idx == 1:  # numpy
            use_sensor = self.cb_reference.isChecked()
            video_path = self.np_video_path
            signal_path = self.np_signal_path
            if video_path is None:
                QMessageBox.critical(self, "Error", "비디오 파일을 선택하세요.")
                return None, None
            if use_sensor:
                if signal_path is None:
                    QMessageBox.critical(self, "Error", "신호 파일을 선택하세요.")
                    return None, None
            else:
                signal_path = None
            input_type = ProcessingThread.INPUT_TYPE_NUMPY
            args = (video_path, signal_path)

        else:  # video file
            video_path = self.file_video_path
            if video_path is None:
                QMessageBox.critical(self, "Error", "비디오 파일을 선택하세요.")
                return None, None
            input_type = ProcessingThread.INPUT_TYPE_VIDEO_FILE
            args = video_path
        return input_type, args

    def on_change_process_mode(self, index):
        self.processing_thread.set_process_mode(self.tab_mode.currentIndex())

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
            self.processing_thread.set_show_mode(self.processing_thread.SHOW_MODE_ORIGIANL)
        elif button == self.b_show_resized:
            self.processing_thread.set_show_mode(self.processing_thread.SHOW_MODE_RESIZED)
        elif button == self.b_show_score:
            self.processing_thread.set_show_mode(self.processing_thread.SHOW_MODE_SCORE)
        elif button == self.b_show_roi:
            self.processing_thread.set_show_mode(self.processing_thread.SHOW_MODE_ROI)


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
        if time_length > self.processing_thread.fps * 1:
            del self.time_stamp[0]
            time_length -= 1

        if time_length > 1:
            processing_time = self.time_stamp[-1] - self.time_stamp[0]
            fps = float(time_length - 1) / processing_time
        else:
            fps = 0.0
        self.lbl_fps.setText('%02d' % round(fps))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_processing()
        event.accept()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(e)
        if e.key() == Qt.Key_1:
            self.processing_thread.set_show_mode(self.processing_thread.SHOW_MODE_ORIGIANL)
        elif e.key() == Qt.Key_2:
            self.processing_thread.set_show_mode(self.processing_thread.SHOW_MODE_RESIZED)
        elif e.key() == Qt.Key_3:
            self.processing_thread.set_show_mode(self.processing_thread.SHOW_MODE_SCORE)
        elif e.key() == Qt.Key_4:
            self.processing_thread.set_show_mode(self.processing_thread.SHOW_MODE_ROI)
