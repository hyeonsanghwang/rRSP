import os
import numpy as np

from PyQt5.QtCore import QSize, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QWidget, QListWidget, QListWidgetItem, QVBoxLayout, QCheckBox, QPushButton, \
    QTabWidget, QSpinBox, QFileDialog, QMessageBox, QLabel
from PyQt5 import uic, QtGui

from path import ui_path
from .video_processing import ProcessingThread
from .window_form import WindowForm
from utils.visualization.signal import signal_to_frame


class MainWindow(WindowForm):
    def __init__(self):
        super().__init__()

        # Init parameters
        self.np_video_path = None
        self.np_signal_path = None
        self.file_video_path = None

        self.video_thread = None

        # Data source
        self.tab_src.currentChanged.connect(self.on_change_data_source)

        self.cb_sensor.clicked.connect(lambda: self.on_click_show_signal(self.cb_sensor))
        self.cb_reference.clicked.connect(lambda: self.on_click_show_signal(self.cb_reference))
        self.b_np_load_video.clicked.connect(lambda: self.on_click_load_np(self.lbl_np_video_name))
        self.b_np_load_signal.clicked.connect(lambda: self.on_click_load_np(self.lbl_np_signal_name))
        self.b_file_load_video.clicked.connect(lambda: self.on_click_load_video_file(self.lbl_file_video_name))

        self.b_start.clicked.connect(self.on_click_start)

        # Parameters
        self.b_set_parameter.clicked.connect(self.on_click_set_parameter)

        # Show bpms
        self.b_show_bpm.clicked.connect(self.on_click_show_bpm)

        # Processing mode
        self.tab_mode.currentChanged.connect(self.on_change_process_mode)


    def on_click_show_signal(self, widget):
        if widget.isChecked():
            self.resize_window_long()
        else:
            self.resize_window_short()

    def on_click_load_np(self, target_label):
        path = QFileDialog.getOpenFileName(self, 'Load npy', filter='Numpy file (*.npy)')[0]
        if path:
            if target_label == self.lbl_np_video_name:
                self.np_video_path = path
            else:
                self.np_signal_path = path
            name = os.path.split(path)[-1]
            target_label.setText(name)

    def on_click_load_video_file(self, target_label):
        path = QFileDialog.getOpenFileName(self, 'Load video')[0]
        if path:
            self.file_video_path = path
            name = os.path.split(path)[-1]
            target_label.setText(name)

    def on_change_data_source(self, index):
        self.stop_processing()
        if index == 0:
            self.on_click_show_signal(self.cb_sensor)
        elif index == 1:
            self.on_click_show_signal(self.cb_reference)
        else:
            self.resize_window_short()

    def on_click_start(self):
        if self.b_start.text() == 'Start':
            input_type, args = self.get_data_source()
            self.start_processing(input_type, args)
        else:
            self.stop_processing()

    def on_click_set_parameter(self):
        self.set_parameters_to_video_thread()

    def on_click_show_bpm(self):
        if self.video_thread is not None:
            bpms = []
            for i, cb in enumerate(self.cb_bpms):
                if cb.isChecked():
                    bpms.append((i + 1) * 5)
            self.video_thread.start_show_bpms(bpms)

    def on_change_process_mode(self, index):
        if self.video_thread is not None:
            self.video_thread.set_process_mode(self.tab_mode.currentIndex())


    def set_parameters_to_video_thread(self):
        fps = self.spin_fps.value()
        process_fps = self.spin_process_fps.value()
        window_size = self.spin_window_size.value()
        resize_ratio = self.spin_resize_ratio.value()
        color_domain = self.combo_color_domain.currentIndex()
        threshold = self.spin_threshold.value()
        self.video_thread.set_parameters(fps, process_fps, window_size, resize_ratio, color_domain, threshold)

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
                return
            if use_sensor:
                if signal_path is None:
                    QMessageBox.critical(self, "Error", "신호 파일을 선택하세요.")
                    return
            else:
                signal_path = None
            input_type = ProcessingThread.INPUT_TYPE_NUMPY
            args = (video_path, signal_path)

        else:  # video file
            video_path = self.file_video_path
            if video_path is None:
                QMessageBox.critical(self, "Error", "비디오 파일을 선택하세요.")
                return
            input_type = ProcessingThread.INPUT_TYPE_VIDEO_FILE
            args = video_path
        return input_type, args

    def start_processing(self, input_type, args):
        self.video_thread = ProcessingThread()
        self.set_parameters_to_video_thread()
        self.on_change_process_mode(None)
        self.video_thread.set_camera_source(input_type, args)
        self.video_thread.signal_get_image.connect(self.show_image)
        self.video_thread.signal_get_signal.connect(self.show_reference_signal)
        self.video_thread.signal_end_process.connect(self.stop_processing)
        self.video_thread.start()
        self.b_start.setText('Stop')

    def stop_processing(self):
        if self.video_thread is not None:
            self.video_thread.is_processing = False
            self.video_thread.wait()
        self.b_start.setText('Start')

    @pyqtSlot(np.ndarray)
    def show_image(self, frame):
        if len(frame.shape) == 2:
            image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.shape[1], QtGui.QImage.Format_Grayscale8)
        else:
            image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap(image)
        self.lbl_frame.setPixmap(pixmap)

    @pyqtSlot(np.ndarray)
    def show_reference_signal(self, signal):
        w = self.lbl_signal_reference.width()
        h = self.lbl_signal_reference.height()

        frame = signal_to_frame(signal, width=w, height=h, foreground=(0, 255, 0))
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap(image)
        self.lbl_signal_reference.setPixmap(pixmap)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_processing()
        event.accept()
